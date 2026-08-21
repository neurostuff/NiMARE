"""Observation distributions for CBMR.

The count model sitting on top of the linear predictor. Separating it from the predictor is
what lets the term structure and the distribution vary independently -- but only so far, and
the limit is mathematical rather than an implementation gap.

Poisson is defined per experiment-voxel cell, so it composes with any design. The two
overdispersion models are not: each is defined on *marginals* of a mean that factorizes as
``mu(v, i) = mu_spatial(v) * mu_moderator(i)``.

- :class:`NegativeBinomial` moment-matches a sum of negative-binomial variables across the
  experiments of a group, so it needs ``sum(mu_moderator)`` over that group.
- :class:`ClusteredNegativeBinomial` places one latent effect per experiment shared over the
  whole brain, so it needs ``sum(mu_spatial)`` over all voxels.

Neither quantity exists if the spatial and experiment parts do not separate. Within one of the
predictor's spatial patterns they always do, by construction, since every experiment sharing a
pattern shares its log-intensity map. So both port cleanly, with "group" becoming "pattern".

What they still require is *several experiments per pattern*. A design with a spatial covariate
gives every experiment its own pattern, and an overdispersion parameter estimated from one
observation is meaningless. That is the honest form of the constraint the old API expressed as
a blanket "voxelwise CBMR requires model=PoissonEstimator": not the moderator's kind, but
whether any experiments share a spatial map.
"""

import math

import numpy as np

from nimare.meta.cbmr._torch import torch
from nimare.meta.cbmr.predictor import _as_dense_array, poisson_log_likelihood


class DistributionError(ValueError):
    """Raised when a distribution cannot be used with a design."""


class Distribution:
    """Base class for CBMR observation distributions."""

    name = "distribution"

    #: Whether the distribution needs several experiments sharing each spatial pattern.
    requires_shared_patterns = False

    def n_nuisance_parameters(self, n_patterns):
        """Return the number of parameters beyond the regression coefficients."""
        return 0

    def initial_nuisance(self, n_patterns):
        """Return starting values for the nuisance parameters, or None if there are none."""
        return None

    def transform_nuisance(self, raw):
        """Map unconstrained optimizer parameters onto the values the likelihood needs.

        Keeps :meth:`log_likelihood` a function of the actual statistical parameters, so it can
        be compared against a reference implementation without knowing how the optimizer
        happens to parameterize them.
        """
        return raw

    def check_design(self, predictor):
        """Raise if this distribution cannot be used with ``predictor``'s design."""
        if not self.requires_shared_patterns:
            return
        counts = np.bincount(
            predictor.patterns.assignment, minlength=predictor.patterns.n_patterns
        )
        if np.any(counts < 2):
            raise DistributionError(
                f"{self.name} estimates one overdispersion parameter per group of experiments "
                "that share a spatial map, and this design leaves "
                f"{int((counts < 2).sum())} of {len(counts)} such groups with a single "
                "experiment, which cannot support one. This happens when a spatial term varies "
                "continuously across experiments -- for example s(age) -- so that no two "
                "experiments share a map. Use Poisson for that design, or make the term "
                "non-spatial."
            )

    def log_likelihood(self, predictor, spatial_coef, global_coef, nuisance, foci):
        """Return the log-likelihood of ``foci``, up to a parameter-free constant."""
        raise NotImplementedError


def _pattern_slices(predictor):
    """Return, per pattern, the indices of the experiments sharing it."""
    assignment = predictor.patterns.assignment
    return [
        np.flatnonzero(assignment == pattern) for pattern in range(predictor.patterns.n_patterns)
    ]


def _pattern_quantities(predictor, spatial_coef, global_coef, foci):
    """Return the per-pattern pieces the marginal likelihoods are written in terms of."""
    log_intensity = predictor.log_intensity_by_pattern(spatial_coef)
    moderator = predictor.moderator_effect(global_coef).to(spatial_coef.dtype)
    foci_per_voxel = torch.as_tensor(
        predictor.patterns.marginal_by_pattern(foci), dtype=spatial_coef.dtype
    )
    foci_per_experiment = torch.as_tensor(
        np.asarray(_as_dense_array(foci).sum(axis=1)).reshape(-1), dtype=spatial_coef.dtype
    )
    return log_intensity, moderator, foci_per_voxel, foci_per_experiment


class Poisson(Distribution):
    """Inhomogeneous Poisson process, the standard CBMR model.

    Voxel-wise foci counts are independent Poisson variables whose rate is the integral of the
    intensity over the voxel. Cheapest and most widely used, but it cannot represent
    overdispersion and may understate standard errors.
    """

    name = "Poisson"

    def log_likelihood(self, predictor, spatial_coef, global_coef, nuisance, foci):
        """Return the Poisson log-likelihood, evaluated on marginals."""
        return poisson_log_likelihood(predictor, spatial_coef, global_coef, foci)


class _OverdispersedDistribution(Distribution):
    """Shared handling of a per-pattern overdispersion parameter."""

    requires_shared_patterns = True
    initial_overdispersion = 1e-2

    def n_nuisance_parameters(self, n_patterns):
        """Return one overdispersion parameter per spatial pattern."""
        return n_patterns

    def initial_nuisance(self, n_patterns):
        """Return log-scale starting values, from the historical 1e-2 overdispersion."""
        return torch.full(
            (n_patterns,),
            math.log(self.initial_overdispersion),
            dtype=torch.float64,
            requires_grad=True,
        )

    def transform_nuisance(self, raw):
        """Exponentiate, so overdispersion stays positive however the optimizer wanders.

        An overdispersion of zero or below makes the likelihood undefined. The older code kept
        one model on a square-root scale and the other unconstrained, which admits zero and puts
        a derivative singularity there; a log scale rules both out for both models.
        """
        return torch.exp(raw)


class NegativeBinomial(_OverdispersedDistribution):
    """Negative binomial counts, allowing excess variance relative to Poisson.

    A latent gamma variable introduces independent variation at each voxel. The per-voxel total
    over a group's experiments is a sum of negative-binomial variables, which has no closed
    form, so it is approximated by moment matching -- the reason this model is written on
    marginals and needs a group of experiments sharing a spatial map.
    """

    name = "NegativeBinomial"

    def log_likelihood(self, predictor, spatial_coef, global_coef, nuisance, foci):
        """Return the moment-matched negative-binomial log-likelihood."""
        log_intensity, moderator, foci_per_voxel, _ = _pattern_quantities(
            predictor, spatial_coef, global_coef, foci
        )
        total = torch.zeros((), dtype=spatial_coef.dtype)
        for pattern, members in enumerate(_pattern_slices(predictor)):
            overdispersion = nuisance[pattern]
            intensity = torch.exp(log_intensity[pattern])
            moderator_effect = torch.exp(moderator[members])
            counts = foci_per_voxel[pattern]

            # Parameters of the single NB variable matching the first two moments of the sum.
            moderator_sum = torch.sum(moderator_effect)
            moderator_square_sum = torch.sum(moderator_effect**2)
            r = moderator_sum**2 / (overdispersion * moderator_square_sum)
            p = 1 / (1 + moderator_sum / (overdispersion * intensity * moderator_square_sum))

            total = total + torch.sum(
                torch.lgamma(counts + r)
                - torch.lgamma(counts + 1)
                - torch.lgamma(r)
                + r * torch.log(1 - p)
                + counts * torch.log(p)
            )
        return total


class ClusteredNegativeBinomial(_OverdispersedDistribution):
    """Random-effects Poisson with one latent effect per experiment.

    Unlike :class:`NegativeBinomial`, the random effect is not an independent voxel-wise
    perturbation but a characteristic of the experiment, shared across the whole brain. That
    brain-wide sum is why this model is also written on marginals.
    """

    name = "ClusteredNegativeBinomial"

    def log_likelihood(self, predictor, spatial_coef, global_coef, nuisance, foci):
        """Return the clustered negative-binomial log-likelihood."""
        log_intensity, moderator, foci_per_voxel, foci_per_experiment = _pattern_quantities(
            predictor, spatial_coef, global_coef, foci
        )
        total = torch.zeros((), dtype=spatial_coef.dtype)
        for pattern, members in enumerate(_pattern_slices(predictor)):
            precision = 1 / nuisance[pattern]
            intensity_sum = torch.sum(torch.exp(log_intensity[pattern]))
            member_moderator = moderator[members]
            member_counts = foci_per_experiment[members]
            mean_per_experiment = intensity_sum * torch.exp(member_moderator)
            n_experiments = member_counts.shape[0]

            total = total + (
                n_experiments * precision * torch.log(precision)
                - n_experiments * torch.lgamma(precision)
                + torch.sum(torch.lgamma(member_counts + precision))
                - torch.sum(
                    (member_counts + precision) * torch.log(mean_per_experiment + precision)
                )
                + torch.dot(foci_per_voxel[pattern], log_intensity[pattern])
                + torch.dot(member_counts, member_moderator)
            )
        return total


DISTRIBUTIONS = {
    "poisson": Poisson,
    "negativebinomial": NegativeBinomial,
    "clusterednegativebinomial": ClusteredNegativeBinomial,
}


def resolve_distribution(distribution):
    """Return a :class:`Distribution` instance from an instance, class, or name."""
    if isinstance(distribution, Distribution):
        return distribution
    if isinstance(distribution, type) and issubclass(distribution, Distribution):
        return distribution()
    if isinstance(distribution, str):
        key = distribution.replace("_", "").replace("-", "").lower()
        if key in DISTRIBUTIONS:
            return DISTRIBUTIONS[key]()
        raise DistributionError(
            f"Unknown distribution {distribution!r}. Choose one of "
            f"{sorted(DISTRIBUTIONS)} or pass a Distribution instance."
        )
    raise DistributionError(
        f"Cannot interpret {distribution!r} as a distribution; pass a name, class, or instance."
    )
