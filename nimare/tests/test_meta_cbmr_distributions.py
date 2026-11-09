"""Tests for the CBMR observation distributions.

The point of these is fidelity. Each distribution is a port of a likelihood that used to be a
method on the old model hierarchy, rewritten to consume the predictor's spatial patterns instead
of a hard-coded notion of groups. A port is only worth having if it computes the same number.

The reference formulas are transcribed here rather than imported, because the code they came
from has been deleted. Transcribing keeps the check independent of the shipped implementation --
the whole value of the comparison -- without keeping a dead module in the package to compare
against. The Poisson case is additionally pinned against statsmodels in
:mod:`nimare.tests.test_meta_cbmr_glm_equivalence`, which no transcription can drift from.
"""

import math

import numpy as np
import pandas as pd
import pytest

from nimare.tests.cbmr_reference import (
    negative_binomial_log_likelihood,
    negative_hessian,
)

try:
    import torch
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr.distributions import (
        ClusteredNegativeBinomial,
        DistributionError,
        NegativeBinomial,
        Poisson,
        resolve_distribution,
    )
    from nimare.meta.cbmr.predictor import CBMRPredictor, FociMarginals
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_GROUP = 5
N_VOXELS = 24
N_BASES = 4
GROUPS = ("schiz", "dep")


def _grouped_setup(n_voxels=N_VOXELS, n_bases=N_BASES):
    """Build a grouped design, a basis, coefficients, and Poisson-drawn foci."""
    rng = np.random.default_rng(17)
    n_experiments = N_PER_GROUP * len(GROUPS)
    annotations = pd.DataFrame(
        {
            "diagnosis": [g for g in GROUPS for _ in range(N_PER_GROUP)],
            "n": rng.normal(size=n_experiments),
            "vol": np.linspace(0.5, 2.0, n_experiments),
        }
    )
    raw = rng.uniform(0.05, 1.0, (n_voxels, n_bases))
    bases = raw / raw.sum(axis=1, keepdims=True)

    predictor = CBMRPredictor(bind(Design.from_formula("~ s(diagnosis) + n"), annotations), bases)
    spatial = torch.tensor(
        rng.normal(1.0, 0.2, (predictor.n_spatial_columns, n_bases)), dtype=torch.float64
    )
    global_coef = torch.tensor(rng.normal(0.0, 0.1, 1), dtype=torch.float64)

    with torch.no_grad():
        eta = predictor.linear_predictor(spatial, global_coef).numpy()
    foci = rng.poisson(np.exp(eta)).astype(float)
    return predictor, spatial, global_coef, foci, annotations


@pytest.fixture
def setup():
    """Build the grouped design most tests share."""
    return _grouped_setup()


def _legacy_pieces(predictor, foci):
    """Split foci into per-group marginals, as the reference formulas are written in terms of.

    For a grouped design the predictor's patterns *are* the groups, which is what makes the
    comparison like-for-like at all.
    """
    per_voxel, per_experiment, members = [], [], []
    for pattern in range(predictor.patterns.n_patterns):
        rows = np.flatnonzero(predictor.patterns.assignment == pattern)
        members.append(rows)
        per_voxel.append(torch.as_tensor(foci[rows].sum(axis=0), dtype=torch.float64))
        per_experiment.append(torch.as_tensor(foci[rows].sum(axis=1), dtype=torch.float64))
    return per_voxel, per_experiment, members


def _pattern_spatial_coefficients(predictor, spatial):
    """Return each pattern's effective spatial coefficients.

    A pattern's coefficient vector is its loading times the term coefficients. Needed because
    ``np.unique`` sorts the pattern rows, so pattern order is not patsy's column order --
    pairing them positionally would compare one group's coefficients against another's foci.
    """
    loadings = torch.as_tensor(predictor.patterns.loadings, dtype=spatial.dtype)
    return loadings @ spatial


def _reference_poisson(predictor, spatial, global_coef, per_voxel, per_experiment, members):
    """Poisson log-likelihood, summed group by group, up to a constant.

    The separable form the old implementation used::

        sum_v y_.v s_v + sum_i y_i. m_i - (sum_v exp(s_v))(sum_i exp(m_i))
    """
    bases = torch.as_tensor(predictor.bases, dtype=spatial.dtype)
    moderators = torch.as_tensor(predictor.global_block, dtype=spatial.dtype)
    per_pattern = _pattern_spatial_coefficients(predictor, spatial)

    total = torch.zeros((), dtype=spatial.dtype)
    for index, rows in enumerate(members):
        log_spatial = bases @ per_pattern[index]
        log_moderator = moderators[rows] @ global_coef
        total = total + (
            torch.dot(per_voxel[index], log_spatial)
            + torch.dot(per_experiment[index], log_moderator)
            - torch.exp(log_spatial).sum() * torch.exp(log_moderator).sum()
        )
    return total


def _reference_negative_binomial(
    predictor, spatial, global_coef, overdispersion, per_voxel, per_experiment, members
):
    """Moment-matched negative-binomial log-likelihood, group by group.

    A group's per-voxel total is a sum of negative-binomial variables, which has no closed form,
    so a single NB matching its first two moments is used instead.
    """
    bases = torch.as_tensor(predictor.bases, dtype=spatial.dtype)
    moderators = torch.as_tensor(predictor.global_block, dtype=spatial.dtype)
    per_pattern = _pattern_spatial_coefficients(predictor, spatial)

    total = torch.zeros((), dtype=spatial.dtype)
    for index, rows in enumerate(members):
        alpha = overdispersion[index]
        mu_spatial = torch.exp(bases @ per_pattern[index])
        mu_moderator = torch.exp(moderators[rows] @ global_coef)
        counts = per_voxel[index]

        r = 1 / alpha * mu_moderator.sum() ** 2 / (mu_moderator**2).sum()
        p = 1 / (1 + mu_moderator.sum() / (alpha * mu_spatial * (mu_moderator**2).sum()))
        total = total + torch.sum(
            torch.lgamma(counts + r)
            - torch.lgamma(counts + 1)
            - torch.lgamma(r)
            + r * torch.log(1 - p)
            + counts * torch.log(p)
        )
    return total


def _reference_clustered_negative_binomial(
    predictor, spatial, global_coef, overdispersion, per_voxel, per_experiment, members
):
    """Clustered negative-binomial log-likelihood, group by group.

    One latent effect per experiment, shared over the whole brain -- hence the brain-wide
    ``sum(mu_spatial)`` rather than a per-voxel term.
    """
    bases = torch.as_tensor(predictor.bases, dtype=spatial.dtype)
    moderators = torch.as_tensor(predictor.global_block, dtype=spatial.dtype)
    per_pattern = _pattern_spatial_coefficients(predictor, spatial)

    total = torch.zeros((), dtype=spatial.dtype)
    for index, rows in enumerate(members):
        precision = 1 / overdispersion[index]
        log_spatial = bases @ per_pattern[index]
        log_moderator = moderators[rows] @ global_coef
        mean_per_experiment = torch.exp(log_spatial).sum() * torch.exp(log_moderator)
        counts = per_experiment[index]
        n_experiments = counts.shape[0]

        total = total + (
            n_experiments * precision * torch.log(precision)
            - n_experiments * torch.lgamma(precision)
            + torch.sum(torch.lgamma(counts + precision))
            - torch.sum((counts + precision) * torch.log(mean_per_experiment + precision))
            + torch.dot(per_voxel[index], log_spatial)
            + torch.dot(counts, log_moderator)
        )
    return total


def test_patterns_recover_the_groups(setup):
    """A grouped design must give exactly one spatial pattern per group."""
    predictor = setup[0]
    assert predictor.patterns.n_patterns == len(GROUPS)
    assert not predictor.patterns.is_degenerate


def test_poisson_matches_the_reference_likelihood(setup):
    """The shipped Poisson likelihood must equal the separable formula it replaced."""
    predictor, spatial, global_coef, foci, _ = setup
    per_voxel, per_experiment, members = _legacy_pieces(predictor, foci)

    actual = Poisson().log_likelihood(predictor, spatial, global_coef, None, foci)
    expected = _reference_poisson(
        predictor, spatial, global_coef, per_voxel, per_experiment, members
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-8)


# Parametrized by name, not by class: the classes are imported inside a torch guard, so
# referencing them in a decorator would be evaluated at collection time and fail without torch.
@pytest.mark.parametrize("name", ["NegativeBinomial", "ClusteredNegativeBinomial"])
def test_overdispersed_likelihoods_match_their_references(setup, name):
    """Both overdispersion ports must equal the marginal likelihoods they came from.

    These are the models that cannot be written for a non-separable predictor at all, so the
    port only had to carry them from "group" to "pattern". Any drift would be a silently
    different model, and there is no external implementation of this moment-matched form to
    catch it.
    """
    distribution = {
        "NegativeBinomial": NegativeBinomial,
        "ClusteredNegativeBinomial": ClusteredNegativeBinomial,
    }[name]
    reference = {
        "NegativeBinomial": _reference_negative_binomial,
        "ClusteredNegativeBinomial": _reference_clustered_negative_binomial,
    }[name]

    predictor, spatial, global_coef, foci, _ = setup
    per_voxel, per_experiment, members = _legacy_pieces(predictor, foci)
    overdispersion = torch.tensor([0.03, 0.07], dtype=torch.float64)

    actual = distribution().log_likelihood(predictor, spatial, global_coef, overdispersion, foci)
    expected = reference(
        predictor, spatial, global_coef, overdispersion, per_voxel, per_experiment, members
    )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-7)


def test_overdispersion_needs_experiments_sharing_a_pattern(setup):
    """A spatial covariate leaves one experiment per pattern, which cannot fit overdispersion.

    The precise form of the constraint the old API stated as "voxelwise CBMR requires
    model=PoissonEstimator". What matters is not the moderator's kind but whether any
    experiments share a spatial map.
    """
    _, _, _, _, annotations = setup
    rng = np.random.default_rng(18)
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(n)"), annotations), raw / raw.sum(axis=1, keepdims=True)
    )

    with pytest.raises(DistributionError, match="single\n?.*experiment|single experiment"):
        NegativeBinomial().check_design(predictor)


def test_poisson_accepts_any_design(setup):
    """Poisson is defined per cell, so no design is out of reach for it."""
    _, _, _, _, annotations = setup
    rng = np.random.default_rng(19)
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(n)"), annotations), raw / raw.sum(axis=1, keepdims=True)
    )
    Poisson().check_design(predictor)  # must not raise


def test_grouped_design_passes_the_overdispersion_check(setup):
    """The check must not fire on the designs overdispersion is meant for."""
    NegativeBinomial().check_design(setup[0])
    ClusteredNegativeBinomial().check_design(setup[0])


def test_nuisance_parameter_shapes(setup):
    """Poisson owns none; the overdispersed models own one per pattern."""
    predictor = setup[0]
    n_patterns = predictor.patterns.n_patterns

    assert Poisson().n_nuisance_parameters(n_patterns) == 0
    assert Poisson().initial_nuisance(n_patterns) is None
    for distribution in (NegativeBinomial(), ClusteredNegativeBinomial()):
        assert distribution.n_nuisance_parameters(n_patterns) == n_patterns
        assert distribution.initial_nuisance(n_patterns).shape == (n_patterns,)


@pytest.mark.parametrize(
    "given,expected",
    [
        ("poisson", "Poisson"),
        ("Poisson", "Poisson"),
        ("negative_binomial", "NegativeBinomial"),
        ("clustered-negative-binomial", "ClusteredNegativeBinomial"),
    ],
)
def test_distributions_resolve_from_names(given, expected):
    """Names, classes, and instances should all be accepted."""
    assert resolve_distribution(given).name == expected
    assert resolve_distribution(type(resolve_distribution(given))).name == expected


def test_unknown_distribution_names_the_options():
    """A typo should list what was available."""
    with pytest.raises(DistributionError, match="Unknown distribution"):
        resolve_distribution("poison")
    with pytest.raises(DistributionError, match="Cannot interpret"):
        resolve_distribution(42)


def test_overdispersion_gradients_flow(setup):
    """Fitting needs gradients for the overdispersion parameters too."""
    predictor, spatial, global_coef, foci, _ = setup
    spatial = spatial.clone().requires_grad_(True)
    overdispersion = torch.tensor([0.05, 0.05], dtype=torch.float64, requires_grad=True)

    NegativeBinomial().log_likelihood(
        predictor, spatial, global_coef, overdispersion, foci
    ).backward()

    assert overdispersion.grad is not None
    assert torch.all(torch.isfinite(overdispersion.grad))
    assert torch.any(overdispersion.grad != 0)


@pytest.mark.parametrize("name", ["Poisson", "NegativeBinomial", "ClusteredNegativeBinomial"])
@pytest.mark.parametrize(
    "formula", ["~ s(diagnosis) + n", "~ 1 + sz(diagnosis)", "~ s(diagnosis) + n + exposure(vol)"]
)
def test_likelihood_stays_on_the_coefficients_device(setup, name, formula):
    """Every tensor a likelihood builds is put on the coefficients' device.

    Evaluated on torch's ``meta`` device, which has shapes but no data. A tensor left on the CPU
    fails here with the "Expected all tensors to be on the same device" error a CUDA fit raises,
    so this runs without a GPU. The second formula has no moderators, which is the path where no
    moderator coefficient exists to take the device from; the third has an exposure. The full
    predictor and mean, which the sandwich covariance reads, are checked too.
    """
    distribution = {
        "Poisson": Poisson,
        "NegativeBinomial": NegativeBinomial,
        "ClusteredNegativeBinomial": ClusteredNegativeBinomial,
    }[name]
    _, _, _, foci, annotations = setup
    rng = np.random.default_rng(5)
    predictor = CBMRPredictor(
        bind(Design.from_formula(formula), annotations), rng.uniform(0.05, 1.0, (N_VOXELS, 4))
    )
    spatial = torch.ones((predictor.n_spatial_columns, N_BASES), device="meta")
    global_coef = (
        torch.ones(predictor.n_global_columns, device="meta")
        if predictor.n_global_columns
        else None
    )
    overdispersion = torch.full((predictor.patterns.n_patterns,), 0.05, device="meta")

    actual = distribution().log_likelihood(predictor, spatial, global_coef, overdispersion, foci)

    assert actual.device.type == "meta"
    assert predictor.linear_predictor(spatial, global_coef).device.type == "meta"
    assert predictor.fitted_mean(spatial, global_coef).device.type == "meta"


@pytest.mark.parametrize("name", ["Poisson", "NegativeBinomial", "ClusteredNegativeBinomial"])
def test_conversions_are_reused_only_while_keeping_tensors(setup, monkeypatch, name):
    """A fit's repeated evaluations reuse the converted inputs; a direct call stays pure.

    A fit evaluates the likelihood hundreds of times, and on a GPU each conversion is a
    host-to-device copy, so inside ``keeping_tensors`` with :class:`FociMarginals` a second
    evaluation converts nothing. Outside it the likelihood depends only on its arguments: foci
    changed in place are seen, and nothing is left cached.
    """
    from nimare.meta.cbmr import predictor as predictor_module

    distribution = {
        "Poisson": Poisson,
        "NegativeBinomial": NegativeBinomial,
        "ClusteredNegativeBinomial": ClusteredNegativeBinomial,
    }[name]()
    predictor, spatial, global_coef, foci, _ = setup
    overdispersion = torch.full((predictor.patterns.n_patterns,), 0.05, dtype=torch.float64)
    conversions = []
    as_tensor = predictor_module._as_tensor
    monkeypatch.setattr(
        predictor_module,
        "_as_tensor",
        lambda *args, **kwargs: conversions.append(args) or as_tensor(*args, **kwargs),
    )

    def evaluate(counts):
        return distribution.log_likelihood(predictor, spatial, global_coef, overdispersion, counts)

    with predictor.keeping_tensors():
        first = evaluate(FociMarginals(predictor.patterns, foci))
        n_conversions = len(conversions)
        marginals = FociMarginals(predictor.patterns, foci)
        assert torch.equal(evaluate(marginals), first)
        # The new marginals and totals only, and for NegativeBinomial each pattern's n_j.
        new = 2 + (predictor.patterns.n_patterns if name == "NegativeBinomial" else 0)
        assert len(conversions) == n_conversions + new
        assert torch.equal(evaluate(marginals), first)
        assert len(conversions) == n_conversions + new
    assert predictor._tensors is None

    counts = foci.copy()
    assert torch.equal(evaluate(counts), first)
    counts *= 2
    assert not torch.equal(evaluate(counts), first)


# The first pair puts one pattern at the boundary a Poisson-like fit reaches. The second puts
# k * mu on both sides of the point where the likelihood switches to a series.
@pytest.mark.parametrize("overdispersion", [[1e-12, 0.05], [1e-4, 1e-3]])
def test_negative_binomial_stays_exact_near_zero_overdispersion(overdispersion):
    """The likelihood and its autodiff Hessian stay exact as an overdispersion approaches zero.

    A fit lands there when one pattern's counts are no more variable than Poisson, as these
    are. The negative-binomial size then grows like one over the overdispersion, and
    ``lgamma(y + r) - lgamma(r)`` subtracts numbers of order ``r log r`` that should cancel to
    order one. The design is kept small because the reference differentiates in mpmath.
    """
    mpmath = pytest.importorskip("mpmath")
    predictor, spatial, global_coef, foci, _ = _grouped_setup(n_voxels=8, n_bases=2)
    n_spatial = spatial.numel()

    def log_likelihood(flat):
        return NegativeBinomial().log_likelihood(
            predictor,
            flat[:n_spatial].reshape(spatial.shape),
            flat[n_spatial:],
            torch.tensor(overdispersion, dtype=torch.float64),
            foci,
        )

    flat = torch.cat([spatial.reshape(-1), global_coef])
    reference = negative_binomial_log_likelihood(predictor, overdispersion, foci, n_spatial)
    with mpmath.workdps(50):
        value = float(reference(*(mpmath.mpf(x) for x in flat.tolist())))
    # The reference leaves out the parameter-free -lgamma(y + 1).
    value -= sum(math.lgamma(y + 1) for y in predictor.patterns.marginal_by_pattern(foci).flat)
    hessian = -negative_hessian(reference, flat.tolist())

    assert log_likelihood(flat).item() == pytest.approx(value, rel=1e-13)
    np.testing.assert_allclose(
        torch.func.hessian(log_likelihood)(flat).numpy(),
        hessian,
        rtol=1e-11,
        atol=1e-11 * np.abs(hessian).max(),
    )


def test_negative_binomial_keeps_fractional_counts(setup):
    """Fractional counts have no finite sum over ``j < y``, so they keep the gamma functions."""
    predictor, spatial, global_coef, foci, _ = setup
    foci = foci * 0.5
    per_voxel, per_experiment, members = _legacy_pieces(predictor, foci)
    overdispersion = torch.tensor([0.03, 0.07], dtype=torch.float64)

    actual = NegativeBinomial().log_likelihood(
        predictor, spatial, global_coef, overdispersion, foci
    )
    expected = _reference_negative_binomial(
        predictor, spatial, global_coef, overdispersion, per_voxel, per_experiment, members
    )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-7)


def test_count_histograms_are_built_once_per_marginals(setup):
    """``n_j`` depends only on the foci, so a fit's evaluations share one copy of it."""
    predictor, _, _, foci, _ = setup
    marginals = FociMarginals(predictor.patterns, foci)

    histograms = marginals.count_histograms(torch.float64, "cpu")

    assert marginals.count_histograms(torch.float64, "cpu") is histograms
    for above, counts in zip(histograms, marginals.by_pattern):
        assert torch.equal(
            above, torch.tensor([(counts > j).sum() for j in range(int(counts.max()))]).double()
        )
