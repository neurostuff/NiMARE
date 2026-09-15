"""Coordinate-based effect-size meta-analysis, driven by the silence of coordinate tables.

Effect-size images supply the magnitude; coordinate tables supply only the pattern of reporting
and non-reporting, which enters a zero-inflated censored likelihood as evidence that the effect
where nothing was reported is small. Reported peak heights are not read. See
:class:`~nimare.meta.cbma.effectsize.CBES`.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from nilearn.maskers import NiftiMasker
from scipy import ndimage
from scipy.special import gammaln, ndtr
from tqdm.auto import tqdm

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.utils import (
    _calculate_cluster_measures,
    _get_mask_flat_to_masked,
    _max_statistic_maps,
    _padded_flat_to_masked,
    sphere_kernel_offsets,
)
from nimare.transforms import d_to_g, t_to_d, t_to_z, z_to_t
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _add_metadata_to_dataframe,
    _check_ncores,
    _mask_img_to_bool,
    _nlogp_to_logp_values,
    get_masker,
    get_masker_mask_image,
    mm2vox,
    validate_coordinate_spaces,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]

#: Distinct arrangements the within-analysis null needs before its p-values mean anything,
#: as a base-10 log. Ten thousand states is where the coarsest attainable p-value, 1e-4, stops
#: being the thing that limits the test; below it the null has too few states for the
#: exceedance count to separate voxels, and a map of p-values would read as inference that was
#: never done.
_MIN_NULL_STATES_LOG10 = 4.0

#: Minimum share of permutations that must attain *distinct* maximum statistics, and minimum
#: coefficient of variation among them, for the voxel-level family-wise correction to be
#: reported. Counting arrangements is not enough: a collection of two-focus studies admits 2^k
#: rearrangements and so clears ``_MIN_NULL_STATES_LOG10`` comfortably, while its permutation
#: distribution attains only a handful of values, because swapping two similar magnitudes within
#: a study barely moves the map's maximum. Measured on simulated global nulls: at two foci per
#: study the family-wise rate was 0.150 against a nominal 0.050, with 6 distinct maxima out of
#: 200 permutations and a coefficient of variation of 0.032; at six foci per study it was exactly
#: nominal, with 57 distinct maxima and a coefficient of variation of 0.106. Disabling the
#: generalized Pareto tail changed neither, so the tail fit is not implicated.
_MIN_NULL_MAXIMA_DISTINCT_FRACTION = 0.10
_MIN_NULL_MAXIMA_CV = 0.05

#: Radius, in mm, inside which a reported focus counts as the study having said something
#: about a voxel. This is the only geometry left in the model: with magnitudes no longer read
#: off coordinate tables, a focus marks a neighbourhood the study was *not* silent about, and
#: nothing else. 20 mm because prevalence rises monotonically with it at every true value and
#: no radius recovers the truth (a true 0.50 reads 0.65, 0.73, 0.76, 0.81 at 8, 14, 20, 28 mm,
#: mean absolute error 0.17 to 0.21 across the range), so the choice is not what limits the
#: estimate; 20 mm is also roughly the extent a paper's peak stands in for.
DEFAULT_COVERAGE_RADIUS_MM = 20.0

#: Default two-tailed reporting threshold, on the z scale, when a study gives no better
#: information. p < .001 uncorrected, the most common screening threshold in the literature.
DEFAULT_REPORTING_THRESHOLD_Z = 3.2905267314919255

DESIGNS = ("one-sample", "two-sample")

SELECTION_MODELS = ("zero-inflated", "none")

#: How the standard error of the pooled estimate is formed. ``"model"`` is the usual
#: inverse-variance expression, which treats the estimated heterogeneity as known;
#: ``"hksj"`` is the Hartung-Knapp-Sidik-Jonkman residual-variance form on ``n_eff - 1``
#: degrees of freedom, which does not.
SE_METHODS = ("model", "hksj")

#: How the interval on ``g`` is obtained. ``"wald"`` reports ``se`` from the observed
#: information with the prevalence profiled out by a Schur complement, referred to a *t*.
#: ``"profile"`` additionally emits ``g_lower`` and ``g_upper`` from the profile likelihood,
#: which inverts nothing and needs no degrees of freedom -- it costs roughly a second fit, and
#: is provisional until measured against the arm table in :class:`CBES`.
INTERVAL_METHODS = ("wald", "profile")

#: How uncorrected p-values are obtained. ``"permute-images"`` scrambles each image study's
#: values among its own voxels, holding the silence pattern fixed; ``"none"`` reports no
#: p-values.
NULL_METHODS = ("permute-images", "none")

#: Resolution of the permutation null histogram for |z|, and where its upper tail is clipped.
_NULL_Z_STEP = 0.01
_NULL_MAX_Z = 50.0

#: EM stops on a voxel once mu and the prevalence both move less than this in one step.
#: Measured on a whole-brain fit: tightening to 1e-5 costs 10% more runtime and moves no
#: voxel's g by more than 0.001, while loosening to 1e-3 buys only a further 11% and starts
#: to distort the map (max |dg| 0.031).
_EM_TOLERANCE = 1e-4
#: A voxel is finished when one EM sweep raises its log-likelihood by less than this, relative
#: to the likelihood itself. Needed alongside the step criterion because the mixture is only
#: weakly identified where a single study reported: mu and the prevalence then trade off along
#: a plateau that the parameter step never leaves, so the loop would otherwise run to max_iter
#: and report whichever point on it the iteration stopped at. Stopping on the likelihood makes
#: that choice reproducible; it does not make the magnitude estimable (see ``CBES.max_iter``).
_EM_LOGLIK_TOLERANCE = 1e-6

#: Rebuild the working set only once this fraction of it has settled, so that compaction
#: (which touches every pair) is amortized rather than run every iteration.
_EM_COMPACTION_FRACTION = 0.05

#: Permutations used to fix the cluster-forming threshold before the main null loop: this
#: fraction of the run, but never fewer than the minimum. The pilot costs a few percent of the
#: main loop where a second full pass would cost 100%.
_NULL_PILOT_ITERS = 20
_NULL_PILOT_DIVISOR = 20

#: Clamp before a logarithm: guards only against a p of exactly zero.
_LOGP_FLOOR = 1e-300
#: Floor on a probability used as a denominator or a mixture responsibility.
_PROBABILITY_FLOOR = 1e-12

#: Prevalence is held inside (0, 1) by this margin: at exactly 0 or 1 the mixture degenerates
#: and the responsibilities stop being informative.
_PREVALENCE_CLAMP = 1e-4
# The profile-likelihood interval. _PROFILE_CRITICAL is the 95% point of a chi-square on one
# degree of freedom, which is what twice the log-likelihood deficit is compared against. The
# multiples are of the reported standard error, which sets where to look for the crossing and
# nothing else; they run past 8 because a voxel whose prevalence is barely identified has a very
# flat profile, and stopping early would report a bound that is really an artifact of the grid.
_PROFILE_CRITICAL = 3.841459
_PROFILE_MULTIPLES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0)
_PROFILE_INNER_ITERS = 10
_PROFILE_FALLBACK_SCALE = 1.0

#: Voxels x studies held in memory at once by the selection-model fit, which allocates several
#: arrays of this size per iteration.
_SELECTION_CHUNK_ELEMENTS = 2_000_000

#: Faces-only connectivity for cluster labelling, matching Nilearn and the other CBMA
#: estimators.
_CLUSTER_CONNECTIVITY = ndimage.generate_binary_structure(rank=3, connectivity=1)

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------- numerical helpers


def _normal_pdf(x):
    """Evaluate the standard normal density in place.

    ``scipy.stats.norm.pdf`` is ~3x slower on large arrays, and the EM below evaluates this on
    an (n_studies, n_voxels) block on every iteration.

    Written in place: on the arrays this sees, the three temporaries the naive expression
    allocates cost more than the exponential.
    """
    out = x * x
    out *= -0.5
    np.exp(out, out=out)
    out *= _INV_SQRT_2PI
    return out


# ---------------------------------------- null histograms and cluster measures


def _null_bin_edges():
    """Bin edges for the permutation null histogram of |z|."""
    return np.arange(0.0, _NULL_MAX_Z + _NULL_Z_STEP, _NULL_Z_STEP)


def _stat_from_histogram(p_value, histogram):
    """Smallest ``|z|`` whose null p-value is at or below ``p_value``.

    Used to turn a cluster-forming p threshold into the statistic threshold the clusters are
    actually defined on. Pooling every voxel of every iteration into one histogram assumes the
    voxels share a null, which they do not exactly -- but a cluster-forming threshold has to be
    a single number, so it is the one place that assumption is unavoidable.
    """
    total = histogram.sum()
    if total <= 0:
        return np.inf
    survival = np.concatenate([np.cumsum(histogram[::-1])[::-1], [0.0]])
    p_by_bin = (survival + 1.0) / (total + 1.0)
    reached = np.flatnonzero(p_by_bin <= p_value)
    if not reached.size:
        return np.inf
    return float(reached[0] * _NULL_Z_STEP)


def _observed_cluster_measures(volume, threshold):
    """Per-voxel cluster size and mass of the cluster each voxel belongs to.

    Voxels below threshold, and voxels in no cluster, get zero -- so they take the largest
    corrected p-value the null can give.
    """
    sizes = np.zeros(volume.shape, dtype=float)
    masses = np.zeros(volume.shape, dtype=float)
    excursion = np.abs(volume) > threshold
    if not excursion.any():
        return sizes, masses

    mass_values = np.abs(volume) - threshold
    for polarity in (volume > threshold, volume < -threshold):
        if not polarity.any():
            continue
        labels, n_clusters = ndimage.label(polarity, _CLUSTER_CONNECTIVITY)
        if not n_clusters:
            continue
        cluster_ids = np.arange(1, n_clusters + 1)
        cluster_sizes = np.bincount(labels.ravel())[1:]
        cluster_masses = np.asarray(ndimage.sum(mass_values, labels=labels, index=cluster_ids))
        inside = labels > 0
        sizes[inside] = cluster_sizes[labels[inside] - 1]
        masses[inside] = cluster_masses[labels[inside] - 1]
    return sizes, masses


# -------------------------------------------------- selection-model likelihood


def _censoring_terms(mu, cutoff_scaled, twice_cutoff_scaled, inv_sigma, inv_sigma_sq, sign):
    r"""Probability of each observed *reporting indicator*, and the pieces of its derivatives.

    A coordinate table carries one bit per study per voxel: the study reported something near
    here, or it did not. Both values of that bit are informative, and the two are complementary
    probabilities of the same event, so they are computed together and told apart by ``sign``:
    ``+1`` for a silent pair, whose probability is :math:`P(|g| < c \mid \mu)`, and ``-1`` for
    a pair that reported, whose probability is :math:`1 - P(|g| < c \mid \mu)` with its height
    discarded.

    **The ``-1`` limb's probability is overstated, and the error peaks mid-window.** A paper
    reports a voxel only if it cleared ``c`` *and* the value there was a local maximum, which
    this :math:`P(|g| \\ge c)` does not require. Measured against a known truth, the model's
    reporting rate exceeded the observed one by 1.00, 1.78, 1.08 and 1.13 at true ``g`` of 0.2,
    0.4, 0.6 and 0.8 -- non-monotone, so no uniform reweighting of the limb can absorb it, and
    raising its probability to a power was measured making every focus worse. Correcting it needs
    the survival of a suprathreshold local maximum, and that needs a field smoothness this model
    does not carry. See ``CBES`` under "Why the indicator and not the heights".

    **Dropping the ``-1`` pairs biases the magnitude down, and hard.** They used to contribute
    nothing at all, on the reasoning that a coordinate carries no usable height -- but omitting
    them leaves the *silent* pairs as the only evidence about the indicator, so the model reads
    the observed silence fraction against a denominator that excludes every study that reported.
    On a one-voxel likelihood with the truth known exactly, 20 studies of which 2 supply images
    and a cutoff of 0.60 g, that returned a mean :math:`\hat\mu` of 0.351 for a true 0.500
    (rmse 0.192); with the indicator restored, 0.524 (rmse 0.129).

    Returned as one dict because the E step and the M step both need these at the same ``mu``.
    This is the most expensive single function in the estimator, over an array with one entry
    per censored ``(study, voxel)`` pair; the cost is spread over four memory-bound kernels, so
    what pays is removing a pass rather than speeding one up.

    Everything that does not move between EM iterations is passed in already divided: ``mu`` is
    the only argument that changes, so ``cutoffs / sigma`` and the reciprocals are hoisted to
    the caller. The lower tail looks negligible and is not -- at a typical cutoff it is a third
    of the score's numerator -- so it is kept. The arithmetic writes into its own temporaries
    wherever numpy allows it, each avoided temporary being hundreds of megabytes of traffic.
    """
    # upper = (c - mu) / sigma;  lower = (-c - mu) / sigma = upper - 2c/sigma
    upper = mu * -inv_sigma
    upper += cutoff_scaled
    lower = upper - twice_cutoff_scaled

    silent_prob = ndtr(upper)
    silent_prob -= ndtr(lower)

    pdf_upper = _normal_pdf(upper)
    pdf_lower = _normal_pdf(lower)

    # d/dmu and d2/dmu2 of P(silent), before normalising by the probability of the event that
    # was actually observed.
    first = pdf_upper - pdf_lower
    first *= -inv_sigma
    # Fold the pdfs into the limits in place: neither is needed afterwards.
    pdf_upper *= upper
    pdf_lower *= lower
    second = pdf_upper - pdf_lower
    second *= -inv_sigma_sq

    # The complement for the pairs that reported: probability and both derivatives flip.
    prob = np.where(sign > 0, silent_prob, 1.0 - silent_prob)
    np.clip(prob, _PROBABILITY_FLOOR, None, out=prob)
    score = first * sign
    score /= prob
    d2_over_prob = second * sign
    d2_over_prob /= prob

    return {"prob": prob, "score": score, "d2_over_prob": d2_over_prob}


def _observed_information(
    *,
    width,
    pi,
    reporting,
    silent,
    mu,
    censoring,
    identified=None,
):
    r"""Observed information for :math:`\mu`, after profiling out the prevalence.

    The EM's own curvature is not this. ``_mu_derivatives`` differentiates the *Q function*,
    with the responsibilities held fixed, so for one observation it keeps :math:`r h` and drops
    the :math:`r(1-r)s^2` that appears when the responsibility is allowed to move with
    :math:`\mu`:

    .. math::

        \frac{\partial^2 \ell}{\partial \mu^2} = r h + r(1-r) s^2,

    where :math:`r` is the posterior probability that the observation came from the active
    component, :math:`s` that component's score in :math:`\mu` and :math:`h` its second
    derivative. Dropping a positive term from a negative curvature overstates the information,
    so the reported error was too small -- measured at 62.5% to 89.8% coverage of nominal-95%
    intervals, and not improving with more studies. This is the missing-information problem of
    :footcite:t:`louis1982finding`, not a degrees-of-freedom adjustment, which is why referring
    ``se`` to a ``t`` could not repair it.

    The prevalence is estimated too, so its uncertainty belongs in :math:`\mu`'s. Writing
    :math:`f_1, f_0` for the two component densities and :math:`f` for the mixture, the cross
    and prevalence blocks reduce to functions of :math:`r` and :math:`\pi` alone, because
    :math:`f_1/f = r/\pi` and :math:`f_0/f = (1-r)/(1-\pi)`:

    .. math::

        \frac{\partial^2\ell}{\partial\mu\,\partial\pi} = \frac{s\,r(1-r)}{\pi(1-\pi)},
        \qquad
        \frac{\partial\ell}{\partial\pi} = \frac{r}{\pi} - \frac{1-r}{1-\pi}.

    What is returned is a triple. The first is the Schur complement
    :math:`I_{\mu\mu} - I_{\mu\pi}^2 / I_{\pi\pi}`, so the caller inverts a scalar for
    :math:`\mu`'s error. Voxels where that is not positive are left to the caller as having no
    usable information.

    The second is the variance of the *product* :math:`\mu\pi`, which is what ``g_marginal``
    reports and the only quantity here that an image-based meta-analysis also estimates. It
    needs the whole inverse rather than the Schur complement, because the two factors covary:
    by the delta method, with :math:`D = I_{\mu\mu} I_{\pi\pi} - I_{\mu\pi}^2`,

    .. math::

        \operatorname{Var}(\mu\pi) =
            \frac{\pi^2 I_{\pi\pi} + \mu^2 I_{\mu\mu} - 2\mu\pi I_{\mu\pi}}{D}.

    The cross term turns out to be small in this model -- measured under 1% of the variance
    across configurations with :math:`\pi` from 0.20 to 1.00 -- so the simpler independent sum
    :math:`\pi^2\operatorname{Var}(\mu) + \mu^2\operatorname{Var}(\pi)` would have been adequate
    in every case tried. The reason is that :math:`I_{\mu\pi}` carries a factor :math:`r(1-r)`,
    and with a threshold that separates the components cleanly the responsibilities sit near 0
    or 1, so the two factors are nearly orthogonal: the reported values identify :math:`\mu`
    and the count of silent studies identifies :math:`\pi`. The full inverse is used anyway
    because it is the correct expression and costs nothing, and because nothing guarantees that
    near-orthogonality on a collection whose thresholds sit close to its effects.

    The third is the share of :math:`I_{\mu\mu}` contributed by the coordinate indicators
    rather than by the images' values, which costs nothing because the two are accumulated
    separately anyway. It is the one diagnostic a reader needs and cannot otherwise get: at 0
    the images carry the estimate alone and the tables changed nothing at this voxel, at 1 the
    indicators carry it. Every caveat in :class:`CBES` -- the prevalence-1 regime, the
    unexplained disagreement between collections, the over-shrinkage at the window -- is about
    when to trust the coordinate channel, and this says where it is even acting.

    The :math:`\pi` block is exact here rather than an approximation. The mixture density is
    linear in :math:`\pi`, so :math:`\partial^2 \log f / \partial\pi^2 = -(\partial \log f /
    \partial\pi)^2` identically, and the outer product of scores *is* the negative second
    derivative -- which is why it can sit in the same matrix as the Hessian-based
    :math:`\mu` block without mixing two different estimators of information.

    References
    ----------
    .. footbibliography::
    """
    safe_pi = np.clip(pi, _PREVALENCE_CLAMP, 1.0 - _PREVALENCE_CLAMP)

    def responsibility_of(voxel, active_density, null_density):
        """Posterior probability of the active component at the ``(mu, pi)`` being reported.

        Recomputed rather than taken from the last E step: the loop takes its step after that
        step, so the stored responsibilities belong to a different ``mu`` than the one being
        written out, and at ``max_iter`` that gap is not small.
        """
        pi_voxel = safe_pi[voxel]
        active = pi_voxel * active_density
        return active / (active + (1.0 - pi_voxel) * null_density + _LOGP_FLOOR)

    def blocks(voxel, weight, responsibility, score, hessian):
        """Accumulate the three information blocks for one kind of observation."""
        r = responsibility
        spread = r * (1.0 - r)
        pi_voxel = safe_pi[voxel]
        i_mu = -np.bincount(
            voxel, weights=weight * (r * hessian + spread * score**2), minlength=width
        )
        cross = -np.bincount(
            voxel,
            weights=weight * score * spread / (pi_voxel * (1.0 - pi_voxel)),
            minlength=width,
        )
        pi_score = r / pi_voxel - (1.0 - r) / (1.0 - pi_voxel)
        i_pi = np.bincount(voxel, weights=weight * pi_score**2, minlength=width)
        return i_mu, cross, i_pi

    def certain_where_unidentified(voxel, responsibility):
        """Force the responsibility to exactly 1 where the prevalence is held at 1.

        Not merely close to 1. With ``pi`` clamped a hair below 1 the responsibility comes out
        a hair below it too, which leaves ``r(1 - r)`` small but non-zero and the collapse to
        the plain likelihood approximate -- about 1 part in 10^4 of the error. Exactly 1 makes
        the cross block vanish identically, so a mixture fit with no indicator anywhere and a
        non-mixture fit of the same images return the same numbers rather than nearly the same.
        """
        if identified is None:
            return responsibility
        return np.where(identified[voxel], responsibility, 1.0)

    density_effect = (
        _normal_pdf((reporting.g - mu[reporting.voxel]) / reporting.sigma) / reporting.sigma
    )
    i_mu, cross, i_pi = blocks(
        reporting.voxel,
        reporting.weight,
        certain_where_unidentified(
            reporting.voxel,
            responsibility_of(reporting.voxel, density_effect, reporting.density_null),
        ),
        (reporting.g - mu[reporting.voxel]) * reporting.precision,
        -reporting.precision,
    )
    censor_score = censoring["score"]
    add_mu, add_cross, add_pi = blocks(
        silent.voxel,
        silent.weight,
        certain_where_unidentified(
            silent.voxel,
            responsibility_of(silent.voxel, censoring["prob"], silent.prob_event_null),
        ),
        censor_score,
        censoring["d2_over_prob"] - censor_score**2,
    )
    # The share of the information about mu that came from the coordinate tables, before the
    # two are summed. Every caveat in ``CBES`` turns on a question a reader cannot otherwise
    # answer -- is the coordinate channel doing anything *here*? -- and this answers it
    # directly: 0 means the images carry the estimate alone and the tables changed nothing at
    # this voxel, 1 means the indicators carry it.
    total_mu = i_mu + add_mu
    coordinate_share = np.divide(add_mu, total_mu, out=np.zeros(width), where=np.abs(total_mu) > 0)
    np.clip(coordinate_share, 0.0, 1.0, out=coordinate_share)

    i_mu = total_mu
    cross += add_cross
    i_pi += add_pi

    profiled = np.where(i_pi > 0, i_mu - cross**2 / np.where(i_pi > 0, i_pi, 1.0), i_mu)

    # Variance of mu*pi by the delta method, from the full 2x2 inverse.
    determinant = i_mu * i_pi - cross**2
    usable = (determinant > 0) & (i_mu > 0) & (i_pi > 0)
    numerator = safe_pi**2 * i_pi + mu**2 * i_mu - 2.0 * mu * safe_pi * cross
    marginal_variance = np.full(width, np.inf, dtype=float)
    np.divide(
        numerator,
        determinant,
        out=marginal_variance,
        where=usable & (numerator > 0),
    )
    marginal_variance[~(usable & (numerator > 0))] = np.inf

    if identified is not None:
        # Where the prevalence is held at 1 rather than fitted there is nothing to profile out,
        # so the information about mu is the plain block and mu*pi is mu. The limit cannot be
        # left to the algebra: with pi clamped just below 1, ``(1 - r) / (1 - pi)`` tends to the
        # ratio of the two component densities rather than to zero, so the cross block stays
        # O(1) and the Schur complement would still subtract a term that no parameter earned.
        plain = np.divide(1.0, i_mu, out=np.full(width, np.inf), where=i_mu > 0)
        profiled = np.where(identified, profiled, i_mu)
        marginal_variance = np.where(identified, marginal_variance, plain)

    return profiled, marginal_variance, coordinate_share


def _mu_derivatives(
    *,
    width,
    mu_rep,
    g_rep,
    precision_rep,
    rep_voxel,
    weight_rep,
    sil_voxel,
    weight_sil,
    censoring,
):
    """Voxelwise first and second derivatives of the weighted log-likelihood in ``mu``.

    Contributions arrive as one entry per weighted ``(study, voxel)`` pair and are summed onto
    voxels with :func:`numpy.bincount`.
    """
    score = np.bincount(
        rep_voxel, weights=weight_rep * (g_rep - mu_rep) * precision_rep, minlength=width
    )
    curvature = -np.bincount(rep_voxel, weights=weight_rep * precision_rep, minlength=width)

    censor_score = censoring["score"]
    score += np.bincount(sil_voxel, weights=weight_sil * censor_score, minlength=width)
    curvature += np.bincount(
        sil_voxel,
        weights=weight_sil * (censoring["d2_over_prob"] - censor_score**2),
        minlength=width,
    )
    return score, curvature


# -------------------------------------------------------- reported-peak theory


def reported_minimum_z(coordinates):
    r"""Smallest reported ``|statistic|`` per study, on the z scale, as a bound on its cutoff.

    This is the one thing the reported statistics are still read for, and it is not a
    magnitude: anything a study reported *cleared* that study's threshold, so

    .. math:: c_k \le \min_j |z_{kj}|

    is a hard inequality rather than an inference. It can clamp an assumed threshold downward
    and can never place one above the truth, which is what separates it from the retired
    ``"study-min"`` rule -- that one tried to *recover* :math:`c_k` by undoing the order
    statistic for the number of peaks, and overshot a cluster-forming cut by about 1 z.

    A ``t`` is mapped to the z with the same tail probability, so a collection that tabulates
    either is handled. Returns an empty series when no statistic column is present, which is a
    perfectly ordinary coordinate table.
    """
    if "z_stat" in coordinates.columns:
        values = np.abs(np.asarray(coordinates["z_stat"], dtype=float))
    elif "t_stat" in coordinates.columns and "sample_size" in coordinates.columns:
        sizes = np.asarray(coordinates["sample_size"], dtype=float)
        values = np.abs(
            t_to_z(np.abs(np.asarray(coordinates["t_stat"], dtype=float)), sizes - 1.0)
        )
    else:
        return pd.Series(dtype=float)

    usable = np.isfinite(values) & (values > 0)
    if not usable.any():
        return pd.Series(dtype=float)
    frame = pd.DataFrame({"id": np.asarray(coordinates["id"])[usable], "z": values[usable]})
    return frame.groupby("id")["z"].min()


def reporting_cutoff_to_g(cutoff_z, sample_size, design="one-sample"):
    r"""Convert a study's reporting threshold from the z scale onto the effect-size scale.

    This is the *only* place a reported test statistic's scale enters the model. Peak heights
    are not read; what is read is each study's reporting threshold, and the censored likelihood
    needs it on the same axis as the effect sizes it is censoring -- "a study of this size,
    applying this cut, would have reported an effect of at least *this* many g".

    Parameters
    ----------
    cutoff_z : array_like
        Two-tailed reporting threshold on the z scale, one per study.
    sample_size : array_like
        Total sample size of each study. ``"two-sample"`` assumes equal groups.
    design : {"one-sample", "two-sample"}, default="one-sample"
        Design behind the collection.

    Returns
    -------
    :class:`numpy.ndarray`
        The same thresholds as Hedges' :math:`g`.

    Notes
    -----
    The ``z`` is treated as a p-value-preserving image of a *t* on ``n - 1`` (or ``n - 2``)
    degrees of freedom and mapped back before conversion, which is what neuroimaging software
    usually produces, and then taken through :func:`~nimare.transforms.t_to_d` and
    :func:`~nimare.transforms.d_to_g` -- the same path the image-based estimators take, so the
    bound and the values are on one scale.

    **The assumed degrees of freedom are load-bearing, because a threshold sits far into the
    tail where that map is steep.** Holding ``n`` at 30 and varying only the assumed residual
    degrees of freedom:

    ============  =======  =======  ========  =========  ======
    cutoff z       df=29    df=60    df=120    df=1000   spread
    ============  =======  =======  ========  =========  ======
    3.30           0.653    0.626     0.614      0.604    1.08x
    4.00           0.830    0.776     0.752      0.733    1.13x
    5.00           1.133    1.009     0.959      0.918    1.23x
    6.00           1.522    1.273     1.178      1.105    1.38x
    ============  =======  =======  ========  =========  ======

    The effective degrees of freedom of a published map are frequently *above* ``n - 1`` --
    variance smoothing raises them, and some mixed-effects tools do that deliberately -- and
    papers seldom state them, so a threshold read off a published z is likely to be placed a
    little too high, which makes a silence look less surprising than it was.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}; got {design!r}.")

    cutoff_z = np.abs(np.asarray(cutoff_z, dtype=float))
    sample_size = np.asarray(sample_size, dtype=float)

    min_n = 4 if design == "one-sample" else 5
    if np.any(sample_size < min_n):
        raise ValueError(
            f"A {design} effect size needs at least {min_n} subjects per study; got a minimum "
            f"of {np.nanmin(sample_size):g}."
        )

    if design == "one-sample":
        t = z_to_t(cutoff_z, sample_size - 1)
        return np.asarray(d_to_g(t_to_d(t, sample_size), sample_size), dtype=float)

    dof = sample_size - 2
    t = z_to_t(cutoff_z, dof)
    half = sample_size / 2.0
    d = t * np.sqrt(1.0 / half + 1.0 / half)
    return (1.0 - 3.0 / (4.0 * dof - 1.0)) * d


def null_effect_variance(sample_size, design="one-sample"):
    """Return the sampling variance of Hedges' g under a null effect, for a silent study.

    A study that reported nothing supplies no effect size, but its *precision* is still known
    from its sample size. That precision is the whole of what a silence contributes, so it is
    the quantity this estimator is built around.

    Written out here rather than obtained by converting a zero statistic, because the conversion
    this used to call existed only to put *reported peak heights* on the effect-size scale, and
    reported peak heights no longer enter the model.

    The two designs do not share a formula, which is worth stating because assuming they did
    got this wrong once. One-sample follows :func:`~nimare.transforms.d_to_g`, whose variance is
    exact rather than the usual approximation -- ``(N - 1)(1 + N d**2) h**2 / (N (N - 3)) - d**2``
    -- and at ``d = 0`` leaves ``(N - 1) h**2 / (N (N - 3))``. Two-sample uses the approximate
    form ``h**2 (1/n1 + 1/n2 + d**2 / (2(n1 + n2)))``, which at ``d = 0`` leaves ``h**2 * 4/N``.
    The first is about 12% larger than the naive ``h**2 / N`` at ``N = 20``.
    """
    n = np.asarray(sample_size, dtype=float)
    if design == "two-sample":
        dof = n - 2.0
        correction = 1.0 - 3.0 / (4.0 * dof - 1.0)
        half = n / 2.0
        return correction**2 * (1.0 / half + 1.0 / half)
    dof = n - 1.0
    correction = 1.0 - 3.0 / (4.0 * dof - 1.0)
    return (n - 1.0) * correction**2 / (n * (n - 3.0))


# --------------------------------------------------------------- heterogeneity


def _null_maxima_diagnostics(max_values):
    """Return ``(usable, n_distinct, cv)`` for a permutation distribution of maxima.

    The family-wise correction refers the observed maximum to this distribution, so what matters
    is not how many arrangements the collection admits but how much the arrangements actually
    move the maximum. A null attaining six values cannot resolve a p-value and, more to the
    point, understates the spread of the quantity it is standing in for.

    Both statistics must be low before the null is called unusable, because either alone can be
    low for a benign reason -- a coarse but wide distribution still separates the observed value
    from the bulk, and a fine but narrow one can arise when every study reports many foci of
    similar size. In the measurements behind the thresholds the two moved together.
    """
    values = np.asarray(max_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return False, int(values.size), 0.0
    n_distinct = int(np.unique(values).size)
    mean = float(np.mean(values))
    cv = float(np.std(values) / abs(mean)) if mean != 0.0 else 0.0
    sparse = n_distinct < max(2, int(np.ceil(_MIN_NULL_MAXIMA_DISTINCT_FRACTION * values.size)))
    narrow = cv < _MIN_NULL_MAXIMA_CV
    return (not (sparse and narrow)), n_distinct, cv


def _hartung_knapp_se(*, g_hat, sum_a, sum_a_g2, n_eff, covered, fallback):
    r"""Hartung-Knapp-Sidik-Jonkman standard error of a kernel-weighted pooled estimate.

    The model-based SE treats :math:`\hat{\tau}^2` as if it were the true heterogeneity, so its
    intervals are too short exactly when heterogeneity is large and the studies are few. HKSJ
    replaces it with the weighted spread of the studies about the pooled value,

    .. math::

        \mathrm{SE}^2 = \frac{\sum_k a_k (g_k - \hat{g})^2}{(k_{\mathrm{eff}} - 1)\sum_k a_k},
        \qquad a_k = \frac{w_k}{s^2_k + \tau^2},

    on :math:`k_{\mathrm{eff}} - 1` degrees of freedom, which gives much better interval
    coverage than the model SE when heterogeneity is large and the studies are few.

    :math:`k_{\mathrm{eff}}` is Kish's :math:`(\sum w)^2 / \sum w^2` -- the ``n_eff`` map --
    and not :math:`\sum w`. The two agree when every weight is one, but only Kish's form is
    invariant to rescaling the weights: at a voxel reached only by distant foci the weights sum
    to less than one, and using that as a study count sends the degrees of freedom to zero.
    Voxels with no effective spread to measure keep the model-based value.
    """
    se = np.array(fallback, dtype=float, copy=True)
    usable = covered & (n_eff > 1.0) & (sum_a > 0)
    if not np.any(usable):
        return se
    # sum a (g - ghat)^2, from the identity noted at the call site. Clipped at zero: the two
    # terms are close where the studies agree, so rounding can make the difference negative.
    residual = np.clip(sum_a_g2[usable] - g_hat[usable] ** 2 * sum_a[usable], 0.0, None)
    se[usable] = np.sqrt(residual / ((n_eff[usable] - 1.0) * sum_a[usable]))
    return se


def _local_dersimonian_laird(sum_w, sum_a, sum_a2, sum_ag, sum_ag2, sum_w2_over_s2, n_studies):
    r"""Kernel-weighted DerSimonian-Laird estimate of between-study heterogeneity.

    With local weights :math:`w_k` and within-study variances :math:`s^2_k`, write
    :math:`a_k = w_k / s^2_k` and :math:`Q = \sum_k a_k (g_k - \bar{g}_a)^2`. Then

    .. math::

        E[Q] = \Big(\sum_k w_k - \tfrac{\sum_k a_k^2 s^2_k}{\sum_k a_k}\Big)
               + \tau^2 \Big(\sum_k a_k - \tfrac{\sum_k a_k^2}{\sum_k a_k}\Big),

    and equating :math:`Q` to its expectation gives the moment estimator below. Setting every
    :math:`w_k = 1` recovers the classical DerSimonian-Laird estimator exactly, which is the
    check :func:`~nimare.tests.test_meta_effectsize.test_local_dl_reduces_to_dersimonian_laird`
    makes.
    """
    tau2 = np.zeros_like(sum_a)
    usable = (n_studies >= 2) & (sum_a > 0)
    if not np.any(usable):
        return tau2

    sum_a_u = sum_a[usable]
    q_stat = sum_ag2[usable] - (sum_ag[usable] ** 2) / sum_a_u
    expected_q = sum_w[usable] - sum_w2_over_s2[usable] / sum_a_u
    scale = sum_a_u - sum_a2[usable] / sum_a_u

    positive = scale > 0
    out = np.zeros_like(sum_a_u)
    out[positive] = (q_stat[positive] - expected_q[positive]) / scale[positive]
    tau2[usable] = np.maximum(out, 0.0)
    return tau2


# ----------------------------------------------------------- estimator support


def _validate_options(
    *,
    design,
    tau2_method,
    selection_model,
    null_method,
    threshold,
    se_method,
    interval,
):
    """Reject unusable option combinations at construction, not at fit time.

    Kept out of ``__init__`` so that reads as the list of what the estimator stores. Everything
    here is a membership or range check on a single argument; anything needing the data belongs
    in :meth:`CBES._fit`.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}; got {design!r}.")
    if tau2_method not in ("dl", "none"):
        raise ValueError(f"tau2_method must be 'dl' or 'none'; got {tau2_method!r}.")
    if selection_model not in SELECTION_MODELS:
        raise ValueError(
            f"selection_model must be one of {SELECTION_MODELS}; got {selection_model!r}."
        )
    if null_method not in NULL_METHODS:
        raise ValueError(f"null_method must be one of {NULL_METHODS}; got {null_method!r}.")
    if se_method not in SE_METHODS:
        raise ValueError(f"se_method must be one of {SE_METHODS}; got {se_method!r}.")
    if interval not in INTERVAL_METHODS:
        raise ValueError(f"interval must be one of {INTERVAL_METHODS}; got {interval!r}.")
    if interval == "profile" and selection_model != "zero-inflated":
        # Refused rather than ignored. Without the mixture there is no prevalence to maximise
        # out, so the "profile" would be the plain likelihood and its interval would differ
        # from the reported se only by the arithmetic used to find it -- which would look like
        # a second opinion while being the same one.
        raise ValueError(
            "interval='profile' needs selection_model='zero-inflated'. With no mixture there "
            "is no prevalence to profile out, so the interval would restate the reported se "
            "rather than provide an independent one."
        )
    if se_method == "hksj" and selection_model != "none":
        # Refused rather than ignored. HKSJ corrects the inverse-variance SE of a weighted
        # mean, and under the selection model that SE is discarded: the reported one comes
        # from the curvature of the censored likelihood at the fitted mu, a different
        # estimator that this correction does not apply to. Accepting the combination would
        # silently return the uncorrected value.
        raise ValueError(
            "se_method='hksj' needs selection_model='none'. HKSJ corrects the "
            "inverse-variance standard error of the pooled mean, but the zero-inflated "
            "selection model reports the curvature of the censored likelihood instead, "
            "which this correction does not apply to."
        )

    # A string is a keyword or the name of a metadata field holding per-study thresholds, and
    # which one it is cannot be known until the collection is in hand.
    if not isinstance(threshold, str) and threshold is not None and not np.isscalar(threshold):
        raise ValueError(
            "threshold must be a metadata field name, a number, or None; got " f"{threshold!r}."
        )


@dataclass
class _ReportingPairs:
    """The ``(study, voxel)`` pairs where a study's kernel reaches the voxel.

    A bag of parallel arrays, but a named one: the EM retires converged voxels and has to drop
    the pairs that pointed at them, which means reindexing every array in step. Doing that by
    hand for a dozen locals is where this loop was easiest to get wrong.
    """

    voxel: np.ndarray
    weight: np.ndarray
    g: np.ndarray
    sigma: np.ndarray
    precision: np.ndarray
    density_null: np.ndarray
    responsibility: np.ndarray

    def compact(self, position):
        """Drop pairs whose voxel has retired, and renumber the rest onto ``position``."""
        moved = position[self.voxel]
        keep = moved >= 0
        return _ReportingPairs(
            voxel=moved[keep],
            weight=self.weight[keep],
            g=self.g[keep],
            sigma=self.sigma[keep],
            precision=self.precision[keep],
            density_null=self.density_null[keep],
            responsibility=self.responsibility[keep],
        )


@dataclass
class _IndicatorPairs:
    """The ``(study, voxel)`` pairs where a coordinate table carries a reporting indicator.

    One entry per study per voxel it says something about *without* supplying a magnitude:
    ``sign = +1`` where the study was silent, ``sign = -1`` where it reported a focus nearby
    and the height was discarded. Both are evidence about the indicator, and keeping them in
    one array keeps the likelihood one vectorised pass (see :func:`_censoring_terms`).

    ``inv_sigma`` and the scaled cutoffs are precomputed because only ``mu`` moves between EM
    iterations, so every division by them is paid once rather than once per iteration.
    """

    voxel: np.ndarray
    weight: np.ndarray
    sign: np.ndarray
    inv_sigma: np.ndarray
    inv_sigma_sq: np.ndarray
    cutoff_scaled: np.ndarray
    twice_cutoff_scaled: np.ndarray
    #: Probability of the *observed* indicator under the null component, which does not depend
    #: on ``mu``: ``P(silent | no effect)`` where ``sign`` is +1, its complement where -1.
    prob_event_null: np.ndarray
    responsibility: np.ndarray

    def compact(self, position):
        """Drop pairs whose voxel has retired, and renumber the rest onto ``position``."""
        moved = position[self.voxel]
        keep = moved >= 0
        return _IndicatorPairs(
            voxel=moved[keep],
            weight=self.weight[keep],
            sign=self.sign[keep],
            inv_sigma=self.inv_sigma[keep],
            inv_sigma_sq=self.inv_sigma_sq[keep],
            cutoff_scaled=self.cutoff_scaled[keep],
            twice_cutoff_scaled=self.twice_cutoff_scaled[keep],
            prob_event_null=self.prob_event_null[keep],
            responsibility=self.responsibility[keep],
        )

    def censoring(self, mu):
        """P(observed indicator) and its derivatives at the current ``mu``."""
        return _censoring_terms(
            mu[self.voxel],
            self.cutoff_scaled,
            self.twice_cutoff_scaled,
            self.inv_sigma,
            self.inv_sigma_sq,
            self.sign,
        )


class CBES(Estimator):
    r"""Coordinate-based effect-size meta-analysis, estimated from what studies did *not* report.

    .. versionadded:: 0.13.0

    Estimates the pooled standardized effect size (Hedges' :math:`g`) at every voxel from
    effect-size images for the studies that share them, corrected by the pattern of
    **reporting and non-reporting** in every other study's coordinate table.

    **Coordinate tables contribute only a reporting indicator, never a magnitude.** Per study
    per voxel: either the study reported a focus here, or it reported nothing within
    ``coverage_radius`` and so was silent about the neighbourhood. Reported peak heights are
    discarded. That is the whole of the coordinate channel, and it is a deliberate narrowing of
    an earlier design that pooled the heights as effect sizes. See Notes for what was measured.

    A collection must therefore supply at least one study with ``g`` and ``g_var`` images.
    Coordinates alone carry no magnitude in this model, and a fit without an image is refused
    rather than returned as a map of zeros.

    Parameters
    ----------
    design : {"one-sample", "two-sample"}, default="one-sample"
        Design behind the collection, used for the sampling variance a silent study is judged
        against.
    tau2_method : {"dl", "none"}, default="dl"
        ``"dl"`` estimates a local between-study variance with a DerSimonian-Laird moment
        estimator over the image studies; ``"none"`` fits a fixed-effects model
        (:math:`\tau^2 \equiv 0`).

        ``"dl"`` is estimated once, about the naive weighted mean, and then held fixed while the
        selection model fits :math:`\mu` -- which keeps each EM iteration one-dimensional and
        concave, at a known cost. Because the weighted mean is by construction the centre that
        *minimises* the moment estimator's ``Q``, taking ``Q`` about the value finally reported
        can only raise :math:`\tau^2`, and the shipped estimate is therefore biased low.
        Measured against a known :math:`\tau = 0.35`, alternating the two recovers
        :math:`\tau^2` of 0.067, 0.083, 0.093, 0.100 over three extra rounds against a true
        0.1225, and moves ``g`` at the focus from 0.883 to 0.822 against a true 0.8. No spurious
        heterogeneity appears where there is none. Not done, because each round is a full refit
        and a principled joint estimate is a larger change than alternation.
    selection_model : {"zero-inflated", "none"}, default="zero-inflated"
        Whether the silence of the coordinate studies is read at all.

        ``"zero-inflated"``
            Each coordinate study's reporting indicator enters a zero-inflated censored
            (Tobit) likelihood, under a mixture in which the study either has a real effect or
            none at all: the probability of staying silent where it has no focus within
            ``coverage_radius``, and the probability of clearing its cut at a voxel it named.
            This is the entire reason the coordinates are in the model. Nothing is imputed.
        ``"none"``
            Silence is not read, so the fit reduces to an inverse-variance random-effects
            meta-analysis of the images alone and the coordinate tables have no effect
            whatever. Useful as the control arm -- it is what the coordinates are being
            credited against -- and roughly ten times faster.
    se_method : {"model", "hksj"}, default="model"
        Standard error of the pooled estimate. ``"model"`` is the inverse-variance expression,
        which treats the estimated :math:`\tau^2` as known; ``"hksj"`` is the
        Hartung-Knapp-Sidik-Jonkman residual-variance form, which does not and covers better
        with few studies. **Requires** ``selection_model="none"``, the zero-inflated model
        reporting the censored likelihood's curvature instead. Changes ``se`` and so ``z``;
        p-values come from the permutation null either way.
    interval : {"wald", "profile"}, default="wald"
        How to bracket ``g``. ``"wald"`` reports ``se`` alone, referred to a *t* on ``dof``.
        ``"profile"`` additionally emits ``g_lower`` and ``g_upper`` from the profile
        likelihood -- the set of :math:`\mu` whose log-likelihood, with the prevalence
        maximised out at each point, sits within half a chi-square critical value of the
        maximum. It inverts no matrix and needs no ``dof``, which is why it is worth having
        here specifically: both of the quantities it replaces degrade as the prevalence becomes
        weakly identified, the Schur complement and the delta method alike. The bounds are
        asymmetric, as a likelihood region generally is, so they are reported rather than
        summarised as a half-width. Costs roughly a second fit, and **requires**
        ``selection_model="zero-inflated"``. Provisional: ``se`` is still what ``z`` is built
        from, and the interval's calibration against the arm table above is not yet measured.
    analysis_mask : :obj:`str` or None, optional
        ``value_type`` of a per-study image marking the voxels that study examined, nonzero
        meaning examined. Studies without one are taken to have examined the whole analysis
        volume, which is what the censoring term assumes of every study by default.

        This is what an ROI or partial-coverage study needs: its silence outside the region it
        analysed is not evidence that nothing is there, and the censoring term would otherwise
        read it as evidence against an effect. Voxels a study did not examine contribute
        neither a value nor a silence for it. Without this the only remedy was
        ``selection_model="none"``, which discards the whole coordinate channel.
    threshold : :obj:`float`, :obj:`str`, or None, optional
        Reporting threshold each study applied, on the z scale. This is the one number a
        silence cannot do without: "study k reported nothing here" is evidence about the
        effect only against how large an effect k would have needed to report it.

        A string names a metadata field holding per-study values, which is the right answer
        when the papers state their thresholds; studies missing the field fall back to the
        median of those that have it. A float applies one cut to every study. None assumes
        two-tailed p < .001, the most common screening threshold in the literature.

        **It is no longer inferred from the reported heights, because the heights are no longer
        read.** The removed rules undid the order statistic on a study's smallest reported
        value, which cannot distinguish a voxelwise height cut from a cluster-forming one and
        overshot the second by about 1 z: against a true forming cut of z = 3.1 the inference
        returned 4.0, and the prevalences it produced were inflated at every site (a true 0.25
        reading 0.48, a true 0.50 reading 0.86) where a plausible constant recovered 0.21 and
        0.47. Assuming a constant is both simpler and more accurate than inferring one.

        What the choice costs divides sharply. ``g`` barely notices -- across a +1.1 z error it
        stays within 10% of its value at the correct threshold, non-monotonically. Supplying
        the real thresholds matters far more for ``prevalence``, which does not survive it:
        0.73, 0.96, 0.99, 1.00 as the threshold given is inflated by 0, 0.4, 0.8 and 1.1 z,
        against a true 0.60.

        This threshold is also what stops a quiet region reading as an effect of zero. It is
        converted onto the effect-size scale by :func:`reporting_cutoff_to_g` before the
        likelihood sees it -- a z of 3.29 is about 0.65 g at ``n = 30`` -- and the silences
        push :math:`\mu` down only as far as they can carry it, which is toward that cut
        rather than toward nothing. Leaving the cut on the z scale would put it eighteen
        sampling standard deviations out, make every silence certain whatever the effect, and
        take the whole coordinate channel inert.
    clamp_threshold : :obj:`bool`, default=True
        Lower each study's assumed threshold to its own smallest reported statistic, where the
        table carries one. Anything a study reported cleared its cut, so this is a hard
        inequality and not an inference: it can only move a cutoff *down*, only for a study
        whose table contradicts the assumption, and never below the truth. That is what
        distinguishes it from the retired ``"study-min"`` rule, which tried to recover the cut
        by undoing an order statistic it could not identify.

        This is the one remaining use of the reported statistics, and it is a bound on the
        threshold rather than a magnitude -- nothing here reaches the pooled effect size.

        On a simulator whose studies applied 2.4, 2.8, 3.29 and 3.8 z against an assumption of
        3.29, the clamp moved 9 of 20 studies' cutoffs and improved the rmse against a known
        truth by 0.014 where the truth is near zero (paired p = 0.0001) and 0.008 in the middle
        stratum (p = 0.015), with no change where the effect is largest (p = 0.79). It was
        **bit-identical** in both regimes where it should do nothing: where every study really
        applied the assumed cut, and where every study thresholded above it so the bound is
        vacuous -- which is also the thin-table case, a paper reporting only its strongest
        peaks. Safe to leave on; turn it off to hold an assumed threshold exactly.

        One caveat, recorded rather than relied on: in that last regime the *true* thresholds
        were worse than the too-low assumption (rmse 0.128 against 0.106 near zero), so a
        cutoff slightly below the truth is compensating for something. The likely cause is that
        this model treats a report as :math:`|g| \ge c` while a reported peak is
        :math:`|g| \ge c` **and** a local maximum, a strictly smaller event -- so the
        probability of reporting is overstated and a lower cut offsets it. Do not read the
        clamp as more accurate than a stated threshold; read it as a bound that cannot hurt.
    coverage_radius : :obj:`float`, default=20.0
        Radius, in mm, within which a reported focus counts as this study having said
        *something* about a voxel; a study with no focus inside it is silent there and
        contributes a censoring term. This is the only geometry left in the model: papers do
        not report cluster extent reliably, so the extent a focus stands in for has to be
        assumed rather than read, and assuming it is not the same as treating it as silence.

        Used only when ``selection_model="zero-inflated"``. ``g`` is insensitive to it at
        realistic focus counts; ``prevalence`` is not. Against a known prevalence the estimate
        rises monotonically with this radius at every true value (a true 0.50 reads 0.65, 0.73,
        0.76, 0.81 at 8, 14, 20 and 28 mm) and no radius recovers the truth: mean absolute
        error runs 0.17 to 0.21 over that range, 14 mm marginally best and 20 mm close behind.
        Left at 20 mm because the differences are small beside the bias itself. On dense focus
        tables ``prevalence`` saturates at 1.0 here.
    max_iter : :obj:`int`, default=25
        Maximum Newton iterations for the censored likelihood. Voxels reached by a single image
        do not converge at any value of this, and raising it does not help: their likelihood is
        flat in the magnitude over the whole plausible range, so the iteration count only
        decides which point on a plateau is reported.
    null_method : {"permute-images", "none"}, default="permute-images"
        How uncorrected p-values are obtained. ``g / se`` is not null-referenced -- the
        standard error treats :math:`\tau^2` as known and ignores the selection the censoring
        term is modelling -- so p comes from a randomization null instead.

        ``"permute-images"`` reassigns each image study's effect sizes among **its own** voxels
        and refits, holding the coordinate tables exactly as they are. The hypothesis is that
        within a study, effect size is unrelated to location. Because nothing moves between
        studies and the silence pattern is identical in the observed fit and in every
        permutation, each voxel keeps its own studies and its own censoring roster throughout
        and is referred to a null of its own -- the exchangeability the test needs
        (:footcite:t:`winkler2014permutation`). Whatever the censoring term contributes cancels
        between observed and null, which is why a null over the coordinates is neither needed
        nor available.

        The p-value is ``(1 + #{null >= observed}) / (1 + n_iters)`` and so cannot fall below
        ``1 / (1 + n_iters)``, which is where the default ``cluster_threshold`` of .001 sits
        unless ``n_iters`` is raised past 1000; familywise correction has no such floor. This
        is deliberately **not** a test of spatial convergence, which is what a null that
        relocates the foci -- ALE's and MKDA's -- would give instead, nor a test of whether the
        effect is zero, which is what sign-flipping the images would give.

        ``"none"`` returns ``p = 1`` everywhere, for inspecting the estimates at no cost.
    cluster_threshold : :obj:`float` or None, default=0.001
        Cluster-forming threshold, as an uncorrected p-value, for the cluster-level FWE null
        that :meth:`fit` builds alongside the voxel-level one. Set to None to skip it, which
        makes :meth:`correct_fwe_montecarlo` pay for a second pass over the permutations if
        cluster correction is then requested.
    n_iters : :obj:`int`, default=1000
        Permutations for the null. Each is a full refit, which makes this the dominant cost of
        the estimator. It also sets the resolution of the uncorrected p, which cannot fall
        below ``1 / (1 + n_iters)``.
    n_cores : :obj:`int`, default=1
        Processes used for the permutation null, which is where nearly all the time goes.
        ``-1`` uses every available core and is close to linear. The iterations run one block
        per core rather than one task per iteration, because the null is accumulated per voxel
        and shipping each iteration's whole map back would cost more than the refits.
    seed : :obj:`int`, default=0
        Seed for the permutation draws.
    memory, memory_level, generate_description
        As in every other :class:`~nimare.estimator.Estimator`.

    Attributes
    ----------
    masker : :class:`~nilearn.maskers.NiftiMasker`
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator.

    Notes
    -----
    Where ALE and (M)KDA ask *where do studies agree something happened*, this asks *how big is
    the effect there*, and answers it from two channels that are deliberately unlike each other:

    **Images give the magnitude.** Each voxel pools the studies supplying ``g``/``g_var`` maps
    by inverse variance, with a local DerSimonian-Laird :math:`\tau^2`:

    .. math::

        \hat{g}(v) = \frac{\sum_k g_k(v) / (s^2_k(v) + \tau^2(v))}
                          {\sum_k 1 / (s^2_k(v) + \tau^2(v))}

    which with one study is that study's map and with several is a textbook random-effects
    meta-analysis. No spatial kernel: an image already says what it says at every voxel.

    **Coordinates give the selection.** Every study on the collection's roster that reported no
    focus within ``coverage_radius`` of :math:`v` enters a zero-inflated censored likelihood
    there, contributing the probability that a study of its sample size, applying its reporting
    threshold, would have stayed silent. Two quantities come out of that fit, and keeping them
    apart is the point of it: :math:`\pi(v)`, the fraction of studies with a non-null effect
    here, and :math:`\mu(v)`, the effect size *among the studies that have one*. A study silent
    in a region either has no effect there or has one that failed to clear its threshold, and
    the mixture lets the data decide, so silence need not be explained as a small-but-real
    common effect -- which is what drags a plain Tobit fit below the truth. Nothing is imputed
    :footcite:p:`tench2017coordinate`.

    Available maps:

    ============== ===============================================================
    "g"            Pooled Hedges' g among the studies with an effect, on the
                   effect-size scale the images arrive on.
    "prevalence"   Fitted fraction of studies with a non-null effect here. Added
                   under the zero-inflated selection model. Read ordinally; see
                   Warnings.
    "g_marginal"   ``g`` times ``prevalence``: the effect averaged over *all*
                   studies rather than over those that have one, which is the
                   estimand an image-based meta-analysis reports. Added alongside
                   ``prevalence``.
    "se_marginal"  Standard error of ``g_marginal``, by the delta method on the
                   same observed information. Zero where there is none.
    "coordinate\_  Share of the Fisher information about ``g`` that came from the
     share"        coordinate tables rather than from the images' values, in [0, 1].
                   **Read this before trusting the coordinate channel anywhere.**
                   Added under the zero-inflated selection model.
    "se"           Standard error of the pooled estimate. See ``se_method``.
    "z"            ``g / se``. Two-tailed.
    "p", "logp"    p-value for ``z``, and its ``-log10``.
    "tau2"         Local between-study variance.
    "n_studies"    Number of image studies contributing a value here.
    "n_eff"        Kish effective number of image studies, ``(sum w)^2 / sum w^2``.
    "dof"          Degrees of freedom to refer ``se`` to; see below.
    ============== ===============================================================

    Why the indicator and not the heights
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    An earlier design pooled the reported peak heights as effect sizes alongside the images,
    with a spatial kernel and a per-study peak-height correction. Stratifying its error by how
    many studies reported near a voxel showed the two channels pulling opposite ways: against a
    held-out reference the bias reduction from adding a coordinate corpus to a two-image
    collection was **54% where no focus reached the voxel and 6% where two studies reported**,
    shrinking monotonically as more reported. That is the signature of the reporting *pattern*
    helping and the magnitudes hurting, not of a magnitude channel working.

    On the 21-study NIDM pain collection, split in half so the reference comes from studies the
    coordinates never touched, with the tables extracted the way a paper would produce them
    (cluster-forming cut, whole clusters kept, one focus per cluster) and eight splits scored
    paired:

    ==========================  ======  ========  ==========  =====  ======
    estimate                      bias  at top      rmse      rank r  AUC
    ==========================  ======  ========  ==========  =====  ======
    images only, pooled         +0.136    +0.137       0.269   0.484   0.893
    CBES with the silence off   +0.142    +0.155       0.272   0.499   0.899
    CBES ``g``                  +0.075    -0.065       0.210   0.486   0.889
    CBES ``g_marginal``         -0.004    -0.226       0.191   0.458   0.861
    ==========================  ======  ========  ==========  =====  ======

    **The coordinate channel corrects the level and leaves the pattern alone.** ``rmse`` falls
    22% against pooling the images by themselves (paired p = 0.0003) and the bias 45%
    (p < 0.0001); at the strongest voxels a +0.137 overestimate becomes a slight under. But the
    ordering barely moves -- rank correlation +0.002 (p = 0.90) and AUC -0.004 (p = 0.32) --
    and Pearson ``r`` costs 0.042 (p = 0.017). Read it as a correction to the magnitude, not as
    a better map.

    The reason the heights were never going to work is in the input rather than the fit.
    Regressing a held-out truth at a focus on the effect size that focus's own table reports
    gives a slope of 0.08 to 0.18 with most of the value in the intercept: **one tabulated
    coordinate explains 5% to 9% of the variance in the effect at its own location.** And the
    level is not even a property of the studies -- holding the studies fixed and changing only
    how a paper would have tabulated them, the ratio of the old ``g`` to the held-out truth ran
    from 0.82 to 2.29 across FDR, voxelwise FWE and cluster-extent thresholding and across
    tabulating a cluster by its maximum or its centre of mass. Measured and rejected as
    remedies before the heights were dropped: the truncated-normal selection correction, which
    is the wrong event for a local maximum and returns 0.26 for a true 0.5; a per-study
    peak-height rescaling; subtracting the censoring floor; reporting by centre of mass; and
    widening the assumed cluster.

    Removing the heights removes what came with them. There is no longer an unidentified
    overall scale -- the images arrive on the effect-size scale, so ``g`` is in the units it
    claims and the ``g_relative``/``g_absolute`` distinction is gone. There is no kernel width
    to choose, no peak-height correction to calibrate, and no threshold to infer.

    Both values of the indicator, over different extents
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A silence and a report are complementary events -- :math:`|g| < c` and :math:`|g| \ge c`
    for the study's own cut :math:`c` -- and **both are evidence**. Dropping the report limb
    leaves the silences as the only evidence about the indicator, so the model reads the
    observed silence fraction against a denominator that excludes every study that reported,
    and the magnitude shrinks past the threshold toward zero. On a field simulator with the
    truth known exactly, where the effect is largest that cost an rmse of 0.091 against 0.074
    and a bias of -0.060 against -0.042; at the focus itself, 0.352 for a true 0.500 against
    0.444.

    But the two assert over **different extents**, and getting that wrong is far worse than
    dropping the limb altogether. A silence is a statement about a neighbourhood: nothing
    within ``coverage_radius`` cleared this study's cut. A report is a statement about one
    voxel, because a reported peak is a local maximum selected for being large and displaced
    from wherever the effect is -- so "someone reported 18 mm away" is not evidence that the
    effect *here* cleared anything. Asserting the report across the same sphere as the silence
    gave an rmse of 0.457 against 0.114 for the named voxel alone, and turned a -0.042 bias
    where the truth is largest into +0.097. So a report asserts its indicator at the voxel it
    names; a voxel a study reached but did not name carries no indicator at all.

    That asymmetry is invisible without a spatial dimension. A one-voxel likelihood, with no
    displacement and no radius, says "restore the report limb" unconditionally -- 0.351 to
    0.524 for a true 0.500 -- and cannot see the radius problem at any signal level.

    **The floor is the threshold, not zero.** Because :math:`\mu` is bounded below the cut only
    as far as the silences push it, a quiet region does not read as an effect of zero; it reads
    as an effect the reporting studies could not detect. The pull toward zero comes through
    :math:`\pi` instead -- many studies, all silent, means few of them have an effect -- which
    is why ``g_marginal`` is the better map where nothing was reported (bias +0.056 against
    +0.095 for the images alone) and the worse one where an effect exists (-0.096).

    The interval
    ~~~~~~~~~~~~
    Refer ``se`` to a *t*, not to a normal, and **mask on coverage before you do**. ``dof`` is
    the number of studies on the censoring roster minus one, zeroed where no study speaks about
    the voxel. The roster rather than a Kish count over the pooling weights, for two reasons,
    and the second is decisive: the likelihood uses every study on the roster -- the images
    through their values, the rest through their silence -- so crediting only the weighted
    contributors denies the model information it demonstrably used; and with the coordinate
    magnitudes gone the only weighted contributions are images at weight 1, so a Kish count is
    just the image count, making ``dof`` **zero with one image** and the recommended interval
    ``nan``. Whether a censored-likelihood observed information really carries the roster's
    degrees of freedom is a genuine question rather than a settled one; a profile-likelihood
    interval would need no ``dof`` at all.

    Under the selection model ``se`` is the observed information of the censored mixture
    likelihood at the fitted point, with the prevalence profiled out by a Schur complement, so
    it carries both the uncertainty about which component an observation came from and the cost
    of not knowing the prevalence. On the estimator's own censored mixture with known variances
    it covers 94.5% to 98.4% of nominal-95% intervals across prevalences, cutoffs and study
    counts, erring conservative.

    **End to end the interval is conservative rather than wrong, and coverage will not tell you
    that either way.** Measured on the field simulator with the truth known exactly, 40
    replications per arm, stratified by the truth because 9204 of 9261 voxels sit near zero and
    a whole-map figure rewards any estimator that shrinks. ``width`` is the half-width of the
    documented interval as a fraction of the truth, so the quiet stratum's is meaningless by
    construction and omitted:

    =========================  ======  =========  ======  ========  =========  ========
    arm                          bias  ``se/sd``  cov(t)  bias      ``se/sd``  width
                                       (quiet)    (quiet) (effect)  (effect)   (effect)
    =========================  ======  =========  ======  ========  =========  ========
    20 studies, 1 image        -0.000       1.51    0.98    -0.089       1.39      0.99
    20 studies, 2 images       +0.001       1.80    0.98    -0.086       1.66      0.89
    20 studies, 5 images       -0.000       2.06    0.99    -0.038       1.83      0.59
    20 studies, 20 images      +0.000       1.16    0.98    -0.023       1.19      0.26
    2 images, silence off      +0.001       1.23    1.00    -0.040       1.22      5.58
    2 images, tau = 0.3        +0.001       1.80    0.98    -0.097       1.78      1.54
    =========================  ======  =========  ======  ========  =========  ========

    **An earlier version of this table was wrong in a way worth recording, because the error
    was in the measurement and not the model.** It reported a quiet-stratum bias of +0.045 to
    +0.114 and ``se/sd`` of 2.05 to 3.74. Both came from taking ``numpy.abs`` of ``g`` before
    the spread across replications was computed. ``g`` is a *signed* inverse-variance mean, so
    at a voxel whose truth is zero the absolute value shrinks the spread to about 0.6 of the
    real one and puts the mean about 0.8 spreads above zero: it halved the denominator of
    ``se/sd`` and manufactured the bias out of nothing, at once. At the foci, where the estimate
    sits far from zero, the absolute value is nearly a no-op -- which is exactly why the old
    table's *quiet* column ran 2.71 to 3.74 while its *effect* column ran 1.55 to 1.97, and
    nobody looked twice. **The estimator is not biased upward where nothing was reported.**

    The defect was confined to this: a spread of :math:`|g|` read as the estimator's sampling
    spread, or a difference of :math:`|g|` from a near-zero truth read as a bias. Comparisons
    between arms that take the absolute value of the *reference* as well -- the collection
    comparisons below, and every ranking against SDM-PSI or an images-only pool -- apply the
    same transformation to every arm and are unaffected.

    The check that would have caught it on the first run, and is now the top row of the arm
    list: give every study an image and switch the selection model off, and the fit is a
    textbook local inverse-variance random-effects meta-analysis whose ``se/sd`` must come out
    near 1. It reads 1.00 to 1.24. A little of the residue above 1 is the bed rather than the
    estimator -- the simulator's per-study noise measures 4% below the ``1/n + g^2/(2n)`` the
    model assumes, so every ``se`` is 4% generous by construction.

    Two things follow.

    **The inflation was the prevalence, not the censoring term.** The old reading -- "the excess
    is in the censoring term, since switching the silence off halves it" -- compared two
    different estimators and was never a localisation. The configuration that settles it is the
    all-image one: with every study carrying an image the reporting indicator is *structurally*
    empty, ``coordinate_share`` is identically zero, and the coordinate channel cannot be
    responsible for anything. That arm was nonetheless the worst measured, ``se/sd`` of 2.21 at
    quiet voxels against 1.14 for the same images fitted without the mixture, in the same bed on
    the same seeds. What it was paying for was a prevalence that nothing identified: only the
    indicator separates "no effect in this study" from "a small effect plus noise", and left
    free against 20 Gaussian values the mixture returned :math:`\pi = 0.577` against a true
    1.000, whose uncertainty was then profiled out of the information about :math:`\mu`. The
    inflation was concentrated at quiet and weak voxels (2.21 and 2.30) and absent at strong
    ones (0.99 and 1.06), which is the signature of that trade-off: where the effect is weak,
    "a small effect in every study" and "a large effect in a few" fit equally well.
    :math:`\pi` is now held at 1 wherever no study contributes an indicator, and the all-image
    arm returns the non-mixture fit exactly -- 1.16 in the table above, and asserted as an
    identity in the tests rather than as a tolerance.

    **What remains is a conservative interval where the coordinates do act, 1.28 to 2.03.** It
    is anomalous in one direction only: the likelihood conditions on where the foci fell while
    the replication spread is marginal over that, so a calibrated *conditional* error should sit
    *below* the marginal spread, not above it. Two candidate causes are now ruled out. It is not
    the reporting rule: the model censors on :math:`|g| < c` while a paper reports local maxima,
    and feeding the estimator its own rule instead -- every supra-threshold voxel reported, 386
    foci per collection against 172 -- moves ``se/sd`` from 1.83 to 1.80 at quiet voxels and
    1.90 to 1.90 at the strongest focus, and the bias not at all. It is not the baseline either,
    at 1.00 to 1.24. The residue tracks how much the indicator says about :math:`\pi`:
    ``se/sd`` falls monotonically with ``coordinate_share``, 1.87 at under 0.05 to 1.33 above
    0.50. So the same channel that identifies the prevalence is what makes its cost bearable,
    and where the tables are thin the interval on :math:`\mu` is wider than the estimate
    deserves.

    **Changing the estimand does not fix it, and this was tested rather than assumed.** The
    ridge runs along curves of roughly constant :math:`\pi\mu`, so the product ought to be the
    determined combination and ``g_marginal`` the sound thing to bracket. Half of that is true
    and the useful half is not. The product really is the steadier *estimate* -- its spread
    across replications is 0.076 at quiet voxels against 0.112 for ``g`` -- but its reported
    error is worse, 0.349 against 0.204, and ``se_marginal/sd`` comes out above ``se/sd`` in
    every stratum but the strongest: 4.58 against 1.83 at quiet voxels, then 2.04, 2.48, 2.43
    and 1.58. The reason is mechanical: :math:`\operatorname{Var}(\mu\pi)` needs the whole
    2x2 inverse rather than a Schur complement, and a near-singular information matrix amplifies
    there instead of cancelling. **So ``g_marginal`` is the more stable estimate and the less
    trustworthy interval.**

    **And the width is not the problem. There is no finite width to be had.** The obvious
    remedy for both -- a profile likelihood, which inverts nothing and needs no ``dof`` -- was
    implemented (``interval="profile"``) and it answers the question in a way that disqualifies
    the question. As :math:`\pi \to 0` the active component explains nothing, the mixture
    density tends to the null one at every observation whose probability depends on
    :math:`\mu`, and the profile log-likelihood approaches a *horizontal asymptote*,

    .. math::

        \ell_\infty = \max_\pi \left[
            \sum_{i \,\notin\, \mathrm{reported}} \log\left((1 - \pi) b_i\right)
            + \sum_{i \,\in\, \mathrm{reported}} \log\left(\pi + (1 - \pi) b_i\right)
        \right],

    where :math:`b_i` is observation :math:`i`'s density or probability under an effect of
    exactly zero. Image values and silences drop their :math:`\mu` dependence because both
    probabilities vanish as :math:`|\mu| \to \infty`; a *report* does not, its probability
    tending to 1 instead, which is the only thing keeping :math:`\pi` in the expression. So the
    interval is bounded exactly when :math:`2(\hat\ell - \ell_\infty)` exceeds the critical
    value -- and at a voxel where no study reported, where the maximum is attained as
    :math:`\pi \to 0`, that is precisely the likelihood-ratio test of :math:`\pi = 0`. With a
    handful of image studies it almost never fires. Measured on the field bed, the fraction of
    voxels where the interval is bounded at all:

    ==============  ================  =============  =============
    image studies   bounded overall   within 10 mm   beyond 30 mm
    ==============  ================  =============  =============
    2                         0.025          0.086          0.029
    5                         0.055          0.148          0.065
    10                        0.072          0.284          0.083
    ==============  ================  =============  =============

    Reparametrising does not escape it: :math:`\pi\mu` has the same asymptote, reached along
    :math:`\pi \to 0` with :math:`\mu \to \infty`.

    **What does escape it is a literature whose studies differ in size, and the reason is
    algebraic.** A silence constrains :math:`(\pi, \mu)` only through the probability of the
    event observed, :math:`P = \pi S(\mu) + (1 - \pi) S(0)` with
    :math:`S(\mu) = P(|g| < c \mid \mu)` -- one equation in two unknowns, which is the ridge.
    Studies with *different* :math:`(\sigma, c)` supply different equations, but not all
    heterogeneity helps, because the reporting cutoff measured in sampling standard deviations
    is just the reported statistic back again: :math:`c / \sigma \approx z`. So

    .. math::

        S(0) = 2\Phi(z) - 1, \qquad
        S(\mu) = \Phi(z - \mu\sqrt{n}) - \Phi(-z - \mu\sqrt{n}).

    The null component's silence probability depends on the *threshold alone*. Varying the
    threshold moves both components together through the same tail and leaves the equations
    nearly collinear; varying the sample size moves the active component through
    :math:`\mu\sqrt{n}` while leaving :math:`S(0)` exactly fixed, which is the contrast that
    separates the two parameters. Measured, 20 studies throughout, with the bounded fraction of
    the profile interval near the signal as the identifiability probe:

    ==========================  ==============  ===================
    studies                     bounded near    fitted :math:`\pi`
    ==========================  ==============  ===================
    alike, n 28-32, one cut              0.422                0.823
    sample size spread 12-120            0.691                0.913
    threshold spread 2.3-4.5             0.414                0.815
    both spread                          0.676                0.901
    ==========================  ==============  ===================

    True :math:`\pi` is 1.0. Spreading the sample size identifies it; spreading the threshold
    does nothing whatever, exactly as the cancellation above says. **So a magnitude is
    recoverable from a literature of widely differing sample sizes and not from a literature of
    uniformly sized studies, however many of them there are** -- which is the opposite of
    treating heterogeneity as a nuisance, and is the one piece of advice here that bears on
    whether to run this estimator on a given collection at all.

    **So ``se`` reports a curvature at the point the EM selected, and outside that regime the
    likelihood does not support that precision about :math:`\mu`.** The estimate's stability
    across replications -- a spread of 0.11 where ``se`` says 0.20 -- comes from the pooled
    image mean it starts from and the ``max_iter`` it stops at, not from the data pinning it
    down. That is worth
    knowing before reading ``g`` as a magnitude, and it is the strongest statement available
    about why the ``se/sd`` question never resolved: it was asking whether an interval was
    calibrated for a parameter the data do not bound.

    None of this touches the p-values, which come from the permutation null and need no
    calibrated ``se``, nor the *ordering* of ``g``, which every collection comparison here
    scores and which holds up. It bears on reading a single voxel's ``g`` as a number with an
    error bar.

    **The ``silence off`` row's width is not comparable.** Without the selection model there is
    no censoring roster, so ``dof`` falls back to the Kish count over the image weights, which
    at two images is 1 -- and a *t* on one degree of freedom has a critical value of 12.71. That
    row is a correct statement about two studies, not a wider interval for the same
    information.

    **Read ``se/sd`` and the width, never coverage.** Every arm covers 0.95 to 1.00, including
    the ones whose interval admits almost any magnitude: at two images the half-width is 0.89 of
    the effect, and coverage alone cannot distinguish that from the all-image row's 0.26.

    **P-values are unaffected by any of this.** They come from the permutation null, which is
    valid for whatever statistic it is computed on and does not require a calibrated ``se``.

    An earlier version reported the curvature of the EM's *Q function* instead of the observed
    information, which holds the responsibilities fixed and so overstates the information; that
    covered 62.5% to 89.8% and did not improve with more studies.

    :meth:`correct_fwe_montecarlo` adds ``logp_level-voxel``,
    ``logp_desc-size_level-cluster`` and ``logp_desc-mass_level-cluster`` (each with a signed
    ``z_*`` companion), matching the names :class:`~nimare.meta.cbma.ale.ALE` uses.
    :class:`~nimare.correct.FDRCorrector` and ``FWECorrector(method="bonferroni")`` work off
    the uncorrected ``"p"`` map instead, and are only meaningful when that map came from the
    permutation null.

    Warnings
    --------
    This estimator is new and has not been validated against a reference implementation.

    **The correction can make ``g`` worse than doing nothing, and the regime where it does is
    not exotic.** Judged against a reference built from subjects used to make no coordinate at
    all -- 786 HCP subjects, 480 cut into 16 synthetic studies of 30 with coordinates extracted
    the way papers produce them, 306 held out to give the truth:

    ======================  ======  ========  =====  ===========
    estimate                     r  rank r     AUC   magnitude
    ======================  ======  ========  =====  ===========
    two images, pooled      +0.845    +0.576  0.973         0.85
    ``g``                   +0.830    +0.575  0.967         0.63
    ``g_marginal``          +0.785    +0.564  0.943         0.54
    ======================  ======  ========  =====  ===========

    Pooling the two images alone wins on every metric, magnitude included -- **and so does an
    independent method given the same data.** SDM-PSI, which takes the same mixture by design,
    was run on the same studies with the same two supplied as maps: it returns r +0.722, AUC
    0.927 and 0.43 of the reference, behind ``g`` on every column and behind the two images
    alone on every column. Two unrelated methods both do worse with the fourteen coordinate
    tables than without them, which makes this a property of thresholded coordinate tables in
    this regime rather than a quirk of this estimator's censoring term. **That design has a
    prevalence of exactly 1**: every synthetic study draws from the same population, so there is
    no between-study absence for the mixture to find, and an unthresholded map of 30 of those
    subjects is already nearly unbiased. There is nothing for a selection correction to correct,
    and it does harm anyway -- fitted prevalence comes back at 0.664 against a true 1.0 (0.929 at
    the strongest decile), because "failed to clear its threshold" and "has no effect" both
    explain a silence and the mixture splits the difference.

    **But the prevalence is the symptom, not the cause.** Refitting the same collection with the
    prevalence *fixed* at 1 -- which is the truth here -- makes the magnitude **worse**, not
    better: 0.60 of the reference against 0.63, and 0.411 against 0.422 for a true 0.5 on the
    simulator. Removing the "this study has no effect" escape forces every silence to be
    explained by a small :math:`\mu`, so :math:`\mu` falls further. So the **censoring term
    over-shrinks :math:`\mu` whatever the prevalence does**, and a fitted :math:`\pi` below 1 is
    the model partly *absorbing* that over-shrinkage rather than adding to it. ``g_marginal``
    comes back worst of all (0.54) because it multiplies the two together.

    Which makes this the same defect as the overstated reporting probability above, from another
    direction: too high a :math:`P(\text{report})` against an under-observed count drags
    :math:`\mu` down, and here there is no genuine absence for the prevalence to absorb it
    into. Pinning the prevalence is therefore not the middle option it sounds like.

    Contrast the 21-study NIDM pain collection above, where the same correction cuts rmse 23%
    and bias 47%. **Why the two disagree is not settled, and the obvious explanations have been
    tested and rejected.**

    It is not the prevalence. Building the regime as a dial -- MOTOR_LH studies that carry the
    effect mixed with EMOTION_FACES studies that do not, so the true :math:`\pi` is designed
    rather than assumed -- ``g`` recovers 0.68, 0.63, 0.56 and 0.53 of :math:`\mu` at true
    prevalences of 1.00, 0.75, 0.50 and 0.25. It degrades *monotonically* as prevalence falls,
    rather than improving.

    Nor is it the statistic. On the pain bed's own whole-map rmse the two arms tie on that dial
    (0.130 against 0.129 at :math:`\pi = 1`) or the images win (0.281 against 0.241 at 0.50).

    Nor is it between-study heterogeneity, the other obvious candidate: adding a relative
    :math:`\tau` of 0.0, 0.3 and 0.6 to the dial's effect studies leaves rmse at 0.126, 0.135
    and 0.159 against the images' 0.126, 0.123 and 0.140 -- no crossing anywhere.

    **What does move it is how the reference is built, and that is a caution about the pain
    number rather than an endorsement of it.** The pain reference was an inverse-variance mean
    of 19 study-level ``g`` maps. Hedges' variance is a function of the *observed* effect, so a
    study that drew high gets less weight and such a reference is itself pulled downward --
    and ``g`` is pulled downward too, so an estimator biased low scores better against a
    reference biased low. Rebuilding the dial's reference the same way, from synthetic
    reference studies rather than pooled subjects, takes ``g`` from tied (0.126 against 0.126)
    to winning (0.122 against 0.125). **But that accounts for about 2 points of the pain gap's
    22**, so the direction of the artefact is demonstrated and its magnitude is not. The one
    remaining difference -- real studies against synthetic, with their different scanners,
    paradigms and sample sizes -- cannot be dialled.

    So the advice here is empirical rather than principled: **fit it both ways.**
    ``selection_model="none"`` reduces to an inverse-variance meta-analysis of the images and
    costs about a tenth of the runtime, so the comparison is cheap. Where the two agree, little
    turns on the choice; where they disagree sharply, the correction is doing something
    load-bearing that nothing measured here can yet vouch for.

    **``coordinate_share`` is the diagnostic every caveat here needs, and it is the one thing a
    reader could not otherwise get.** The warnings above are all about *when* to trust the
    coordinate channel -- the prevalence-1 regime where it does harm, the unexplained
    disagreement between collections, the over-shrinkage at the window. None of them can be
    checked on a given collection. What can be checked is whether the channel is even acting at
    a voxel: ``coordinate_share`` is the fraction of the information about ``g`` contributed by
    the indicators rather than the images' values, and it comes free because the two are
    accumulated separately inside the likelihood. At 0 the images carry the estimate alone and
    the tables changed nothing here, so none of the coordinate caveats apply; at 1 the
    indicators carry it and all of them do.

    On a 20-study collection with two image donors it runs from 0.02 to 1.00, with a **median of
    0.10 and 0.79 at the focus** -- so on a typical map the images carry the estimate almost
    everywhere and the coordinates take over exactly where studies reported. That is the
    stratification the design rests on, now readable per voxel rather than only in aggregate.

    **It also says where the interval is trustworthy, in the direction opposite to the obvious
    guess.** If the ``se``'s conservatism were produced by the censoring term, ``se/sd`` would
    be worst where the share is high. It is the reverse, and monotonically so: across bands of
    the share from below 0.05 to above 0.50 it runs 1.87, 1.82, 1.87, 1.63 and 1.33, with the
    top decile at 1.54 against the bottom's 1.86. That is not a signal effect masquerading as a
    share effect -- only a few hundred of these 15625 voxels carry any truth, so the band from
    0.25 to 0.50 is almost entirely quiet, and the gradient holds inside it. The reason is the
    one the interval section gives: what the ``se`` is paying for is an imprecisely known
    prevalence, the indicator is the only thing that identifies it, and the share is precisely
    how much indicator reached this voxel. So a high share is the regime where the coordinate
    caveats bite *and* where the interval is soundest, and the two readings do not conflict.

    **Read ``prevalence`` ordinally, not as a fraction, and not within one map.** On the
    designed-prevalence dial just described -- real subjects, a true :math:`\pi` set by how many
    studies carry the effect -- it reads 0.918, 0.714, 0.590 and 0.540 against true values of
    1.00, 0.75, 0.50 and 0.25, measured at the strongest decile. So it tracks well down to about
    0.5 and then **floors near 0.54**, which is the compression in its least flattering place:
    a rare effect and a common one come back nearly the same. Against a simulator the same
    compression is worse still -- a true 0.25 comes back as 0.49 to 0.60 depending on
    ``coverage_radius``, a true 0.50 as 0.65 to 0.81, a true 1.00 as 0.74 to 0.94.
    Its map-wide median sits near 0.4 whatever the truth,
    so a map cannot be summarised by it.

    Worse, the ordinal reading holds on average over many maps and rarely within any one of
    them. Tested as the claim is made -- four sites in a single fit at true prevalences 0.25,
    0.50, 0.75 and 1.00 -- the rank correlation against the truth averages +0.76 for a strong
    effect, but the four-site ranking is exactly right in only **19%** of maps; for a weak
    effect it averages +0.33 with a standard deviation of 0.60 and is exactly right in **6%**.
    The per-voxel scatter is largest at the low end, so rare sites are both biased upward and
    noisier -- the worst combination for the use this invites, picking out which region is the
    least consistent.

    That is structural rather than a calibration that could be fixed. :math:`\pi` and
    :math:`\mu` are separably estimable only in a window of detectability: where a study's
    effect lands near its own reporting threshold, so that the chance of reporting responds to
    the magnitude. Below that window nothing is detected and the prevalence is not identified
    at all; above it detection saturates, the magnitude stops being constrained from above and
    the prevalence absorbs the level instead -- which is why a strongly reported site returns a
    prevalence near 1 whatever its truth. A map spans magnitudes and therefore spans the
    window, so comparing two voxels compares quantities identified to different degrees. A
    spread of sample sizes and reporting thresholds across the collection widens the window; a
    roster of identically powered studies narrows it.

    **``g`` and ``g_marginal`` are different estimands, and the second is the one an image-based
    meta-analysis reports.** Every IBMA -- DerSimonian-Laird, Hedges, weighted least squares,
    the likelihood estimators -- pools per-study effect maps around a single mean, so a study
    with no effect at a voxel enters that average as a zero and the quantity estimated is
    :math:`\pi(v)\,\mu(v)`. ``g`` is :math:`\mu(v)`, the effect over the studies that have one.
    The two differ by a factor of :math:`1/\pi`, which on the NIDM pain collection is 1.4 to
    1.8. So ``g`` cannot be checked against an image-based reference even in principle;
    ``g_marginal`` can, and is the map to compare. Conversely :math:`\mu(v)` may not be
    identifiable from images at all: computing it requires classifying every study as having an
    effect at every voxel or not, which is a thresholding decision and reintroduces the
    selection this estimator exists to correct.

    **``g_marginal`` does not work for the reason its name gives.** In a held-out-subject design
    where every synthetic study is drawn from one population, the true prevalence is exactly 1
    and ``g_marginal`` should equal ``g``; instead ``prevalence`` comes back near 0.68 and the
    product is the better estimate. Multiplying by it is shrinking a magnitude by a data-driven
    factor, not averaging over studies that have no effect. That the two errors cancel is why it
    is worth reporting and also why it should not be trusted outside the regimes it has been
    measured in. It degrades when studies report few foci, because ``prevalence`` then falls
    toward its floor: at six foci per study it came back at 0.70 times the held-out truth.

    **The dynamic range is recovered, which the earlier design's was not.** Compression was that
    design's headline failure: a reference effect spanning elevenfold across its strata came
    back spanning about 1.2-fold, and an unknown overall scale would have left that ratio alone,
    so it was a real defect and not a units problem. Re-measured here on four well-separated
    foci at true ``g`` of 0.2, 0.4, 0.6 and 0.8 inside one map, 8 collections, regressing the
    estimate on the truth over the voxels carrying signal:

    Values are the estimate at each focus's own voxel, with its standard error across the 24
    collections and its relative error:

    ==============  =====  =========  ==================  ==================
    estimate        slope  intercept  @ 0.2               @ 0.4
    ==============  =====  =========  ==================  ==================
    images only     0.911     +0.043  0.230 +- .025 (+15%)  0.419 +- .029 (+5%)
    ``g``           0.759     +0.049  0.194 +- .019 (-3%)   0.329 +- .022 (-18%)
    ``g_marginal``  0.769     +0.018  0.151 +- .018 (-25%)  0.267 +- .024 (-33%)
    ==============  =====  =========  ==================  ==================

    ==============  ==================  ==================  =========  ==========
    estimate        @ 0.6               @ 0.8               range      mean |err|
    ==============  ==================  ==================  =========  ==========
    images only     0.635 +- .027 (+6%)   0.792 +- .025 (-1%)  3.44-fold        6.7%
    ``g``           0.634 +- .016 (+6%)   0.825 +- .017 (+3%)  4.26-fold        7.5%
    ``g_marginal``  0.576 +- .015 (-4%)   0.780 +- .019 (-3%)  5.18-fold       16.1%
    ==============  ==================  ==================  =========  ==========

    **The range is recovered and the level is not uniformly better.** ``g`` returns 4.26-fold
    for a true 4-fold against 3.44-fold for pooling the images alone, so the compression that
    was the earlier design's headline failure is gone. But its *mean* absolute relative error
    over the four foci is 7.5% against the images' 6.7% -- a wash, slightly the wrong way --
    because the two arms trade errors focus by focus rather than one dominating.

    Where they trade is informative, and it maps onto the window of detectability. At the
    weakest focus, where **no** study reports (0 of 18 in a representative collection), the
    images are inflated 15% and ``g`` is 3% off: the correction is working, though note that the
    coordinate channel contributes nothing usable there -- with nothing reported and silence
    nearly flat in :math:`\mu` below the cut, ``g`` is the two-image estimate and its accuracy
    is the images' sampling noise. At the strongest focus, where 12 of 18 report, both are
    within a few percent.

    **The 0.4 focus is the clear remaining defect: 18% low, about three standard errors, where
    the images are 5% high. It is located, and it is the reporting model's functional form.**
    The censored likelihood gives a reported voxel the probability :math:`P(|g| \ge c)`, but a
    paper reports a voxel only if it cleared :math:`c` **and** was a local maximum -- a strictly
    smaller event. Instrumenting the same bed to compare the observed reporting rate against the
    rate the model computes at the true :math:`\mu`:

    =======  ========  ==============  =============  =====================
    truth    reports   observed rate   model's rate   ratio (95% CI)
    =======  ========  ==============  =============  =====================
    0.2             3           0.007          0.007  0.97 (unmeasured)
    0.4            21           0.050          0.086  **1.72 [1.19, 3.03]**
    0.6           163           0.392          0.392  1.00 [0.87, 1.18]
    0.8           299           0.702          0.789  1.12 [1.01, 1.27]
    =======  ========  ==============  =============  =====================

    Counts are totals over 24 collections and the interval is Poisson on them, because this is a
    ratio of two small rates and a point estimate would not be a measurement. **The
    over-statement is real in the middle of the window** -- the 0.4 interval excludes 1 -- and
    it is what pulls :math:`\mu` down there, which is the whole of that focus's deficit. Above
    the window it is absent or mild, and the two intervals overlap, so the shape is "well below
    1 at the window, at or just past 1 above it" rather than a peak with two sides. At 0.2
    nothing is reported, so nothing is measured.

    **The mechanism is confirmed by intervention, and the shape of the fix is known.** The
    correction is not a reweighting -- that was tried twice and made every focus worse. It is
    that both limbs must be complementary probabilities of the same event: writing
    :math:`E = P(|g| \ge c \mid \mu)` and :math:`q` for the chance that a study exceeding here
    actually *names* this voxel, a reported pair carries :math:`qE` and a silent one
    :math:`1 - qE`. That is still a proper likelihood, and the arithmetic says where it acts --
    the report limb's score is :math:`(qE)'/(qE) = E'/E`, so :math:`q` cancels there, while the
    silent limb's becomes :math:`-qE'/(1 - qE)`, weakened by roughly :math:`q`. Exactly the
    over-shrinkage above.

    Measured with :math:`q` fixed at 0.56, the reciprocal of the 1.78 at the 0.4 focus, that
    focus closes precisely: -11% to +1%. **But no constant :math:`q` helps overall** -- mean
    absolute error over the four foci runs 8.9%, 9.4%, 10.3%, 11.0% and 12.2% at
    :math:`q` of 1.00, 0.90, 0.80, 0.70 and 0.56, because a scalar lifts the weak foci and
    overshoots the strong ones. The shipped :math:`q = 1` is the best constant.

    So the fix needs :math:`q(\mu)`, the probability that an exceeding voxel is the one actually
    named. **A random-field expected-maxima density is not that function, and gets its sign
    backwards.** The usual clump argument gives :math:`q \sim 1/\text{clump size} \sim u^3`
    for a standardised threshold :math:`u = (c - \mu)/\sigma`, which *falls* as :math:`\mu`
    rises -- :math:`u` runs +1.10, 0.00, -1.10 across the three foci above -- while the measured
    :math:`q` *rises* (0.58, 1.00, 0.89). That density describes a zero-mean field, and these are
    signal peaks: at a strong focus the blob's own curvature makes that voxel the local maximum,
    so :math:`q` approaches 1, while at a marginal focus the noise decides which of several
    exceeding voxels is the maximum. The governing quantity is the signal's curvature against the
    noise smoothness, which a coordinate table does not carry and a reported FWHM does not
    supply. Open, and harder than a missing metadata field.

    Worth stating alongside, because it is easy to assume otherwise: **the indicator channel does
    not dominate the fit.** At the 0.4 focus the observed count alone implies
    :math:`\mu = 0.161` and the images imply about 0.4; the fit lands at 0.337. Below the window
    the count implies a nonsensical :math:`\mu = -0.144` and the fit is within 6% of the truth.
    The images carry the magnitude; the indicators move it.

    The slope of 0.759 is not the foci: it comes from the blob skirts, where the truth runs 0.05
    to 0.2 and both arms are dominated by the floor that reading a map as ``|g|`` imposes. Read
    the per-focus columns, not the slope.

    ``g_marginal`` should be read narrowly: within 4% at the two strong foci and 25% to 33% low
    at the two weak ones, because ``prevalence`` falls toward its floor exactly where few studies
    reported. Prefer it to ``g`` only when comparing against an image-based reference, where it
    is the matching estimand.

    A corollary for reading any three-bin summary of this estimator, including the ones above
    under "The interval": a top bin spanning 0.25 to 0.50 of truth averages voxels whose
    estimate is slightly high with voxels whose estimate is low, and reports the mixture as a
    bias. The slope, the intercept and the per-focus values are the honest summary.

    **What the null tests is not what a reader may expect.** The null is that *within a study,
    effect size is unrelated to location*. A voxel is significant when the image studies'
    effects near it are large relative to what the same studies show elsewhere -- not when the
    pooled effect differs from zero, and not when studies converge there. A collection with a
    genuine effect of the same size everywhere has nothing for this null to find. The
    zero-effect null is not available: a reported peak exists only because it cleared a
    threshold, so "no effect anywhere" predicts no coordinates at all and the observed table
    falsifies it before any voxel is examined; testing it needs subject-level images, which is
    what :footcite:t:`albajes2019meta` imputes in order to permute. Sign-flipping the image
    studies while shuffling the coordinates would test it for part of the collection only, and
    the two hypotheses then combine into a rejection either can cause -- with 20 coordinate and
    5 image analyses the sign flips alone floored the p-value at 1/32 whatever the locations
    said.

    **The collection must supply an image, and one image is a thin basis for a magnitude.**
    ``g`` at a voxel reached by a single image is that image's value corrected by the others'
    silence, with no between-study spread to estimate and no replication to average over. That
    configuration is supported, and measured better than two images on real collections, but the
    magnitude it reports rests on one study's map.

    References
    ----------
    .. footbibliography::
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        design="one-sample",
        tau2_method="dl",
        selection_model="zero-inflated",
        se_method="model",
        interval="wald",
        analysis_mask=None,
        threshold=None,
        clamp_threshold=True,
        coverage_radius=DEFAULT_COVERAGE_RADIUS_MM,
        max_iter=25,
        null_method="permute-images",
        cluster_threshold=0.001,
        n_iters=1000,
        n_cores=1,
        seed=0,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        generate_description=True,
        *,
        mask=None,
    ):
        super().__init__(
            memory=memory, memory_level=memory_level, generate_description=generate_description
        )

        _validate_options(
            design=design,
            tau2_method=tau2_method,
            selection_model=selection_model,
            null_method=null_method,
            threshold=threshold,
            se_method=se_method,
            interval=interval,
        )

        self.design = design
        self.tau2_method = tau2_method
        self.selection_model = selection_model
        self.threshold = threshold
        self.clamp_threshold = clamp_threshold
        self.coverage_radius = coverage_radius
        self.max_iter = max_iter
        self.null_method = null_method
        self.cluster_threshold = cluster_threshold
        self.n_iters = n_iters
        self.n_cores = n_cores
        self.se_method = se_method
        self.interval = interval
        self.analysis_mask = analysis_mask
        self.seed = seed

        if mask is not None:
            mask = get_masker(mask, memory=memory, memory_level=memory_level)
        self.masker = mask

    # ------------------------------------------------------------------ inputs

    def _collect_inputs(self, dataset, drop_invalid=True):
        """Collect the declared inputs, redirecting a collection that has only images.

        Without coordinates there is nothing for this estimator to do that an image-based one
        does not do better. The selection model has no censored observations to explain, the
        kernel has no peaks to spread, and the null has nothing to permute -- an empty focus
        table is invariant under every permutation, so the p-values would look perfectly
        calibrated and mean nothing. The fit would silently reduce to a random-effects
        meta-analysis of the images, which :mod:`nimare.meta.ibma` already does directly and
        with valid inference.

        So this is an error rather than a quiet reduction, and it names the alternatives.
        """
        from nimare.studyset import normalize_collection

        dataset = normalize_collection(dataset)
        coordinates = getattr(dataset, "coordinates", None)
        images = getattr(dataset, "images", None)
        if coordinates is None or not len(coordinates):
            usable_images = (
                images is not None
                and {"g", "g_var"}.issubset(images.columns)
                and bool(images[["g", "g_var"]].notna().all(axis=1).any())
            )
            if usable_images:
                raise ValueError(
                    "This collection has images but no coordinates, and CBES exists to "
                    "correct an image-based estimate by what the coordinate studies did not "
                    "report. With no coordinate table there is no silence to read, so what it "
                    "would return is a random-effects meta-analysis of the images and nothing "
                    "more. Use an image-based estimator instead: "
                    "nimare.meta.ibma.DerSimonianLaird or nimare.meta.ibma.Hedges for random "
                    "effects on beta/varcope maps, WeightedLeastSquares for fixed effects, or "
                    "Stouffers on z maps. CBES is for collections that have coordinate tables "
                    "and at least one study sharing a g image."
                )
        super()._collect_inputs(dataset, drop_invalid=drop_invalid)

    def _preprocess_input(self, dataset):
        """Attach voxel indices and per-study sample sizes to the coordinates table."""
        validate_coordinate_spaces(self.inputs_["coordinates"])
        masker, mask_img = get_masker_mask_image(
            self.masker,
            dataset=dataset,
            message=(
                "A masker is required for coordinate-based meta-analysis. "
                "Provide a `mask` to the Estimator (e.g., CBES(mask=...)) or initialize the "
                "Dataset with a `target` and/or `mask` so `dataset.masker` is defined."
            ),
        )
        self.masker = masker

        xyz = self.inputs_["coordinates"][["x", "y", "z"]].values
        self.inputs_["coordinates"][["i", "j", "k"]] = mm2vox(xyz, mask_img.affine)

        self.inputs_["coordinates"] = _add_metadata_to_dataframe(
            dataset,
            self.inputs_["coordinates"],
            metadata_field=("sample_sizes", "sample_size"),
            target_column="sample_size",
            filter_func=np.sum if self.design == "two-sample" else np.mean,
        )
        self._reported_thresholds_ = self._threshold_metadata(dataset)
        self._analysis_masks_ = self._load_analysis_masks(dataset)

    def _threshold_metadata(self, dataset):
        """Per-study reporting thresholds, when ``threshold`` names a metadata field.

        Papers that state their threshold are the easy case, and they should not be forced
        through an inference that exists only for the ones that do not. Values are read on the
        z scale, the same convention as a float ``threshold``; studies missing the field fall
        back to the median of those that have it.
        """
        if not isinstance(self.threshold, str):
            return None

        available = set(dataset.get_metadata())
        if self.threshold not in available:
            raise ValueError(
                f"threshold={self.threshold!r} is not a metadata field of the collection. "
                f"Available fields: {sorted(available)}."
            )

        ids = np.asarray(dataset.ids, dtype=object)
        values = dataset.get_metadata(field=self.threshold, ids=list(ids))
        cleaned = []
        for value in values:
            if isinstance(value, (list, tuple, np.ndarray)):
                value = float(np.mean(value)) if len(value) else np.nan
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                cleaned.append(np.nan)

        series = pd.Series(cleaned, index=ids, dtype=float)
        series = series[~series.index.duplicated()]
        if not np.isfinite(series.values).any():
            raise ValueError(
                f"Metadata field {self.threshold!r} holds no usable numeric threshold for any "
                "study."
            )
        return series

    def _load_image_studies(self, dataset):
        """Return ``{study_id: (g, var_g)}`` for studies supplying both images.

        Masked to the analysis volume, so the vectors line up with every other per-voxel array
        in the estimator.
        """
        images = getattr(dataset, "images", None)
        if images is None or "g" not in images.columns or "g_var" not in images.columns:
            named = sorted(
                c
                for c in (images.columns if images is not None else [])
                if c not in ("id", "study_id", "contrast_id", "space")
                and not c.endswith("__relative")
                and images[c].notna().any()
            )
            raise ValueError(
                "CBES needs at least one study supplying both a 'g' and a 'g_var' image, and "
                f"this collection supplies none (image columns present: {named or 'none'}). "
                "Coordinates carry no magnitude in this model -- they say where a study "
                "reported and, by omission, where it did not -- so without an image there is "
                "nothing to put on an effect-size scale. CBES reads effect-size maps, not "
                "test statistics: convert them with "
                "nimare.transforms.transform_images(target='g') (and 'g_var'), or label them "
                "value_type='g'/'g_var' in the NIMADS collection."
            )

        loaded = {}
        for study_id, g_path, var_path in zip(
            images["id"].astype(str), images["g"], images["g_var"]
        ):
            # ``pd.isna`` rather than ``is None``: a study with no image carries NaN in these
            # columns, which ``os.path.isfile`` then rejects as the string "nan" -- one
            # spurious "missing on disk" warning per coordinate-only study.
            if pd.isna(g_path) or pd.isna(var_path):
                continue
            if not (os.path.isfile(str(g_path)) and os.path.isfile(str(var_path))):
                LGR.warning(f"Study {study_id} names g images that are missing on disk.")
                continue
            g = self.masker.transform(str(g_path)).ravel().astype(float)
            var_g = self.masker.transform(str(var_path)).ravel().astype(float)
            usable = np.isfinite(g) & np.isfinite(var_g) & (var_g > 0)
            if not usable.any():
                LGR.warning(f"Study {study_id} has no usable g image voxels.")
                continue
            g = np.where(usable, g, 0.0)
            var_g = np.where(usable, var_g, np.inf)
            loaded[study_id] = (g, var_g, usable)

        if not loaded:
            raise ValueError(
                "CBES needs at least one study supplying both a 'g' and a 'g_var' image. This "
                "collection names those value types but none of them could be read -- see the "
                "warnings above for which studies and why. Coordinates carry no magnitude in "
                "this model, so without an image there is nothing to put on an effect-size "
                "scale."
            )

        total = len(set(images["id"].astype(str)))
        rest = "" if len(loaded) >= total else "; the rest contribute silence only"
        LGR.info(f"Magnitudes from {len(loaded)} image studies{rest}.")
        return loaded

    def _load_analysis_masks(self, dataset):
        """Return ``{study_id: examined}`` for studies declaring which voxels they analysed.

        Read from an image whose ``value_type`` matches ``analysis_mask``, nonzero meaning
        examined. Without one, a study is assumed to have examined the whole analysis volume,
        which is what the censoring term has always assumed of every study.

        Keyed per *analysis*, not per study, matching the unit the rest of the estimator uses
        for ``n_studies`` and for the censoring roster. A paper contributing several contrasts
        therefore has to declare the mask on each one it applies to; declaring it on some and
        not others leaves the others' silence read as evidence.

        This is what an ROI or partial-coverage study needs. Its silence outside the region it
        analysed is not evidence that nothing is there -- it never looked -- and the censoring
        term would otherwise read it as evidence against an effect. The only previous remedy was
        ``selection_model="none"`` for the entire collection, which throws away the correction
        for the studies that did examine the whole brain.
        """
        if not self.analysis_mask:
            return {}

        images = getattr(dataset, "images", None)
        available = [] if images is None else [c for c in images.columns if c != "id"]
        if images is None or self.analysis_mask not in images.columns:
            # Requested and not found is a silent no-op otherwise, and the ways to land here are
            # easy: a typo, or a loader that skipped the value type because it is not one of the
            # ones NiMARE recognises. Both leave every study's silence read as evidence, which
            # is the behaviour the caller asked to switch off.
            LGR.warning(
                f"analysis_mask={self.analysis_mask!r} matches no image value type in this "
                f"collection, so every study is treated as whole-brain and partial coverage is "
                f"not honoured. Value types present: {sorted(available)}."
            )
            return {}

        loaded = {}
        for study_id, path in zip(images["id"].astype(str), images[self.analysis_mask]):
            if path is None or not os.path.isfile(str(path)):
                continue
            values = self.masker.transform(str(path)).ravel()
            examined = np.isfinite(values) & (values != 0)
            if not examined.any():
                # Kept, not dropped. Dropping it fell back to whole-brain, which inverts what
                # the mask says: a study that examined nothing outside the analysis volume
                # contributes no value and no silence, rather than everything.
                LGR.warning(
                    f"Study {study_id} declares an analysis mask covering no in-mask voxel, so "
                    "it contributes neither a value nor a silence anywhere."
                )
            if examined.all():
                continue  # whole-brain, which is the default anyway
            loaded[study_id] = examined

        if loaded:
            LGR.info(
                f"{len(loaded)} studies declare a partial analysis mask; their silence outside "
                "it is not read as evidence."
            )
        return loaded

    def _build_focus_table(self):
        """Reduce the coordinates table to what the model consumes: positions and study ids.

        A focus is now a statement that a study *reported something here*, and nothing more. Its
        reported height is not read, so no statistic column is required and none is looked for --
        which is what lets this estimator consume the tables the coordinate literature actually
        publishes, most of which tabulate a location and nothing usable beside it.

        ``sample_size`` is still required, because a silence is only informative against the
        precision of the study that stayed silent, and that comes from its N.
        """
        coords = self.inputs_["coordinates"]
        if "sample_size" not in coords.columns:
            raise ValueError(
                "CBES needs a sample size for every study: a study's silence is only "
                "informative against its own precision. Populate the metadata field "
                "'sample_sizes' or 'sample_size'."
            )

        usable = coords["sample_size"].notna()
        dropped = int((~usable).sum())
        if dropped:
            level = LGR.warning if self._drop_invalid else LGR.info
            level(f"Dropping {dropped} of {len(coords)} foci with no sample size.")
        if not usable.any() and not getattr(self, "_image_studies_", None):
            raise ValueError(
                "No focus has a sample size and no study supplies an image; nothing to fit."
            )
        return coords.loc[usable, ["id", "i", "j", "k", "sample_size"]].copy()

    def _size_reduction(self):
        """How a study's per-group sample sizes collapse to the number the model wants.

        A two-sample design's N is a *total*, split into equal groups downstream, so the
        reduction has to be a sum: metadata of ``[30, 30]`` means sixty subjects, and reducing
        it by mean gave thirty, which was then read as two groups of fifteen. That error
        reached the sampling variances, the censoring cutoffs and the null variances alike --
        it inflated the old peak conversion's ``g`` by 39% at ``t = 3``. A lone value is
        already a total either way, so summing is right in both cases.

        One-sample designs want the mean, which is what a single number or a repeated one gives.
        """
        return "sum" if self.design == "two-sample" else "mean"

    def _all_sample_sizes(self, dataset):
        """Sample size of every analysis in the collection, reporting foci or not.

        This is the difference between an effect-size meta-analysis and a convergence one. A
        study that reported no peak at all never reaches ``inputs_`` -- ``_collect_inputs``
        drops it as having no coordinates (see neurostuff/NiMARE#294) -- but for this model its
        silence is the single most informative observation about how common the effect is. So
        the roster of studies comes from the collection, and only the reported values come from
        the coordinates table.
        """
        ids = np.asarray(dataset.ids, dtype=object)
        sample_sizes = np.asarray(dataset.sample_sizes(reduce=self._size_reduction()), dtype=float)
        series = pd.Series(sample_sizes, index=ids)
        series = series[series.notna()]
        if series.empty:
            raise ValueError(
                "CBES needs a sample size for every study in order to place reported "
                "statistics on an effect-size scale. Populate the metadata field "
                "'sample_sizes' or 'sample_size'."
            )
        return series[~series.index.duplicated()]

    def _study_cutoffs_z(self, table, sample_sizes):
        """Per-study reporting threshold on the z scale, one entry per study on the roster.

        This is the one number a silence cannot do without. "Study k reported nothing near voxel
        v" is only evidence about the effect there if we know how large an effect would have had
        to be for k to report it, and that is the threshold k applied.

        It can no longer be inferred. The old rules read it off the smallest reported statistic,
        undoing the order statistic for the number of peaks; with the heights no longer read,
        there is nothing to read it from. So it is **supplied or assumed**, which is also the
        honest position: the earlier inference was measured at 0.201 of prevalence error against
        0.008 for a fixed constant on cluster-extent tables, because it cannot tell a
        cluster-forming cut from a voxelwise one and overshoots the first by about 1 z.

        Pass the field name to ``threshold`` to use per-study values from metadata, a float to
        apply one cut to every study, or leave it at the default two-tailed p < 0.001.
        """
        index = sample_sizes.index
        if isinstance(self.threshold, str):
            supplied = getattr(self, "_reported_thresholds_", None)
            if supplied is None:
                raise ValueError(
                    f"threshold={self.threshold!r} names a metadata field, but no per-study "
                    "thresholds were read from the collection. Supply that field, or pass a "
                    "float to assume one cut for every study."
                )
            finite = supplied.values[np.isfinite(supplied.values)]
            fallback = float(np.nanmedian(finite)) if finite.size else np.nan
            cutoff_z = supplied.reindex(index).astype(float).fillna(fallback).values
        else:
            value = DEFAULT_REPORTING_THRESHOLD_Z if self.threshold is None else self.threshold
            cutoff_z = np.full(len(index), float(value))

        cutoff_z = np.asarray(cutoff_z, dtype=float)
        cutoff_z = np.where(
            np.isfinite(cutoff_z) & (cutoff_z > 0), cutoff_z, DEFAULT_REPORTING_THRESHOLD_Z
        )

        if self.clamp_threshold:
            bound = (
                reported_minimum_z(self.inputs_["coordinates"]).reindex(index).astype(float).values
            )
            usable = np.isfinite(bound) & (bound > 0)
            lowered = int(np.sum(usable & (bound < cutoff_z - 1e-9)))
            cutoff_z = np.where(usable, np.minimum(cutoff_z, bound), cutoff_z)
            if lowered:
                LGR.info(
                    f"Lowered the assumed reporting threshold for {lowered} of {len(index)} "
                    "studies to their own smallest reported statistic, which they must have "
                    "cleared. Pass clamp_threshold=False to use the assumed value as given."
                )
        return pd.Series(cutoff_z, index=index)

    def _accumulate(self, table, image_studies=None):
        """Walk the image studies, returning per-study voxel contributions and voxel sums.

        **Only images contribute a magnitude.** A coordinate table says where a study reported
        and, by omission, where it did not; the pooled mean is built from the images alone and
        the coordinates enter through the selection model instead. So there is no kernel here
        and no per-focus value: a reported height was the only thing a kernel had to spread,
        and spreading it was measured to cost accuracy on every collection tested.

        ``table`` is still taken, and still decides the silence geometry downstream, so the
        signature does not change and the caller does not have to know which channel is which.
        """
        mask_img = self.masker.mask_img
        mask_flat_to_masked = _get_mask_flat_to_masked(mask_img)
        n_voxels = int(mask_flat_to_masked.max()) + 1 if mask_flat_to_masked.size else 0

        sums = {
            name: np.zeros(n_voxels, dtype=float)
            for name in ("w", "w2", "a", "a2", "ag", "ag2", "w2_over_s2", "n")
        }
        contributions = []

        for study_id, (g, var_g, usable) in (image_studies or {}).items():
            cols = np.flatnonzero(usable).astype(np.int64)
            if not cols.size:
                continue
            weights = np.ones(cols.size, dtype=float)
            contributions.append((study_id, cols, weights, g[cols], var_g[cols]))

            a = weights / var_g[cols]
            for name, value in (
                ("w", weights),
                ("w2", weights),
                ("a", a),
                ("a2", a**2),
                ("ag", a * g[cols]),
                ("ag2", a * g[cols] ** 2),
                ("w2_over_s2", a),
                ("n", weights),
            ):
                sums[name][cols] += value

        if not contributions:
            raise ValueError("No study contributed any in-mask voxels.")

        return contributions, sums, n_voxels

    # ----------------------------------------- pooling and the selection model

    def _pool(self, table, image_studies=None):
        """Run the two-pass local random-effects fit. Returns a dict of masked-voxel arrays."""
        contributions, sums, n_voxels = self._accumulate(table, image_studies)

        if self.tau2_method == "dl":
            tau2 = _local_dersimonian_laird(
                sums["w"],
                sums["a"],
                sums["a2"],
                sums["ag"],
                sums["ag2"],
                sums["w2_over_s2"],
                sums["n"],
            )
        else:
            tau2 = np.zeros(n_voxels, dtype=float)

        # Second pass: tau2 is voxel-specific, so the pooling weights cannot be accumulated
        # alongside the moment sums above.
        numerator = np.zeros(n_voxels, dtype=float)
        denominator = np.zeros(n_voxels, dtype=float)
        variance_numerator = np.zeros(n_voxels, dtype=float)
        # Sum of a_k g_k^2, which turns into the weighted residual sum of squares without a
        # third pass: sum a (g - ghat)^2 = sum a g^2 - ghat^2 sum a, because ghat is itself
        # sum(a g) / sum(a).
        weighted_square = np.zeros(n_voxels, dtype=float)

        for _, cols, weights, g, var_g in contributions:
            total_var = var_g + tau2[cols]
            pooling_weight = weights / total_var
            numerator += np.bincount(cols, weights=pooling_weight * g, minlength=n_voxels)
            denominator += np.bincount(cols, weights=pooling_weight, minlength=n_voxels)
            variance_numerator += np.bincount(
                cols, weights=weights**2 / total_var, minlength=n_voxels
            )
            weighted_square += np.bincount(
                cols, weights=pooling_weight * g * g, minlength=n_voxels
            )

        covered = denominator > 0
        g_hat = np.zeros(n_voxels, dtype=float)
        se = np.full(n_voxels, np.inf, dtype=float)
        g_hat[covered] = numerator[covered] / denominator[covered]
        se[covered] = np.sqrt(variance_numerator[covered]) / denominator[covered]

        n_eff = np.zeros(n_voxels, dtype=float)
        positive_w = sums["w2"] > 0
        n_eff[positive_w] = sums["w"][positive_w] ** 2 / sums["w2"][positive_w]

        if self.se_method == "hksj":
            se = _hartung_knapp_se(
                g_hat=g_hat,
                sum_a=denominator,
                sum_a_g2=weighted_square,
                n_eff=n_eff,
                covered=covered,
                fallback=se,
            )

        return {
            "contributions": contributions,
            "n_voxels": n_voxels,
            "covered": covered,
            "g": g_hat,
            "se": se,
            "tau2": tau2,
            "n_studies": sums["n"],
            "n_eff": n_eff,
            "denominator": denominator,
            "sum_w": sums["w"],
        }

    def _indicator_entries(self, table, study_ids, active, n_voxels, image_ids=()):
        """Pair each voxel with every coordinate study's reporting indicator there.

        Returns ``(voxel, study_position, sign)``: ``sign = +1`` where the study has no focus
        within ``coverage_radius`` of the voxel, ``-1`` at a voxel the study actually named.
        Both values are the coordinate channel -- a table says whether a study reported at a
        location, and that is all this estimator reads from it.

        **The two assert over different extents, and that is the point.** A silence is a
        statement about a neighbourhood: nothing within the radius cleared this study's cut. A
        report is a statement about one voxel, because a reported peak is a local maximum
        selected for being large and displaced from wherever the effect actually is -- so
        "someone reported 18 mm away" is not evidence that the effect *here* cleared anything.
        Measured on a known truth, asserting the report across the sphere gave an rmse of
        0.457 against 0.114 for the named voxel alone.

        Omitting the ``-1`` limb entirely is worse than including it at the named voxel: the
        silent pairs are then the only evidence about the indicator, so the model reads the
        observed silence fraction against a denominator that excludes every study that
        reported, and over-shrinks. Where the truth is largest that cost 0.091 of rmse against
        0.074 and -0.060 of bias against -0.042.

        Four kinds of pair are omitted rather than given an indicator:

        * **An image study's**, at every voxel. Its magnitude enters through its value, and its
          map reports everywhere, so it has no indicator to contribute.
        * **A voxel a study never examined**, where ``analysis_mask`` says so. An ROI study's
          silence outside its region is not evidence that nothing is there.
        * **A voxel outside the analysis volume**, which nothing reads.
        * **A voxel a study reached but did not name**, which is neither silent nor reported.

        The radius is separate from anything the pooled mean uses, and deliberately generous: a
        study whose peak sits 6 mm away has plainly not been silent about the region, and
        papers do not report cluster extent reliably enough to read the true neighbourhood off
        the table.
        """
        mask_img = self.masker.mask_img
        # ``shape[:3]``: a mask image may carry a trailing singleton volume axis.
        shape = np.asarray(mask_img.shape[:3], dtype=np.int64)

        radius = (
            DEFAULT_COVERAGE_RADIUS_MM if self.coverage_radius is None else self.coverage_radius
        )
        offsets = sphere_kernel_offsets(radius, mask_img.header.get_zooms()[:3])

        # Dilation on a padded grid, so that a study's reached voxels come out of one add and
        # one gather per (focus, sphere offset) pair rather than an array of candidate
        # coordinates and six comparisons against the shape. With a 20 mm sphere that array is
        # the largest thing this method would otherwise allocate.
        padded_lookup, padded_shape, pad = _padded_flat_to_masked(mask_img, offsets)
        padded_strides = np.array(
            [padded_shape[1] * padded_shape[2], padded_shape[2], 1], dtype=np.int64
        )
        flat_offsets = offsets.astype(np.int64) @ padded_strides
        reach = np.abs(offsets).max(axis=0)

        active_lookup = np.full(n_voxels, -1, dtype=np.int64)
        active_lookup[active] = np.arange(active.size)
        # Reused across studies to deduplicate the voxels a study's spheres reach. A scratch
        # bitmap costs one pass over the hits; ``np.unique`` sorts or hashes them, and with
        # a 20 mm sphere per focus there are a great many hits.
        reached_flag = np.zeros(active.size, dtype=bool)

        image_ids = set(image_ids)
        analysis_masks = getattr(self, "_analysis_masks_", None) or {}
        cols, positions, signs = [], [], []
        for position, study_id in enumerate(study_ids):
            if study_id in image_ids:
                continue
            examined = analysis_masks.get(study_id)
            if examined is None:
                candidates = np.arange(active.size, dtype=np.int64)
            else:
                candidates = np.flatnonzero(examined[active]).astype(np.int64)
                if not candidates.size:
                    continue

            ijk = table.loc[table["id"] == study_id, ["i", "j", "k"]].values.astype(np.int64)
            # A focus further outside the image than the sphere's own reach cannot touch an
            # in-mask voxel, so dropping it here loses nothing and keeps every remaining index
            # inside the padded grid.
            if ijk.size:
                ijk = ijk[np.all((ijk >= -reach) & (ijk < shape + reach), axis=1)]
            if ijk.size:
                base = (ijk + pad) @ padded_strides
                hit = padded_lookup[(base[:, None] + flat_offsets).ravel()]
                hit = hit[hit >= 0].astype(np.int64)
                local = active_lookup[hit]
                local = local[local >= 0]
                reached_flag[local] = True

            # A report asserts its indicator at the voxel it names and nowhere else, while a
            # silence asserts one over the whole neighbourhood. That asymmetry is not a
            # convenience: a silence really is a statement about a region -- nothing within
            # the radius cleared the cut -- whereas "someone reported 18 mm away" says nothing
            # about the effect here, the reported peak being a local maximum selected for
            # being large and displaced from wherever the effect is.
            #
            # Swept, and there is no middle ground: the named voxel is optimal and the penalty
            # starts at the first ring. Against a known truth, rmse where the effect is ran
            # 0.070 at the named voxel, 0.113 at 4 mm, 0.122 at 6 mm and 0.126 at 20 mm, and
            # the bias flipped from -0.039 to +0.094 at 4 mm alone -- one ring overshoots by
            # more than the original undershoot, because a 4 mm sphere asserts the indicator at
            # seven voxels rather than one. A voxel a study reached but did not name therefore
            # gets no indicator either way.
            at_focus = np.zeros(active.size, dtype=bool)
            if ijk.size:
                named = padded_lookup[(ijk + pad) @ padded_strides]
                named = named[named >= 0].astype(np.int64)
                local = active_lookup[named]
                at_focus[local[local >= 0]] = True
            sign = np.where(
                at_focus[candidates], -1.0, np.where(reached_flag[candidates], 0.0, 1.0)
            )
            reached_flag[:] = False
            informative = sign != 0
            candidates, sign = candidates[informative], sign[informative]
            if not candidates.size:
                continue
            cols.append(candidates)
            positions.append(np.full(candidates.size, position, dtype=np.int64))
            signs.append(sign)

        if not cols:
            empty = np.array([], dtype=np.int64)
            return empty, empty, np.array([], dtype=float)

        return np.concatenate(cols), np.concatenate(positions), np.concatenate(signs)

    def _value_entries(self, fit, study_ids, active, n_voxels):
        """``(local_voxel, study_position, w, g, var)`` for every voxel the kernel reaches."""
        position = {study_id: i for i, study_id in enumerate(study_ids)}
        active_lookup = np.full(n_voxels, -1, dtype=np.int64)
        active_lookup[active] = np.arange(active.size)

        parts = {name: [] for name in ("col", "pos", "w", "g", "var")}
        for study_id, cols, weights, g, var_g in fit["contributions"]:
            local = active_lookup[cols]
            keep = local >= 0
            if not np.any(keep):
                continue
            parts["col"].append(local[keep])
            parts["pos"].append(np.full(int(keep.sum()), position[study_id], dtype=np.int64))
            parts["w"].append(weights[keep])
            parts["g"].append(g[keep])
            parts["var"].append(var_g[keep])

        if not parts["col"]:
            empty = np.array([], dtype=np.int64)
            return {name: empty for name in parts}

        return {name: np.concatenate(values) for name, values in parts.items()}

    def _apply_selection_model(self, fit, table, thresholds, sample_sizes, image_ids=()):
        r"""Refit each voxel under the selection model, replacing the naive weighted mean.

        Two quantities come out, and keeping them apart is the point of the model:

        ``prevalence``
            :math:`\pi(v)`, the fraction of studies with a non-null effect here. This is what
            a convergence-based estimator is implicitly measuring.
        ``g``
            :math:`\mu(v)`, the effect size *among the studies that have an effect*.

        A study silent in this region either has no effect here or has one that failed to clear
        its reporting threshold; the zero-inflated model lets the data decide, so silence need
        not be explained as a small-but-real common effect -- which is what drags a plain Tobit
        fit below the truth. ``tau2`` is held at its moment estimate, so each EM iteration
        optimizes only :math:`\mu` alongside a closed-form update for :math:`\pi`.
        """
        active = np.flatnonzero(fit["covered"])
        n_voxels = fit["n_voxels"]
        fit["prevalence"] = np.zeros(n_voxels, dtype=float)
        if not active.size:
            return

        study_ids = list(sample_sizes.index)
        # Which studies were silent where is decided by the foci positions, which the
        # permutation null never moves, so this survives across refits like the geometry does.
        #
        # The key has to name everything that decides coverage, not just its shape. It used to
        # be (active extent, number of analyses), which the calibration sequence defeats: that
        # fits the coordinates alone and then each image donor alone, with the same roster and
        # the same active voxels but an empty focus table, so a donor fit reused the coordinate
        # fit's censoring matrix. Measured donor-only estimates of [0.2840, 0.3708] against
        # [0.2658, 0.3421] once invalidated.
        masks = getattr(self, "_analysis_masks_", None) or {}
        coverage_key = (
            active.size,
            int(active[0]),
            int(active[-1]),
            tuple(study_ids),
            tuple(sorted(image_ids)),
            table[["i", "j", "k"]].values.astype(np.int64).tobytes() if len(table) else b"",
            np.asarray(table["id"].values, dtype=object).tobytes() if len(table) else b"",
            tuple(sorted((study, mask.tobytes()) for study, mask in masks.items())),
        )
        cached = getattr(self, "_coverage_", None)
        if cached is not None and cached[0] == coverage_key:
            ind_col, ind_pos, ind_sign = cached[1]
        else:
            ind_col, ind_pos, ind_sign = self._indicator_entries(
                table, study_ids, active, n_voxels, image_ids=image_ids
            )
            self._coverage_ = (coverage_key, (ind_col, ind_pos, ind_sign))
        values = self._value_entries(fit, study_ids, active, n_voxels)
        n_studies = len(study_ids)

        # No rho: the observations are not rescaled, so the null component needs no matching
        # factor. It used to be multiplied by each study's peak-height correction to stay on the
        # same axis as its rescaled reported heights, and there are no reported heights now.
        null_var = null_effect_variance(sample_sizes.values, design=self.design)[:, None]
        cutoffs = np.abs(thresholds.loc[study_ids].values)[:, None]
        value_order = np.argsort(values["col"], kind="mergesort")
        values = {name: array[value_order] for name, array in values.items()}
        ind_order = np.argsort(ind_col, kind="mergesort")
        ind_col, ind_pos, ind_sign = ind_col[ind_order], ind_pos[ind_order], ind_sign[ind_order]

        mu_out = np.zeros(active.size, dtype=float)
        pi_out = np.zeros(active.size, dtype=float)
        se_out = np.full(active.size, np.inf, dtype=float)
        se_marginal_out = np.full(active.size, np.inf, dtype=float)
        share_out = np.zeros(active.size, dtype=float)
        lower_out = np.full(active.size, -np.inf, dtype=float)
        upper_out = np.full(active.size, np.inf, dtype=float)

        # Dense blocks are (n_studies, chunk); cap their element count rather than their width
        # so that a studyset with many experiments simply takes more, smaller chunks.
        chunk = max(1, int(_SELECTION_CHUNK_ELEMENTS // max(n_studies, 1)))
        for lo in range(0, active.size, chunk):
            hi = min(lo + chunk, active.size)
            width = hi - lo

            weights = np.zeros((n_studies, width), dtype=float)
            g_obs = np.zeros((n_studies, width), dtype=float)
            var_obs = np.ones((n_studies, width), dtype=float)
            v_lo, v_hi = np.searchsorted(values["col"], [lo, hi])
            if v_hi > v_lo:
                rows = values["pos"][v_lo:v_hi]
                cols = values["col"][v_lo:v_hi] - lo
                weights[rows, cols] = values["w"][v_lo:v_hi]
                g_obs[rows, cols] = values["g"][v_lo:v_hi]
                var_obs[rows, cols] = values["var"][v_lo:v_hi]

            # 0 means "no indicator here": an image study, a voxel nobody examined, or a
            # study off this chunk. +1 is silent, -1 reported nearby with its height discarded.
            indicator = np.zeros((n_studies, width), dtype=float)
            c_lo, c_hi = np.searchsorted(ind_col, [lo, hi])
            if c_hi > c_lo:
                indicator[ind_pos[c_lo:c_hi], ind_col[c_lo:c_hi] - lo] = ind_sign[c_lo:c_hi]

            mu, pi, se, se_marginal, share, lower, upper = self._fit_chunk(
                weights=weights,
                g_obs=g_obs,
                var_obs=var_obs,
                indicator=indicator,
                tau2=fit["tau2"][active[lo:hi]],
                null_var=null_var,
                cutoffs=cutoffs,
                start=fit["g"][active[lo:hi]],
            )
            mu_out[lo:hi], pi_out[lo:hi] = mu, pi
            se_out[lo:hi], se_marginal_out[lo:hi] = se, se_marginal
            share_out[lo:hi] = share
            if lower is not None:
                lower_out[lo:hi], upper_out[lo:hi] = lower, upper

        fit["g"] = np.zeros(n_voxels, dtype=float)
        fit["g"][active] = mu_out
        fit["prevalence"][active] = pi_out
        fit["se"] = np.full(n_voxels, np.inf, dtype=float)
        fit["se"][active] = se_out
        fit["se_marginal"] = np.full(n_voxels, np.inf, dtype=float)
        fit["se_marginal"][active] = se_marginal_out
        fit["coordinate_share"] = np.zeros(n_voxels, dtype=float)
        fit["coordinate_share"][active] = share_out
        if self.interval == "profile":
            fit["g_lower"] = np.full(n_voxels, -np.inf, dtype=float)
            fit["g_lower"][active] = lower_out
            fit["g_upper"] = np.full(n_voxels, np.inf, dtype=float)
            fit["g_upper"][active] = upper_out

    def _working_sets(self, *, weights, g_obs, var_obs, indicator, tau2, null_var, cutoffs):
        """Split the block into the value-bearing and indicator-bearing pairs the EM uses.

        Works on the pairs that carry information rather than on the dense study-by-voxel
        block. A study either supplied a value here (an image), or carries a reporting
        indicator here (a coordinate table, silent or not), or carries neither -- and the third
        case is common, because a voxel nobody examined and an image study's own indicator both
        fall into it. Evaluating normal CDFs across the full block and then multiplying most of
        them by zero was 97% of the runtime.
        """
        width = weights.shape[1]
        value_pairs = np.flatnonzero(weights > 0)
        indicator_pairs = np.flatnonzero(indicator != 0)
        rep_voxel = value_pairs % width
        ind_voxel, ind_study = indicator_pairs % width, indicator_pairs // width

        var_rep = var_obs.ravel()[value_pairs]
        sigma_rep = np.sqrt(var_rep + tau2[rep_voxel])
        g_rep = g_obs.ravel()[value_pairs]
        w_rep = weights.ravel()[value_pairs]

        sign_ind = indicator.ravel()[indicator_pairs]
        cutoff_ind = cutoffs.ravel()[ind_study]
        null_var_ind = null_var.ravel()[ind_study]
        inv_sigma_ind = 1.0 / np.sqrt(null_var_ind + tau2[ind_voxel])

        # An indicator pair enters at the weight of an average value-bearing study at this
        # voxel, rather than at 1. Values arrive at whatever weight the image gave them, so an
        # indicator entering at full weight would count for more than a study that actually
        # measured something -- the indicators would outvote the evidence.
        n_values = np.bincount(rep_voxel, minlength=width)
        sum_values = np.bincount(rep_voxel, weights=w_rep, minlength=width)
        value_scale = np.divide(sum_values, n_values, out=np.ones(width), where=n_values > 0)

        # Probability of the observed indicator when the study has no effect at all: silence is
        # near-certain whenever the threshold is several sigma, and reporting near-impossible.
        # Fixed across iterations, because it does not depend on mu.
        silent_null = ndtr(cutoff_ind / np.sqrt(null_var_ind)) - ndtr(
            -cutoff_ind / np.sqrt(null_var_ind)
        )

        reporting_pairs = _ReportingPairs(
            voxel=rep_voxel,
            weight=w_rep,
            g=g_rep,
            sigma=sigma_rep,
            precision=1.0 / sigma_rep**2,
            density_null=_normal_pdf(g_rep / np.sqrt(var_rep)) / np.sqrt(var_rep),
            responsibility=np.ones(value_pairs.size),
        )
        censored_pairs = _IndicatorPairs(
            voxel=ind_voxel,
            weight=value_scale[ind_voxel],
            sign=sign_ind,
            inv_sigma=inv_sigma_ind,
            inv_sigma_sq=inv_sigma_ind * inv_sigma_ind,
            cutoff_scaled=cutoff_ind * inv_sigma_ind,
            twice_cutoff_scaled=cutoff_ind * inv_sigma_ind * 2.0,
            prob_event_null=np.clip(
                np.where(sign_ind > 0, silent_null, 1.0 - silent_null),
                _PROBABILITY_FLOOR,
                None,
            ),
            responsibility=np.ones(indicator_pairs.size),
        )
        return reporting_pairs, censored_pairs

    @staticmethod
    def _update_prevalence(reporting, silent, mu, pi, total_weight, censoring):
        """One E-step: the responsibilities, the prevalence they imply, and the log-likelihood.

        The log-likelihood comes free with the E step. Each observation's mixture density is
        exactly the normaliser the responsibility is divided by, so summing its log onto voxels
        costs two logarithms and two ``bincount`` calls, against the two normal CDFs per silent
        pair that the iteration is already paying for.
        """
        pi_rep, pi_sil = pi[reporting.voxel], pi[silent.voxel]

        density_effect = (
            _normal_pdf((reporting.g - mu[reporting.voxel]) / reporting.sigma) / reporting.sigma
        )
        resp_rep = pi_rep * density_effect
        mixture_rep = resp_rep + (1.0 - pi_rep) * reporting.density_null + _LOGP_FLOOR
        resp_rep = resp_rep / mixture_rep
        resp_sil = pi_sil * censoring["prob"]
        mixture_sil = resp_sil + (1.0 - pi_sil) * silent.prob_event_null + _LOGP_FLOOR
        resp_sil = resp_sil / mixture_sil

        log_likelihood = np.bincount(
            reporting.voxel, weights=reporting.weight * np.log(mixture_rep), minlength=mu.size
        ) + np.bincount(
            silent.voxel, weights=silent.weight * np.log(mixture_sil), minlength=mu.size
        )

        claimed = np.bincount(
            reporting.voxel, weights=reporting.weight * resp_rep, minlength=mu.size
        ) + np.bincount(silent.voxel, weights=silent.weight * resp_sil, minlength=mu.size)
        updated = np.clip(
            np.divide(claimed, total_weight, out=np.zeros(mu.size), where=total_weight > 0),
            _PREVALENCE_CLAMP,
            1.0 - _PREVALENCE_CLAMP,
        )
        return resp_rep, resp_sil, updated, log_likelihood

    def _profile_log_likelihood(self, reporting, silent, total_weight, identified, mu):
        r"""Log-likelihood at ``mu`` with the prevalence maximised out, one value per voxel.

        The inner maximisation is the same E step the fit already uses. That is sound rather
        than convenient: the mixture log-likelihood is a sum of logs of terms *linear* in
        :math:`\pi`, so it is concave in :math:`\pi` alone and the fixed point is the global
        conditional maximum. The censoring probabilities depend on ``mu`` and not on
        :math:`\pi`, so the two normal CDFs per silent pair are paid once for the whole inner
        loop, which is what makes a profile affordable here at all.
        """
        censoring = silent.censoring(mu)
        pi = np.where(identified, 0.5, 1.0)
        log_likelihood = np.zeros(mu.size)
        for _ in range(_PROFILE_INNER_ITERS):
            _, _, updated, log_likelihood = self._update_prevalence(
                reporting, silent, mu, pi, total_weight, censoring
            )
            pi = np.where(identified, updated, 1.0)
        # _update_prevalence reports the likelihood at the pi it was given, so one more pass is
        # needed to read it at the converged one.
        _, _, _, log_likelihood = self._update_prevalence(
            reporting, silent, mu, pi, total_weight, censoring
        )
        return log_likelihood

    def _profile_bound(
        self, *, reporting, silent, total_weight, identified, mu_hat, peak, scale, direction
    ):
        r"""Edge of the likelihood region containing ``mu_hat``, in one direction.

        Walked outward on a grid of multiples of ``scale`` and then interpolated on the
        deficit. The grid costs one censoring evaluation per point against the fit's own
        twenty-five, and a bisection would spend most of its steps re-deciding voxels whose
        bound was already bracketed.

        **The deficit is not monotone, and what is returned is therefore the edge of the
        connected component around** ``mu_hat``, not the supremum of the region. As
        :math:`\pi \to 0` the active component explains nothing and the mixture density tends
        to the null one at every observation, so the profile log-likelihood has a *horizontal
        asymptote* at the null-only value, independent of :math:`\mu`. The deficit therefore
        rises away from the maximum and then falls back to
        :math:`2(\ell(\hat\mu, \hat\pi) - \ell_{\mathrm{null}})`. Two consequences, both
        intended:

        * If that asymptote sits below the critical value -- equivalently, if the data do not
          reject :math:`\pi = 0` -- no crossing is ever found and the bound is reported
          infinite. That is the correct answer and not a failure of the search: the data really
          do not exclude an arbitrarily large effect present in almost no studies.
        * If the deficit does cross and later falls back, the region is disconnected. The first
          crossing is reported, which is the component the estimate lives in and the only part
          a reader can act on, but it is a lower bound on the width rather than the width.
        """
        width = mu_hat.size
        inside_at = np.zeros(width)
        inside_deficit = np.zeros(width)
        outside_at = np.full(width, np.nan)
        outside_deficit = np.full(width, np.nan)

        for multiple in _PROFILE_MULTIPLES:
            pending = np.isnan(outside_at)
            if not pending.any():
                break
            trial = mu_hat + direction * multiple * scale
            deficit = 2.0 * (
                peak
                - self._profile_log_likelihood(reporting, silent, total_weight, identified, trial)
            )
            # A negative deficit means mu_hat was not quite the maximiser. The region is still
            # the right set to report, so the floor keeps the interpolation monotone instead of
            # discarding the voxel.
            np.maximum(deficit, 0.0, out=deficit)
            crossed = pending & (deficit >= _PROFILE_CRITICAL)
            outside_at[crossed] = multiple
            outside_deficit[crossed] = deficit[crossed]
            held = pending & ~crossed
            inside_at[held] = multiple
            inside_deficit[held] = deficit[held]

        span = outside_deficit - inside_deficit
        fraction = np.divide(
            _PROFILE_CRITICAL - inside_deficit,
            span,
            out=np.zeros(width),
            where=np.isfinite(span) & (span > 0),
        )
        np.clip(fraction, 0.0, 1.0, out=fraction)
        reached = np.isfinite(outside_at)
        multiple = np.where(reached, inside_at + (outside_at - inside_at) * fraction, np.inf)
        return mu_hat + direction * multiple * scale

    def _profile_interval(self, *, reporting, silent, total_weight, identified, mu_hat, se):
        """Lower and upper profile-likelihood bounds on ``mu``, one pair per voxel.

        The ``se`` sets only the *scale* of the search, not the answer: it is used to choose
        where to look for the crossing, and a quantity known to be miscalibrated by up to a
        factor of two is still a perfectly good ruler for that. Where it is not finite the
        search falls back to a fixed span.
        """
        scale = np.where(np.isfinite(se) & (se > 0), se, _PROFILE_FALLBACK_SCALE)
        peak = self._profile_log_likelihood(reporting, silent, total_weight, identified, mu_hat)
        shared = dict(
            reporting=reporting,
            silent=silent,
            total_weight=total_weight,
            identified=identified,
            mu_hat=mu_hat,
            peak=peak,
            scale=scale,
        )
        lower = self._profile_bound(direction=-1.0, **shared)
        upper = self._profile_bound(direction=1.0, **shared)
        return lower, upper

    def _fit_chunk(self, *, weights, g_obs, var_obs, indicator, tau2, null_var, cutoffs, start):
        """EM for one block of voxels. Returns ``(mu, prevalence, se)``, one value per voxel.

        Voxels converge at very different rates: most settle within a handful of iterations
        while a few drift for dozens. Iterating the whole block until the slowest voxel is done
        wastes nearly all of the work, and stopping on a global criterion instead leaves the
        stragglers short of the MLE. So settled voxels are retired from the working set and the
        rest keep going, which is what the ``compact`` calls below are doing.
        """
        zero_inflated = self.selection_model == "zero-inflated"
        width = weights.shape[1]
        reporting, silent = self._working_sets(
            weights=weights,
            g_obs=g_obs,
            var_obs=var_obs,
            indicator=indicator,
            tau2=tau2,
            null_var=null_var,
            cutoffs=cutoffs,
        )

        total_weight = np.bincount(
            reporting.voxel, weights=reporting.weight, minlength=width
        ) + np.bincount(silent.voxel, weights=silent.weight, minlength=width)

        # Where no study contributes a reporting indicator, the prevalence is not identified and
        # is held at 1 rather than fitted. Only the indicator separates "no effect in this
        # study" from "a small effect plus noise": a two-component mixture fitted to a handful
        # of Gaussian values with known variances will happily explain noise as a mixture, and
        # does. With every study carrying an image -- so that the indicator is empty everywhere
        # -- it returned a prevalence of 0.577 against a true 1.0, and profiling that
        # unidentified parameter out of the observed information inflated ``se/sd`` to 2.21,
        # the worst of any configuration measured, against 1.14 for the same images fitted
        # without the mixture. Holding it at 1 makes the mixture collapse to the plain
        # likelihood through the existing algebra: the responsibilities go to 1, so
        # ``r(1 - r)`` goes to 0, so the cross block vanishes and the Schur complement with it.
        identified = (
            np.bincount(silent.voxel, weights=silent.weight, minlength=width) > 0
            if zero_inflated
            else np.zeros(width, dtype=bool)
        )

        # Both of these are compacted as voxels retire, so the profile pass needs the
        # uncompacted originals rather than whatever the loop leaves behind.
        full_identified = identified
        full_total_weight = total_weight
        mu = start.copy()
        pi = np.where(identified, 0.5, 1.0) if zero_inflated else np.ones(width)
        mu_out = np.zeros(width)
        pi_out = np.zeros(width)
        se_out = np.full(width, np.inf)
        se_marginal_out = np.full(width, np.inf)
        share_out = np.zeros(width)
        voxel_ids = np.arange(width)

        def retire(positions):
            """Write out voxels that have converged.

            The error comes from the *observed* information at the point being written out, not
            from the EM's own curvature. They are different quantities: the EM differentiates
            the Q function with responsibilities fixed, which overstates the information by the
            part attributable to not knowing which component an observation came from, and says
            nothing about the jointly estimated prevalence. Using it as an error understated the
            uncertainty enough to cover 62.5% to 89.8% of nominal-95% intervals.

            The EM's curvature still drives the *step*; only what is reported changes.
            """
            ids = voxel_ids[positions]
            mu_out[ids] = mu[positions]
            pi_out[ids] = pi[positions]
            information, marginal_variance, share = _observed_information(
                width=mu.size,
                pi=pi,
                reporting=reporting,
                silent=silent,
                mu=mu,
                censoring=silent.censoring(mu),
                identified=identified if zero_inflated else None,
            )
            information = information[positions]
            marginal_variance = marginal_variance[positions]
            share_out[ids] = share[positions]
            informative = information > 0
            se_out[ids[informative]] = 1.0 / np.sqrt(information[informative])
            marginal = np.isfinite(marginal_variance) & (marginal_variance > 0)
            se_marginal_out[ids[marginal]] = np.sqrt(marginal_variance[marginal])

        def derivatives(censoring):
            """Score and curvature of the weighted log-likelihood in mu."""
            return _mu_derivatives(
                width=mu.size,
                mu_rep=mu[reporting.voxel],
                g_rep=reporting.g,
                precision_rep=reporting.precision,
                rep_voxel=reporting.voxel,
                weight_rep=reporting.weight * reporting.responsibility,
                sil_voxel=silent.voxel,
                weight_sil=silent.weight * silent.responsibility,
                censoring=censoring,
            )

        curvature = np.zeros(width)
        log_likelihood = np.full(width, -np.inf)
        for _ in range(self.max_iter):
            if not mu.size:
                break
            censoring = silent.censoring(mu)
            pi_shift = np.zeros(mu.size)
            stalled = np.zeros(mu.size, dtype=bool)
            if zero_inflated:
                previous_pi, previous_ll = pi, log_likelihood
                (
                    reporting.responsibility,
                    silent.responsibility,
                    pi,
                    log_likelihood,
                ) = self._update_prevalence(reporting, silent, mu, pi, total_weight, censoring)
                # Unidentified voxels never leave pi = 1, and every observation there belongs
                # to the active component with certainty.
                pi = np.where(identified, pi, 1.0)
                reporting.responsibility = np.where(
                    identified[reporting.voxel], reporting.responsibility, 1.0
                )
                silent.responsibility = np.where(
                    identified[silent.voxel], silent.responsibility, 1.0
                )
                pi_shift = np.abs(pi - previous_pi)
                # EM increases the likelihood monotonically, so a voxel whose likelihood has
                # stopped rising has finished, whatever its parameters are still doing. Where a
                # single study reported, the step criterion alone never fires and the loop runs
                # to max_iter; this makes the answer reproducible, and that is all it makes it.
                # The surface there is a plateau, not a ridge with a peak along it, because one
                # report cannot separate a moderate effect from a false positive at mu = 0, so
                # no stopping rule and no optimizer recovers a magnitude. See ``max_iter``.
                gain = log_likelihood - previous_ll
                stalled = np.isfinite(previous_ll) & (
                    gain <= _EM_LOGLIK_TOLERANCE * (np.abs(log_likelihood) + 1.0)
                )

            score, curvature = derivatives(censoring)
            step = np.where(curvature < 0, -score / curvature, 0.0)
            # The likelihood is concave but flat far from the data; cap the step so a voxel
            # with almost no reporting weight cannot run away.
            mu = mu + np.clip(step, -1.0, 1.0)

            settled = stalled | ((np.abs(step) < _EM_TOLERANCE) & (pi_shift < _EM_TOLERANCE))
            if settled.all():
                retire(np.flatnonzero(settled))
                mu = mu[:0]
                break
            # Compaction touches every pair, so it is amortized rather than run every iteration.
            if settled.mean() < _EM_COMPACTION_FRACTION:
                continue

            retire(np.flatnonzero(settled))
            keep = ~settled
            position = np.full(mu.size, -1, dtype=np.int64)
            position[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
            reporting = reporting.compact(position)
            silent = silent.compact(position)
            mu, pi = mu[keep], pi[keep]
            identified = identified[keep]
            log_likelihood = log_likelihood[keep]
            total_weight = total_weight[keep]
            voxel_ids = voxel_ids[keep]

        if mu.size:
            retire(np.arange(mu.size))

        if self.interval != "profile":
            return mu_out, pi_out, se_out, se_marginal_out, share_out, None, None

        # The working sets above have been compacted as voxels retired, so they no longer span
        # the chunk. Rebuilding them costs one setup pass and keeps the profile a read-only
        # postscript to the fit rather than something the EM has to carry along.
        reporting, silent = self._working_sets(
            weights=weights,
            g_obs=g_obs,
            var_obs=var_obs,
            indicator=indicator,
            tau2=tau2,
            null_var=null_var,
            cutoffs=cutoffs,
        )
        lower, upper = self._profile_interval(
            reporting=reporting,
            silent=silent,
            total_weight=full_total_weight,
            identified=full_identified,
            mu_hat=mu_out,
            se=se_out,
        )
        return mu_out, pi_out, se_out, se_marginal_out, share_out, lower, upper

    # ----------------------------------------------------------- the statistic

    def _statistic(self, table, sample_sizes, thresholds, image_studies=None):
        """Fit one configuration of foci and return ``(fit, z)``.

        The observed map and every permutation go through this, so the null is built from
        exactly the statistic being tested. Running the null off the naive weighted mean while
        the observed map came from the selection model would compare two different quantities.
        """
        fit = self._pool(table, image_studies)
        if self.selection_model != "none":
            self._apply_selection_model(
                fit, table, thresholds, sample_sizes, image_ids=tuple(image_studies or ())
            )

        z_values = np.divide(
            fit["g"], fit["se"], out=np.zeros_like(fit["g"]), where=np.isfinite(fit["se"])
        )
        z_values[~fit["covered"]] = 0.0
        return fit, z_values

    # ---------------------------------------------------------------- the null

    def _mask_bool(self):
        """Boolean analysis mask, cached: the null loop unmasks a volume every iteration."""
        cached = getattr(self, "_mask_bool_", None)
        if cached is None:
            cached = _mask_img_to_bool(self.masker.mask_img)
            self._mask_bool_ = cached
        return cached

    def _permute_image_values(self, rng):
        """Reassign each image study's own values among its own voxels.

        The same action the coordinate side takes, applied to the one other kind of study, so
        that both are randomized under one hypothesis. An image has values everywhere rather
        than at a handful of peaks, so "its arrangement over its own locations is arbitrary"
        is a shuffle of its voxels; the variance travels with the value it belongs to, because
        the pair is one observation.

        Sign-flipping is what an image admits on its own, and is what this used to do. It is a
        null for a different hypothesis -- that the effect is zero -- and mixing the two makes
        the combined test reject for either reason: with 20 coordinate analyses and 5 image
        analyses all carrying the same real effect, the sign flips alone put a floor of 1/32 on
        the p-value, which was read as evidence about location.
        """
        images = getattr(self, "_image_studies_", None)
        if not images:
            return images
        out = {}
        for study_id, (g, var_g, usable) in images.items():
            where = np.flatnonzero(usable)
            donor = rng.permutation(where)
            g_null, var_null = g.copy(), var_g.copy()
            g_null[where] = g[donor]
            var_null[where] = var_g[donor]
            out[study_id] = (g_null, var_null, usable)
        return out

    def _null_has_states(self):
        """Report ``(n_arrangements_log10, n_contributing)`` for the within-study shuffle.

        Only the image studies contribute. The null scrambles each image's values among its own
        voxels and leaves the coordinate tables exactly as they are, which is what keeps it a
        test of the magnitude channel: the silence pattern is identical in the observed fit and
        in every permutation, so whatever the censoring term contributes cancels between them.
        A collection with no images therefore admits one arrangement and cannot be tested.

        The count is returned as a log because one whole-brain image already overflows a float.
        """
        log10_states, contributing = 0.0, 0
        for _, _, usable in (getattr(self, "_image_studies_", None) or {}).values():
            n_usable = int(usable.sum())
            if n_usable > 1:
                contributing += 1
                log10_states += float(gammaln(n_usable + 1.0)) / np.log(10.0)
        return log10_states, contributing

    def _null_is_usable(self):
        """Refuse to build the null when no image study can be shuffled.

        The shuffle acts on image values. Without an image every permutation reproduces the
        observed map and every p-value comes back at 1.0, which reads as a null result rather
        than as a test that could not be run. Saying so is the difference between the two.
        """
        log10_states, contributing = self._null_has_states()
        if log10_states >= _MIN_NULL_STATES_LOG10:
            return True
        # Recomputed rather than cached, so that it cannot go stale against a collection that
        # changed; only the warning is remembered, because ``fit`` and the description both ask.
        if getattr(self, "_null_refusal_logged_", False):
            return False
        self._null_refusal_logged_ = True
        LGR.warning(
            "No p-values were computed: the within-study null admits about "
            f"1e{log10_states:.1f} arrangements, from {contributing} image studies. The null "
            "scrambles image values among their own voxels, so a collection without effect-size "
            "images cannot be tested -- the maps are still estimated, and 'p' is 1.0 everywhere "
            "to say that nothing was tested."
        )
        return False

    def _permutation_chunk(self, seeds, sample_sizes, thresholds, observed, cluster_stat):
        """Run a block of permutations, reducing as it goes.

        Chunked rather than one iteration per task because the per-voxel null is accumulated,
        not stored: returning each iteration's whole ``|z|`` map would cost ``n_voxels x
        n_iters`` floats, which on a whole brain at the default ``n_iters`` is gigabytes. A
        chunk keeps one counter per voxel instead, and the counters sum across chunks.
        """
        exceedances = np.zeros(observed.size, dtype=np.int64)
        histogram = np.zeros(len(_null_bin_edges()) - 1, dtype=float)
        peaks, sizes, masses = [], [], []
        mask_bool = self._mask_bool() if cluster_stat is not None else None
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # The focus table is passed through unchanged, which is the point. Only the image
            # values move, because they are the only magnitudes in the model. The coordinate
            # silence pattern is therefore identical in the observed fit and in every null
            # draw, so whatever structure it contributes appears on both sides and cancels --
            # which is what makes this a valid null for a silence-based estimator without
            # relocating any focus. Relocation was tried and rejected: it is not conditional on
            # multiplicity, and permuting whole rows put two foci of one study on a voxel and
            # dropped the null's study count.
            _, z_null = self._statistic(
                self._focus_table_,
                sample_sizes,
                thresholds,
                self._permute_image_values(rng),
            )
            absolute = np.abs(z_null)
            exceedances += absolute >= observed
            counts, _ = np.histogram(np.clip(absolute, 0, _NULL_MAX_Z), bins=_null_bin_edges())
            histogram += counts
            peaks.append(float(absolute.max()) if absolute.size else 0.0)
            if cluster_stat is not None and np.isfinite(cluster_stat):
                volume = np.zeros(mask_bool.shape, dtype=float)
                volume[mask_bool] = z_null
                size, mass = _calculate_cluster_measures(
                    volume, cluster_stat, _CLUSTER_CONNECTIVITY, tail="two"
                )
                sizes.append(float(size))
                masses.append(float(mass))
        return exceedances, histogram, peaks, sizes, masses

    def _compute_permutation_null(
        self, n_iters, n_cores, seed, observed, cluster_stat=None, cluster_threshold=None
    ):
        """Per-voxel null from permuting the magnitudes over fixed positions.

        Per voxel, which is the point. Because the positions never move, a voxel is reached by
        the same studies in every iteration, so it has a null of its own and is compared only
        against itself. Pooling ``|z|`` over the brain into one histogram would refer a voxel
        carrying thirty studies to a distribution made mostly of voxels carrying two, and since
        the standard error falls with the number of contributing studies it would be the study
        count rather than the effect sizes driving significance -- a convergence test wearing an
        effect-size statistic.

        The uncorrected p is ``(1 + #{null >= observed}) / (1 + n_iters)`` and so cannot fall
        below ``1 / (1 + n_iters)``. The pooled histogram is still accumulated, but only to
        resolve the cluster-forming threshold, which needs one ``|z|`` cutoff rather than a
        per-voxel one.
        """
        sample_sizes = getattr(self, "_sample_sizes_", None)
        thresholds = getattr(self, "_thresholds_", None)
        n_cores = _check_ncores(n_cores)
        observed = np.asarray(observed, dtype=float)

        if cluster_stat is None and cluster_threshold is not None:
            n_pilot = int(min(max(_NULL_PILOT_ITERS, n_iters // _NULL_PILOT_DIVISOR), n_iters))
            _, pilot_histogram, _, _, _ = self._permutation_chunk(
                range(seed + n_iters, seed + n_iters + n_pilot),
                sample_sizes,
                thresholds,
                observed,
                None,
            )
            cluster_stat = _stat_from_histogram(cluster_threshold, pilot_histogram)
            self.null_distributions_["cluster_forming_stat"] = cluster_stat

        chunks = [
            range(seed + start, seed + min(start + -(-n_iters // n_cores), n_iters))
            for start in range(0, n_iters, -(-n_iters // n_cores))
        ]
        results = Parallel(n_jobs=n_cores)(
            delayed(self._permutation_chunk)(
                chunk, sample_sizes, thresholds, observed, cluster_stat
            )
            for chunk in tqdm(chunks, disable=len(chunks) < 2, desc="CBES permutation null")
        )

        exceedances = np.sum([counts for counts, _, _, _, _ in results], axis=0)
        histogram = np.sum([hist for _, hist, _, _, _ in results], axis=0).astype(np.float64)
        max_values = np.array(
            [peak for _, _, peaks, _, _ in results for peak in peaks], dtype=float
        )
        p_values = (1.0 + exceedances) / (1.0 + n_iters)

        self.null_distributions_["histogram_bins"] = _null_bin_edges()
        self.null_distributions_["histweights_corr-none_method-montecarlo"] = histogram
        self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"] = max_values
        if cluster_stat is not None and np.isfinite(cluster_stat):
            self.null_distributions_[
                "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
            ] = np.array([size for _, _, _, sizes, _ in results for size in sizes], dtype=float)
            self.null_distributions_[
                "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
            ] = np.array([mass for _, _, _, _, masses in results for mass in masses], dtype=float)
        return p_values, max_values

    # --------------------------------------------------------------------- fit

    def _prepare_focus_table(self, dataset):
        """Build the focus table and the per-study quantities the model reads off it.

        The roster, the thresholds and the peak-height correction have to be resolved in that
        order and are therefore resolved together: ``rho_k`` is a function of a study's
        threshold, and the threshold is then rescaled by ``rho_k`` so that the censored
        likelihood compares an observation against a bound on the same scale.

        Sets ``_focus_table_``, ``_sample_sizes_`` and ``_thresholds_``, which the null reads
        back to refit exactly the statistic that was tested.
        """
        table = self._build_focus_table()
        # A study that supplies an image has nothing to gain from its own coordinates: the
        # image gives the effect everywhere its peaks do and more, without the localization
        # uncertainty or the selection.
        if self._image_studies_:
            table = table[~table["id"].isin(self._image_studies_)].copy()

        # The scale needs no settling: the pooled mean is built from images, which arrive on the
        # effect-size scale, so there is no unidentified constant to calibrate and no
        # absolute-versus-relative distinction to carry.
        if self.selection_model != "none":
            roster = self._all_sample_sizes(dataset)
            self._cutoffs_z_ = self._study_cutoffs_z(table, roster)
            # The censored likelihood compares an effect size against a bound, so the bound has
            # to be an effect size. Leaving it on the z scale saturates the censoring term --
            # a z of 3.29 is about 18 sampling standard deviations at N = 30, so every silence
            # becomes certain and the whole coordinate channel goes inert.
            thresholds = pd.Series(
                reporting_cutoff_to_g(self._cutoffs_z_.values, roster.values, design=self.design),
                index=roster.index,
            )
        else:
            roster, thresholds = None, None
            self._cutoffs_z_ = None

        # Without the selection model the fit never asks what a study would have reported, so
        # the roster and the thresholds are not merely unused but meaningless, and are dropped
        # rather than left lying around for the null to pick up.
        self._focus_table_ = table
        self._sample_sizes_ = roster if self.selection_model != "none" else None
        self._thresholds_ = thresholds if self.selection_model != "none" else None

    def _fit(self, dataset):
        """Estimate the effect size at every voxel, and refer it to the permutation null."""
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator."
            )

        self.null_distributions_ = {}
        self._mask_bool_ = None
        self._coverage_ = None
        self._null_refusal_logged_ = False
        self._image_studies_ = self._load_image_studies(dataset)
        self._prepare_focus_table(dataset)

        fit, z_values = self._statistic(
            self._focus_table_, self._sample_sizes_, self._thresholds_, self._image_studies_
        )

        if self.null_method == "permute-images" and self._null_is_usable():
            p_values, _ = self._compute_permutation_null(
                self.n_iters,
                self.n_cores,
                self.seed,
                np.abs(z_values),
                cluster_threshold=self.cluster_threshold,
            )
        else:
            # No null was built, so there is nothing to refer z to. Returning 1 rather than a
            # normal-theory p-value keeps a caller from mistaking the absence of inference for
            # inference: the former parametric option was anticonservative by a factor of eight.
            p_values = np.ones_like(z_values)
        p_values[~fit["covered"]] = 1.0

        # `dof` is taken from the censoring roster, not from a Kish count over the pooled
        # weights. Two reasons, and the second is decisive. The likelihood uses every study on
        # the roster -- the images through their values, the rest through their silence -- so
        # crediting only the weighted contributors denies the model information it demonstrably
        # used. And with coordinate magnitudes gone, the only weighted contributions are images
        # at weight 1, so a Kish count is just the image count: `n_eff = k`, `dof = k - 1`,
        # which is **0 with one image** and makes the recommended interval `nan`. The roster is
        # the only reference that survives the design.
        roster_size = getattr(self, "_sample_sizes_", None)
        if roster_size is not None and len(roster_size):
            dof = np.full(fit["g"].shape, float(len(roster_size)) - 1.0)
            # A voxel no study speaks about has no interval, whatever the roster says.
            dof = np.where(fit["covered"], dof, 0.0)
        else:
            dof = np.clip(fit["n_eff"] - 1.0, 0.0, None)

        maps = {
            "g": fit["g"].astype(DEFAULT_FLOAT_DTYPE),
            "se": np.where(np.isfinite(fit["se"]), fit["se"], 0).astype(DEFAULT_FLOAT_DTYPE),
            "z": z_values.astype(DEFAULT_FLOAT_DTYPE),
            "p": p_values.astype(DEFAULT_FLOAT_DTYPE),
            "logp": _nlogp_to_logp_values(np.log(np.clip(p_values, _LOGP_FLOOR, None))),
            "tau2": fit["tau2"].astype(DEFAULT_FLOAT_DTYPE),
            "n_studies": fit["n_studies"].astype(DEFAULT_FLOAT_DTYPE),
            "n_eff": fit["n_eff"].astype(DEFAULT_FLOAT_DTYPE),
            "dof": np.clip(dof, 0.0, None).astype(DEFAULT_FLOAT_DTYPE),
        }
        if "prevalence" in fit:
            maps["prevalence"] = fit["prevalence"].astype(DEFAULT_FLOAT_DTYPE)
            maps["g_marginal"] = (fit["g"] * fit["prevalence"]).astype(DEFAULT_FLOAT_DTYPE)
            # Zero where there is no usable information, matching how "se" is emitted, so that
            # a reader is not handed an infinity to divide by.
            maps["coordinate_share"] = fit["coordinate_share"].astype(DEFAULT_FLOAT_DTYPE)
            if "g_lower" in fit:
                maps["g_lower"] = fit["g_lower"].astype(DEFAULT_FLOAT_DTYPE)
                maps["g_upper"] = fit["g_upper"].astype(DEFAULT_FLOAT_DTYPE)
            marginal_se = fit.get("se_marginal")
            if marginal_se is not None:
                maps["se_marginal"] = np.where(np.isfinite(marginal_se), marginal_se, 0).astype(
                    DEFAULT_FLOAT_DTYPE
                )
        return maps, {}, self._generate_description()

    def correct_fwe_montecarlo(
        self,
        result,
        voxel_thresh=0.001,
        n_iters=None,
        n_cores=None,
        seed=None,
        vfwe_only=False,
        tail_approximation=True,
    ):
        r"""FWE correction from maximum-statistic nulls, at voxel and cluster level.

        Each iteration reassigns each analysis's reported values among its own locations and
        refits, the same randomization the uncorrected map came from, so voxel-level and
        familywise inference test the same hypothesis. This runs even when the estimator was
        fitted with ``null_method="none"``, since a maximum statistic has to come from
        somewhere -- but not when the collection admits too few arrangements to permute, since
        then there is no null to build at either level.

        Three nulls come out of the same refits: the maximum ``|z|``, the maximum cluster size
        and the maximum cluster mass. Clusters are formed on ``|z|`` at the statistic
        corresponding to ``voxel_thresh``, read off the uncorrected null rather than assumed --
        CBES's ``z`` is not standard normal, so a nominal 3.29 would not be a p of .001. When
        :meth:`fit` ran the same iterations at the same threshold, all three are reused and this
        is nearly free.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result of a previous :meth:`fit`.
        voxel_thresh : :obj:`float`, default=0.001
            Cluster-forming threshold, as an uncorrected p-value.
        n_iters, n_cores, seed : optional
            Override the estimator's own settings for this correction.
        tail_approximation : :obj:`bool`, default=True
            Fit a generalized Pareto distribution to the tail of the maximum-statistic null
            and read corrected p-values off it, rather than off the empirical tail alone. A
            permutation p cannot fall below ``1 / (1 + n_iters)``, so without this a corrected
            p of 1e-4 needs ten thousand permutations; extreme value theory says the exceedances
            of a high threshold are generalized Pareto whatever the parent distribution, so the
            tail can be modelled from far fewer \\citep{Winkler2016}. The fit is tested and the
            tail shortened until it is acceptable; if none passes, the empirical tail is used
            unchanged, so this can refine a quantized p-value but never manufacture one. It is
            applied only well above the empirical floor, where validation found it reliable.
        vfwe_only : :obj:`bool`, default=False
            Only compute voxel-level correction.

        Returns
        -------
        maps, tables, description
        """
        if getattr(self, "_focus_table_", None) is None:
            raise ValueError("correct_fwe_montecarlo requires a fitted estimator.")
        if not self._null_is_usable():
            # Refusing here and not only in ``fit`` because this path builds its own null: with
            # too few arrangements the maximum statistic is the observed one in almost every
            # iteration, which reads as a corrected p far below the uncorrected one.
            raise ValueError(
                "correct_fwe_montecarlo has no null to build: the reported foci admit too few "
                "within-analysis arrangements to permute. See the warning from fit()."
            )

        n_iters = self.n_iters if n_iters is None else n_iters
        n_cores = self.n_cores if n_cores is None else n_cores
        seed = self.seed if seed is None else seed

        cached = self.null_distributions_.get("values_level-voxel_corr-fwe_method-montecarlo")
        reusable = cached is not None and len(cached) == n_iters and seed == self.seed

        cluster_stat = self.null_distributions_.get("cluster_forming_stat")
        already_clustered = (
            reusable
            and cluster_stat is not None
            and voxel_thresh == self.cluster_threshold
            and "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
            in self.null_distributions_
        )

        observed_abs = np.abs(result.maps["z"])
        if vfwe_only:
            if not reusable:
                _, cached = self._compute_permutation_null(n_iters, n_cores, seed, observed_abs)
        elif not already_clustered:
            # fit() either did not permute, or did so at a different cluster-forming
            # threshold, so the cluster nulls have to be built here.
            _, cached = self._compute_permutation_null(
                n_iters, n_cores, seed, observed_abs, cluster_threshold=voxel_thresh
            )
            cluster_stat = self.null_distributions_.get("cluster_forming_stat")

        if not vfwe_only and (cluster_stat is None or not np.isfinite(cluster_stat)):
            LGR.warning(
                f"No statistic reaches p < {voxel_thresh} under the null from {n_iters} "
                "iterations, so no cluster can form. Reporting voxel-level correction only; "
                "raise n_iters or voxel_thresh."
            )
            vfwe_only = True

        observed = np.abs(result.maps["z"])
        sign = np.sign(result.maps["z"])
        maps = {}
        # The correction is only as good as the spread of the null it refers to, and that is
        # knowable only once the permutations have run -- unlike ``_null_is_usable``, which
        # counts arrangements before building anything and cannot see that a collection of
        # two-focus studies barely moves its own maximum.
        usable, n_distinct, cv = _null_maxima_diagnostics(cached)
        self.null_distributions_["max_statistic_distinct_values"] = n_distinct
        self.null_distributions_["max_statistic_cv"] = cv
        if usable:
            maps["logp_level-voxel"], maps["z_level-voxel"] = _max_statistic_maps(
                observed, cached, sign, tail_approximation=tail_approximation
            )
        else:
            LGR.warning(
                f"No voxel-level family-wise correction was computed: across {n_iters} "
                f"permutations the maximum statistic attained only {n_distinct} distinct "
                f"values with a coefficient of variation of {cv:.3f}. Rearranging magnitudes "
                "within a study that reported only two or three foci barely moves the map's "
                "maximum, so the permutation distribution understates the spread of the "
                "quantity it stands in for -- on simulated global nulls that configuration "
                "rejected at 0.150 against a nominal 0.050. 'logp_level-voxel' is 0 "
                "everywhere to say that nothing was corrected, rather than reporting a "
                "family-wise p-value that does not hold its level. A collection whose studies "
                "report more foci each, or an uncorrected 'p' map read with a different "
                "multiplicity correction, are the alternatives."
            )
            maps["logp_level-voxel"] = np.zeros_like(observed)
            maps["z_level-voxel"] = np.zeros_like(observed)
            # The cluster-level nulls are left alone. They are built from the same permutations
            # but refer a different quantity -- a cluster's size or mass rather than the map's
            # maximum -- whose own degeneracy has not been measured, and refusing it here on the
            # strength of the voxel-level measurement would be an assumption rather than a
            # finding. It would also make the description below claim a voxel-level scope for a
            # run whose voxel level is exactly what was withheld.

        if not vfwe_only:
            mask_bool = self._mask_bool()
            volume = np.zeros(mask_bool.shape, dtype=float)
            volume[mask_bool] = result.maps["z"]
            sizes, masses = _observed_cluster_measures(volume, cluster_stat)
            for label, observed_measure, key in (
                ("size", sizes, "values_desc-size_level-cluster_corr-fwe_method-montecarlo"),
                ("mass", masses, "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"),
            ):
                null = self.null_distributions_[key]
                logp, z_corrected = _max_statistic_maps(
                    observed_measure[mask_bool], null, sign, tail_approximation=tail_approximation
                )
                maps[f"logp_desc-{label}_level-cluster"] = logp
                maps[f"z_desc-{label}_level-cluster"] = z_corrected

        scope = "voxel-level" if vfwe_only else "voxel- and cluster-level"
        description = (
            f"Family-wise error rate correction was performed with a {scope} permutation "
            f"procedure using {n_iters} iterations, in which each image study's effect sizes "
            "were reassigned among its own voxels with the pattern of coordinate silence held "
            "fixed."
        )
        if not vfwe_only:
            description += (
                f" Clusters were formed at an uncorrected p of {voxel_thresh}, which "
                f"corresponds to |z| > {cluster_stat:.2f} under that null, and were compared "
                "against the null distributions of maximum cluster size and mass."
            )
        if tail_approximation:
            description += (
                " Corrected p-values in the tail were obtained by fitting a generalized Pareto "
                "distribution to the exceedances of the maximum-statistic null "
                "\\citep{Winkler2016}, which resolves p-values below the "
                f"{1 / (1 + n_iters):.2g} floor that {n_iters} permutations would otherwise "
                "impose; where no acceptable fit was found the empirical tail was retained."
            )
        return maps, {}, description

    # ------------------------------------------------------------- description

    def _generate_description(self):
        heterogeneity = (
            "a locally estimated between-study variance (DerSimonian-Laird)"
            if self.tau2_method == "dl"
            else "a fixed-effects model, with no between-study variance"
        )
        radius = (
            DEFAULT_COVERAGE_RADIUS_MM if self.coverage_radius is None else self.coverage_radius
        )
        if self.selection_model == "zero-inflated":
            selection = (
                f" Studies that reported no peak within {radius:g}"
                " mm of a voxel contributed the probability of that non-report to a "
                "zero-inflated censored (Tobit) likelihood there, which separates the "
                "proportion of studies with a non-null effect from the size of that effect "
                "and corrects the estimate for the within-study thresholding that decided "
                "what was reported. No effect-size images were imputed."
            )
        else:
            selection = (
                " Non-reports were not modelled, so the estimate is biased away from zero by "
                "the within-study thresholding that decided what was reported."
            )
        n_foci = len(getattr(self, "_focus_table_", []))
        roster = getattr(self, "_sample_sizes_", None)
        n_studies = 0 if roster is None else len(roster)
        if self.null_method == "permute-images" and self._null_is_usable():
            inference = (
                " Uncorrected p-values were obtained from a permutation null distribution, in "
                "which each image study's effect sizes were reassigned among its own voxels "
                f"{self.n_iters} times with the pattern of coordinate silence held fixed, "
                "each voxel being referred to its own null."
            )
        elif self.null_method == "permute-images":
            inference = (
                " No null distribution was computed, because the collection admits too few "
                "within-study arrangements to test, so no p-values are reported."
            )
        else:
            inference = " No null distribution was computed, so no p-values are reported."
        return (
            "A coordinate-based effect-size meta-analysis was performed with NiMARE "
            f"{__version__} (RRID:SCR_017398; \\citealt{{Salo2023}}). Effect-size (Hedges' g) "
            "images supplied the magnitude at every voxel; the coordinate tables supplied only "
            "the pattern of reporting and non-reporting, their peak heights being discarded. "
            f"Voxel-wise pooling used {heterogeneity}.{selection}{inference} "
            f"The input dataset included {n_foci} foci from "
            f"{n_studies} experiments."
        )
