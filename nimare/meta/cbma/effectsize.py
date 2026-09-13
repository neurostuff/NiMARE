"""Coordinate-based effect-size meta-analysis (CBES).

Where ALE, (M)KDA and CBMR ask *where do studies agree that something happened*, this module
asks *how big is the effect there*. The unit of analysis is not the density of reported foci
but the standardized effect size (Hedges' :math:`g`) implied by each reported peak's test
statistic and the study's sample size.

The model
---------
A study :math:`k` reports a peak at :math:`x_{ki}` with statistic :math:`T_{ki}`. Two
transformations turn that into a meta-analytic observation:

1. **Statistic to effect size.** :math:`T_{ki}` and :math:`N_k` give Hedges'
   :math:`g_{ki}` and its sampling variance :math:`s^2_{ki}`
   (:func:`peak_stat_to_hedges_g`). This is the same conversion NiMARE's image-based
   estimators use; nothing here is specific to coordinates.

2. **Peak to field.** A reported peak is a point observation of a spatially continuous effect,
   localized only up to spatial uncertainty. Each focus is therefore given a kernel weight
   :math:`w_{ki}(v) = K_h(\\|v - x_{ki}\\|)`, equal to 1 at the peak and decaying with
   distance. This is the Nadaraya-Watson smoother applied to the *marks* of a marked spatial
   point process -- the standard estimator for "what is the average mark near here".

At voxel :math:`v` the estimator then solves a **local random-effects meta-analysis**, using
:math:`w_{ki}(v)` as local-likelihood weights:

.. math::

    \\hat{g}(v) = \\frac{\\sum_k W_k(v) g_k}{\\sum_k W_k(v)},
    \\qquad W_k(v) = \\frac{w_k(v)}{s^2_k + \\tau^2(v)}

with :math:`\\tau^2(v)` a locally estimated between-study heterogeneity (a kernel-weighted
generalization of DerSimonian-Laird; see :func:`_local_dersimonian_laird`). With all
:math:`w = 1` this reduces exactly to a textbook random-effects meta-analysis, which is the
sense in which the spatial model is a *weighting scheme* rather than a separate algorithm.

Selection
---------
Reported peaks are not a random sample of the effect-size field: they are the values that
survived a within-study threshold, so pooling them naively is a spatial winner's curse. But a
study that reported nothing near :math:`v` usually has *no effect there*, not a small one, and
a plain censored model cannot say so -- with one shared :math:`\\mu` per voxel, silence can only
be read as evidence that :math:`\\mu` is small, which drags the estimate below the truth.

``selection_model="zero-inflated"`` (the default) therefore fits a mixture at each voxel. With
probability :math:`\\pi(v)` a study has a real effect :math:`\\delta_k \\sim N(\\mu(v),
\\tau^2(v))`; otherwise it has none. Either way it reports a peak only if
:math:`|\\hat{g}_k| > c_k`. A study that reported nothing contributes the *probability of that
silence* under both components, so the data decide which explanation it supports:

.. math::

    \\ell(\\mu, \\pi; v) = \\sum_{k \\in R} w_k(v)\\, \\log\\big[
            \\pi \\phi_{\\sigma_k}(g_k - \\mu) + (1 - \\pi) \\phi_{s_k}(g_k) \\big]
        + \\sum_{k \\notin R} m_k(v)\\, \\log\\big[
            \\pi P(|\\hat{g}| < c_k \\mid \\mu) + (1 - \\pi) P(|\\hat{g}| < c_k \\mid 0)
            \\big],

maximized by EM with :math:`\\tau^2` held at its moment estimate. No effect-size images are
imputed at any point; non-reporting enters only through those probabilities, as in
:footcite:t:`tench2017coordinate`. ``selection_model="none"`` pools only the reported peaks.

A plain Tobit -- the same censoring without the zero component -- was tried and dropped. It has
to explain every silence as a small common effect rather than as no effect, which drags the
estimate to -0.16 and -0.34 where the truth is positive. Zero inflation is what fixes that, so
there is no regime in which the plain version is the right choice.

Two details that matter more than the choice of likelihood. Membership of :math:`R`, the set of
studies that reported *something* near :math:`v`, is judged over a wider radius than the kernel
:math:`w_k` that weights their values -- a study whose peak landed 6 mm away should have its
value discounted but has plainly not been silent. And :math:`m_k`, the weight on a silent
study, is the average :math:`w` of the reporting studies at that voxel rather than 1: reporting
studies are kernel-discounted, so giving silence full weight lets it outvote evidence.

Inference
---------
``g / se`` is not a null-referenced statistic -- the peaks being pooled were selected for being
large, so under a global null the pooled effect at a focus is large by construction. Uncorrected
p-values therefore come from a spatial null: the same null the convergence-based estimators
use, that reported coordinates fall at random within the mask. By default
(``null_method="approximate"``) a voxel's null is drawn directly from the studies' foci counts
and kernel geometry, which costs nothing in the size of the mask; ``"montecarlo"`` relocates
every focus and refits, and is what the approximation is validated against. See the
:class:`CBES` notes for what a normal-theory alternative does to the false positive rate.

References
----------
.. footbibliography::
"""

import logging
import os

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from nilearn.maskers import NiftiMasker
from scipy import ndimage, stats
from scipy.optimize import brentq
from scipy.special import ndtr
from tqdm.auto import tqdm

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.utils import (
    _calculate_cluster_measures,
    _get_mask_flat_to_masked,
    _kernel_to_sparse_support,
    get_ale_kernel,
    sphere_kernel_offsets,
)
from nimare.transforms import d_to_g, t_to_d, t_to_z, z_to_t
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _mask_img_to_bool,
    _add_metadata_to_dataframe,
    _check_ncores,
    _nlogp_to_logp_values,
    get_masker,
    get_masker_mask_image,
    mm2vox,
    validate_coordinate_spaces,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]

#: Smallest sampling variance we will attribute to a reported peak. Guards the pooling weights
#: against division by zero for implausibly large sample sizes.
_MIN_VARIANCE = 1e-8

#: Typical effect magnitude by sample size, from 258 unthresholded group T/Z maps on
#: NeuroVault (134 collections, 85 cognitive paradigms, N from 10 to 1369). Each entry is
#: ``(representative N, median mean |g| in that map's top decile)``.
#:
#: Studies with a large N investigate smaller effects, because that is what they are powered
#: for: ``corr(log N, log magnitude) = -0.395``. Matching on sample size cuts the error in
#: predicting a held-out map's magnitude by 28% against using the corpus median, where matching
#: on *spatial* similarity does not help at all -- it explains 0.26% of the variance and the
#: fifteen most similar maps predict worse than ignoring similarity. Binned medians rather than
#: a fitted line because the relationship is not log-linear: the line recovers only 0.91 of the
#: baseline error where these bins recover 0.72.
REFERENCE_MAGNITUDE_BY_N = (
    (17.1, 0.761),
    (21.9, 0.564),
    (28.6, 0.578),
    (38.3, 0.399),
    (50.9, 0.408),
    (74.3, 0.338),
    (117.1, 0.223),
    (264.6, 0.283),
)

#: Spread of the reference relationship, in log units: predictions are good to about a factor
#: of two, and no scale derived from it should be quoted more precisely than that.
REFERENCE_MAGNITUDE_LOG_SD = 0.673


def reference_magnitude(sample_sizes):
    """Effect magnitude a collection of this size would typically show, from the reference corpus.

    Returns the geometric mean over studies of the typical ``|g|`` for each study's sample size,
    interpolated in log-log space between :data:`REFERENCE_MAGNITUDE_BY_N`. This is an external
    prior, not a measurement of the collection: it says researchers who ran this many subjects
    were usually studying effects of about this size.
    """
    sizes = np.asarray(sample_sizes, dtype=float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if not sizes.size:
        return None
    grid = np.log([entry[0] for entry in REFERENCE_MAGNITUDE_BY_N])
    values = np.log([entry[1] for entry in REFERENCE_MAGNITUDE_BY_N])
    return float(np.exp(np.mean(np.interp(np.log(sizes), grid, values))))


#: Keywords ``threshold`` understands; any other string names a metadata field.
THRESHOLD_KEYWORDS = ("pooled-min", "study-min")

#: Default two-tailed reporting threshold, on the z scale, when a study gives no better
#: information. p < .001 uncorrected, the most common screening threshold in the literature.
DEFAULT_REPORTING_THRESHOLD_Z = 3.2905267314919255

DESIGNS = ("one-sample", "two-sample")

SELECTION_MODELS = ("zero-inflated", "none")

NULL_METHODS = ("montecarlo", "approximate", "none")

#: Resolution of the Monte Carlo null histogram for |z|, and where its upper tail is clipped.
_NULL_Z_STEP = 0.01
_NULL_MAX_Z = 50.0

#: EM stops on a voxel once mu and the prevalence both move less than this in one step.
#: Measured on a whole-brain fit: tightening to 1e-5 costs 10% more runtime and moves no
#: voxel's g by more than 0.001, while loosening to 1e-3 buys only a further 11% and starts
#: to distort the map (max |dg| 0.031).
_EM_TOLERANCE = 1e-4
#: Rebuild the working set only once this fraction of it has settled, so that compaction
#: (which touches every pair) is amortized rather than run every iteration.
_EM_COMPACTION_FRACTION = 0.05

#: Minimum relocations used to fix the cluster-forming threshold before the main null loop.
_NULL_PILOT_ITERS = 20

#: Independent draws used by ``null_method='approximate'`` when ``n_iters`` is small.
_MIN_APPROXIMATE_DRAWS = 200_000

#: Excess of the mean reported peak height over the null peak height, in z units, below which
#: the reported magnitudes are treated as carrying no usable effect-size information.
_MIN_PEAK_EXCESS_Z = 0.25

#: Faces-only connectivity for cluster labelling, matching Nilearn and the other CBMA
#: estimators.
_CLUSTER_CONNECTIVITY = ndimage.generate_binary_structure(rank=3, connectivity=1)

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _normal_pdf(x):
    """Standard normal density. ``scipy.stats.norm.pdf`` is ~3x slower on large arrays, and
    the EM below evaluates it on an (n_studies, n_voxels) block on every iteration.

    Written in place: on the arrays this sees, the three temporaries the naive expression
    allocates cost more than the exponential.
    """
    out = x * x
    out *= -0.5
    np.exp(out, out=out)
    out *= _INV_SQRT_2PI
    return out


def _null_bin_edges():
    """Bin edges for the Monte Carlo null histogram of |z|."""
    return np.arange(0.0, _NULL_MAX_Z + _NULL_Z_STEP, _NULL_Z_STEP)


def _stat_from_histogram(p_value, histogram):
    """Smallest ``|z|`` whose null p-value is at or below ``p_value``.

    The inverse of :func:`_p_from_histogram`, used to turn a cluster-forming p threshold into
    the statistic threshold the clusters are actually defined on.
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


def _gpd_goodness_of_fit(excess, shape, scale, n_boot=200, seed=0):
    """p-value for "these exceedances are generalized Pareto", by parametric bootstrap.

    The parameters were estimated from the same data, so the textbook Cramer-von Mises null
    distribution does not apply -- using it accepts fits it should reject, which is how a tail
    approximation ends up anticonservative. Simulating from the fitted distribution and
    refitting each replicate gives the right reference.
    """
    rng = np.random.default_rng(seed)
    observed_stat = stats.cramervonmises(
        excess, stats.genpareto(shape, loc=0.0, scale=scale).cdf
    ).statistic
    n = excess.size

    worse = 0
    for _ in range(n_boot):
        sample = stats.genpareto.rvs(shape, loc=0.0, scale=scale, size=n, random_state=rng)
        try:
            boot_shape, _, boot_scale = stats.genpareto.fit(sample, floc=0.0)
            if not np.isfinite(boot_shape) or boot_scale <= 0:
                continue
            statistic = stats.cramervonmises(
                sample, stats.genpareto(boot_shape, loc=0.0, scale=boot_scale).cdf
            ).statistic
        except Exception:  # noqa: BLE001
            continue
        worse += statistic >= observed_stat
    return (1 + worse) / (1 + n_boot)


def _gpd_tail_p(observed, null_maxima, min_exceedances=30, alpha=0.05):
    """Corrected p-values from a generalized Pareto fit to the tail of the null maxima.

    A permutation p-value cannot go below ``1 / (1 + n_iters)``, so resolving a corrected p of
    1e-4 needs ten thousand relocations however uninteresting the other 9999 are. Extreme value
    theory says the exceedances of a high threshold converge to a generalized Pareto
    distribution whatever the parent, so the tail can be *modelled* rather than counted. This
    is the tail approximation of Winkler et al. (2016), which they recommend specifically for
    familywise error.

    Validated here rather than taken on faith, and the result bounds what it may do. Above five
    times the empirical floor the fitted p is 0.85-1.00 of a 40000-permutation truth; at and
    below the floor it runs about twice anticonservative in 36-60% of runs, on every parent
    distribution tried. So it refines p-values only in the range where it was shown to work and
    defers to the empirical tail below -- which gives up the extrapolation past the floor that
    the published method is prized for. With a few hundred exceedances this implementation did
    not earn it.

    The threshold is chosen the way they choose it: start with the largest tenth of the null
    maxima, test the fit, and if it is rejected drop the smallest exceedance and refit, until
    the fit is acceptable or too few points remain. Falling back to the empirical tail when no
    fit is accepted is what keeps this safe -- it can only ever refine a p-value it would
    otherwise have quantized, never invent one on a tail that is not Pareto.

    Returns ``None`` when no acceptable fit exists, leaving the caller on the empirical tail.
    """
    maxima = np.sort(np.asarray(null_maxima, dtype=float))
    n_total = maxima.size
    if n_total < 100:
        return None  # too few to say anything about a tail

    n_exceed = max(int(round(0.10 * n_total)), min_exceedances)
    while n_exceed >= min_exceedances:
        threshold = maxima[n_total - n_exceed - 1]
        excess = maxima[n_total - n_exceed :] - threshold
        if not np.all(np.isfinite(excess)) or excess.max() <= 0:
            n_exceed -= max(1, n_exceed // 20)
            continue
        try:
            shape, _, scale = stats.genpareto.fit(excess, floc=0.0)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                raise ValueError
            fitted = stats.genpareto(shape, loc=0.0, scale=scale)
            goodness = _gpd_goodness_of_fit(excess, shape, scale, seed=n_exceed)
        except Exception:  # noqa: BLE001 -- any failure just means try a shorter tail
            n_exceed -= max(1, n_exceed // 20)
            continue

        if goodness > alpha:
            rate = n_exceed / n_total
            observed = np.asarray(observed, dtype=float)
            in_tail = observed > threshold
            p_corrected = np.empty(observed.shape, dtype=float)
            # Below the threshold the empirical tail is well resolved, so keep it there.
            p_corrected[~in_tail] = (
                1 + np.sum(maxima[None, :] >= observed[~in_tail][:, None], axis=1)
            ) / (1 + n_total)
            p_corrected[in_tail] = rate * fitted.sf(observed[in_tail] - threshold)
            # How far this can be trusted was measured, not assumed: against a
            # 40000-permutation reference, across three parent distributions, 25 repetitions
            # each. Comfortably above the empirical floor it is accurate -- at p = .05 and .01
            # the fitted value is 0.85-1.00 of the truth and essentially never more than twice
            # too small. At and below the floor (1/501 for a 500-relocation run) it runs about
            # twice anticonservative in 36-60% of runs, on every parent tried.
            #
            # So the fit is used only where it was shown to work, five times the floor and
            # above, and the empirical tail is kept below that. That forgoes the extrapolation
            # past the floor which is the published method's main selling point; with a few
            # hundred exceedances this implementation did not earn it, and an anticonservative
            # familywise p is worse than a quantized one.
            floor = 5.0 / (1.0 + n_total)
            p_corrected[in_tail] = np.maximum(p_corrected[in_tail], floor)
            return np.clip(p_corrected, 0.0, 1.0)
        n_exceed -= max(1, n_exceed // 20)

    return None


def _max_statistic_maps(observed, null_maxima, sign, tail_approximation=False):
    """Corrected ``-log10(p)`` and signed z for a statistic against its maximum-statistic null."""
    p_corrected = None
    if tail_approximation:
        p_corrected = _gpd_tail_p(observed, null_maxima)
    if p_corrected is None:
        p_corrected = (1 + np.sum(null_maxima[None, :] >= observed[:, None], axis=1)) / (
            1 + len(null_maxima)
        )
    logp = _nlogp_to_logp_values(np.log(np.clip(p_corrected, 1e-300, None)))
    z_corrected = stats.norm.isf(np.clip(p_corrected, 1e-16, 1.0) / 2.0) * sign
    return (
        logp.astype(DEFAULT_FLOAT_DTYPE),
        z_corrected.astype(DEFAULT_FLOAT_DTYPE),
    )


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


def _censoring_terms(mu, cutoff_scaled, twice_cutoff_scaled, inv_sigma, inv_sigma_sq):
    """P(|g| < c | mu) and the pieces of its derivatives, for a set of silent observations.

    Returned together because the E step and the M step both need them at the same ``mu``, and
    this is the single most expensive thing the estimator does -- about 63% of a whole-brain
    fit, over an array with one entry per silent ``(study, voxel)`` pair.

    Everything that does not move between EM iterations is passed in already divided: ``mu`` is
    the only argument that changes, so ``cutoffs / sigma`` and the reciprocals are hoisted to
    the caller. What is left per iteration is one multiply and two subtracts to form the
    standardized limits, and the two normal CDFs that cannot be avoided. The lower tail looks
    negligible and is not -- at a typical cutoff it is a third of the score's numerator -- so
    it is kept.

    The arithmetic writes into its own temporaries wherever numpy allows it. At tens of
    millions of pairs each avoided temporary is hundreds of megabytes of traffic, and this runs
    tens of times per fit and once per Monte Carlo relocation.
    """
    # upper = (c - mu) / sigma;  lower = (-c - mu) / sigma = upper - 2c/sigma
    upper = mu * -inv_sigma
    upper += cutoff_scaled
    lower = upper - twice_cutoff_scaled

    prob = ndtr(upper)
    prob -= ndtr(lower)
    np.clip(prob, 1e-12, None, out=prob)

    pdf_upper = _normal_pdf(upper)
    pdf_lower = _normal_pdf(lower)

    score = pdf_upper - pdf_lower
    score *= -inv_sigma
    score /= prob

    # d2_over_prob = -(upper * pdf_upper - lower * pdf_lower) / sigma^2 / prob. Fold the pdfs
    # into the limits in place: neither is needed afterwards.
    pdf_upper *= upper
    pdf_lower *= lower
    d2_over_prob = pdf_upper - pdf_lower
    d2_over_prob *= -inv_sigma_sq
    d2_over_prob /= prob

    return {"prob": prob, "score": score, "d2_over_prob": d2_over_prob}


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


def _p_from_histogram(values, histogram):
    """Two-tailed p for ``|z|`` against a histogram of null ``|z|`` values.

    Pooling every voxel of every iteration into one histogram is what ALE and MKDA do for
    their uncorrected nulls; it assumes voxels share a null distribution, which is only
    approximately true here because coverage varies across the brain.
    """
    total = histogram.sum()
    if total <= 0:
        return np.ones_like(values, dtype=float)

    # Survival counts: how many null values land at or above each bin's lower edge.
    survival = np.concatenate([np.cumsum(histogram[::-1])[::-1], [0.0]])
    index = np.clip(
        np.floor(np.asarray(values) / _NULL_Z_STEP).astype(np.int64), 0, len(histogram)
    )
    return (survival[index] + 1.0) / (total + 1.0)


def peak_stat_to_hedges_g(stat, sample_size, stat_type="z", design="one-sample"):
    """Convert a reported peak test statistic into Hedges' g and its sampling variance.

    Parameters
    ----------
    stat : array_like
        Reported (signed) test statistic for each peak.
    sample_size : array_like
        Total sample size of the study the peak came from.
    stat_type : {"z", "t"}, default="z"
        Scale of ``stat``. A ``"z"`` statistic is first mapped back onto the t scale with
        :func:`~nimare.transforms.z_to_t`, matching tail probabilities, so that the sample-size
        correction is applied on the scale the statistic was actually computed on.
    design : {"one-sample", "two-sample"}, default="one-sample"
        Design the statistic came from. ``"two-sample"`` assumes equal group sizes, i.e.
        ``n1 = n2 = sample_size / 2``.

    Returns
    -------
    g : :class:`numpy.ndarray`
        Hedges' g, carrying the sign of ``stat``.
    var_g : :class:`numpy.ndarray`
        Sampling variance of ``g``.

    Notes
    -----
    For a one-sample design this is ``t / sqrt(N)`` followed by the usual small-sample bias
    correction, i.e. :func:`~nimare.transforms.t_to_d` then
    :func:`~nimare.transforms.d_to_g` -- the same path the image-based estimators take, so a
    coordinate-based and an image-based estimate of the same contrast are on one scale.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}; got {design!r}.")
    if stat_type not in ("z", "t"):
        raise ValueError(f"stat_type must be 'z' or 't'; got {stat_type!r}.")

    stat = np.asarray(stat, dtype=float)
    sample_size = np.asarray(sample_size, dtype=float)

    min_n = 4 if design == "one-sample" else 5
    if np.any(sample_size < min_n):
        raise ValueError(
            f"A {design} effect size needs at least {min_n} subjects per study; got a minimum "
            f"of {np.nanmin(sample_size):g}."
        )

    if design == "one-sample":
        dof = sample_size - 1
        t = z_to_t(stat, dof) if stat_type == "z" else stat
        d = t_to_d(t, sample_size)
        g, var_g = d_to_g(d, sample_size, return_variance=True)
    else:
        dof = sample_size - 2
        t = z_to_t(stat, dof) if stat_type == "z" else stat
        n1 = n2 = sample_size / 2.0
        d = t * np.sqrt(1.0 / n1 + 1.0 / n2)
        bias = 1.0 - (3.0 / (4.0 * dof - 1.0))
        g = bias * d
        var_g = (bias**2) * ((n1 + n2) / (n1 * n2) + d**2 / (2.0 * (n1 + n2)))

    return g, np.maximum(var_g, _MIN_VARIANCE)


def null_peak_overshoot(threshold_z):
    """Mean height of a suprathreshold local maximum of a smooth null field, in z units.

    For a smooth 3D Gaussian field the survival function of a peak above ``u`` is
    :math:`S(z) = (z^2-1)e^{-z^2/2} / [(u^2-1)e^{-u^2/2}]`
    :footcite:p:`chumbley2009false`, from which the mean follows by quadrature. A peak drawn
    from pure noise sits about ``1/u`` above the threshold -- roughly 0.3 z units at the usual
    p < .001.
    """
    u = float(threshold_z)
    if u <= 1.9:  # the high-threshold form is not positive below sqrt(3)
        return u + 1.0 / max(u, 1e-6)
    grid = np.linspace(u, u + 12.0, 4000)
    density = grid * (grid**2 - 3.0) * np.exp(-0.5 * grid**2)
    density = np.clip(density, 0.0, None)
    mass = np.trapezoid(density, grid)
    return float(np.trapezoid(grid * density, grid) / mass) if mass > 0 else u


def peak_information(stats_z, threshold_z):
    """How much effect-size information the reported peak heights actually carry.

    Returns ``(observed_mean, null_mean, excess)`` on the z scale. ``excess`` is what is left
    once the height a pure-noise peak would have reached is accounted for, and it is the only
    part of a reported peak height that speaks to the size of the effect.

    On real collections this is often indistinguishable from zero. Measured on the 21 NIDM
    pain studies: reported peaks average z = 3.639 against a null-peak expectation of 3.63, an
    excess of +0.009, where the effect actually present at those locations would have produced
    +1.10. When that happens the reported *magnitudes* are uninformative -- they are a function
    of the reporting threshold and the sample size, not of the effect -- and no correction
    computed from them can recover the effect size, because the information is not there.
    """
    observed = float(np.mean(np.abs(np.asarray(stats_z, dtype=float))))
    expected = null_peak_overshoot(threshold_z)
    return observed, expected, observed - expected


def null_peak_mean_g(threshold_z, sample_size, design="one-sample"):
    """Effect size a study would report from a *pure noise* peak above its threshold.

    A reported peak is a local maximum that cleared the study's threshold, and on
    underpowered data its height is set almost entirely by that threshold rather than by any
    effect (see :func:`peak_information`). This is the expected ``|g|`` of such a peak, taken
    over the RFT null peak-height distribution above ``threshold_z`` and converted with the
    study's own sample size.

    It is the scale a study contributes *by construction*, so dividing by it removes the part
    of a reported effect size that is an artefact of how strictly the paper thresholded and how
    many subjects it had -- neither of which is a fact about the brain.
    """
    u = float(threshold_z)
    grid = np.linspace(u, u + 12.0, 2000)
    density = np.clip(grid * (grid**2 - 3.0) * np.exp(-0.5 * grid**2), 0.0, None)
    mass = np.trapezoid(density, grid)
    if mass <= 0:  # threshold below sqrt(3): fall back to the exponential overshoot
        grid = np.linspace(u, u + 12.0, 2000)
        density = u * np.exp(-u * (grid - u))
        mass = np.trapezoid(density, grid)

    sizes = np.full(grid.shape, float(sample_size))
    g_of_z, _ = peak_stat_to_hedges_g(grid, sizes, stat_type="z", design=design)
    return float(np.trapezoid(np.abs(g_of_z) * density, grid) / mass)


def _expected_min_peak(u, n_peaks, span=10.0, n_grid=500):
    """E[smallest of ``n_peaks`` heights drawn above ``u``] under the RFT null."""
    grid = np.linspace(u, u + span, n_grid)
    survival = np.clip(
        (grid**2 - 1.0) * np.exp(-0.5 * grid**2) / ((u**2 - 1.0) * np.exp(-0.5 * u**2)), 0.0, 1.0
    )
    return u + np.trapezoid(survival**n_peaks, grid)


def infer_threshold_from_minimum(min_stat_z, n_peaks):
    """Recover a study's reporting threshold from its smallest reported statistic.

    Papers often do not state the threshold, and the smallest statistic they report is an
    *upper* bound on it: with only a handful of peaks the smallest of them still sits well
    above the cut. The minimum of ``n_peaks`` draws from the null peak-height distribution
    above ``u`` exceeds ``u`` by a computable amount, so that bias can be inverted rather than
    absorbed. With many reported peaks the correction vanishes, as it should.

    This assumes ``n_peaks`` is what the study's *height* threshold admitted. Any filter that
    removes low peaks for another reason is indistinguishable from a stricter height threshold,
    and this will return the filter rather than the threshold. Reporting one local maximum per
    cluster is fine; a cluster-extent threshold is not, and biases the result high by 0.19 z at
    k >= 10 voxels, rising to 0.57 z at k >= 50.

    Parameters
    ----------
    min_stat_z : :obj:`float`
        The study's smallest reported statistic, on the z scale, in absolute value.
    n_peaks : :obj:`int`
        How many peaks the study reported.

    Returns
    -------
    :obj:`float`
        The inferred threshold, never above ``min_stat_z``.
    """
    z_min = float(min_stat_z)
    n_peaks = int(n_peaks)
    if n_peaks <= 0 or not np.isfinite(z_min) or z_min <= 1.95:
        return z_min
    if _expected_min_peak(z_min, n_peaks) <= z_min:  # already consistent
        return z_min
    try:
        return float(
            brentq(lambda u: _expected_min_peak(u, n_peaks) - z_min, 1.95, z_min, xtol=1e-4)
        )
    except ValueError:
        return z_min


def null_effect_variance(sample_size, design="one-sample"):
    """Sampling variance of Hedges' g under a null effect, for a study that reported nothing.

    A study that did not report a peak supplies no effect size, but its *precision* is still
    known from its sample size. That precision is what makes the censoring term in the Tobit
    likelihood informative, so it is computed here at ``d = 0``.
    """
    zeros = np.zeros_like(np.asarray(sample_size, dtype=float))
    _, var_g = peak_stat_to_hedges_g(zeros, sample_size, stat_type="t", design=design)
    return var_g


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


class CBES(Estimator):
    """Coordinate-based effect-size meta-analysis.

    .. versionadded:: 0.13.0

    Estimates the pooled standardized effect size (Hedges' :math:`g`) at every voxel from
    reported peak coordinates *and* their reported test statistics, rather than from the
    spatial density of the coordinates alone. See the :mod:`module docstring
    <nimare.meta.cbma.effectsize>` for the model.

    Parameters
    ----------
    fwhm : :obj:`float` or None, default=10.0
        Full width at half maximum, in mm, of the Gaussian kernel expressing spatial
        uncertainty about each reported peak. If None, an ALE-style sample-size-dependent
        kernel is used instead, so that larger studies localize their peaks more tightly.
    use_images : :obj:`bool`, default=True
        Use per-study ``g`` and ``g_var`` images for any study that has them, in place of that
        study's coordinates. An image is the limiting case of a coordinate: it gives the effect
        at a voxel with no localization uncertainty and no reporting threshold, so it enters
        with kernel weight 1 everywhere and never contributes a censoring term. Studies without
        images are unaffected, so a collection may mix the two freely.

        Supply the images with ``ImageTransformer(target=["g", "g_var"])``, which derives them
        from ``t`` maps and sample sizes on the same scale
        :func:`peak_stat_to_hedges_g` puts the coordinates on.

        .. warning::
            Mixing is only sound once ``peak_bias`` is set. A reported peak is a local maximum
            and is inflated relative to the field around it -- measured at 2.05x on the NIDM
            pain images -- so an uncorrected coordinate study and an image study disagree about
            the same region by roughly a factor of two, and the pooled estimate then depends on
            how many studies of each kind the collection happens to contain.
    peak_bias : :obj:`float`, "per-study", or None, optional
        Correction for the fact that a reported peak is a local maximum that cleared a
        threshold. A reported statistic is treated as measuring ``g_true / rho_k``, so the
        effect size, its standard deviation and the study's reporting threshold are all scaled
        by ``rho_k`` -- a rescaling of the effect-size axis for coordinate studies, which
        leaves the censored likelihood coherent. Images are never rescaled; they are already
        unbiased.

        ``float``
            One ``rho`` for every study. Corrects the overall scale but assumes every study
            thresholded alike, which a literature search cannot guarantee.
        ``"per-study"``
            ``rho_k`` proportional to ``1 / null_peak_mean_g(u_k, N_k)``, normalized so the
            median reporting study sits at ``peak_bias_scale``. This divides out the part of
            the inflation that is a function of *how strictly study k thresholded and how many
            subjects it had*, which is the part that varies across a heterogeneous collection
            and the part coordinates can identify on their own. A study that reported at
            z > 4.3 with n = 12 is then discounted harder than one that reported at z > 2.3
            with n = 40, as it should be.

            What it cannot fix is the common scale, which no coordinate-only model can:
            rescaling ``g``, its variance and the censoring threshold together leaves the
            likelihood unchanged, so the constant is exactly non-identified. With
            ``peak_bias_scale=1.0`` the ``g`` and ``se`` maps are therefore readable only up
            to that constant -- on the NIDM pain collection they run about 2x high where the
            reference says there is something to estimate. (An earlier 2.65x came from a ratio
            of means over every covered voxel, whose denominator collapses wherever the truth
            is near zero; scored where the signal is, it is 2.02 at the top quartile and 1.72
            at the top decile.) It is a
            limit of the data rather than of the model: the deconvolution that would identify
            the scale needs peak heights that carry signal, and in the usual underpowered
            regime they do not (see :func:`peak_information`).

            This costs less than it sounds, because the constant divides out of everything
            except the magnitude. ``z``, the p-values and every corrected map are unchanged by
            it, since ``z = g / se`` scales top and bottom alike; so is ``prevalence``, which
            is a probability and cancels from the mixture responsibilities. An uncalibrated
            fit still gives valid inference and a valid prevalence map, with ``g`` read as a
            relative quantity.
        ``None``
            No correction. A reported peak overstates the *local* effect about fivefold -- the
            pooled truth at a peak is 0.22-0.27 where the reporting study says 1.22 -- but the
            pooled map runs about 2x high rather than fivefold, because the kernel and the
            selection model already absorb most of it.

        The size of the overall correction is not a modelling choice, it is measurable. On the
        NIDM pain collection, the leave-one-out pooled effect at a reported peak is 0.22-0.27
        where the reporting study says 1.22, giving a scale near 0.2. Calibrate it on your own
        data with the procedure in ``docs/notes/validate_cbes.py``.
    peak_bias_scale : :obj:`float`, "auto", "images", or "rates", default=1.0
        Overall scale of the ``"per-study"`` correction: the ``rho_k`` given to the median
        reporting study. Ignored unless ``peak_bias="per-study"``. Between-study differences in
        threshold and sample size are identified from the coordinates; this one number is not,
        because rescaling every coordinate study by the same constant leaves the
        coordinate-only likelihood unchanged.

        For a coordinate-only fit that is harmless: 1.0 gives a map correct up to a single
        multiplicative constant, which is what a relative effect-size map is. Once images are
        in the same fit it stops being harmless, because images are on the true ``g`` scale and
        a mismatched constant makes the two kinds of study disagree about the same voxel.
        ``"auto"`` (equivalently ``"images"``) then reads the constant off the studies that
        supplied images, by fitting images and coordinates separately and taking the ratio over
        the voxels both cover; expect it to need five or more image studies to be stable (two
        gives +-145%, five +-17%, sixteen +-6%). Mixing images with the default 1.0 warns.

        ``"rates"``, ``"mle"`` and ``"rate-match"`` all try to fix the scale without images, and
        none of them works. Seven attempts are recorded in ``docs/notes/effect_size_cbma.md``
        section 15; the short version is that converting reporting rates into an absolute effect
        size needs a local resels-per-voxel map, which is computed from images. Within-brain
        smoothness varies more than between-study smoothness (95/5 ratio 1.99x against 1.6x, an
        8x spread in resels), so a global value mis-predicts reporting in a spatial pattern no
        single scale can absorb. They are kept as documented negative results, warn when they
        rail, and are never selected by ``"auto"``.

        ``"rates"`` fixes the scale from how *often* studies reported, needing no images at all
        -- see :meth:`_calibrate_scale_from_rates`. **Experimental, opt-in only.** The
        identification is established: fitting the effect size from reporting indicators alone
        recovers a simulated truth of 0.50 as 0.496 and 0.80 as 0.810, using no peak heights.
        The estimator built on it is not: pooling one global constant as a ratio of means is
        dragged down by voxels holding no effect, where the rates correctly say zero but a
        reported noise peak still converts to a positive ``g``. Until that is fixed ``"auto"``
        will not select it.
    stat_column : :obj:`str` or None, optional
        Column of the coordinates table holding the reported statistic. When None, ``z_stat``
        is used if present, otherwise ``t_stat``.
    design : {"one-sample", "two-sample"}, default="one-sample"
        Design behind the reported statistics, used for the effect-size conversion.
    tau2_method : {"dl", "none"}, default="dl"
        ``"dl"`` estimates a local between-study variance with a kernel-weighted
        DerSimonian-Laird moment estimator; ``"none"`` fits a fixed-effects model
        (:math:`\\tau^2 \\equiv 0`).
    selection_model : {"zero-inflated", "none"}, default="zero-inflated"
        ``"censored"`` adds, for every study that reported nothing in this region, a term for
        the probability that it would have reported nothing, and returns the maximizer of that
        Tobit log-likelihood. ``"none"`` pools only the reported peaks, and is biased away from
        zero by the within-study thresholding that produced them. Nothing is imputed under
        either option.
    threshold : :obj:`float`, :obj:`str`, or None, default="pooled-min"
        Reporting threshold, on the z scale, assumed for each study. It enters twice: it
        decides how surprising a study's silence is, and with ``peak_bias="per-study"`` it sets
        how far that study's reported peaks are discounted.

        ``"pooled-min"``
            The smallest absolute statistic reported anywhere in the collection. One threshold
            for everyone, and the tightest bound available if that assumption holds.
        ``"study-min"``
            Each study's own smallest reported statistic, with the order statistic undone. The
            smallest of ``m_k`` peaks above ``u_k`` sits above ``u_k`` by an amount that grows
            as ``m_k`` shrinks, so :func:`infer_threshold_from_minimum` solves for the ``u_k``
            whose expected minimum is what the study actually reported -- never worse than the
            raw minimum, and reducing to it once a study reports enough peaks. Use this when
            studies plainly thresholded differently and none of them says how, which is the
            usual case in a literature search.

            Both per-study rules assume the reported peaks are whatever cleared the study's
            height threshold. Reporting one local maximum per cluster, the common convention,
            satisfies that: on the 21 NIDM pain studies it recovers every imposed threshold to
            within 0.02 z. A **cluster-extent threshold does not**, and it is just as common.
            Requiring k >= 10, 20 or 50 voxels lifts the smallest reported statistic by 0.19,
            0.32 and 0.57 z respectively, and the order-statistic correction removes only about
            a tenth of that -- an extent threshold preferentially drops small clusters, which
            are the ones with low peaks, so it acts like a stricter height threshold and is
            indistinguishable from one.

            The consequence is an inferred threshold biased high, which makes silence look less
            surprising than it was and under-corrects the selection. Prefer the metadata field
            when the papers state their extent threshold, and treat ``"study-min"`` as a
            lower bound on the bias when they do not.
        any other string
            The name of a metadata field holding each study's threshold on the z scale, for
            collections where the papers state it. Studies missing the field take the median
            of those that have it.

        A float applies one z threshold to every study; None falls back to
        ``DEFAULT_REPORTING_THRESHOLD_Z``.
    coverage_radius : :obj:`float` or None, optional
        Radius, in mm, within which a reported peak counts as this study having reported
        *something* about this location. A study with no focus inside that radius is treated as
        silent here and contributes a censoring term; a study with one is not, however far its
        peak is downweighted by the spatial kernel. Keeping the two radii separate matters: a
        study whose peak sits 6 mm away should have its *value* discounted, but it has plainly
        not been silent. Defaults to twice the kernel FWHM (20 mm when ``fwhm`` is None). Only
        used when ``selection_model="censored"``.
    kernel_min_weight : :obj:`float`, default=0.01
        Truncate the spatial kernel below this fraction of its peak. A focus then reaches only
        voxels it says something about (about 13 mm for a 10 mm FWHM), which is what keeps
        ``n_studies`` interpretable and the fit affordable.
    max_iter : :obj:`int`, default=25
        Maximum Newton iterations for the censored likelihood.
    null_method : {"approximate", "montecarlo", "none"}, default="approximate"
        How uncorrected p-values are obtained.

        ``"approximate"``
            Sample each study's *local configuration* instead of refitting the brain. At one
            voxel the studies are independent under relocation, and each study's contribution
            follows from its foci count and the kernel geometry alone, so a voxel's null can be
            drawn directly. Costs ``n_draws x n_studies`` with no dependence on the size of the
            mask, where relocation costs ``n_voxels x n_studies x n_iters``; on a whole brain
            with 40 studies that is 4e7 study-voxel pairs against 9.1e9. The draws are also
            independent, where neighbouring voxels of a relocation share studies. Agrees with
            the relocation null to within 2% of the ``|z|`` threshold at every p from .05 to
            1e-4, and to r = 0.999 on the p-values themselves. ``n_iters`` sets the draws, at
            1000 each.

            The default, because it matches the relocation null's calibration at a quarter of
            the cost: over 15 global-null simulations the two return the same uncorrected
            rejection rates, 0.042 and 0.041 at a nominal .05 and 0.0010 apiece at a nominal
            .001, for 14.0 s/fit against 53.7.

        ``"montecarlo"``
            Relocate every focus to a random in-mask voxel ``n_iters`` times, keeping its
            effect size and study membership, and read p off the resulting null distribution
            of ``|z|``. This is the null the convergence-based estimators use -- that reported
            coordinates fall at random within the mask -- and it is what ``"approximate"`` is
            validated against, so it remains the reference when the two could disagree: a mask
            small enough that relocation is cheap anyway, or a studyset whose kernel weights
            make the per-voxel independence assumption doubtful. It is also what makes
            :class:`~nimare.correct.FDRCorrector` and ``FWECorrector(method="bonferroni")``
            meaningful, since both simply operate on these p-values.
        ``"none"``
            No uncorrected p-values at all: ``p`` comes back as 1 everywhere and the effect
            size, prevalence and ``z`` maps are produced without a null. For inspecting the
            estimates when inference is not wanted, at no cost.

            This replaces a former ``"parametric"`` option, which referred ``g / se`` to a
            normal distribution. That was fast and wrong: on a global null it returned
            uncorrected ``p < .05`` for 41% of voxels with ``selection_model="none"`` and 11%
            with ``"zero-inflated"`` against a nominal 5%, and FDR and Bonferroni built on it
            rejected somewhere in every null simulation. Returning no p-value is the honest
            version of being fast.

    cluster_threshold : :obj:`float` or None, default=0.001
        Cluster-forming threshold, as an uncorrected p-value, for the cluster-level FWE null
        that :meth:`fit` builds alongside the voxel-level one. Set to None to skip it, which
        makes :meth:`correct_fwe_montecarlo` pay for a second pass over the permutations if
        cluster correction is then requested.
    n_iters : :obj:`int`, default=1000
        Iterations for the null, read differently by the two methods. Under
        ``null_method="approximate"`` it sets the draws per voxel, at 1000 each, and costs
        little. Under ``"montecarlo"`` each iteration is a full refit and this becomes the
        dominant cost of the estimator -- far more so than for ALE, whose per-iteration
        statistic is much cheaper -- so reduce it when exploring with that null.
    n_cores : :obj:`int`, default=1
        Processes used for the Monte Carlo null, which is where nearly all the time goes.
        ``-1`` uses every available core and is close to linear, since the relocations are
        independent; it is the single largest speedup available to a caller. Note the null is
        intrinsically dearer than the observed fit -- relocation scatters the foci over far
        more of the brain than the real, clustered configuration does, which made a null
        iteration 2.4x the cost of the fit it came from in one whole-brain benchmark.
        Cores for the Monte Carlo null. ``-1`` uses all available.
    seed : :obj:`int`, default=0
        Seed for the relocation draws.
    mask : Niimg-like object or None, optional
        Mask to use. If None, the collection's masker is used.
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
    Available maps:

    ============== ===============================================================
    "g"            Pooled Hedges' g.
    "se"           Standard error of the pooled estimate.
    "z"            ``g / se``. Two-tailed.
    "p", "logp"    p-value for ``z``, and its ``-log10``.
    "tau2"         Local between-study variance.
    "n_studies"    Number of studies with a focus inside the kernel support.
    "n_eff"        Kish effective number of studies, ``(sum w)^2 / sum w^2``.
    ============== ===============================================================

    :meth:`correct_fwe_montecarlo` adds ``logp_level-voxel``,
    ``logp_desc-size_level-cluster`` and ``logp_desc-mass_level-cluster`` (each with a
    signed ``z_*`` companion), matching the names
    :class:`~nimare.meta.cbma.ale.ALE` uses. :class:`~nimare.correct.FDRCorrector` and
    ``FWECorrector(method="bonferroni")`` work off the uncorrected ``"p"`` map instead,
    and are only meaningful when that map came from the Monte Carlo null.

    Warnings
    --------
    This estimator is new and has not been validated against a reference implementation.

    On a global null (30 studies of pure noise foci, 20 simulations) the Monte Carlo null
    returned uncorrected ``p < .05`` for 5.2% of voxels with ``selection_model="none"`` and
    4.2% with ``"zero-inflated"``, and every correction built on it rejected in 1 or 2 of the
    20 simulations -- consistent with a nominal 0.05, though 20 simulations cannot resolve a
    rate more finely than that.

    The effect-size maps are well ranked but poorly calibrated in magnitude. Against a
    random-effects pooling of the 21 NIDM pain studies' full t images, CBES run on peaks
    thresholded out of those same images reached rho = 0.84 (ALE, 0.18) but overestimated the
    effect roughly twofold (0.80 against 0.41 where the reference exceeded 0.2). Reported
    peaks are local maxima, and that inflation is not yet modelled. Treat ``g`` as a relative
    map until it is.

    References
    ----------
    .. footbibliography::
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        fwhm=10.0,
        use_images=True,
        peak_bias=None,
        peak_bias_scale=1.0,
        stat_column=None,
        design="one-sample",
        tau2_method="dl",
        selection_model="zero-inflated",
        threshold="pooled-min",
        coverage_radius=None,
        kernel_min_weight=0.01,
        max_iter=25,
        null_method="approximate",
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
        if isinstance(peak_bias_scale, str) and peak_bias_scale not in (
            "auto",
            "images",
            "reference",
        ):
            raise ValueError(
                "peak_bias_scale must be 'auto', 'images', 'reference', or a positive "
                f"number; got {peak_bias_scale!r}."
            )
        if not isinstance(peak_bias_scale, str) and not float(peak_bias_scale) > 0:
            raise ValueError(
                "peak_bias_scale must be 'auto', 'images', 'reference', or a positive "
                f"number; got {peak_bias_scale!r}."
            )
        if isinstance(peak_bias, str):
            if peak_bias != "per-study":
                raise ValueError(
                    f"peak_bias must be None, 'per-study', or a number in (0, 1]; got "
                    f"{peak_bias!r}."
                )
        elif peak_bias is not None and not 0.0 < float(peak_bias) <= 1.0:
            raise ValueError(
                f"peak_bias must be None, 'per-study', or a number in (0, 1]; got "
                f"{peak_bias!r}."
            )
        if isinstance(threshold, str):
            pass  # a keyword, or the name of a metadata field holding per-study thresholds
        elif threshold is not None and not np.isscalar(threshold):
            raise ValueError(
                f"threshold must be one of {list(THRESHOLD_KEYWORDS)}, a metadata field name, "
                f"a number, or None; got {threshold!r}."
            )

        self.fwhm = fwhm
        self.use_images = use_images
        self.peak_bias = peak_bias
        self.peak_bias_scale = (
            peak_bias_scale if isinstance(peak_bias_scale, str) else float(peak_bias_scale)
        )
        self.stat_column = stat_column
        self.design = design
        self.tau2_method = tau2_method
        self.selection_model = selection_model
        self.threshold = threshold
        self.coverage_radius = coverage_radius
        self.kernel_min_weight = kernel_min_weight
        self.max_iter = max_iter
        self.null_method = null_method
        self.cluster_threshold = cluster_threshold
        self.n_iters = n_iters
        self.n_cores = n_cores
        self.seed = seed

        if mask is not None:
            mask = get_masker(mask, memory=memory, memory_level=memory_level)
        self.masker = mask

    # ------------------------------------------------------------------ inputs

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
            filter_func=np.mean,
        )
        self._reported_thresholds_ = self._threshold_metadata(dataset)

    def _threshold_metadata(self, dataset):
        """Per-study reporting thresholds, when ``threshold`` names a metadata field.

        Papers that state their threshold are the easy case, and they should not be forced
        through an inference that exists only for the ones that do not. Values are read on the
        z scale, the same convention as a float ``threshold``; studies missing the field fall
        back to the median of those that have it.
        """
        if not isinstance(self.threshold, str) or self.threshold in THRESHOLD_KEYWORDS:
            return None

        available = set(dataset.get_metadata())
        if self.threshold not in available:
            raise ValueError(
                f"threshold={self.threshold!r} is neither one of "
                f"{list(THRESHOLD_KEYWORDS)} nor a metadata field of the collection. "
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

    def _resolve_stat_column(self, coords):
        """Pick the column holding the reported statistic, and say what scale it is on."""
        if self.stat_column is not None:
            if self.stat_column not in coords.columns:
                raise ValueError(
                    f"stat_column={self.stat_column!r} is not a column of the input "
                    f"coordinates. Available columns: {sorted(coords.columns)}."
                )
            column = self.stat_column
        elif "z_stat" in coords.columns and coords["z_stat"].notna().any():
            column = "z_stat"
        elif "t_stat" in coords.columns and coords["t_stat"].notna().any():
            column = "t_stat"
        elif getattr(self, "_image_studies_", None):
            return None, "z"  # images carry the fit; the coordinates are unused
        else:
            raise ValueError(
                "CBES needs a reported test statistic for each peak, but the input "
                "coordinates have no usable 'z_stat' or 't_stat' column. Convergence-based "
                "estimators (ALE, MKDADensity, KDA) do not require one; effect-size "
                "estimation does."
            )

        return column, "t" if column.startswith("t") else "z"

    def _load_image_studies(self, dataset):
        """Return ``{study_id: (g, var_g)}`` for studies supplying both images.

        Masked to the analysis volume, so the vectors line up with every other per-voxel array
        in the estimator.
        """
        if not self.use_images:
            return {}

        images = getattr(dataset, "images", None)
        if images is None or "g" not in images.columns or "g_var" not in images.columns:
            return {}

        loaded = {}
        for study_id, g_path, var_path in zip(
            images["id"].astype(str), images["g"], images["g_var"]
        ):
            if g_path is None or var_path is None:
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

        if loaded:
            LGR.info(f"Using images for {len(loaded)} studies; coordinates for the rest.")
        return loaded

    def _build_focus_table(self):
        """Reduce the coordinates table to the per-focus quantities the model consumes."""
        coords = self.inputs_["coordinates"]
        column, stat_type = self._resolve_stat_column(coords)
        if column is None:
            return coords.iloc[:0].assign(
                stat=np.array([], dtype=float),
                g=np.array([], dtype=float),
                var_g=np.array([], dtype=float),
                stat_type=np.array([], dtype=object),
            )

        if "sample_size" not in coords.columns:
            raise ValueError(
                "CBES needs a sample size for every study in order to place reported "
                "statistics on an effect-size scale. Populate the metadata field "
                "'sample_sizes' or 'sample_size'."
            )

        usable = coords[column].notna() & coords["sample_size"].notna()
        dropped = int((~usable).sum())
        if dropped:
            level = LGR.warning if self._drop_invalid else LGR.info
            level(
                f"Dropping {dropped} of {len(coords)} foci with no reported {column} or no "
                "sample size."
            )
        if not usable.any() and not getattr(self, "_image_studies_", None):
            raise ValueError(
                f"No focus has both a reported {column} and a sample size; nothing to pool."
            )

        table = coords.loc[usable, ["id", "i", "j", "k", "sample_size", column]].copy()
        table = table.rename(columns={column: "stat"})

        g, var_g = peak_stat_to_hedges_g(
            table["stat"].values,
            table["sample_size"].values,
            stat_type=stat_type,
            design=self.design,
        )
        table["g"] = g
        table["var_g"] = var_g
        table["stat_type"] = stat_type
        return table

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
        sample_sizes = np.asarray(dataset.sample_sizes(), dtype=float)
        series = pd.Series(sample_sizes, index=ids)
        series = series[series.notna()]
        if series.empty:
            raise ValueError(
                "CBES needs a sample size for every study in order to place reported "
                "statistics on an effect-size scale. Populate the metadata field "
                "'sample_sizes' or 'sample_size'."
            )
        return series[~series.index.duplicated()]

    def _reported_z(self, table):
        """Reported statistics on the z scale, where studies are comparable to one another.

        A t of 3.5 means something different in a study of 15 than in one of 80, so every
        threshold inference and every peak-height correction happens here, not on the raw
        reported scale.
        """
        if not len(table):
            return np.array([], dtype=float)

        stat_type = table["stat_type"].iloc[0]
        return np.abs(
            table["stat"].values
            if stat_type == "z"
            else t_to_z(table["stat"].values, table["sample_size"].values - 1)
        )

    def _study_cutoffs_z(self, table, sample_sizes):
        """Per-study reporting threshold on the z scale, one entry per study on the roster.

        Studies that reported nothing anywhere still need a threshold -- it is what makes their
        silence quantitative. Since they reported no statistic to infer one from, they are
        given the median threshold of the studies that did report.
        """
        index = sample_sizes.index
        if not len(table):
            # Every study supplied an image, so there are no reported peaks and nothing is
            # censored. The cutoffs are unused but must still line up with the roster.
            return pd.Series(np.zeros(len(index)), index=index)

        reported_z = self._reported_z(table)
        study_ids = np.asarray(table["id"].values, dtype=object)

        if self.threshold == "pooled-min":
            # The smallest statistic reported anywhere is the tightest available upper bound on
            # a threshold shared by every study.
            cutoff_z = np.full(
                len(index), float(np.nanmin(reported_z)) if reported_z.size else np.nan
            )
        elif self.threshold == "study-min":
            grouped = pd.Series(reported_z, index=study_ids).groupby(level=0)
            per_study = grouped.min().astype(float)
            # A study's smallest reported peak is the minimum of however many peaks it
            # reported, so it sits above the threshold by an amount that depends on that count.
            # Undoing the order statistic is never worse than taking the minimum at face value,
            # and reduces to it when the study reported enough peaks for the gap to vanish.
            per_study = pd.Series(
                [
                    infer_threshold_from_minimum(minimum, count)
                    for minimum, count in zip(per_study.values, grouped.size().values)
                ],
                index=per_study.index,
                dtype=float,
            )
            fallback = float(np.nanmedian(per_study.values)) if len(per_study) else np.nan
            cutoff_z = per_study.reindex(index).astype(float).fillna(fallback).values
        elif isinstance(self.threshold, str):
            supplied = getattr(self, "_reported_thresholds_", None)
            if supplied is None:
                raise ValueError(
                    f"threshold={self.threshold!r} names a metadata field, but no per-study "
                    "thresholds were read from the collection."
                )
            fallback = (
                float(np.nanmedian(supplied.values[np.isfinite(supplied.values)]))
                if np.isfinite(supplied.values).any()
                else np.nan
            )
            cutoff_z = supplied.reindex(index).astype(float).fillna(fallback).values
        else:
            value = DEFAULT_REPORTING_THRESHOLD_Z if self.threshold is None else self.threshold
            cutoff_z = np.full(len(index), float(value))

        cutoff_z = np.asarray(cutoff_z, dtype=float)
        cutoff_z = np.where(
            np.isfinite(cutoff_z) & (cutoff_z > 0), cutoff_z, DEFAULT_REPORTING_THRESHOLD_Z
        )
        self._check_peak_information(reported_z, cutoff_z)
        return pd.Series(cutoff_z, index=index)

    def _peak_bias_factors(self, cutoff_z, sample_sizes, reporting_ids):
        """Per-study shrinkage ``rho_k`` for the peak-height bias, one entry per study.

        A reported peak is a local maximum that cleared the study's own threshold, so its
        height is set partly by the effect and partly by ``(u_k, N_k)``: the stricter the
        threshold and the smaller the sample, the larger the effect size a study reports for
        the same underlying truth. :func:`null_peak_mean_g` says exactly how large that
        artefact is -- the effect size a study would report from a peak of *pure noise*.

        ``peak_bias="per-study"`` divides it out. ``rho_k`` is inversely proportional to
        ``null_peak_mean_g(u_k, N_k)``, normalized so that the median reporting study is left
        at ``peak_bias_scale``. That removes the *between-study* artefact, which is what
        coordinates alone can identify; the one remaining number, the overall scale, is
        ``peak_bias_scale`` and still needs images (or a willingness to read the map as
        relative). A scalar ``peak_bias`` sets every ``rho_k`` to the same value instead,
        correcting the scale but not the heterogeneity.
        """
        index = sample_sizes.index
        if self.peak_bias is None:
            return pd.Series(np.ones(len(index)), index=index)
        if not isinstance(self.peak_bias, str):
            return pd.Series(np.full(len(index), float(self.peak_bias)), index=index)

        scale = float(getattr(self, "_peak_bias_scale_", 1.0))

        null_g = np.array(
            [
                null_peak_mean_g(cutoff, size, design=self.design)
                for cutoff, size in zip(cutoff_z.values, sample_sizes.values)
            ],
            dtype=float,
        )
        # Only studies that actually reported peaks carry the artefact, so they set the anchor;
        # silent studies contribute nothing to rescale.
        reporting = np.isin(
            np.asarray(index, dtype=object), np.asarray(reporting_ids, dtype=object)
        )
        reference = null_g[reporting] if reporting.any() else null_g
        finite = reference[np.isfinite(reference) & (reference > 0)]
        anchor = float(np.median(finite)) if finite.size else 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            rho = scale * anchor / null_g
        rho = np.where(np.isfinite(rho) & (rho > 0), rho, scale)
        return pd.Series(rho, index=index)

    def _resolve_peak_bias_scale(self, table, sample_sizes, reporting_ids):
        """Settle ``peak_bias_scale`` before it is used, calibrating it if asked and able.

        Calibration needs a provisional fit, which needs a ``rho``, so the scale is resolved
        at 1.0 first and the answer applied afterwards. That is exact rather than iterative:
        the fit is linear in the scale, so a fit at 1.0 times the calibrated scale *is* the
        fit at the calibrated scale.
        """
        if self.peak_bias is None:
            return 1.0
        if self.peak_bias_scale not in ("auto", "images", "reference"):
            if self._image_studies_ and self.peak_bias_scale == 1.0:
                LGR.warning(  # noqa: E501
                    "This fit mixes images with coordinates but leaves peak_bias_scale at "
                    "1.0, so the coordinate studies are on a relative scale while the images "
                    "are on the true Hedges' g scale. The two then disagree about the same "
                    "voxel -- by a factor of about two on the NIDM pain images -- and the "
                    "pooled value depends on how many studies of each kind the collection "
                    "holds. Use peak_bias_scale='auto' to read the constant off the images."
                )
            return float(self.peak_bias_scale)

        self._peak_bias_scale_ = 1.0
        provisional = self._peak_bias_factors(self._cutoffs_z_, sample_sizes, reporting_ids)
        scaled, thresholds = self._apply_peak_bias(
            table, self._cutoffs_z_, sample_sizes, provisional
        )

        if self.peak_bias_scale == "reference":
            return self._calibrate_scale_from_reference(scaled, sample_sizes)

        if not self._image_studies_:
            LGR.warning(
                "peak_bias_scale needs images to calibrate against, and this collection "
                "supplies none. Falling back to 1.0, which leaves the effect-size map correct "
                "up to one multiplicative constant. peak_bias_scale='reference' sets the scale "
                "from an external corpus instead, to within about a factor of two."
            )
            return 1.0

        return self._calibrate_peak_bias_scale(
            scaled, sample_sizes, thresholds, self._image_studies_
        )

    def _calibrate_scale_from_reference(self, table, sample_sizes):
        """Set the overall scale from what studies of this size typically find.

        No route from the coordinates' own reporting behaviour recovers the scale; seven were
        tried and section 15 records why. This takes it from outside instead. Sample size
        predicts effect magnitude across a reference corpus of 258 group maps, because studies
        are powered for the effects they set out to find, and matching on it cuts the error in
        predicting a held-out magnitude by 28%. Spatial similarity, which looks like the more
        natural key, does not work at all.

        What this assumes is a fact about how research is designed, not about the brain:
        researchers who ran this many subjects were usually studying effects of about this size.
        It would be wrong for a collection unusually over- or under-powered for its effect, and
        it is good to about a factor of two either way -- so the resulting magnitudes should be
        read as an order of scale, not a calibrated value.

        .. warning::
            On field-simulated collections with a known truth this made the estimate *worse*,
            by 1.5 to 2.0x, in every case tried. The reason is the assumption failing as
            advertised: those collections had true effects of 0.5 to 1.2 where the reference
            expects about 0.57 at their sample sizes, so the prior pulled them down. A
            collection atypical of the reference is penalised for it.

            Those same runs once looked like they contradicted the inflation measured against
            the image reference. They no longer do: the 2.65x they were compared against was an
            artefact of a ratio of means over near-zero truth, and scored where the signal is,
            the two instruments broadly agree -- 1.72 on pain at the top decile against 1.22 to
            1.52 on the simulator at its focus. What gap remains is explained by how much the
            peaks carry: simulated peaks hold real signal, while the pain peaks average z =
            3.639 against 3.625 for peaks of pure noise, so on real data the scale is being
            reconstructed from thresholds and sample sizes rather than read off the values.
            Hence opt-in, and never chosen by ``"auto"``.
        """
        expected = reference_magnitude(sample_sizes.values)
        if expected is None:
            LGR.warning("No usable sample sizes; cannot set the scale from the reference.")
            return 1.0

        fit = self._pool(table, None)
        covered = fit["covered"]
        if not covered.any():
            return 1.0
        magnitude = np.abs(fit["g"][covered])
        if not magnitude.size or magnitude.max() <= 0:
            return 1.0
        # Matched to how the reference was summarised: the mean over each map's own top decile,
        # since a whole-brain mean measures how much of the brain is active rather than how
        # strong the effect is where it is present.
        top = magnitude[magnitude >= np.percentile(magnitude, 90)]
        observed = float(top.mean())
        if observed <= 0:
            return 1.0

        scale = expected / observed
        spread = float(np.exp(REFERENCE_MAGNITUDE_LOG_SD))
        LGR.info(
            f"Reference calibration: studies of this size typically show |g| ~ {expected:.3f}, "
            f"this fit shows {observed:.3f}, so peak_bias_scale = {scale:.3f}. The reference "
            f"is good to about a factor of {spread:.1f}, so read the magnitudes as an order of "
            "scale rather than a calibrated value."
        )
        return scale

    def _calibrate_peak_bias_scale(self, table, sample_sizes, thresholds, image_studies):
        """Read the overall peak-to-field ratio off the studies that supplied images.

        Between-study differences in the peak-height bias are identified from the coordinates
        alone (:meth:`_peak_bias_factors`), but the one common scale is not: rescaling every
        coordinate study by the same constant leaves the coordinate-only likelihood unchanged.
        That is harmless for a coordinate-only map, which is then correct up to a constant, and
        *not* harmless once images are in the same fit -- images sit on the true ``g`` scale, so
        a mismatched constant makes the two kinds of study disagree about the same voxel and
        the pooled value depends on how many of each the collection happens to hold. On the
        NIDM pain images that disagreement is a factor of 2.05.

        So when images are present the constant is no longer free, and they are what fixes it:
        fit the images alone and the coordinates alone, and take the ratio of the two over the
        voxels both cover. The fit is exactly linear in the scale, so a ratio of summaries
        recovers it -- a regression slope would be attenuated by the many voxels where a study
        peaked and the images say nothing.
        """
        coordinate_only = self._statistic(table, sample_sizes, thresholds, image_studies=None)[0]
        image_only = self._statistic(
            table.iloc[:0], sample_sizes, thresholds, image_studies=image_studies
        )[0]

        both = (
            coordinate_only["covered"]
            & image_only["covered"]
            & np.isfinite(coordinate_only["g"])
            & np.isfinite(image_only["g"])
        )
        from_coordinates = np.abs(coordinate_only["g"][both])
        from_images = np.abs(image_only["g"][both])
        if not both.any() or from_coordinates.mean() <= 0:
            LGR.warning(
                "Cannot calibrate peak_bias_scale: the image studies and the coordinate "
                "studies share no voxel. Falling back to 1.0, which leaves the two kinds of "
                "study on different scales."
            )
            return 1.0

        # Scored where the images say there is something to estimate, and by a paired median
        # rather than a ratio of means. This version divided the two means over every shared
        # voxel, and a covered brain is mostly voxels holding no effect, so the image mean
        # collapsed toward zero and the scale came out far too small -- 0.16 against a true 0.8
        # on simulated data, which made adding images *worse* the more of them there were. The
        # same unsound summary is described at length in section 17 of the design note.
        strong = from_images >= np.percentile(from_images, 75)
        if strong.sum() < 50:
            strong = np.ones_like(from_images, dtype=bool)
        ratios = from_images[strong] / np.clip(from_coordinates[strong], 1e-12, None)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if not ratios.size:
            return 1.0

        scale = float(np.median(ratios))
        LGR.info(
            f"Calibrated peak_bias_scale = {scale:.3f} from {int(strong.sum())} voxels where "
            f"the {len(image_studies)} image studies show a substantial effect."
        )
        return scale

    def _apply_peak_bias(self, table, cutoff_z, sample_sizes, peak_bias):
        """Rescale reported effect sizes by ``rho_k``, and put the cutoffs on the g scale.

        The threshold lives on the same axis as the values it censored, so it takes the same
        factor -- otherwise the censored likelihood would be comparing a rescaled observation
        against an unrescaled bound.
        """
        threshold_g, _ = peak_stat_to_hedges_g(
            cutoff_z.values, sample_sizes.values, stat_type="z", design=self.design
        )
        thresholds = pd.Series(threshold_g * peak_bias.values, index=sample_sizes.index)

        if len(table) and not np.allclose(peak_bias.values, 1.0):
            default = float(np.median(peak_bias.values)) if len(peak_bias) else 1.0
            factor = (
                peak_bias.reindex(np.asarray(table["id"].values, dtype=object))
                .astype(float)
                .fillna(default)
                .values
            )
            table = table.copy()
            table["g"] = table["g"].values * factor
            table["var_g"] = table["var_g"].values * factor**2
            table["peak_bias"] = factor

        return table, thresholds

    def _check_peak_information(self, reported_z, cutoff_z):
        """Warn when the reported peak heights say nothing about the size of the effect."""
        if not len(reported_z):
            return
        threshold = float(np.nanmedian(cutoff_z))
        observed, expected, excess = peak_information(reported_z, threshold)
        self.peak_information_ = {
            "observed_mean_z": observed,
            "null_peak_mean_z": expected,
            "excess_z": excess,
        }
        if excess < _MIN_PEAK_EXCESS_Z:
            LGR.warning(
                f"Reported peak heights average z = {observed:.3f} against {expected:.3f} for "
                "peaks of pure noise at the same threshold, an excess of "
                f"{excess:+.3f}. Their magnitudes are therefore close to uninformative about "
                "the effect size -- they are largely a function of the reporting threshold and "
                "the sample size. The estimated scale will be far too large and no correction "
                "computed from the peak values can repair it. Because the bias is a function "
                "of (threshold, sample size), peak_bias='per-study' removes the part of it "
                "that varies between studies without needing images; fixing the common scale "
                "still needs images, via peak_bias_scale or a scalar peak_bias. The spatial "
                "pattern is driven by where the peaks are, which none of this touches."
            )

    # ------------------------------------------------------- spatial machinery

    def _kernel_support(self, sample_size=None):
        """Sparse (offsets, weights) for the spatial-uncertainty kernel, peak-normalized.

        Truncated at ``kernel_min_weight`` of the peak. :func:`get_ale_kernel` keeps every
        voxel above floating-point zero, which for a 10 mm FWHM kernel is a radius of about
        26 mm -- so on a whole-brain mask every voxel ends up "reached" by several studies at
        weights of order 1e-10, ``n_studies`` stops meaning anything, and the censoring term
        is evaluated at voxels no study says anything about.
        """
        mask_img = self.masker.mask_img
        if self.fwhm is not None:
            _, kernel = get_ale_kernel(mask_img, fwhm=self.fwhm)
        else:
            _, kernel = get_ale_kernel(mask_img, sample_size=sample_size)

        kernel = kernel / kernel.max()
        kernel[kernel < self.kernel_min_weight] = 0.0
        offsets, values = _kernel_to_sparse_support(kernel)
        return offsets, values

    def _study_voxel_weights(self, study_table, offsets, values, mask_flat_to_masked, shape):
        """Expand one study's foci onto masked voxels, keeping the nearest focus per voxel.

        Returns ``(cols, w, g, var_g)``, one entry per voxel the study reaches. A study that
        reports two peaks close together would otherwise contribute twice to the same voxel and
        be counted as two independent studies; keeping only the largest weight (i.e. the
        nearest peak) enforces one observation per study per voxel, as ALE does when it takes
        the maximum over a study's kernels.
        """
        ijk = study_table[["i", "j", "k"]].values.astype(np.int64)
        n_foci = len(ijk)

        candidates = ijk[:, None, :] + offsets[None, :, :].astype(np.int64)
        in_bounds = np.all((candidates >= 0) & (candidates < np.asarray(shape)), axis=-1)

        flat = (
            candidates[..., 0] * shape[1] * shape[2]
            + candidates[..., 1] * shape[2]
            + candidates[..., 2]
        )
        flat = np.where(in_bounds, flat, 0)
        cols = mask_flat_to_masked[flat]
        keep = in_bounds & (cols >= 0)
        if not np.any(keep):
            empty_i = np.array([], dtype=np.int64)
            empty_f = np.array([], dtype=float)
            return empty_i, empty_f, empty_f, empty_f

        focus_idx = np.broadcast_to(np.arange(n_foci)[:, None], (n_foci, len(values)))
        cols = cols[keep].astype(np.int64)
        weights = np.broadcast_to(values, (n_foci, len(values)))[keep].astype(float)
        focus_idx = focus_idx[keep]

        # One observation per (study, voxel): keep the focus with the largest weight.
        order = np.lexsort((weights, cols))
        cols, weights, focus_idx = cols[order], weights[order], focus_idx[order]
        last_of_group = np.r_[cols[1:] != cols[:-1], True]
        cols, weights, focus_idx = (
            cols[last_of_group],
            weights[last_of_group],
            focus_idx[last_of_group],
        )

        return (
            cols,
            weights,
            study_table["g"].values[focus_idx],
            study_table["var_g"].values[focus_idx],
        )

    def _accumulate(self, table, image_studies=None):
        """Walk the studies once, returning per-study voxel contributions and voxel sums.

        An image study is appended as a contribution covering every voxel at weight 1.
        Everything downstream -- the moment sums, the second pooling pass, the selection
        model -- then treats it exactly like a very well localized reported peak, which is
        what it is.
        """
        mask_img = self.masker.mask_img
        # ``shape[:3]``: a mask image may carry a trailing singleton volume axis.
        shape = np.asarray(mask_img.shape[:3], dtype=np.int64)
        mask_flat_to_masked = _get_mask_flat_to_masked(mask_img)
        n_voxels = int(mask_flat_to_masked.max()) + 1 if mask_flat_to_masked.size else 0

        fixed_support = self._kernel_support() if self.fwhm is not None else None

        sums = {
            name: np.zeros(n_voxels, dtype=float)
            for name in ("w", "w2", "a", "a2", "ag", "ag2", "w2_over_s2", "n")
        }
        contributions = []

        for study_id, study_table in table.groupby("id", sort=False):
            if fixed_support is not None:
                offsets, values = fixed_support
            else:
                offsets, values = self._kernel_support(
                    sample_size=float(study_table["sample_size"].iloc[0])
                )

            cols, weights, g, var_g = self._study_voxel_weights(
                study_table, offsets, values, mask_flat_to_masked, shape
            )
            if not cols.size:
                LGR.info(f"Study {study_id} contributes no in-mask voxels; skipping.")
                continue

            contributions.append((study_id, cols, weights, g, var_g))

            a = weights / var_g
            for name, value in (
                ("w", weights),
                ("w2", weights**2),
                ("a", a),
                ("a2", a**2),
                ("ag", a * g),
                ("ag2", a * g**2),
                ("w2_over_s2", weights**2 / var_g),
                ("n", np.ones_like(weights)),
            ):
                sums[name] += np.bincount(cols, weights=value, minlength=n_voxels)

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

    # ------------------------------------------------------------------ fitting

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

        for _, cols, weights, g, var_g in contributions:
            total_var = var_g + tau2[cols]
            pooling_weight = weights / total_var
            numerator += np.bincount(cols, weights=pooling_weight * g, minlength=n_voxels)
            denominator += np.bincount(cols, weights=pooling_weight, minlength=n_voxels)
            variance_numerator += np.bincount(
                cols, weights=weights**2 / total_var, minlength=n_voxels
            )

        covered = denominator > 0
        g_hat = np.zeros(n_voxels, dtype=float)
        se = np.full(n_voxels, np.inf, dtype=float)
        g_hat[covered] = numerator[covered] / denominator[covered]
        se[covered] = np.sqrt(variance_numerator[covered]) / denominator[covered]

        n_eff = np.zeros(n_voxels, dtype=float)
        positive_w = sums["w2"] > 0
        n_eff[positive_w] = sums["w"][positive_w] ** 2 / sums["w2"][positive_w]

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

    def _coverage_entries(self, table, study_ids, active, n_voxels, image_ids=()):
        """``(local_voxel, study_position)`` pairs: did this study report anything near here?

        Separate from the pooling kernel on purpose. The kernel answers "how much does this
        study's reported value tell me about this voxel", and falls off quickly with distance.
        Coverage answers a different question -- "was this study silent about this region" --
        and a study whose peak landed 6 mm away was not silent. Judging both with the same
        kernel makes such a study argue against its own reported effect, which pulls the
        estimate toward zero.
        """
        mask_img = self.masker.mask_img
        # ``shape[:3]``: a mask image may carry a trailing singleton volume axis.
        shape = np.asarray(mask_img.shape[:3], dtype=np.int64)
        mask_flat_to_masked = _get_mask_flat_to_masked(mask_img)

        radius = self.coverage_radius
        if radius is None:
            radius = 2.0 * (self.fwhm if self.fwhm is not None else 10.0)
        offsets = sphere_kernel_offsets(radius, mask_img.header.get_zooms()[:3])

        active_lookup = np.full(n_voxels, -1, dtype=np.int64)
        active_lookup[active] = np.arange(active.size)
        # Reused across studies to deduplicate the voxels a study's spheres cover. A scratch
        # bitmap costs one pass over the hits; ``np.unique`` sorts or hashes them, and with
        # a 20 mm sphere per focus there are a great many hits.
        seen = np.zeros(active.size, dtype=bool)

        image_ids = set(image_ids)
        cols, positions = [], []
        for position, study_id in enumerate(study_ids):
            if study_id in image_ids:
                # An image reports everywhere, so it is silent nowhere and contributes no
                # censoring term. Marking it covered at every active voxel says exactly that.
                cols.append(np.arange(active.size, dtype=np.int64))
                positions.append(np.full(active.size, position, dtype=np.int64))
                continue
            ijk = table.loc[table["id"] == study_id, ["i", "j", "k"]].values.astype(np.int64)
            if not ijk.size:
                continue  # reported nothing anywhere: silent at every voxel
            candidates = ijk[:, None, :] + offsets[None, :, :].astype(np.int64)
            in_bounds = np.all((candidates >= 0) & (candidates < shape), axis=-1)
            flat = (
                candidates[..., 0] * shape[1] * shape[2]
                + candidates[..., 1] * shape[2]
                + candidates[..., 2]
            )
            reached = mask_flat_to_masked[np.where(in_bounds, flat, 0)]
            reached = reached[in_bounds & (reached >= 0)].astype(np.int64)
            if not reached.size:
                continue
            local = active_lookup[reached]
            local = local[local >= 0]
            if not local.size:
                continue
            seen[local] = True
            local = np.flatnonzero(seen)
            seen[local] = False
            cols.append(local)
            positions.append(np.full(local.size, position, dtype=np.int64))

        if not cols:
            empty = np.array([], dtype=np.int64)
            return empty, empty

        return np.concatenate(cols), np.concatenate(positions)

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

        A study that reported nothing in this region is silent for one of two reasons: it has
        no effect here at all, or it has one that failed to clear its reporting threshold. The
        zero-inflated model lets the data decide between them, so silence no longer has to be
        explained as a small-but-real common effect -- which is what drags a plain Tobit fit
        below the truth.

        ``tau2`` is held at its moment estimate throughout, so each EM iteration optimizes only
        :math:`\mu` (concave, one-dimensional) alongside a closed-form update for
        :math:`\pi`.
        """
        active = np.flatnonzero(fit["covered"])
        n_voxels = fit["n_voxels"]
        fit["prevalence"] = np.zeros(n_voxels, dtype=float)
        if not active.size:
            return

        study_ids = list(sample_sizes.index)
        cov_col, cov_pos = self._coverage_entries(
            table, study_ids, active, n_voxels, image_ids=image_ids
        )
        values = self._value_entries(fit, study_ids, active, n_voxels)
        n_studies = len(study_ids)

        null_var = null_effect_variance(sample_sizes.values, design=self.design)[:, None]
        peak_bias = getattr(self, "_peak_bias_", None)
        if peak_bias is not None:
            # The null component lives on the same rescaled axis as the observations, so it
            # takes each study's own rho -- not one shared factor.
            null_var = null_var * peak_bias.loc[study_ids].values[:, None] ** 2
        cutoffs = np.abs(thresholds.loc[study_ids].values)[:, None]
        value_order = np.argsort(values["col"], kind="mergesort")
        values = {name: array[value_order] for name, array in values.items()}
        cov_order = np.argsort(cov_col, kind="mergesort")
        cov_col, cov_pos = cov_col[cov_order], cov_pos[cov_order]

        mu_out = np.zeros(active.size, dtype=float)
        pi_out = np.zeros(active.size, dtype=float)
        se_out = np.full(active.size, np.inf, dtype=float)

        # Dense blocks are (n_studies, chunk); cap their element count rather than their width
        # so that a studyset with many experiments simply takes more, smaller chunks.
        chunk = max(1, int(2e6 // max(n_studies, 1)))
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

            covered = np.zeros((n_studies, width), dtype=bool)
            c_lo, c_hi = np.searchsorted(cov_col, [lo, hi])
            if c_hi > c_lo:
                covered[cov_pos[c_lo:c_hi], cov_col[c_lo:c_hi] - lo] = True

            mu, pi, se = self._fit_chunk(
                weights=weights,
                g_obs=g_obs,
                var_obs=var_obs,
                covered=covered,
                tau2=fit["tau2"][active[lo:hi]],
                null_var=null_var,
                cutoffs=cutoffs,
                start=fit["g"][active[lo:hi]],
            )
            mu_out[lo:hi], pi_out[lo:hi], se_out[lo:hi] = mu, pi, se

        fit["g"] = np.zeros(n_voxels, dtype=float)
        fit["g"][active] = mu_out
        fit["prevalence"][active] = pi_out
        fit["se"] = np.full(n_voxels, np.inf, dtype=float)
        fit["se"][active] = se_out

    def _fit_chunk(self, *, weights, g_obs, var_obs, covered, tau2, null_var, cutoffs, start):
        """EM for one block of voxels. Returns ``(mu, prevalence, se)``, one value per voxel.

        Works on the ``(study, voxel)`` pairs that carry weight rather than on the dense
        study-by-voxel block. At any given voxel a study either reported nearby or was silent
        there, and in a real studyset most studies are neither -- they reported in the region
        but outside this voxel's kernel, so they inform neither term. Evaluating normal CDFs
        across the full block and then multiplying most of them by zero was 97% of the runtime.
        The arithmetic is unchanged; only the entries that contribute are visited.
        """
        zero_inflated = self.selection_model == "zero-inflated"
        width = weights.shape[1]

        reporting = np.flatnonzero(weights > 0)
        # Covered but out of kernel range: the study said something about this region but
        # nothing about this voxel. It informs neither term, and is in neither index.
        silence = np.flatnonzero(~covered)
        rep_voxel, rep_study = reporting % width, reporting // width
        sil_voxel, sil_study = silence % width, silence // width
        del rep_study

        w_rep = weights.ravel()[reporting]
        g_rep = g_obs.ravel()[reporting]
        var_rep = var_obs.ravel()[reporting]
        sigma_rep = np.sqrt(var_rep + tau2[rep_voxel])
        precision_rep = 1.0 / sigma_rep**2

        cutoff_sil = cutoffs.ravel()[sil_study]
        null_var_sil = null_var.ravel()[sil_study]
        sigma_sil = np.sqrt(null_var_sil + tau2[sil_voxel])
        # Hoisted out of the EM loop: sigma and the cutoffs do not move between iterations,
        # only mu does, so every division by them is paid once instead of tens of times.
        inv_sigma_sil = 1.0 / sigma_sil
        inv_sigma_sq_sil = inv_sigma_sil * inv_sigma_sil
        cutoff_scaled_sil = cutoff_sil * inv_sigma_sil
        twice_cutoff_scaled_sil = cutoff_scaled_sil * 2.0

        def censoring_at(values):
            """P(silent) and its derivatives for a set of silent observations at ``values``."""
            return _censoring_terms(
                values,
                cutoff_scaled_sil,
                twice_cutoff_scaled_sil,
                inv_sigma_sil,
                inv_sigma_sq_sil,
            )

        # A reporting study's log-likelihood is discounted by the spatial kernel, so a silent
        # study entering at full weight would count for more than a study that actually
        # measured something -- silence would outvote evidence, and the estimate would sit well
        # below the truth however many studies reported. Put a silent study on the same footing
        # as an average reporting study at this voxel instead.
        n_reporting = np.bincount(rep_voxel, minlength=width)
        sum_reported = np.bincount(rep_voxel, weights=w_rep, minlength=width)
        reporter_scale = np.divide(
            sum_reported, n_reporting, out=np.ones(width), where=n_reporting > 0
        )
        w_sil = reporter_scale[sil_voxel]

        # Probability a silent study stays silent when it has no effect at all. Fixed across
        # iterations, and close to one whenever the threshold is the usual several sigma.
        null_sd_sil = np.sqrt(null_var_sil)
        prob_silent_null = np.clip(
            ndtr(cutoff_sil / null_sd_sil) - ndtr(-cutoff_sil / null_sd_sil), 1e-12, None
        )
        density_null = _normal_pdf(g_rep / np.sqrt(var_rep)) / np.sqrt(var_rep)

        mu = start.copy()
        pi = np.full(width, 0.5 if zero_inflated else 1.0)
        total_weight = np.bincount(rep_voxel, weights=w_rep, minlength=width) + np.bincount(
            sil_voxel, weights=w_sil, minlength=width
        )
        resp_rep = np.ones(reporting.size)
        resp_sil = np.ones(silence.size)

        # Voxels converge at very different rates: most settle within a handful of iterations
        # while a few drift for dozens. Iterating the whole block until the slowest voxel is
        # done wastes nearly all of the work, and stopping on a global criterion instead leaves
        # the stragglers short of the MLE. So settled voxels are retired from the working set
        # and the rest keep going.
        mu_out = np.zeros(width)
        pi_out = np.zeros(width)
        se_out = np.full(width, np.inf)
        voxel_ids = np.arange(width)

        def _retire(positions, curvature):
            """Write out voxels that have converged."""
            ids = voxel_ids[positions]
            mu_out[ids] = mu[positions]
            pi_out[ids] = pi[positions]
            curv = curvature[positions]
            informative = curv < 0
            se_out[ids[informative]] = 1.0 / np.sqrt(-curv[informative])

        curvature = np.zeros(width)
        for _ in range(self.max_iter):
            if not mu.size:
                break
            censoring = censoring_at(mu[sil_voxel])
            pi_shift = np.zeros(mu.size)
            if zero_inflated:
                pi_rep, pi_sil = pi[rep_voxel], pi[sil_voxel]
                density_effect = _normal_pdf((g_rep - mu[rep_voxel]) / sigma_rep) / sigma_rep
                resp_rep = pi_rep * density_effect
                resp_rep /= resp_rep + (1.0 - pi_rep) * density_null + 1e-300
                resp_sil = pi_sil * censoring["prob"]
                resp_sil /= resp_sil + (1.0 - pi_sil) * prob_silent_null + 1e-300

                numerator = np.bincount(
                    rep_voxel, weights=w_rep * resp_rep, minlength=mu.size
                ) + np.bincount(sil_voxel, weights=w_sil * resp_sil, minlength=mu.size)
                previous_pi = pi
                pi = np.clip(
                    np.divide(
                        numerator,
                        total_weight,
                        out=np.zeros(mu.size),
                        where=total_weight > 0,
                    ),
                    1e-4,
                    1.0 - 1e-4,
                )
                pi_shift = np.abs(pi - previous_pi)

            score, curvature = _mu_derivatives(
                width=mu.size,
                mu_rep=mu[rep_voxel],
                g_rep=g_rep,
                precision_rep=precision_rep,
                rep_voxel=rep_voxel,
                weight_rep=w_rep * resp_rep,
                sil_voxel=sil_voxel,
                weight_sil=w_sil * resp_sil,
                censoring=censoring,
            )
            step = np.where(curvature < 0, -score / curvature, 0.0)
            # The likelihood is concave but flat far from the data; cap the step so a voxel
            # with almost no reporting weight cannot run away.
            mu = mu + np.clip(step, -1.0, 1.0)

            settled = (np.abs(step) < _EM_TOLERANCE) & (pi_shift < _EM_TOLERANCE)
            if settled.all():
                _retire(np.flatnonzero(settled), curvature)
                mu = mu[:0]
                break
            if settled.mean() < _EM_COMPACTION_FRACTION:
                continue

            _retire(np.flatnonzero(settled), curvature)
            keep = ~settled
            position = np.full(mu.size, -1, dtype=np.int64)
            position[np.flatnonzero(keep)] = np.arange(int(keep.sum()))

            moved = position[rep_voxel]
            kept_pairs = moved >= 0
            rep_voxel = moved[kept_pairs]
            w_rep, g_rep = w_rep[kept_pairs], g_rep[kept_pairs]
            sigma_rep, precision_rep = sigma_rep[kept_pairs], precision_rep[kept_pairs]
            density_null, resp_rep = density_null[kept_pairs], resp_rep[kept_pairs]

            moved = position[sil_voxel]
            kept_pairs = moved >= 0
            sil_voxel = moved[kept_pairs]
            w_sil, prob_silent_null = w_sil[kept_pairs], prob_silent_null[kept_pairs]
            inv_sigma_sil = inv_sigma_sil[kept_pairs]
            inv_sigma_sq_sil = inv_sigma_sq_sil[kept_pairs]
            cutoff_scaled_sil = cutoff_scaled_sil[kept_pairs]
            twice_cutoff_scaled_sil = twice_cutoff_scaled_sil[kept_pairs]
            resp_sil = resp_sil[kept_pairs]

            mu, pi = mu[keep], pi[keep]
            total_weight = total_weight[keep]
            voxel_ids = voxel_ids[keep]

        if mu.size:
            _, curvature = _mu_derivatives(
                width=mu.size,
                mu_rep=mu[rep_voxel],
                g_rep=g_rep,
                precision_rep=precision_rep,
                rep_voxel=rep_voxel,
                weight_rep=w_rep * resp_rep,
                sil_voxel=sil_voxel,
                weight_sil=w_sil * resp_sil,
                censoring=censoring_at(mu[sil_voxel]),
            )
            _retire(np.arange(mu.size), curvature)

        return mu_out, pi_out, se_out

    def _statistic(self, table, sample_sizes, thresholds, image_studies=None):
        """Fit one configuration of foci and return ``(fit, z)``.

        The observed map and every Monte Carlo relocation go through this, so the null is
        built from exactly the statistic being tested. Running the null off the naive
        weighted mean while the observed map came from the selection model would compare two
        different quantities.
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

    def _mask_bool(self):
        """Boolean analysis mask, cached: the null loop unmasks a volume every iteration."""
        cached = getattr(self, "_mask_bool_", None)
        if cached is None:
            cached = _mask_img_to_bool(self.masker.mask_img)
            self._mask_bool_ = cached
        return cached

    def _in_mask_ijk(self):
        """Voxel indices a relocated focus may land on."""
        # ``[:, :3]``: a mask image may carry a trailing singleton volume axis.
        return np.argwhere(np.asarray(self.masker.mask_img.dataobj) > 0)[:, :3]

    def _null_iteration(self, seed, in_mask_ijk, sample_sizes, thresholds, cluster_stat=None):
        """One relocation of every focus.

        Returns ``(histogram of |z|, max |z|, max cluster size, max cluster mass)``; the two
        cluster measures are zero unless ``cluster_stat`` gives a cluster-forming threshold.

        Relocating the foci while keeping their effect sizes and study membership is the same
        null the convergence-based estimators use -- that reported coordinates fall at random
        within the mask -- but evaluated with a statistic that is sensitive to magnitude.
        """
        rng = np.random.default_rng(seed)
        permuted = self._focus_table_.copy()
        permuted[["i", "j", "k"]] = in_mask_ijk[
            rng.integers(0, len(in_mask_ijk), size=len(permuted))
        ]
        _, z_null = self._statistic(
            permuted, sample_sizes, thresholds, getattr(self, "_image_studies_", None)
        )

        absolute = np.abs(z_null)
        counts, _ = np.histogram(np.clip(absolute, 0, _NULL_MAX_Z), bins=_null_bin_edges())
        peak = float(absolute.max()) if absolute.size else 0.0

        max_size = max_mass = 0.0
        if cluster_stat is not None and np.isfinite(cluster_stat):
            mask_bool = self._mask_bool()
            volume = np.zeros(mask_bool.shape, dtype=float)
            volume[mask_bool] = z_null
            max_size, max_mass = _calculate_cluster_measures(
                volume, cluster_stat, _CLUSTER_CONNECTIVITY, tail="two"
            )
        return counts, peak, float(max_size), float(max_mass)

    def _compute_montecarlo_null(
        self, n_iters, n_cores, seed, cluster_stat=None, cluster_threshold=None
    ):
        """Accumulate the null distributions in one pass over the relocations.

        The voxelwise histogram (for uncorrected p), the maximum ``|z|`` (for voxel-level FWE)
        and, when clusters are wanted, the maximum cluster size and mass all come from the same
        refits. Computing them separately would mean permuting two or three times, and a
        permutation here is a full refit.

        Clusters need a forming threshold, and an honest one can only be read off the null that
        this pass is producing. Rather than permute twice, ``cluster_threshold`` (a p-value) is
        resolved against a short pilot run first; the pilot costs a few percent of the main
        loop where a second full pass would cost 100%.
        """
        in_mask_ijk = self._in_mask_ijk()
        sample_sizes = getattr(self, "_sample_sizes_", None)
        thresholds = getattr(self, "_thresholds_", None)
        n_cores = _check_ncores(n_cores)

        if cluster_stat is None and cluster_threshold is not None:
            n_pilot = int(min(max(_NULL_PILOT_ITERS, n_iters // 20), n_iters))
            pilot = Parallel(n_jobs=n_cores)(
                # Seeded past the main loop's range so the pilot draws are disjoint from
                # it, and never negative, which ``default_rng`` rejects.
                delayed(self._null_iteration)(
                    seed + n_iters + i, in_mask_ijk, sample_sizes, thresholds
                )
                for i in range(n_pilot)
            )
            pilot_histogram = np.sum([counts for counts, _, _, _ in pilot], axis=0).astype(
                np.float64
            )
            cluster_stat = _stat_from_histogram(cluster_threshold, pilot_histogram)
            self.null_distributions_["cluster_forming_stat"] = cluster_stat

        results = Parallel(n_jobs=n_cores)(
            delayed(self._null_iteration)(
                seed + i, in_mask_ijk, sample_sizes, thresholds, cluster_stat
            )
            for i in tqdm(range(n_iters), disable=n_iters < 50, desc="CBES null")
        )

        histogram = np.sum([counts for counts, _, _, _ in results], axis=0).astype(np.float64)
        max_values = np.array([peak for _, peak, _, _ in results], dtype=float)
        self.null_distributions_["histogram_bins"] = _null_bin_edges()
        self.null_distributions_["histweights_corr-none_method-montecarlo"] = histogram
        self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"] = max_values
        if cluster_stat is not None:
            self.null_distributions_[
                "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
            ] = np.array([size for _, _, size, _ in results], dtype=float)
            self.null_distributions_[
                "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
            ] = np.array([mass for _, _, _, mass in results], dtype=float)
        return histogram, max_values

    def _approximate_null(self, table, sample_sizes, thresholds, n_draws, seed):
        """Null distribution of ``|z|`` from factorised per-voxel draws, not brain refits.

        The relocation null moves every focus to a uniform in-mask voxel and refits the whole
        brain. But at any *single* voxel the studies are independent under that null, and each
        study's local configuration has a distribution that follows from its foci count and the
        kernel geometry alone -- nothing about the brain is needed. For study ``k`` with
        ``m_k`` foci, each focus independently lands inside this voxel's kernel support with
        probability ``|support| / |mask|``, or inside the coverage sphere without reaching the
        kernel, and a focus that lands is a uniformly chosen one of that study's foci carrying
        its own ``g``. So the configuration can be sampled directly and the voxel fitted on its
        own.

        This is the same idea behind ALE's analytic null (Eickhoff et al., 2012), which
        convolves per-study histograms because its statistic is a product over independent
        studies. CBES's statistic is an EM fit rather than a product, so there is no
        convolution to do -- but the independence that makes convolution valid also makes
        direct sampling valid.

        What it buys is a change of scaling, not a constant factor. Relocation costs
        ``n_voxels x n_studies x n_iters``; this costs ``n_draws x n_studies``, with no
        dependence on the size of the mask. On a whole brain with 40 studies that is about
        9.1e9 study-voxel pairs against 4e7, and the draws are genuinely independent rather
        than spatially correlated as neighbouring voxels are. Validated against the relocation
        null to within 2% of the ``|z|`` threshold at every p from .05 down to 1e-4.

        Only the *uncorrected* null comes from here. Familywise error needs the maximum over
        the brain, which is a property of the spatial correlation this deliberately discards,
        so :meth:`correct_fwe_montecarlo` still relocates.
        """
        rng = np.random.default_rng(seed)
        study_ids = list(sample_sizes.index)
        n_studies = len(study_ids)

        offsets, kernel_weights = self._kernel_support()
        radius = self.coverage_radius
        if radius is None:
            radius = 2.0 * (self.fwhm if self.fwhm is not None else 10.0)
        zooms = self.masker.mask_img.header.get_zooms()[:3]
        n_sphere = len(sphere_kernel_offsets(radius, zooms))
        n_mask = int(self._mask_bool().sum())
        p_kernel = min(len(kernel_weights) / n_mask, 1.0)
        p_sphere = min(n_sphere / n_mask, 1.0)

        weights = np.zeros((n_studies, n_draws))
        g_obs = np.zeros((n_studies, n_draws))
        var_obs = np.ones((n_studies, n_draws))
        covered = np.zeros((n_studies, n_draws), dtype=bool)

        by_study = {str(key): value for key, value in table.groupby("id")} if len(table) else {}
        for position, study_id in enumerate(study_ids):
            sub = by_study.get(str(study_id))
            if sub is None or not len(sub):
                continue  # silent everywhere, so it is covered nowhere
            n_foci = len(sub)
            g_k = sub["g"].to_numpy()
            var_k = sub["var_g"].to_numpy()

            hit = rng.binomial(n_foci, p_kernel, size=n_draws) > 0
            covered[position] = rng.binomial(n_foci, p_sphere, size=n_draws) > 0
            covered[position] |= hit  # reaching the kernel implies reaching the sphere
            if not hit.any():
                continue
            chosen = rng.integers(0, n_foci, size=int(hit.sum()))
            weights[position, hit] = rng.choice(kernel_weights, size=int(hit.sum()))
            g_obs[position, hit] = g_k[chosen]
            var_obs[position, hit] = var_k[chosen]

        safe_var = np.where(var_obs > 0, var_obs, 1.0)
        a = weights / safe_var
        tau2 = (
            _local_dersimonian_laird(
                weights.sum(0),
                a.sum(0),
                (a**2).sum(0),
                (a * g_obs).sum(0),
                (a * g_obs**2).sum(0),
                (weights**2 / safe_var).sum(0),
                (weights > 0).sum(0).astype(float),
            )
            if self.tau2_method == "dl"
            else np.zeros(n_draws)
        )

        pooling = np.where(weights > 0, weights / (var_obs + tau2), 0.0)
        denominator = pooling.sum(0)
        start = np.divide(
            (pooling * g_obs).sum(0), denominator, out=np.zeros(n_draws), where=denominator > 0
        )

        if self.selection_model == "none":
            numerator = np.sqrt((weights**2 / (var_obs + tau2)).sum(0))
            se = np.divide(
                numerator, denominator, out=np.full(n_draws, np.inf), where=denominator > 0
            )
            mu = start
        else:
            null_var = null_effect_variance(sample_sizes.values, design=self.design)[:, None]
            peak_bias = getattr(self, "_peak_bias_", None)
            if peak_bias is not None:
                null_var = null_var * peak_bias.loc[study_ids].values[:, None] ** 2
            mu, _, se = self._fit_chunk(
                weights=weights,
                g_obs=g_obs,
                var_obs=var_obs,
                covered=covered,
                tau2=tau2,
                null_var=null_var,
                cutoffs=np.abs(thresholds.loc[study_ids].to_numpy())[:, None],
                start=start,
            )

        z_values = np.abs(
            np.divide(mu, se, out=np.zeros(n_draws), where=np.isfinite(se) & (se > 0))
        )
        # A voxel no relocated focus reached contributes |z| = 0 to the relocation null, so
        # uncovered draws have to enter the histogram the same way rather than be dropped.
        z_values[denominator <= 0] = 0.0
        histogram, _ = np.histogram(z_values, bins=_null_bin_edges())
        return histogram.astype(float)

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator."
            )

        self.null_distributions_ = {}
        self._mask_bool_ = None
        self._image_studies_ = self._load_image_studies(dataset)
        table = self._build_focus_table()
        # A study that supplies an image has nothing to gain from its own coordinates: the
        # image gives the effect everywhere its peaks do and more, without the localization
        # uncertainty or the selection.
        if self._image_studies_:
            table = table[~table["id"].isin(self._image_studies_)].copy()
        # The roster, the thresholds and the peak-height correction have to be resolved in
        # that order: rho_k is a function of a study's threshold, and the threshold itself is
        # rescaled by rho_k so the censored likelihood stays on one scale.
        if self.selection_model != "none" or self.peak_bias is not None:
            roster = self._all_sample_sizes(dataset)
            self._cutoffs_z_ = self._study_cutoffs_z(table, roster)
            reporting_ids = table["id"].unique() if len(table) else []

            self._peak_bias_scale_ = self._resolve_peak_bias_scale(table, roster, reporting_ids)
            self._peak_bias_ = self._peak_bias_factors(self._cutoffs_z_, roster, reporting_ids)
            table, thresholds = self._apply_peak_bias(
                table, self._cutoffs_z_, roster, self._peak_bias_
            )
        else:
            roster, thresholds = None, None
            self._cutoffs_z_, self._peak_bias_ = None, None

        self._focus_table_ = table
        self._sample_sizes_ = roster if self.selection_model != "none" else None
        self._thresholds_ = thresholds if self.selection_model != "none" else None

        fit, z_values = self._statistic(
            table, self._sample_sizes_, self._thresholds_, self._image_studies_
        )

        if self.null_method == "montecarlo":
            histogram, _ = self._compute_montecarlo_null(
                self.n_iters,
                self.n_cores,
                self.seed,
                cluster_threshold=self.cluster_threshold,
            )
            p_values = _p_from_histogram(np.abs(z_values), histogram)
        elif self.null_method == "approximate":
            n_draws = max(int(self.n_iters) * 1000, _MIN_APPROXIMATE_DRAWS)
            histogram = self._approximate_null(
                table, self._sample_sizes_, self._thresholds_, n_draws, self.seed
            )
            self.null_distributions_["histweights_corr-none_method-approximate"] = histogram
            p_values = _p_from_histogram(np.abs(z_values), histogram)
        else:
            # No null was built, so there is nothing to refer z to. Returning 1 rather than a
            # normal-theory p-value keeps a caller from mistaking the absence of inference for
            # inference: the former parametric option was anticonservative by a factor of eight.
            p_values = np.ones_like(z_values)
        p_values[~fit["covered"]] = 1.0

        maps = {
            "g": fit["g"].astype(DEFAULT_FLOAT_DTYPE),
            "se": np.where(np.isfinite(fit["se"]), fit["se"], 0).astype(DEFAULT_FLOAT_DTYPE),
            "z": z_values.astype(DEFAULT_FLOAT_DTYPE),
            "p": p_values.astype(DEFAULT_FLOAT_DTYPE),
            "logp": _nlogp_to_logp_values(np.log(np.clip(p_values, 1e-300, None))),
            "tau2": fit["tau2"].astype(DEFAULT_FLOAT_DTYPE),
            "n_studies": fit["n_studies"].astype(DEFAULT_FLOAT_DTYPE),
            "n_eff": fit["n_eff"].astype(DEFAULT_FLOAT_DTYPE),
        }
        if "prevalence" in fit:
            maps["prevalence"] = fit["prevalence"].astype(DEFAULT_FLOAT_DTYPE)
            maps["g_marginal"] = (fit["g"] * fit["prevalence"]).astype(DEFAULT_FLOAT_DTYPE)
        return maps, {}, self._description_text()

    # ------------------------------------------------------------- correction

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
        """FWE correction from maximum-statistic nulls, at voxel and cluster level.

        Each iteration moves every focus to a uniformly drawn in-mask voxel, carrying its
        effect size and study membership with it, and refits. Three null distributions come out
        of the same refits: the maximum ``|z|``, the maximum cluster size, and the maximum
        cluster mass. Clusters are formed on ``|z|`` at the statistic corresponding to
        ``voxel_thresh``, read off the uncorrected null rather than assumed -- CBES's ``z`` is
        not standard normal, so a nominal 3.29 would not be a p of .001.

        When :meth:`fit` ran the same relocations for ``null_method="montecarlo"`` at the same
        cluster-forming threshold, all three nulls are reused and this is nearly free. Asking
        for a different ``voxel_thresh`` than the estimator's ``cluster_threshold``, or fitting
        without a Monte Carlo null, means permuting again here.

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
            p of 1e-4 needs ten thousand relocations; extreme value theory says the exceedances
            of a high threshold are generalized Pareto whatever the parent distribution, so the
            tail can be modelled from far fewer \\citep{Winkler2016}. The fit is tested and
            the tail shortened until it is acceptable; if no fit passes, the empirical tail is
            used unchanged, so this can refine a quantized p-value but never manufacture one.
            Validation here showed the fit to be about twice anticonservative at and below the
            empirical floor, so it is applied only above five times that floor and the
            empirical tail is kept below -- more cautious than the published method, and
            accordingly this sharpens corrected p-values without reaching past ``n_iters``.
        vfwe_only : :obj:`bool`, default=False
            Only compute voxel-level correction.

        Returns
        -------
        maps, tables, description
        """
        if getattr(self, "_focus_table_", None) is None:
            raise ValueError("correct_fwe_montecarlo requires a fitted estimator.")

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

        if vfwe_only:
            if not reusable:
                _, cached = self._compute_montecarlo_null(n_iters, n_cores, seed)
        elif not already_clustered:
            # fit() either did not permute, or did so at a different cluster-forming
            # threshold, so the cluster nulls have to be built here.
            _, cached = self._compute_montecarlo_null(
                n_iters, n_cores, seed, cluster_threshold=voxel_thresh
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
        maps["logp_level-voxel"], maps["z_level-voxel"] = _max_statistic_maps(
            observed, cached, sign, tail_approximation=tail_approximation
        )

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
            f"Family-wise error rate correction was performed with a {scope} Monte Carlo "
            f"procedure using {n_iters} iterations, in which every focus was relocated to a "
            "uniformly drawn voxel within the analysis mask while retaining its effect size "
            "and study membership."
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
                f"{1 / (1 + n_iters):.2g} floor that {n_iters} relocations would otherwise "
                "impose; where no acceptable fit was found the empirical tail was retained."
            )
        return maps, {}, description

    # ------------------------------------------------------------ description

    def _generate_description(self):
        kernel_description = (
            f"a Gaussian kernel with {self.fwhm} mm FWHM"
            if self.fwhm is not None
            else "a study-specific Gaussian kernel whose width decreased with sample size"
        )
        heterogeneity = (
            "a locally estimated between-study variance (a kernel-weighted DerSimonian-Laird "
            "moment estimator)"
            if self.tau2_method == "dl"
            else "a fixed-effects model, with no between-study variance"
        )
        if self.selection_model == "zero-inflated":
            selection = (
                " Studies that reported no peak in a region contributed the probability of that "
                "non-report to a zero-inflated censored (Tobit) likelihood there, which "
                "separates the proportion of studies with a non-null effect from the size of "
                "that effect and corrects the estimate for the within-study thresholding that "
                "generated the reported peaks. No effect-size images were imputed."
            )
        else:
            selection = (
                " Only reported peaks were pooled, so the estimate is biased away from zero by "
                "the within-study thresholding that generated them."
            )
        if self.peak_bias == "per-study":
            bias = (
                " Because a reported peak is a local maximum that cleared the reporting study's "
                "own threshold, each study's effect sizes were rescaled by a factor inversely "
                "proportional to the effect size a peak of pure noise would have produced at "
                "that study's threshold and sample size, normalized to "
                f"{self.peak_bias_scale} at the median reporting study."
            )
        elif self.peak_bias is not None:
            bias = (
                " Reported effect sizes were rescaled by a factor of "
                f"{self.peak_bias} to correct for the inflation of a reported local maximum "
                "relative to the effect in the surrounding region."
            )
        else:
            bias = (
                " Reported peaks were not corrected for the inflation of a local maximum "
                "relative to the effect in the surrounding region, so the magnitudes are "
                "overestimates."
            )
        n_foci = len(getattr(self, "_focus_table_", []))
        n_studies = (
            self._focus_table_["id"].nunique() if hasattr(self, "_focus_table_") else "an unknown"
        )
        if self.null_method == "montecarlo":
            inference = (
                " Uncorrected p-values were obtained from a Monte Carlo null distribution, in "
                f"which every focus was relocated to a random voxel within the analysis mask "
                f"{self.n_iters} times while retaining its effect size and study membership."
            )
        elif self.null_method == "approximate":
            inference = (
                " Uncorrected p-values were obtained from a null distribution in which each "
                "study's local configuration of foci was sampled directly from its focus count "
                "and the kernel geometry, rather than by refitting the brain for every "
                "relocation."
            )
        else:
            inference = " No null distribution was computed, so no p-values are reported."
        return (
            "A coordinate-based effect-size meta-analysis was performed with NiMARE "
            f"{__version__} (RRID:SCR_017398; \\citealt{{Salo2023}}). Each reported peak "
            f"statistic was converted to Hedges' g using the study's sample size and a "
            f"{self.design} design, and peaks were assigned spatial uncertainty with "
            f"{kernel_description}. Voxel-wise pooling used {heterogeneity}.{selection}"
            f"{bias}{inference} "
            f"The input dataset included {n_foci} foci with reported statistics from "
            f"{n_studies} experiments."
        )
