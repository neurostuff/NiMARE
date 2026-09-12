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
:footcite:t:`tench2017coordinate`. ``selection_model="tobit"`` drops the zero component, and
``"none"`` pools only the reported peaks.

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
p-values therefore come from a spatial null by default (``null_method="montecarlo"``), which
relocates every focus within the mask and refits, exactly as the convergence-based estimators
do. See the :class:`CBES` warnings for what the parametric alternative does to the false
positive rate.

References
----------
.. footbibliography::
"""

import logging

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from nilearn.maskers import NiftiMasker
from scipy import ndimage, stats
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

#: Default two-tailed reporting threshold, on the z scale, when a study gives no better
#: information. p < .001 uncorrected, the most common screening threshold in the literature.
DEFAULT_REPORTING_THRESHOLD_Z = 3.2905267314919255

DESIGNS = ("one-sample", "two-sample")

SELECTION_MODELS = ("zero-inflated", "tobit", "none")

NULL_METHODS = ("montecarlo", "parametric")

#: Resolution of the Monte Carlo null histogram for |z|, and where its upper tail is clipped.
_NULL_Z_STEP = 0.01
_NULL_MAX_Z = 50.0

#: EM stops on a voxel once mu and the prevalence both move less than this in one step.
_EM_TOLERANCE = 1e-5
#: Rebuild the working set only once this fraction of it has settled, so that compaction
#: (which touches every pair) is amortized rather than run every iteration.
_EM_COMPACTION_FRACTION = 0.05

#: Minimum relocations used to fix the cluster-forming threshold before the main null loop.
_NULL_PILOT_ITERS = 20

#: Faces-only connectivity for cluster labelling, matching Nilearn and the other CBMA
#: estimators.
_CLUSTER_CONNECTIVITY = ndimage.generate_binary_structure(rank=3, connectivity=1)

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _normal_pdf(x):
    """Standard normal density. ``scipy.stats.norm.pdf`` is ~3x slower on large arrays, and
    the EM below evaluates it on an (n_studies, n_voxels) block on every iteration."""
    return np.exp(-0.5 * x * x) * _INV_SQRT_2PI


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


def _max_statistic_maps(observed, null_maxima, sign):
    """Corrected ``-log10(p)`` and signed z for a statistic against its maximum-statistic null."""
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


def _censoring_terms(mu, cutoffs, sigma):
    """P(|g| < c | mu) and the pieces of its derivatives, for a set of silent observations.

    Returned together because the E step and the M step both need them at the same ``mu``, and
    the normal CDFs here are the single most expensive thing the estimator does.
    """
    upper = (cutoffs - mu) / sigma
    lower = (-cutoffs - mu) / sigma
    prob = np.clip(ndtr(upper) - ndtr(lower), 1e-12, None)
    pdf_upper = _normal_pdf(upper)
    pdf_lower = _normal_pdf(lower)
    return {
        "prob": prob,
        "score": -(pdf_upper - pdf_lower) / sigma / prob,
        "d2_over_prob": -(upper * pdf_upper - lower * pdf_lower) / sigma**2 / prob,
    }


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
    stat_column : :obj:`str` or None, optional
        Column of the coordinates table holding the reported statistic. When None, ``z_stat``
        is used if present, otherwise ``t_stat``.
    design : {"one-sample", "two-sample"}, default="one-sample"
        Design behind the reported statistics, used for the effect-size conversion.
    tau2_method : {"dl", "none"}, default="dl"
        ``"dl"`` estimates a local between-study variance with a kernel-weighted
        DerSimonian-Laird moment estimator; ``"none"`` fits a fixed-effects model
        (:math:`\\tau^2 \\equiv 0`).
    selection_model : {"censored", "none"}, default="censored"
        ``"censored"`` adds, for every study that reported nothing in this region, a term for
        the probability that it would have reported nothing, and returns the maximizer of that
        Tobit log-likelihood. ``"none"`` pools only the reported peaks, and is biased away from
        zero by the within-study thresholding that produced them. Nothing is imputed under
        either option.
    threshold : :obj:`float`, "pooled-min", "study-min", or None, default="pooled-min"
        Reporting threshold, on the z scale, assumed for each study. Coordinates alone never
        state it, so it has to be inferred, and the estimate matters: too high a threshold makes
        silence unsurprising and the selection correction does nothing.

        ``"pooled-min"``
            The smallest absolute statistic reported anywhere in the collection. The tightest
            bound available, and the recommended default.
        ``"study-min"``
            The smallest absolute statistic each study reported. Biased upward for studies that
            reported few peaks -- a study with one focus has no information about its own
            threshold at all -- but appropriate when studies plainly used different thresholds.

        A float applies one z threshold to every study; None falls back to
        ``DEFAULT_REPORTING_THRESHOLD_Z``. Unused when ``selection_model="none"``.
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
    null_method : {"montecarlo", "parametric"}, default="montecarlo"
        How uncorrected p-values are obtained.

        ``"montecarlo"``
            Relocate every focus to a random in-mask voxel ``n_iters`` times, keeping its
            effect size and study membership, and read p off the resulting null distribution
            of ``|z|``. This is the null the convergence-based estimators use -- that reported
            coordinates fall at random within the mask -- and it is the only one validated
            here. It is also what makes :class:`~nimare.correct.FDRCorrector` and
            ``FWECorrector(method="bonferroni")`` meaningful, since both simply operate on
            these p-values.
        ``"parametric"``
            ``g / se`` referred to a normal distribution. Fast, and **anticonservative**: see
            the warning below. Useful for inspecting the effect-size maps, not for inference.

    cluster_threshold : :obj:`float` or None, default=0.001
        Cluster-forming threshold, as an uncorrected p-value, for the cluster-level FWE null
        that :meth:`fit` builds alongside the voxel-level one. Set to None to skip it, which
        makes :meth:`correct_fwe_montecarlo` pay for a second pass over the permutations if
        cluster correction is then requested.
    n_iters : :obj:`int`, default=1000
        Monte Carlo iterations for the null. Each is a full refit, so this is the dominant
        cost of the estimator -- far more so than for ALE, whose per-iteration statistic is
        much cheaper. Reduce it, or use ``null_method="parametric"``, when exploring.
    n_cores : :obj:`int`, default=1
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

    Do not use ``null_method="parametric"`` for inference. Measured on a global null (30
    studies of pure noise foci, 20 simulations), it returned uncorrected ``p < .05`` for 41% of
    voxels with ``selection_model="none"`` and 11% with ``"zero-inflated"``, against a nominal
    5%, and FDR and Bonferroni built on those p-values rejected somewhere in 100% of null
    simulations. The Monte Carlo null returned 0.052 and 0.042 respectively, and every
    correction built on it rejected in 1 or 2 of the 20 simulations -- consistent with a
    nominal 0.05, though 20 simulations cannot resolve a rate more finely than that.

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
        stat_column=None,
        design="one-sample",
        tau2_method="dl",
        selection_model="zero-inflated",
        threshold="pooled-min",
        coverage_radius=None,
        kernel_min_weight=0.01,
        max_iter=25,
        null_method="montecarlo",
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
        if null_method == "parametric":
            LGR.warning(
                "null_method='parametric' produces anticonservative p-values: the standard "
                "error treats tau-squared and the mixture weights as known, and the reported "
                "peaks it pools were selected for being large. Under a global null it flags "
                "roughly 40% of voxels at p < .05 (10% with selection_model='zero-inflated') "
                "instead of 5%, and FDR and Bonferroni built on top of it reject in every "
                "null simulation. Use null_method='montecarlo' for inference."
            )
        if isinstance(threshold, str):
            if threshold not in ("pooled-min", "study-min"):
                raise ValueError(
                    "threshold must be 'pooled-min', 'study-min', a number, or None; got "
                    f"{threshold!r}."
                )
        elif threshold is not None and not np.isscalar(threshold):
            raise ValueError(
                f"threshold must be 'pooled-min', 'study-min', a number, or None; got "
                f"{threshold!r}."
            )

        self.fwhm = fwhm
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
        else:
            raise ValueError(
                "CBES needs a reported test statistic for each peak, but the input "
                "coordinates have no usable 'z_stat' or 't_stat' column. Convergence-based "
                "estimators (ALE, MKDADensity, KDA) do not require one; effect-size "
                "estimation does."
            )

        return column, "t" if column.startswith("t") else "z"

    def _build_focus_table(self):
        """Reduce the coordinates table to the per-focus quantities the model consumes."""
        coords = self.inputs_["coordinates"]
        column, stat_type = self._resolve_stat_column(coords)

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
        if not usable.any():
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

    def _study_thresholds(self, table, sample_sizes):
        """Per-study reporting threshold, expressed on the Hedges' g scale.

        Studies that reported nothing anywhere still need a threshold -- it is what makes their
        silence quantitative. Since they reported no statistic to infer one from, they are
        given the median threshold of the studies that did report.
        """
        stat_type = table["stat_type"].iloc[0]

        # Thresholds are inferred on the z scale, where they are comparable across studies;
        # a t of 3.5 means something different in a study of 15 than in a study of 80.
        reported_z = np.abs(
            table["stat"].values
            if stat_type == "z"
            else t_to_z(table["stat"].values, table["sample_size"].values - 1)
        )

        if self.threshold == "pooled-min":
            # The smallest statistic reported anywhere is the tightest available upper bound on
            # the common threshold. Per-study minima look tempting but are badly biased upward
            # whenever a study reports only a handful of peaks -- with one peak, a study's
            # minimum *is* its single reported value, which says nothing about its threshold.
            cutoff_z = np.full(
                len(sample_sizes), float(np.nanmin(reported_z)) if reported_z.size else np.nan
            )
        elif self.threshold == "study-min":
            per_study = pd.Series(reported_z, index=table["id"].values).groupby(level=0).min()
            fallback = float(np.nanmedian(per_study.values)) if len(per_study) else np.nan
            cutoff_z = per_study.reindex(sample_sizes.index).fillna(fallback).values
        else:
            value = DEFAULT_REPORTING_THRESHOLD_Z if self.threshold is None else self.threshold
            cutoff_z = np.full(len(sample_sizes), float(value))

        cutoff_z = np.where(np.isfinite(cutoff_z), cutoff_z, DEFAULT_REPORTING_THRESHOLD_Z)
        threshold_g, _ = peak_stat_to_hedges_g(
            cutoff_z, sample_sizes.values, stat_type="z", design=self.design
        )
        return pd.Series(threshold_g, index=sample_sizes.index)

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

    def _accumulate(self, table):
        """Walk the studies once, returning per-study voxel contributions and voxel sums."""
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

        if not contributions:
            raise ValueError("No study contributed any in-mask voxels.")

        return contributions, sums, n_voxels

    # ------------------------------------------------------------------ fitting

    def _pool(self, table):
        """Run the two-pass local random-effects fit. Returns a dict of masked-voxel arrays."""
        contributions, sums, n_voxels = self._accumulate(table)

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

    def _coverage_entries(self, table, study_ids, active, n_voxels):
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

        cols, positions = [], []
        for position, study_id in enumerate(study_ids):
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

    def _apply_selection_model(self, fit, table, thresholds, sample_sizes):
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
        below the truth. With ``selection_model="tobit"`` the zero component is switched off
        and every study is assumed to share one effect, censored by the threshold.

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
        cov_col, cov_pos = self._coverage_entries(table, study_ids, active, n_voxels)
        values = self._value_entries(fit, study_ids, active, n_voxels)
        n_studies = len(study_ids)

        null_var = null_effect_variance(sample_sizes.values, design=self.design)[:, None]
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
            censoring = _censoring_terms(mu[sil_voxel], cutoff_sil, sigma_sil)
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
            w_sil, cutoff_sil = w_sil[kept_pairs], cutoff_sil[kept_pairs]
            sigma_sil, prob_silent_null = sigma_sil[kept_pairs], prob_silent_null[kept_pairs]
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
                censoring=_censoring_terms(mu[sil_voxel], cutoff_sil, sigma_sil),
            )
            _retire(np.arange(mu.size), curvature)

        return mu_out, pi_out, se_out

    def _statistic(self, table, sample_sizes, thresholds):
        """Fit one configuration of foci and return ``(fit, z)``.

        The observed map and every Monte Carlo relocation go through this, so the null is
        built from exactly the statistic being tested. Running the null off the naive
        weighted mean while the observed map came from the selection model would compare two
        different quantities.
        """
        fit = self._pool(table)
        if self.selection_model != "none":
            self._apply_selection_model(fit, table, thresholds, sample_sizes)

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
        _, z_null = self._statistic(permuted, sample_sizes, thresholds)

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
        table = self._build_focus_table()
        self._focus_table_ = table

        self._sample_sizes_ = (
            self._all_sample_sizes(dataset) if self.selection_model != "none" else None
        )
        self._thresholds_ = (
            self._study_thresholds(table, self._sample_sizes_)
            if self.selection_model != "none"
            else None
        )

        fit, z_values = self._statistic(table, self._sample_sizes_, self._thresholds_)

        if self.null_method == "montecarlo":
            histogram, _ = self._compute_montecarlo_null(
                self.n_iters,
                self.n_cores,
                self.seed,
                cluster_threshold=self.cluster_threshold,
            )
            p_values = _p_from_histogram(np.abs(z_values), histogram)
        else:
            p_values = stats.norm.sf(np.abs(z_values)) * 2.0
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
        with ``null_method="parametric"``, means permuting again here.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result of a previous :meth:`fit`.
        voxel_thresh : :obj:`float`, default=0.001
            Cluster-forming threshold, as an uncorrected p-value.
        n_iters, n_cores, seed : optional
            Override the estimator's own settings for this correction.
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
            observed, cached, sign
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
                logp, z_corrected = _max_statistic_maps(observed_measure[mask_bool], null, sign)
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
        elif self.selection_model == "tobit":
            selection = (
                " Studies that reported no peak in a region contributed the probability of that "
                "non-report to a censored (Tobit) likelihood there, correcting the estimate for "
                "the within-study thresholding that generated the reported peaks. No "
                "effect-size images were imputed."
            )
        else:
            selection = (
                " Only reported peaks were pooled, so the estimate is biased away from zero by "
                "the within-study thresholding that generated them."
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
        else:
            inference = (
                " Uncorrected p-values were obtained by referring the pooled estimate to a "
                "normal distribution; these are anticonservative and should not be used for "
                "inference."
            )
        return (
            "A coordinate-based effect-size meta-analysis was performed with NiMARE "
            f"{__version__} (RRID:SCR_017398; \\citealt{{Salo2023}}). Each reported peak "
            f"statistic was converted to Hedges' g using the study's sample size and a "
            f"{self.design} design, and peaks were assigned spatial uncertainty with "
            f"{kernel_description}. Voxel-wise pooling used {heterogeneity}.{selection}"
            f"{inference} "
            f"The input dataset included {n_foci} foci with reported statistics from "
            f"{n_studies} experiments."
        )
