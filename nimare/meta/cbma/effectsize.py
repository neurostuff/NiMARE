"""Coordinate-based effect-size meta-analysis."""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from nilearn.maskers import NiftiMasker
from scipy import ndimage
from scipy.optimize import brentq
from scipy.special import gammaln, ndtr
from tqdm.auto import tqdm

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.utils import (
    _calculate_cluster_measures,
    _get_mask_flat_to_masked,
    _kernel_to_sparse_support,
    _max_statistic_maps,
    _padded_flat_to_masked,
    get_ale_kernel,
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

#: Smallest sampling variance we will attribute to a reported peak. Guards the pooling weights
#: against division by zero for implausibly large sample sizes.
_MIN_VARIANCE = 1e-8

#: Percentile of ``|g|`` over covered voxels that the relative map is divided by, so that 1.0
#: reads as "as strong as the top few percent of this collection". A high percentile rather than
#: the median because a covered brain is mostly voxels holding no effect, and dividing by a
#: near-zero summary gives a map that describes its own denominator.
_RELATIVE_NORMALIZATION_PERCENTILE = 95

#: An inferred reporting threshold above this is reported as suspicious. Conventional height
#: thresholds run from about z = 3.1 (p < .001 uncorrected) to around 5 for whole-brain FWE
#: correction, so a value here does not prove anything is wrong -- but it is also what
#: cluster-extent reporting looks like, which the inference cannot distinguish from strict
#: height thresholding and which wrecks ``prevalence``. Worth saying so once.
_SUSPICIOUS_INFERRED_THRESHOLD_Z = 4.0

#: Image studies needed before an absolute-scale ``g`` map is emitted. Two, not one, because two
#: is the fewest at which the spread of the per-donor scale estimates can be measured at all, so
#: that the caller can see how well determined the constant is rather than taking one study's
#: word for it.
_MIN_SCALE_DONORS = 2

#: Distinct arrangements the within-analysis null needs before its p-values mean anything,
#: as a base-10 log. Ten thousand states is where the coarsest attainable p-value, 1e-4, stops
#: being the thing that limits the test; below it the null has too few states for the
#: exceedance count to separate voxels, and a map of p-values would read as inference that was
#: never done.
_MIN_NULL_STATES_LOG10 = 4.0

#: Keywords ``peak_bias_scale`` understands; anything else must be a positive number.
PEAK_BIAS_SCALE_KEYWORDS = ("auto", "images")

#: Keywords ``threshold`` understands; any other string names a metadata field.
THRESHOLD_KEYWORDS = ("pooled-min", "study-min")

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

NULL_METHODS = ("permute-magnitudes", "none")

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

#: Quadrature for the RFT null peak-height integrals. The density is negligible more than this
#: far above the threshold, and the grids are sized so the integral is stable to 1e-4.
_PEAK_GRID_SPAN_Z = 12.0
_PEAK_OVERSHOOT_GRID = 4000
_PEAK_MEAN_GRID = 2000
#: Below this the high-threshold peak-height form is not positive (it fails under sqrt(3)),
#: so the 1/u approximation is used instead.
_PEAK_OVERSHOOT_MIN_Z = 1.9

#: Bracket and tolerance for inverting the minimum reported peak back to a study's threshold.
#: A threshold below this is not a plausible reporting cut and the minimum is returned as is.
_MIN_INFERRED_THRESHOLD_Z = 1.95
_THRESHOLD_SEARCH_XTOL = 1e-4

#: Prevalence is held inside (0, 1) by this margin: at exactly 0 or 1 the mixture degenerates
#: and the responsibilities stop being informative.
_PREVALENCE_CLAMP = 1e-4

#: Voxels used to calibrate the effect-size scale: the image/coordinate ratio is taken over
#: the strongest image voxels, of which there must be enough for the ratio to mean anything.
_CALIBRATION_PERCENTILE = 75
_MIN_CALIBRATION_VOXELS = 50

#: Voxels x studies held in memory at once by the selection-model fit, which allocates several
#: arrays of this size per iteration.
_SELECTION_CHUNK_ELEMENTS = 2_000_000


#: Excess of the mean reported peak height over the null peak height, in z units, below which
#: the reported magnitudes are treated as carrying no usable effect-size information.
_MIN_PEAK_EXCESS_Z = 0.25

#: Faces-only connectivity for cluster labelling, matching Nilearn and the other CBMA
#: estimators.
_CLUSTER_CONNECTIVITY = ndimage.generate_binary_structure(rank=3, connectivity=1)

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------- numerical helpers


def _trapezoid(y, x):
    """Integrate ``y`` over ``x`` by the trapezoidal rule.

    NumPy renamed ``trapz`` to ``trapezoid`` in 2.0 and dropped the old name, while the oldest
    NumPy this package supports (1.22) has only ``trapz``. Neither name works everywhere, and
    the rule is one line, so it is written out rather than branched on a version.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


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


def _censoring_terms(mu, cutoff_scaled, twice_cutoff_scaled, inv_sigma, inv_sigma_sq):
    """P(|g| < c | mu) and the pieces of its derivatives, for a set of silent observations.

    Returned together because the E step and the M step both need them at the same ``mu``. This
    is the most expensive single function in the estimator, over an array with one entry per
    silent ``(study, voxel)`` pair, though only about a fifth of a whole-brain fit -- the cost
    is spread over four kernels and all of them are memory-bound, so what pays is removing a
    pass rather than speeding one up.

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

    prob = ndtr(upper)
    prob -= ndtr(lower)
    np.clip(prob, _PROBABILITY_FLOOR, None, out=prob)

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


def _observed_information(
    *,
    width,
    pi,
    reporting,
    silent,
    mu,
    censoring,
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

    What is returned is a pair. The first is the Schur complement
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

    density_effect = (
        _normal_pdf((reporting.g - mu[reporting.voxel]) / reporting.sigma) / reporting.sigma
    )
    i_mu, cross, i_pi = blocks(
        reporting.voxel,
        reporting.weight,
        responsibility_of(reporting.voxel, density_effect, reporting.density_null),
        (reporting.g - mu[reporting.voxel]) * reporting.precision,
        -reporting.precision,
    )
    censor_score = censoring["score"]
    add_mu, add_cross, add_pi = blocks(
        silent.voxel,
        silent.weight,
        responsibility_of(silent.voxel, censoring["prob"], silent.prob_silent_null),
        censor_score,
        censoring["d2_over_prob"] - censor_score**2,
    )
    i_mu += add_mu
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
    return profiled, marginal_variance


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
    if u <= _PEAK_OVERSHOOT_MIN_Z:
        return u + 1.0 / max(u, 1e-6)
    grid = np.linspace(u, u + _PEAK_GRID_SPAN_Z, _PEAK_OVERSHOOT_GRID)
    density = grid * (grid**2 - 3.0) * np.exp(-0.5 * grid**2)
    density = np.clip(density, 0.0, None)
    mass = _trapezoid(density, grid)
    return float(_trapezoid(grid * density, grid) / mass) if mass > 0 else u


def peak_information(stats_z, threshold_z):
    """How much effect-size information the reported peak heights actually carry.

    Returns ``(observed_mean, null_mean, excess)`` on the z scale. ``excess`` is what is left
    once the height a pure-noise peak would have reached is accounted for, and it is the only
    part of a reported peak height that speaks to the size of the effect.

    On real collections the excess is often indistinguishable from zero. When that happens the
    reported *magnitudes* are uninformative -- a function of the reporting threshold and the
    sample size, not of the effect -- and no correction computed from them can recover the
    effect size, because the information is not there.
    """
    observed = float(np.mean(np.abs(np.asarray(stats_z, dtype=float))))
    expected = null_peak_overshoot(threshold_z)
    return observed, expected, observed - expected


def null_peak_mean_g(threshold_z, sample_size, design="one-sample"):
    """Effect size a study would report from a *pure noise* peak above its threshold.

    The expected ``|g|`` of such a peak, over the RFT null peak-height distribution above
    ``threshold_z`` and converted with the study's own sample size. It is the scale a study
    contributes *by construction*, so dividing by it removes the part of a reported effect size
    that is an artefact of how strictly the paper thresholded and how many subjects it had --
    neither of which is a fact about the brain.
    """
    u = float(threshold_z)
    grid = np.linspace(u, u + _PEAK_GRID_SPAN_Z, _PEAK_MEAN_GRID)
    density = np.clip(grid * (grid**2 - 3.0) * np.exp(-0.5 * grid**2), 0.0, None)
    mass = _trapezoid(density, grid)
    if mass <= 0:  # threshold below sqrt(3): fall back to the exponential overshoot
        grid = np.linspace(u, u + _PEAK_GRID_SPAN_Z, _PEAK_MEAN_GRID)
        density = u * np.exp(-u * (grid - u))
        mass = _trapezoid(density, grid)

    sizes = np.full(grid.shape, float(sample_size))
    g_of_z, _ = peak_stat_to_hedges_g(grid, sizes, stat_type="z", design=design)
    return float(_trapezoid(np.abs(g_of_z) * density, grid) / mass)


def _expected_min_peak(u, n_peaks, span=10.0, n_grid=500):
    """E[smallest of ``n_peaks`` heights drawn above ``u``] under the RFT null."""
    grid = np.linspace(u, u + span, n_grid)
    survival = np.clip(
        (grid**2 - 1.0) * np.exp(-0.5 * grid**2) / ((u**2 - 1.0) * np.exp(-0.5 * u**2)), 0.0, 1.0
    )
    return u + _trapezoid(survival**n_peaks, grid)


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
    cluster is fine. A cluster-extent threshold is not: measured on smooth fields, this recovers
    a true height threshold to +0.06 z, but comes out +0.41 z high when clusters of at least ten
    voxels are kept and +1.10 z at fifty, because extent thresholding keeps the broad clusters
    whose peaks run higher and drops isolated low ones. See ``CBES.threshold`` for what that
    costs downstream.

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
    if n_peaks <= 0 or not np.isfinite(z_min) or z_min <= _MIN_INFERRED_THRESHOLD_Z:
        return z_min
    if _expected_min_peak(z_min, n_peaks) <= z_min:  # already consistent
        return z_min
    try:
        return float(
            brentq(
                lambda u: _expected_min_peak(u, n_peaks) - z_min,
                _MIN_INFERRED_THRESHOLD_Z,
                z_min,
                xtol=_THRESHOLD_SEARCH_XTOL,
            )
        )
    except ValueError:
        return z_min


def null_effect_variance(sample_size, design="one-sample"):
    """Return the sampling variance of Hedges' g under a null effect, for a silent study.

    A study that did not report a peak supplies no effect size, but its *precision* is still
    known from its sample size. That precision is what makes the censoring term in the Tobit
    likelihood informative, so it is computed here at ``d = 0``.
    """
    zeros = np.zeros_like(np.asarray(sample_size, dtype=float))
    _, var_g = peak_stat_to_hedges_g(zeros, sample_size, stat_type="t", design=design)
    return var_g


# --------------------------------------------------------------- heterogeneity


def _relative_g(g, covered):
    """``g`` divided by a high percentile of its own magnitude, so the units cancel.

    The overall scale of a coordinate-only fit is not identified, so dividing by a summary of
    the map's own magnitude removes the unknown constant exactly and leaves the part the
    coordinates do identify: the pattern, and the ratios between voxels.

    Normalized always rather than only where the scale is unknown -- a map that is sometimes in
    Hedges' g and sometimes in units of itself cannot be compared across collections, or read
    without checking which it is.
    """
    out = np.zeros_like(g, dtype=float)
    if not np.any(covered):
        return out
    magnitude = np.abs(g[covered])
    magnitude = magnitude[np.isfinite(magnitude)]
    if not magnitude.size:
        return out
    reference = float(np.percentile(magnitude, _RELATIVE_NORMALIZATION_PERCENTILE))
    if not np.isfinite(reference) or reference <= 0:
        return out
    out[covered] = g[covered] / reference
    return out


def _scale_confidence_interval(per_donor, alpha=0.05):
    """Return a confidence interval for a scale constant estimated from per-donor ratios.

    The scale is a multiplicative quantity, so the interval is built on ``log`` and exponentiated
    back: the donors' log-ratios are treated as a sample, and the interval is the point estimate
    times ``exp(+/- t * s / sqrt(K))`` on ``K - 1`` degrees of freedom.

    This replaces reporting ``(min, max)`` of the per-donor estimates, which is a *sample range*
    and not an interval at all. A range answers "how far apart did these donors land", and its
    relationship to the uncertainty in their central value runs the wrong way with the number of
    donors: on simulated collections it was 0.51 times an honest interval at two donors and 4.32
    times it at twenty, so the error changed sign somewhere in between. A range shrinks toward
    the truth's own spread as donors accumulate, while the uncertainty in their centre shrinks
    like ``1 / sqrt(K)``.

    Two donors give ``t = 12.71``, so the interval is very wide. That is the honest answer rather
    than a defect: a scale resting on two studies is barely pinned, which is what the estimator's
    single-donor warning says in words. The one case this shares with the old range is that
    donors agreeing exactly give a zero-width interval, because the sample spread is the only
    evidence available about the spread -- with a handful of donors that agreement can be
    coincidence, so an interval of zero width should be read as "too few donors to tell", not as
    a pinned scale.
    """
    from scipy.stats import t as student_t

    ratios = np.asarray([r for r in per_donor if np.isfinite(r) and r > 0], dtype=float)
    if ratios.size < 2:
        return None
    logs = np.log(ratios)
    centre = float(np.median(logs))
    spread = float(np.std(logs, ddof=1)) / np.sqrt(ratios.size)
    half = float(student_t.ppf(1.0 - alpha / 2.0, ratios.size - 1)) * spread
    return (float(np.exp(centre - half)), float(np.exp(centre + half)))


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
    peak_bias,
    peak_bias_scale,
    threshold,
    se_method,
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

    scale_is_keyword = isinstance(peak_bias_scale, str)
    if (scale_is_keyword and peak_bias_scale not in PEAK_BIAS_SCALE_KEYWORDS) or (
        not scale_is_keyword and not float(peak_bias_scale) > 0
    ):
        raise ValueError(
            f"peak_bias_scale must be one of {list(PEAK_BIAS_SCALE_KEYWORDS)} or a positive "
            f"number; got {peak_bias_scale!r}."
        )

    bias_is_keyword = isinstance(peak_bias, str)
    if (bias_is_keyword and peak_bias != "per-study") or (
        not bias_is_keyword and peak_bias is not None and not 0.0 < float(peak_bias) <= 1.0
    ):
        raise ValueError(
            f"peak_bias must be None, 'per-study', or a number in (0, 1]; got {peak_bias!r}."
        )

    # A string is a keyword or the name of a metadata field holding per-study thresholds, and
    # which one it is cannot be known until the collection is in hand.
    if not isinstance(threshold, str) and threshold is not None and not np.isscalar(threshold):
        raise ValueError(
            f"threshold must be one of {list(THRESHOLD_KEYWORDS)}, a metadata field name, a "
            f"number, or None; got {threshold!r}."
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
class _SilentPairs:
    """The ``(study, voxel)`` pairs where a study reported in the region but not at the voxel.

    ``inv_sigma`` and the scaled cutoffs are precomputed because only ``mu`` moves between EM
    iterations, so every division by them is paid once rather than once per iteration.
    """

    voxel: np.ndarray
    weight: np.ndarray
    inv_sigma: np.ndarray
    inv_sigma_sq: np.ndarray
    cutoff_scaled: np.ndarray
    twice_cutoff_scaled: np.ndarray
    prob_silent_null: np.ndarray
    responsibility: np.ndarray

    def compact(self, position):
        """Drop pairs whose voxel has retired, and renumber the rest onto ``position``."""
        moved = position[self.voxel]
        keep = moved >= 0
        return _SilentPairs(
            voxel=moved[keep],
            weight=self.weight[keep],
            inv_sigma=self.inv_sigma[keep],
            inv_sigma_sq=self.inv_sigma_sq[keep],
            cutoff_scaled=self.cutoff_scaled[keep],
            twice_cutoff_scaled=self.twice_cutoff_scaled[keep],
            prob_silent_null=self.prob_silent_null[keep],
            responsibility=self.responsibility[keep],
        )

    def censoring(self, mu):
        """P(silent) and its derivatives at the current ``mu``."""
        return _censoring_terms(
            mu[self.voxel],
            self.cutoff_scaled,
            self.twice_cutoff_scaled,
            self.inv_sigma,
            self.inv_sigma_sq,
        )


class CBES(Estimator):
    r"""Coordinate-based effect-size meta-analysis.

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
        Use per-study ``g``/``g_var`` images for any study that has them, in place of that
        study's coordinates. An image is the limiting case of a coordinate -- no localization
        uncertainty and no reporting threshold -- so it enters at kernel weight 1 and
        contributes no censoring term. Supply them with
        ``ImageTransformer(target=["g", "g_var"])``; a collection may mix the two freely.
    peak_bias : :obj:`float`, "per-study", or None, optional
        Divide reported effect sizes by ``rho_k`` before pooling, to undo the inflation of a
        reported peak: a peak is a local maximum that cleared a threshold, so its height
        overstates the local effect. ``"per-study"`` sets ``rho_k`` from each study's own
        threshold and sample size, removing the between-study part of the bias -- the part
        coordinates can identify. A float sets every ``rho_k`` to the same value. The common
        scale is *not* identified and must come from ``peak_bias_scale`` or be accepted, which
        is why ``g_relative`` is the map to read by default.

        **A float leaves the inference alone; ``"per-study"`` does not.** One shared factor
        scales every study's variance identically, so every inverse-variance weight is scaled
        together: ``g`` scales exactly by the factor while ``z`` moves by at most 0.5%. A
        per-study factor scales each study's variance by its own ``rho_k**2``, which reweights
        the studies against each other, so it is a change to the model rather than a rescaling
        of the output. Over the voxels reaching ``|z| > 1`` on a 24-study collection it shifted
        ``z`` by a median of 18% and by 64% at the 95th percentile with sample sizes from 15 to
        400, and by 10% and 62% with them from 20 to 40. Narrowing the sample sizes is
        therefore not a remedy: it shrinks the typical shift and leaves the tail.

        ``rho_k`` is also derived at :math:`\mu = 0`: :func:`null_peak_mean_g` is the effect
        size a *pure-noise* peak would report, so ``rho_k`` grows like :math:`\sqrt{N_k}` and
        a study whose effect is large enough to clear the threshold easily -- where there is
        barely any winner's curse to undo -- is rescaled the hardest. Against a median
        :math:`N` of 30 the factor reaches 2.8 at :math:`N = 200` and 6.4 at
        :math:`N = 1000`. The correction is therefore sound where the reported heights are
        noise-dominated, which is where :func:`peak_information` reports they carry no
        effect-size information anyway, and is an overcorrection where they are not. Neither
        mode is on by default, and a shared float is the safer of the two: it cannot move the
        inference, only the magnitude scale that was never identified to begin with.
    peak_bias_scale : :obj:`float`, "auto", or "images", default=1.0
        The overall scale of the ``"per-study"`` correction. ``"images"`` reads it off any
        studies in the collection that supply images and ``"auto"`` does the same when images
        are present, leaving it at 1.0 otherwise. Ignored unless ``peak_bias="per-study"``.
        Nothing recovers this constant from coordinates alone, so with none supplied read
        ``g_relative`` rather than ``g``.
    stat_column : :obj:`str` or None, optional
        Column of the coordinates table holding the reported statistic. When None, ``z_stat``
        is used if present, otherwise ``t_stat``.
    design : {"one-sample", "two-sample"}, default="one-sample"
        Design behind the reported statistics, used for the effect-size conversion.
    tau2_method : {"dl", "none"}, default="dl"
        ``"dl"`` estimates a local between-study variance with a kernel-weighted
        DerSimonian-Laird moment estimator; ``"none"`` fits a fixed-effects model
        (:math:`\\tau^2 \\equiv 0`).

        ``"dl"`` is estimated once, about the naive weighted mean, and then held fixed while the
        selection model fits :math:`\\mu` -- which keeps each EM iteration one-dimensional and
        concave, at a known cost. Because the weighted mean is by construction the centre that
        *minimises* the moment estimator's ``Q``, taking ``Q`` about the value finally reported
        can only raise :math:`\\tau^2`, and the shipped estimate is therefore biased low.
        Measured against a known :math:`\\tau = 0.35`, alternating the two recovers
        :math:`\\tau^2` of 0.067, 0.083, 0.093, 0.100 over three extra rounds against a true
        0.1225, and moves ``g`` at the focus from 0.883 to 0.822 against a true 0.8. No spurious
        heterogeneity appears where there is none. Not done, because each round is a full refit
        and a principled joint estimate is a larger change than alternation.
    se_method : {"model", "hksj"}, default="model"
        Standard error of the pooled estimate. ``"model"`` is the inverse-variance expression,
        which treats the estimated :math:`\\tau^2` as known; ``"hksj"`` is the
        Hartung-Knapp-Sidik-Jonkman residual-variance form on ``n_eff - 1`` degrees of freedom,
        which does not and covers better with few studies. **Requires**
        ``selection_model="none"``, the zero-inflated model reporting the censored likelihood's
        curvature instead. Changes ``se`` and so ``z``; p-values come from the permutation null
        either way.
    selection_model : {"zero-inflated", "none"}, default="zero-inflated"
        How a study that reported nothing near a voxel is handled. Nothing is imputed under
        either option.

        ``"zero-inflated"``
            Silence contributes the probability of being silent, under a mixture in which the
            study either has a real effect or none at all. This corrects the spatial winner's
            curse, and assumes the study *examined* the voxel. Where only one study reported,
            the magnitude is not estimable at all; see ``max_iter``.
        ``"none"``
            Only the reported peaks are pooled, so the estimate keeps the bias of the
            thresholding that selected them. Right when silence is *not* informative, because
            studies examined only part of the brain -- an ROI study says nothing about voxels
            it never analysed, and there is no per-study coverage flag yet. Also useful as a
            diagnostic, and roughly ten times faster.
    analysis_mask : :obj:`str` or None, optional
        ``value_type`` of a per-study image marking the voxels that study examined, nonzero
        meaning examined. Studies without one are taken to have examined the whole analysis
        volume, which is what the censoring term assumes of every study by default.

        This is what an ROI or partial-coverage study needs: its silence outside the region it
        analysed is not evidence that nothing is there, and the censoring term would otherwise
        read it as evidence against an effect. Voxels a study did not examine contribute neither
        a value nor a silence for it. Without this the only remedy was
        ``selection_model="none"`` for the whole collection, which also discards the correction
        for the studies that did examine the whole brain.
    threshold : :obj:`float`, :obj:`str`, or None, default="study-min"
        Reporting threshold assumed for each study, on the z scale. It decides how surprising a
        study's silence is and, with ``peak_bias="per-study"``, how far its peaks are
        discounted. ``"study-min"`` takes each study's own smallest absolute statistic with the
        order statistic undone by :func:`infer_threshold_from_minimum`; ``"pooled-min"`` takes
        the smallest reported anywhere in the collection, which applies the most liberal
        study's cut to every study. A string naming a metadata field holding the real
        thresholds is better than either. A float applies one threshold to every study.

        ``"study-min"`` is the default because it adapts to the threshold the table it is given
        actually reflects. A paper reporting only its top handful of peaks has an effective cut
        far above its nominal one, and the per-study rule recovers that where a pooled or fixed
        threshold cannot: on peak tables thinned to ten per study it roughly triples the rank
        correlation against known image truth, whether or not the studies really shared a
        threshold, and on complete tables the two rules agree exactly. Note that ``prevalence``
        moves a great deal with this choice and there is no truth to check it against.

        Both inference rules assume a *height* threshold, one local maximum per cluster. A
        cluster-extent threshold biases them upward by more than is comfortable: on smooth
        simulated fields the inversion recovers a true height threshold to +0.06 z but
        overshoots by +0.41 z when clusters of at least ten voxels are kept, and +1.10 z at
        fifty, because extent thresholding keeps broad clusters, whose peaks are higher, and
        discards isolated low ones.

        What that costs divides sharply. ``g`` barely notices -- across a +1.1 z error it stays
        within 10% of its value at the correct threshold, non-monotonically. ``prevalence`` does
        not survive it: 0.73, 0.96, 0.99, 1.00 as the threshold given is inflated by 0, 0.4, 0.8
        and 1.1 z, against a true 0.60. Supplying the real threshold from metadata therefore
        matters far more for ``prevalence`` than for the effect-size map.
    coverage_radius : :obj:`float` or None, optional
        Radius, in mm, within which a reported peak counts as this study having reported
        *something* about this location; a study with no focus inside it is treated as silent
        and contributes a censoring term. Keeping this separate from the kernel matters: a
        study whose peak sits 6 mm away should have its *value* discounted, but it has plainly
        not been silent. Defaults to twice the kernel FWHM (20 mm when ``fwhm`` is None), and
        is used only when ``selection_model="zero-inflated"``. ``g`` is insensitive to it at
        realistic peak counts; ``prevalence`` is not. Against a known prevalence the estimate
        rises monotonically with this radius at every true value (a true 0.50 reads 0.65, 0.73,
        0.76, 0.81 at 8, 14, 20 and 28 mm) and no radius recovers the truth: mean absolute error
        runs 0.17 to 0.21 over that range, 14 mm marginally best and the 20 mm default close
        behind. Left at 20 mm because the differences are small beside the bias itself. On dense
        peak tables ``prevalence`` saturates at 1.0 here.
    kernel_min_weight : :obj:`float`, default=0.01
        Truncate the spatial kernel below this fraction of its peak. A focus then reaches only
        voxels it says something about (about 13 mm for a 10 mm FWHM), which is what keeps
        ``n_studies`` interpretable and the fit affordable.
    max_iter : :obj:`int`, default=25
        Maximum Newton iterations for the censored likelihood. Voxels where a single study
        reported do not converge at any value of this, and raising it does not help: their
        likelihood is flat in the magnitude over the whole plausible range, so the iteration
        count only decides which point on a plateau is reported.
    null_method : {"permute-magnitudes", "none"}, default="permute-magnitudes"
        How uncorrected p-values are obtained. ``g / se`` is not null-referenced -- the standard
        error treats :math:`\\tau^2` as known and ignores that the peaks being pooled were
        selected for being large -- so p comes from a randomization null instead.

        ``"permute-magnitudes"`` reassigns each analysis's reported effect sizes among **its
        own** reported locations, holding the positions and study membership fixed, and refits;
        the hypothesis is that within a study, effect size is unrelated to location. An image
        study takes the same action over its own voxels. Because nothing moves between studies,
        each voxel keeps its own studies in every iteration and is referred to a null of its
        own, and a study's sample size, threshold and ``rho_k`` stay attached to its values --
        the exchangeability the test needs (:footcite:t:`winkler2014permutation`).

        The p-value is ``(1 + #{null >= observed}) / (1 + n_iters)`` and so cannot fall below
        ``1 / (1 + n_iters)``, which is where the default ``cluster_threshold`` of .001 sits
        unless ``n_iters`` is raised past 1000; familywise correction has no such floor. This is
        deliberately **not** a test of spatial convergence, which is what a null that relocates
        the foci -- ALE's and MKDA's -- would give instead, nor a test of whether the effect is
        zero, which is what sign-flipping the images would give.

        The price is that an analysis reporting a single focus has one arrangement and
        contributes no randomness. Where the collection admits fewer than about ``1e4``
        arrangements in total the null is not built at all and ``p`` is 1.0 everywhere, with a
        warning naming how many analyses contributed; the effect-size maps are unaffected.

        That floor is map-wide, and each voxel has a tighter one of its own: a voxel reached by
        two studies of three foci draws from 36 arrangements, so its p cannot fall below 1/37
        however many iterations are run. Such voxels simply never reach significance, which
        costs power rather than validity, and is why sparsely covered edges of a map stay
        non-significant no matter how large their ``g``.

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
    scale_interval_ : :obj:`tuple` of :obj:`float`, or None
        Multiplicative bounds the overall effect-size scale is identified to, or None when it
        is not identified at all -- which is the case for any coordinate-only fit, since
        rescaling every study by one constant leaves the coordinate likelihood unchanged.
        Reported because a point estimate of a partially identified parameter invites being
        read as a measurement. With images it is a 95% confidence interval for the scale,
        built on the log of the donor studies' individual estimates and exponentiated back, so
        it narrows as donors accumulate. It is not the range of those estimates: a range
        describes how far the donors landed apart and was measured at 0.51 times an honest
        interval with two donors and 4.32 times it with twenty. With two donors the interval is
        very wide, which is the correct statement about a scale resting on two studies, and a
        zero-width interval means the donors happened to agree exactly rather than that the
        scale is pinned.
    peak_information_ : :obj:`dict`
        ``observed_mean_z``, ``null_peak_mean_z`` and ``excess_z`` for the reported peaks. When
        the excess is small the heights carry no information about the size of the effect and
        only the spatial pattern is interpretable; :meth:`fit` says so in its description.
    masker : :class:`~nilearn.maskers.NiftiMasker`
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator.

    Notes
    -----
    Where ALE and (M)KDA ask *where do studies agree something happened*, this asks *how big is
    the effect there*. A peak's statistic and its study's sample size give Hedges' :math:`g`
    (:func:`peak_stat_to_hedges_g`), and each voxel solves a local random-effects meta-analysis
    over the foci whose kernels reach it:

    .. math::

        \\hat{g}(v) = \\frac{\\sum_k W_k(v) g_k}{\\sum_k W_k(v)},
        \\qquad W_k(v) = \\frac{w_k(v)}{s^2_k + \\tau^2(v)}

    for a kernel weight :math:`w_k(v)` and a kernel-weighted DerSimonian-Laird
    :math:`\\tau^2(v)`. With every :math:`w = 1` it reduces to a textbook random-effects
    meta-analysis, so the spatial part is a weighting scheme rather than a separate algorithm.
    Nothing is imputed; non-reporting enters only through the selection model
    :footcite:p:`tench2017coordinate`.

    Available maps:

    ============== ===============================================================
    "g"            Pooled Hedges' g, on whatever scale the fit could identify.
    "g_relative"   ``g`` over the 95th percentile of ``|g|``, so the unidentified
                   scale cancels. Always emitted, and the map to read by default.
    "g_absolute"   ``g`` with its overall scale pinned, which needs at least two
                   image studies in the collection or an explicit numeric
                   ``peak_bias_scale``; absent otherwise, there being nothing then
                   to distinguish it from ``g_relative`` times an unknown constant.
                   Pinning fixes the map's average level and not its values: see the
                   compression described under Warnings.
    "g_marginal"   ``g`` times ``prevalence``: the effect averaged over *all* studies
                   rather than over those that have one. Added under the zero-inflated
                   selection model. Closest of the magnitude maps to an independent
                   reference; see below.
    "se_marginal"  Standard error of ``g_marginal``, by the delta method on the same
                   observed information. Added alongside it. Zero where there is none.
    "se"           Standard error of the pooled estimate. See ``se_method``.
    "z"            ``g / se``. Two-tailed. Unaffected by the scale.
    "p", "logp"    p-value for ``z``, and its ``-log10``.
    "tau2"         Local between-study variance.
    "n_studies"    Number of studies with a focus inside the kernel support.
    "n_eff"        Kish effective number of studies, ``(sum w)^2 / sum w^2``.
    "dof"          ``n_eff - 1``, the degrees of freedom to refer ``se`` to. See below.
    ============== ===============================================================

    Build an interval from ``se`` against a *t* on ``dof``, not against a normal. Under the
    selection model ``se`` is the observed information of the censored mixture likelihood at the
    fitted point, with the prevalence profiled out by a Schur complement -- so it carries both
    the uncertainty about which component an observation came from and the cost of not knowing
    the prevalence. On the estimator's own censored mixture with known variances it covers 94.5%
    to 98.4% of nominal-95% intervals across prevalences, cutoffs and study counts, erring
    conservative.

    An earlier version reported the curvature of the EM's *Q function* instead, which holds the
    responsibilities fixed and therefore overstates the information; that covered 62.5% to 89.8%
    and did not improve with more studies. The p-values are unaffected either way -- they come
    from the permutation null, not from referring ``z`` to any distribution.

    ``prevalence`` is added under the zero-inflated selection model.
    ``prevalence`` is scale-free, so unlike ``g`` it does not depend on the constant the
    coordinates cannot identify -- but **read it ordinally, not as a fraction**. Against a
    simulator drawing a known prevalence it is compressed toward the middle of the range: a true
    0.25 comes back as 0.49 to 0.60 depending on ``coverage_radius``, a true 0.50 as 0.65 to
    0.81, a true 1.00 as 0.74 to 0.94. Its map-wide median sits near 0.4 whatever the truth, so a
    map cannot be summarised by it.

    **The ordinal reading holds on average over many maps, and rarely within any one of them.**
    Tested as the claim is made -- four sites in a single fit at true prevalences 0.25, 0.50,
    0.75 and 1.00, 24 studies -- the rank correlation against the truth averages +0.76 for a
    strong effect, but the four-site ranking is exactly right in only **19%** of maps; for a weak
    effect it averages +0.33 with a standard deviation of 0.60 and is exactly right in **6%**.
    Individual weak-effect maps run from anti-ordered to ordered, so a reader comparing two
    voxels in one of them is reading noise. The spacing carries less still: a true 0.25 comes
    back near 0.31 while a true 0.50 comes back near 0.68, so the map is inflated in the middle
    of the range and a difference between two voxels is not a difference in prevalence even
    approximately. And the per-voxel scatter is largest at the low end, so rare sites are both
    biased upward and noisier -- the worst combination for the use this invites, picking out which
    region is the least consistent.

    The assumed reporting threshold moves the level of this map and not its order. Supplying one
    rather than inferring it changed a true 0.50 from 0.64 to 0.68 and left the rank correlation
    and the exact-ordering rate identical to three decimals, because a change of cutoff applies a
    roughly common inflation across voxels. Under cluster-extent reporting, however, inference is
    badly wrong in level: the smallest value a study reports is then its smallest cluster
    *maximum* rather than anything near its threshold, ``threshold="study-min"`` came back at
    z = 4.0 against a true forming cut of z = 3.1, and the prevalences it produced were inflated
    at every site (a true 0.25 reading 0.48, a true 0.50 reading 0.86). Passing the
    cluster-forming threshold explicitly, or leaving ``threshold`` at a plausible constant,
    recovered a true 0.25 as 0.21 and a true 0.50 as 0.47. **On a collection whose tables came
    from cluster-extent correction, supply the threshold.**

    The reason is structural rather than a calibration that could be fixed. ``prevalence`` and
    ``g`` are separably estimable only in a window of detectability: where a study's effect lands
    near its own reporting threshold, so that the chance of reporting responds to the magnitude.
    Below that window nothing is detected and the prevalence is not identified at all; above it
    detection saturates, the magnitude stops being constrained from above and the prevalence
    absorbs the level instead -- which is why a strongly reported site returns a prevalence near
    1 whatever its truth. A map spans magnitudes and therefore spans the window, so comparing two
    voxels compares quantities identified to different degrees. A spread of sample sizes and
    reporting thresholds across the collection widens the window; a roster of identically
    powered studies narrows it.

    ``se_marginal`` is its standard error, from the delta method on the same observed
    information that gives ``se``, using the full two-by-two inverse rather than the Schur
    complement because the product depends on the prevalence as well as the magnitude. It
    covers only the *sampling* uncertainty at the fitted point, exactly as ``se`` does: neither
    includes the effect-size scale, which coordinates do not identify at all, so an interval
    built from either is an interval about a quantity on an unknown scale unless images pinned
    it. What it does buy is that ``g_marginal`` -- the one magnitude here an image-based
    meta-analysis also estimates -- can now be compared with one interval against another
    rather than point against point.

    ``g_marginal`` is ``g`` times ``prevalence``, and estimates a different quantity from ``g``:
    the effect averaged over every study, including those with none here, rather than over the
    studies that have one. It inherits the unidentified scale of ``g`` and the compression of
    ``prevalence``, so it is no more absolute than either -- but measured against references
    built from studies the coordinates never touched it is consistently the closest of the
    magnitude maps. On the 21-study NIDM pain collection split in half, with coordinates taken
    the way a paper would tabulate them, it correlates +0.32 to +0.48 with the held-out truth
    against +0.14 to +0.28 for ``g``, and where that truth is largest its ratio to it is 1.10 to
    1.23 against 1.46 to 2.29.

    Three caveats hold it to the same reading as everything else here.

    It does not escape the compression described under Warnings: across the truth's strata it
    moves about as little as ``g`` does, and below the median it is still several times the
    truth. What it calibrates is the upper part of the range -- on held-out HCP subjects under
    cluster-extent reporting it is within 10% to 20% of the truth above that truth's 75th
    percentile, and four to five times it below the median.

    It degrades when studies report few foci, because ``prevalence`` then falls toward its floor:
    at six foci per study it came back at 0.70 times the held-out truth overall, overshooting
    downward.

    And it does not work for the reason its name gives. In the held-out HCP design every
    synthetic study is drawn from one population, so the true prevalence is exactly 1 and
    ``g_marginal`` should equal ``g``; instead ``prevalence`` comes back near 0.68 and the
    product is the better estimate. Multiplying by it is shrinking an inflated magnitude by a
    data-driven factor rather than averaging over studies that have no effect. That the two
    happen to cancel -- ``g`` inflated upward by peak selection, ``prevalence`` compressed
    downward toward the middle of its range -- is why this is worth using and also why it should
    not be trusted beyond the regimes it has been measured in. Read it ordinally, and prefer it
    to ``g`` when a magnitude map is wanted.

    :meth:`correct_fwe_montecarlo` adds ``logp_level-voxel``,
    ``logp_desc-size_level-cluster`` and ``logp_desc-mass_level-cluster`` (each with a
    signed ``z_*`` companion), matching the names
    :class:`~nimare.meta.cbma.ale.ALE` uses. :class:`~nimare.correct.FDRCorrector` and
    ``FWECorrector(method="bonferroni")`` work off the uncorrected ``"p"`` map instead,
    and are only meaningful when that map came from the permutation null.

    Warnings
    --------
    This estimator is new and has not been validated against a reference implementation.

    **Treat ``g_relative`` as the effect-size output.** The magnitude of ``g`` is not
    calibrated: ``peak_bias`` corrects only the part of the peak-height inflation that varies
    between studies, and the common scale is not identified from coordinates at all. How badly
    it is off depends on the collection, from about twofold to an order of magnitude, and
    :func:`peak_information` says in advance when the reported heights carry no effect-size
    information at all -- which on real collections is often. ``g_absolute`` is emitted only
    when images or an explicit ``peak_bias_scale`` pin the scale, and even then
    ``scale_interval_`` reports how well.

    **A reported z carries a degrees-of-freedom assumption, and it is load-bearing.** A reported
    statistic is converted to an effect size through :func:`peak_stat_to_hedges_g`; a ``z`` is
    treated as a p-value-preserving image of a *t* on ``n - 1`` degrees of freedom and mapped
    back before conversion, which is what neuroimaging software usually produces. Reported peaks
    sit far into the tail, where that map is steep, so the assumed degrees of freedom matter.
    Holding ``n`` at 30 and varying only the assumed residual degrees of freedom:

    ============  =======  =======  ========  =========  ======
    reported z     df=29    df=60    df=120    df=1000   spread
    ============  =======  =======  ========  =========  ======
    3.30           0.653    0.626     0.614      0.604    1.08x
    4.00           0.830    0.776     0.752      0.733    1.13x
    5.00           1.133    1.009     0.959      0.918    1.23x
    6.00           1.522    1.273     1.178      1.105    1.38x
    ============  =======  =======  ========  =========  ======

    The sensitivity grows with the reported height, so it is largest exactly where the
    peak-height inflation is largest and in the same direction. The effective degrees of freedom
    of a published z map are frequently *above* ``n - 1`` -- variance smoothing raises them, and
    some mixed-effects tools do that deliberately -- and papers seldom state them, so the likely
    direction of the error is a further over-statement of magnitude. A study reporting a ``t`` is
    unaffected, since that conversion is direct, which is a reason to prefer
    ``stat_column="t_stat"`` on a collection that offers both.

    **No image-based meta-analysis estimates what ``g`` estimates**, so ``g_absolute`` cannot be
    checked against one even in principle. Every IBMA -- DerSimonian-Laird, Hedges, weighted least
    squares, the likelihood estimators -- pools per-study effect maps around a single mean, so a
    study with no effect at a voxel enters that average as a zero and the quantity estimated is
    :math:`\pi(v)\,\mu(v)`, the effect over *all* studies. ``g`` is :math:`\mu(v)`, the effect
    over the studies that have one. The two differ by a factor of :math:`1/\pi`, which on the
    NIDM pain collection is 1.4 to 1.8, so part of what looks like inflation when ``g`` is scored
    against an image-based reference is the two maps answering different questions.

    Worse for validation, :math:`\mu(v)` may not be identifiable from images at all: computing it
    requires classifying every study as having an effect at every voxel or not, which is a
    thresholding decision and reintroduces the selection this estimator exists to correct.
    ``g_marginal`` is therefore the map to compare against images, being the same estimand, and
    it is the one that validates -- not because it is better estimated but because it answers the
    reference's question.

    **The magnitude is also compressed, and pinning the scale does not uncompress it.** Judged
    against references built from studies the coordinates never touched -- held-out HCP subjects,
    and split halves of the 21-study NIDM pain collection and of NeuroVault collections sharing a
    cognitive paradigm -- a reference effect spanning elevenfold across its strata comes back
    spanning about 1.2-fold. An unknown constant would leave that ratio alone; it does not.

    Nor is the level a property of the studies. On one collection, holding the studies fixed and
    changing only how a paper would have tabulated them, the ratio of ``g`` to the held-out truth
    where that truth is largest runs from 0.82 to 2.29 -- across thresholding by FDR, by voxelwise
    family-wise error and by cluster extent, and across tabulating a cluster by its maximum or by
    its centre of mass. Two collections reporting the same effects under different conventions
    will not agree.

    The cause is the input rather than the fit, so no option here changes it. Regressing the
    held-out truth at a focus on the effect size that focus's own table reports gives a slope of
    0.08 to 0.18 with most of the value in the intercept: one tabulated coordinate explains 5% to
    9% of the variance in the effect at its own location. The pooled map already does slightly
    better than that ceiling, so the estimator is extracting more than a coordinate carries, not
    less. Measured and rejected as remedies: the truncated-normal selection correction, which is
    the wrong event for a local maximum and returns 0.26 for a true 0.5; ``peak_bias="per-study"``;
    subtracting the censoring floor; reporting by centre of mass, which carries no winner's curse
    and still does not decompress; and widening the assumed cluster, which costs correlation.

    Read ``g`` and ``g_relative`` as ordering voxels within one collection, which they do: against
    the same held-out references they correlate +0.21 to +0.45, and better than any single
    coordinate does.

    What the null tests is not what a reader of a coordinate-based meta-analysis may expect. The
    estimand is :math:`\mu(v)`, the effect size at a voxel among the studies that have an
    effect there, on a scale the coordinates do not identify; the null is that **within a
    study, effect size is unrelated to location**. A voxel is significant when the effects
    reported near it are large relative to what *the same studies* reported elsewhere -- not
    when the pooled effect differs from zero, and not when studies converge there.

    That has two consequences worth stating plainly. A collection with a genuine effect of the
    same size everywhere has nothing for this null to find, though neither would any method
    built on reported peaks, since a peak is only reported where the effect is locally large.
    And a study reporting a single focus admits one arrangement, so it contributes no
    randomness: where the whole collection admits too few, ``p`` comes back at 1.0 with a
    warning rather than as inference that was never done.

    The zero-effect null is not available here. A reported peak exists only because it cleared
    a threshold, so "no effect anywhere" predicts no coordinates at all and the observed table
    falsifies it before any voxel is examined; testing it needs subject-level images, which is
    what :footcite:t:`albajes2019meta` imputes in order to permute. Sign-flipping the image
    studies while shuffling the coordinates would test it for part of the collection only, and
    the two hypotheses then combine into a rejection either can cause -- with 20 coordinate and
    5 image analyses the sign flips alone floored the p-value at 1/32 whatever the locations
    said, which is why the images now take the same within-study shuffle as the coordinates.

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
        se_method="model",
        analysis_mask=None,
        threshold="study-min",
        coverage_radius=None,
        kernel_min_weight=0.01,
        max_iter=25,
        null_method="permute-magnitudes",
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
            peak_bias=peak_bias,
            peak_bias_scale=peak_bias_scale,
            threshold=threshold,
            se_method=se_method,
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
        self.se_method = se_method
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
                    "This collection has images but no coordinates, and CBES is a "
                    "coordinate-based estimator: with nothing to pool from peaks it would "
                    "reduce to a random-effects meta-analysis of the images, and its null "
                    "would have no foci to permute -- every permutation reproduces the "
                    "observed map, so the p-values would be meaningless. Use an image-based "
                    "estimator instead: nimare.meta.ibma.DerSimonianLaird or "
                    "nimare.meta.ibma.Hedges for random effects on beta/varcope maps, "
                    "WeightedLeastSquares for fixed effects, or Stouffers on z maps. CBES is "
                    "for collections that have coordinates, optionally with images alongside "
                    "them for a subset of studies."
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

    def _warn_if_threshold_implausible(self, cutoff_z):
        """Say so when an inferred threshold lands where extent-based reporting would put it.

        The inference cannot tell a strict height threshold from a cluster-extent one -- both
        leave few, high peaks -- and guesses high when reporting was by extent, by +1.1 z at a
        fifty-voxel threshold. ``g`` tolerates that; ``prevalence`` does not, saturating toward
        1.0. So the warning names the output at risk and the remedy, rather than pretending the
        threshold can be recovered.
        """
        finite = np.asarray(cutoff_z, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return
        median = float(np.median(finite))
        if median <= _SUSPICIOUS_INFERRED_THRESHOLD_Z:
            return
        LGR.warning(
            f"Inferred reporting threshold z = {median:.2f}, above the usual range for a height "
            "threshold. Either these studies thresholded unusually strictly, or they reported "
            "by cluster extent, which this inference cannot distinguish and which it overshoots "
            "by about 1 z. 'g' tolerates that error to within about 10%, but 'prevalence' does "
            "not -- it saturates toward 1.0 -- so pass the real thresholds via a metadata field "
            "if 'prevalence' is going to be read."
        )

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
            total = len(set(images["id"].astype(str)))
            rest = "" if len(loaded) >= total else "; coordinates for the rest"
            LGR.info(f"Using images for {len(loaded)} studies{rest}.")
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

    def _size_reduction(self):
        """How a study's per-group sample sizes collapse to the number the model wants.

        ``peak_stat_to_hedges_g`` takes a *total* N for a two-sample design and splits it into
        equal groups, so the reduction has to be a sum: metadata of ``[30, 30]`` means sixty
        subjects, and reducing it by mean gave thirty, which the converter then read as two
        groups of fifteen. That inflated ``g`` by 39% at ``t = 3`` (1.066 against 0.765) and the
        same error reached the sampling variances, the cutoff conversion and the null variances.
        A lone value is already a total either way, so summing is right in both cases.

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

    def _reported_z(self, table):
        """Put reported statistics on the z scale, where studies are comparable.

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
            self._warn_if_threshold_implausible(cutoff_z)
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
            self._warn_if_threshold_implausible(cutoff_z)
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

        Also sets ``scale_interval_``: the multiplicative bounds the scale is identified to,
        or None when it is not identified at all. A point estimate of a partially identified
        parameter invites being read as a measurement, which the scale is not.

        Calibration needs a provisional fit, which needs a ``rho``, so the scale is resolved
        at 1.0 first and the answer applied afterwards. That is exact rather than iterative:
        the fit is linear in the scale, so a fit at 1.0 times the calibrated scale *is* the
        fit at the calibrated scale.
        """
        self.scale_interval_ = None
        self.scale_source_ = "unset"
        self.n_scale_donors_ = 0
        if self.peak_bias is None:
            return 1.0
        if self.peak_bias_scale not in ("auto", "images"):
            if self._image_studies_ and self.peak_bias_scale == 1.0:
                LGR.warning(  # noqa: E501
                    "This fit mixes images with coordinates but leaves peak_bias_scale at "
                    "1.0, so the coordinate studies are on a relative scale while the images "
                    "are on the true Hedges' g scale. The two then disagree about the same "
                    "voxel -- by a factor of about two on the NIDM pain images -- and the "
                    "pooled value depends on how many studies of each kind the collection "
                    "holds. Use peak_bias_scale='auto' to read the constant off the images."
                )
            # An explicit number is the caller asserting the scale, which is as much
            # identification as any collection of coordinates can offer.
            if float(self.peak_bias_scale) != 1.0:
                self.scale_source_ = "supplied"
            return float(self.peak_bias_scale)

        self._peak_bias_scale_ = 1.0
        provisional = self._peak_bias_factors(self._cutoffs_z_, sample_sizes, reporting_ids)
        scaled, thresholds = self._apply_peak_bias(
            table, self._cutoffs_z_, sample_sizes, provisional
        )

        if not self._image_studies_:
            LGR.warning(
                "peak_bias_scale needs images to calibrate against, and this collection "
                "supplies none. Falling back to 1.0, which leaves the effect-size map correct "
                "up to one multiplicative constant -- read 'g_relative' rather than 'g'."
            )
            return 1.0

        return self._calibrate_peak_bias_scale(
            scaled, sample_sizes, thresholds, self._image_studies_
        )

    def _calibrate_peak_bias_scale(self, table, sample_sizes, thresholds, image_studies):
        """Read the overall peak-to-field ratio off the studies that supplied images.

        The one common scale is not identified from coordinates, and that is harmless for a
        coordinate-only map but not once images are in the same fit: images sit on the true
        ``g`` scale, so a mismatched constant makes the two kinds of study disagree about the
        same voxel and the pooled value depends on how many of each the collection holds.

        Images are what fix it: fit the coordinates alone, fit each image study alone, and take
        the ratio over the voxels both cover. The fit is exactly linear in the scale, so a ratio
        of summaries recovers it -- a regression slope would be attenuated by the many voxels
        where a study peaked and the images say nothing.

        **Each fit receives only the studies it contains.** A study absent from the table a fit
        is given falls through :meth:`_coverage_entries` as having examined every voxel and
        reported nothing, which is right in the real fit and wrong here: it made every
        donor-only fit carry one censored-silent observation per coordinate study, dragging the
        donor's magnitude down and the ratio with it. On the pain collection the scale came out
        0.61 against 0.71 with two donors, 0.61 against 0.66 with three and 0.62 against 0.66
        with five, so the absolute magnitudes were 6 to 14% too small, least with the most
        donors. The drag partly cancels between the two fits, which is why it is a modest bias
        rather than the factor of 1.5 a reimplementation of the ratio suggested.

        **One ratio per donor, pooled across donors**, rather than one ratio against all the
        images pooled together: pooling the images first makes the answer depend on how many
        there are, because a single image's map keeps its own peaks while averaging several
        flattens them and the coordinate fit stays winner's-curse inflated. Per-donor ratios
        are estimates of one constant, so adding donors sharpens rather than moves it -- with
        dense peak tables. On sparse ones the pooled median still slides, which is what
        ``scale_interval_`` reports.
        """
        donor_ids = list(image_studies or {})
        # The coordinate fit is every study that speaks through coordinates, which is the whole
        # roster less the donors -- a coordinate study that reported nothing anywhere is
        # genuinely silent and belongs here, unlike a donor, which speaks through its image.
        coordinate_ids = [study for study in sample_sizes.index if study not in set(donor_ids)]
        coordinate_only = self._statistic(
            table[table["id"].isin(coordinate_ids)],
            sample_sizes.loc[coordinate_ids],
            thresholds.loc[coordinate_ids],
            image_studies=None,
        )[0]

        per_donor = []
        for study_id, payload in (image_studies or {}).items():
            single = self._statistic(
                table.iloc[:0],
                sample_sizes.loc[[study_id]],
                thresholds.loc[[study_id]],
                image_studies={study_id: payload},
            )[0]
            both = (
                coordinate_only["covered"]
                & single["covered"]
                & np.isfinite(coordinate_only["g"])
                & np.isfinite(single["g"])
            )
            if not both.any():
                continue
            from_coordinates = np.abs(coordinate_only["g"][both])
            from_image = np.abs(single["g"][both])
            # Scored where this image says there is something to estimate, and by a paired
            # median rather than a ratio of means. An earlier version divided the two means
            # over every shared voxel, and a covered brain is mostly voxels holding no effect,
            # so the image mean collapsed toward zero and the scale came out far too small --
            # 0.16 against a true 0.8 on simulated data.
            strong = from_image >= np.percentile(from_image, _CALIBRATION_PERCENTILE)
            if strong.sum() < _MIN_CALIBRATION_VOXELS:
                strong = np.ones_like(from_image, dtype=bool)
            ratios = from_image[strong] / np.clip(
                from_coordinates[strong], _PROBABILITY_FLOOR, None
            )
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            if ratios.size:
                per_donor.append(float(np.median(ratios)))

        if not per_donor:
            LGR.warning(
                "Cannot calibrate peak_bias_scale: no image study shares a voxel with the "
                "coordinate studies where both carry a usable estimate. Falling back to 1.0, "
                "which leaves the two kinds of study on different scales."
            )
            return 1.0

        # Median across donors: with a handful of them one atypical study should not carry the
        # constant, and the spread is what a caller supplying a single image is exposed to.
        scale = float(np.median(per_donor))
        spread = (max(per_donor) / min(per_donor)) if min(per_donor) > 0 else float("inf")
        self.scale_source_ = "images"
        self.n_scale_donors_ = len(per_donor)
        # With one donor there is no spread to measure and the interval is unknown, not zero.
        self.scale_interval_ = _scale_confidence_interval(per_donor)
        LGR.info(
            f"Calibrated peak_bias_scale = {scale:.3f} from {len(per_donor)} image "
            f"{'study' if len(per_donor) == 1 else 'studies'}, whose individual estimates span "
            f"a factor of {spread:.2f}."
        )
        if len(per_donor) == 1:
            LGR.warning(
                f"peak_bias_scale was calibrated from a single image study, so the whole "
                f"effect-size scale rests on how representative that one study is. Its value "
                f"is {scale:.3f}; two or three donors would show whether that is typical."
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

        Returns ``(cols, w, focus_index)``, one entry per voxel the study reaches. A study that
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
            empty = np.array([], dtype=np.int64)
            return empty, np.array([], dtype=float), empty

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

        return cols, weights, focus_idx

    def _focus_geometry(self, table, fixed_support, mask_flat_to_masked, shape):
        """Per-study ``(study_id, voxels, weights, focus_index)``, cached across permutations.

        Which voxels a study reaches, and with what kernel weight, is a function of where its
        foci are and nothing else. The permutation null holds every position fixed and moves
        only the reported values, so this is identical on every one of ``n_iters`` refits and is
        computed once. What the caller gathers per refit is ``g`` and ``var_g``, via the focus
        index returned here.

        Keyed on the positions themselves rather than assumed valid, so a caller that passes a
        differently arranged table gets a rebuild instead of a wrong answer.
        """
        masks = getattr(self, "_analysis_masks_", None) or {}
        key = (
            table[["i", "j", "k"]].values.astype(np.int64).tobytes(),
            np.asarray(table["id"].values, dtype=object).tobytes(),
            # The masks clip the geometry below, so two fits under different masks must not
            # share a cache entry.
            tuple(sorted((study, mask.tobytes()) for study, mask in masks.items())),
        )
        cached = getattr(self, "_geometry_", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        geometry = []
        for study_id, study_table in table.groupby("id", sort=False):
            if fixed_support is not None:
                offsets, values = fixed_support
            else:
                offsets, values = self._kernel_support(
                    sample_size=float(study_table["sample_size"].iloc[0])
                )
            cols, weights, focus_idx = self._study_voxel_weights(
                study_table, offsets, values, mask_flat_to_masked, shape
            )
            # A kernel reaches about 13 mm for a 10 mm FWHM, so a peak just inside a declared
            # region spills outside it. Suppressing that study's *silence* out there while
            # still letting its *value* be pooled there is the worst of both: the voxel gets a
            # number from a study that never examined it. Clip the geometry to what was
            # examined, which is where every later sum is built from.
            examined = masks.get(study_id)
            if examined is not None and cols.size:
                inside = examined[cols]
                cols, weights, focus_idx = cols[inside], weights[inside], focus_idx[inside]
            if not cols.size:
                LGR.info(f"Study {study_id} contributes no in-mask voxels; skipping.")
                continue
            geometry.append((study_id, cols, weights, focus_idx))

        self._geometry_ = (key, geometry)
        return geometry

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
        # Only the values move between permutations; the geometry below is reused.
        geometry_values = {
            study_id: (group["g"].values, group["var_g"].values)
            for study_id, group in table.groupby("id", sort=False)
        }

        sums = {
            name: np.zeros(n_voxels, dtype=float)
            for name in ("w", "w2", "a", "a2", "ag", "ag2", "w2_over_s2", "n")
        }
        contributions = []

        for study_id, cols, weights, focus_idx in self._focus_geometry(
            table, fixed_support, mask_flat_to_masked, shape
        ):
            values_g, values_var = geometry_values[study_id]
            g, var_g = values_g[focus_idx], values_var[focus_idx]

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

    def _coverage_entries(self, table, study_ids, active, n_voxels, image_ids=()):
        """Pair each voxel with the studies that reported anything near it.

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

        radius = self.coverage_radius
        if radius is None:
            radius = 2.0 * (self.fwhm if self.fwhm is not None else 10.0)
        offsets = sphere_kernel_offsets(radius, mask_img.header.get_zooms()[:3])

        # Dilation on a padded grid, so that a study's covered voxels come out of one add and
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
        # Reused across studies to deduplicate the voxels a study's spheres cover. A scratch
        # bitmap costs one pass over the hits; ``np.unique`` sorts or hashes them, and with
        # a 20 mm sphere per focus there are a great many hits.
        seen = np.zeros(active.size, dtype=bool)

        image_ids = set(image_ids)
        analysis_masks = getattr(self, "_analysis_masks_", None) or {}
        cols, positions = [], []
        for position, study_id in enumerate(study_ids):
            # A voxel a study never examined is marked covered, which is how the model says
            # "contributes nothing": covered suppresses the censoring term, and the kernel
            # weight is already zero there, so neither silence nor a value is read from it.
            examined = analysis_masks.get(study_id)
            if examined is not None:
                outside = np.flatnonzero(~examined[active])
                if outside.size:
                    cols.append(outside.astype(np.int64))
                    positions.append(np.full(outside.size, position, dtype=np.int64))
            if study_id in image_ids:
                # An image reports everywhere, so it is silent nowhere and contributes no
                # censoring term. Marking it covered at every active voxel says exactly that.
                cols.append(np.arange(active.size, dtype=np.int64))
                positions.append(np.full(active.size, position, dtype=np.int64))
                continue
            ijk = table.loc[table["id"] == study_id, ["i", "j", "k"]].values.astype(np.int64)
            if not ijk.size:
                continue  # reported nothing anywhere: silent at every voxel
            # A focus further outside the image than the sphere's own reach cannot touch an
            # in-mask voxel, so dropping it here loses nothing and keeps every remaining index
            # inside the padded grid.
            ijk = ijk[np.all((ijk >= -reach) & (ijk < shape + reach), axis=1)]
            if not ijk.size:
                continue
            base = (ijk + pad) @ padded_strides
            reached = padded_lookup[(base[:, None] + flat_offsets).ravel()]
            reached = reached[reached >= 0].astype(np.int64)
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
            cov_col, cov_pos = cached[1]
        else:
            cov_col, cov_pos = self._coverage_entries(
                table, study_ids, active, n_voxels, image_ids=image_ids
            )
            self._coverage_ = (coverage_key, (cov_col, cov_pos))
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
        se_marginal_out = np.full(active.size, np.inf, dtype=float)

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

            covered = np.zeros((n_studies, width), dtype=bool)
            c_lo, c_hi = np.searchsorted(cov_col, [lo, hi])
            if c_hi > c_lo:
                covered[cov_pos[c_lo:c_hi], cov_col[c_lo:c_hi] - lo] = True

            mu, pi, se, se_marginal = self._fit_chunk(
                weights=weights,
                g_obs=g_obs,
                var_obs=var_obs,
                covered=covered,
                tau2=fit["tau2"][active[lo:hi]],
                null_var=null_var,
                cutoffs=cutoffs,
                start=fit["g"][active[lo:hi]],
            )
            mu_out[lo:hi], pi_out[lo:hi] = mu, pi
            se_out[lo:hi], se_marginal_out[lo:hi] = se, se_marginal

        fit["g"] = np.zeros(n_voxels, dtype=float)
        fit["g"][active] = mu_out
        fit["prevalence"][active] = pi_out
        fit["se"] = np.full(n_voxels, np.inf, dtype=float)
        fit["se"][active] = se_out
        fit["se_marginal"] = np.full(n_voxels, np.inf, dtype=float)
        fit["se_marginal"][active] = se_marginal_out

    def _working_sets(self, *, weights, g_obs, var_obs, covered, tau2, null_var, cutoffs):
        """Split the block into the reporting and silent ``(study, voxel)`` pairs the EM uses.

        Works on the pairs that carry weight rather than on the dense study-by-voxel block. At
        any given voxel a study either reported nearby or was silent there, and in a real
        studyset most studies are neither -- they reported in the region but outside this
        voxel's kernel, so they inform neither term. Evaluating normal CDFs across the full
        block and then multiplying most of them by zero was 97% of the runtime.
        """
        width = weights.shape[1]
        reporting = np.flatnonzero(weights > 0)
        silence = np.flatnonzero(~covered)
        rep_voxel = reporting % width
        sil_voxel, sil_study = silence % width, silence // width

        var_rep = var_obs.ravel()[reporting]
        sigma_rep = np.sqrt(var_rep + tau2[rep_voxel])
        g_rep = g_obs.ravel()[reporting]
        w_rep = weights.ravel()[reporting]

        cutoff_sil = cutoffs.ravel()[sil_study]
        null_var_sil = null_var.ravel()[sil_study]
        inv_sigma_sil = 1.0 / np.sqrt(null_var_sil + tau2[sil_voxel])

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

        reporting_pairs = _ReportingPairs(
            voxel=rep_voxel,
            weight=w_rep,
            g=g_rep,
            sigma=sigma_rep,
            precision=1.0 / sigma_rep**2,
            density_null=_normal_pdf(g_rep / np.sqrt(var_rep)) / np.sqrt(var_rep),
            responsibility=np.ones(reporting.size),
        )
        silent_pairs = _SilentPairs(
            voxel=sil_voxel,
            weight=reporter_scale[sil_voxel],
            inv_sigma=inv_sigma_sil,
            inv_sigma_sq=inv_sigma_sil * inv_sigma_sil,
            cutoff_scaled=cutoff_sil * inv_sigma_sil,
            twice_cutoff_scaled=cutoff_sil * inv_sigma_sil * 2.0,
            # Probability a silent study stays silent when it has no effect at all. Fixed
            # across iterations, and close to one whenever the threshold is several sigma.
            prob_silent_null=np.clip(
                ndtr(cutoff_sil / np.sqrt(null_var_sil))
                - ndtr(-cutoff_sil / np.sqrt(null_var_sil)),
                _PROBABILITY_FLOOR,
                None,
            ),
            responsibility=np.ones(silence.size),
        )
        return reporting_pairs, silent_pairs

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
        mixture_sil = resp_sil + (1.0 - pi_sil) * silent.prob_silent_null + _LOGP_FLOOR
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

    def _fit_chunk(self, *, weights, g_obs, var_obs, covered, tau2, null_var, cutoffs, start):
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
            covered=covered,
            tau2=tau2,
            null_var=null_var,
            cutoffs=cutoffs,
        )

        total_weight = np.bincount(
            reporting.voxel, weights=reporting.weight, minlength=width
        ) + np.bincount(silent.voxel, weights=silent.weight, minlength=width)

        mu = start.copy()
        pi = np.full(width, 0.5 if zero_inflated else 1.0)
        mu_out = np.zeros(width)
        pi_out = np.zeros(width)
        se_out = np.full(width, np.inf)
        se_marginal_out = np.full(width, np.inf)
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
            information, marginal_variance = _observed_information(
                width=mu.size,
                pi=pi,
                reporting=reporting,
                silent=silent,
                mu=mu,
                censoring=silent.censoring(mu),
            )
            information = information[positions]
            marginal_variance = marginal_variance[positions]
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
            log_likelihood = log_likelihood[keep]
            total_weight = total_weight[keep]
            voxel_ids = voxel_ids[keep]

        if mu.size:
            retire(np.arange(mu.size))

        return mu_out, pi_out, se_out, se_marginal_out

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

    def _permute_magnitudes(self, rng):
        """Reassign each analysis's reported values among its own reported locations.

        The randomization an effect-size estimate admits: each focus keeps where it is and
        gives up what it said, so the hypothesis is that effect size is unrelated to location.
        Only the value and its variance move -- study membership, sample size and position all
        stay, which leaves the spatial design exactly invariant. Moving the study label too
        would look more thorough and is wrong: a voxel keeps one observation per study, so
        relabelling can land two foci of one study on a voxel and quietly drop its count, which
        makes a site significant on multiplicity alone.

        **Within an analysis, not across the table.** A value is exchangeable only with values
        drawn from the same distribution, and a study's reported magnitudes carry its sample
        size, its reporting threshold and its ``rho_k`` -- so a large-N study's peak landing on
        a small-N study's voxel is an arrangement the null should never have contained. The
        across-table shuffle this used to do rejected at 96.7% where the nominal rate is 5%
        once precision varied across studies, because the null's spread came from the roster's
        heterogeneity rather than from the observed map. Restricting the shuffle to within an
        analysis is the standard remedy for exchangeability under nuisance structure
        (:footcite:t:`winkler2014permutation`); it also costs power, since a study reporting a
        single focus has one arrangement and contributes nothing.

        Sign-flipping, the natural randomization for a one-sample effect, cannot be applied to
        reported peaks: a peak is in the table only because it cleared a threshold, so the
        coordinate side is not sign-symmetric under the null.
        """
        table = self._focus_table_
        permuted = table.copy()
        groups = self._permutation_groups()
        if groups is None:
            return permuted
        positions, labels = groups
        # One sort rather than a loop over studies: keyed on (analysis, random), both arrays
        # come out grouped by analysis in the same order, so assigning one onto the other is a
        # permutation within each analysis and nothing crosses between them.
        donor = np.lexsort((rng.random(len(table)), labels))
        # ``peak_bias`` is present only when the rescaling was applied, and travels with the
        # value it rescaled. Column by column rather than as one block: a mixed-dtype
        # ``.values`` would come back as object and cost more than the permutation itself.
        for column in ("g", "var_g", "stat", "peak_bias"):
            if column in table.columns:
                values = table[column].values
                shuffled = values.copy()
                shuffled[positions] = values[donor]
                permuted[column] = shuffled
        return permuted

    def _permutation_groups(self):
        """Return cached ``(positions, labels)`` for the within-analysis shuffle.

        ``labels`` is an integer per row naming its analysis; ``positions`` orders the rows by
        that label, so the two sorts line up group for group. Built once because the focus
        table does not change across permutations -- only which value sits at which row.
        """
        cached = getattr(self, "_permutation_groups_", None)
        if cached is None:
            table = self._focus_table_
            if not len(table):
                return None
            ids = np.asarray(table["id"].values, dtype=object)
            _, labels = np.unique(ids, return_inverse=True)
            cached = (np.argsort(labels, kind="stable"), labels)
            self._permutation_groups_ = cached
        return cached

    def _null_has_states(self):
        """Report ``(n_arrangements_log10, n_contributing)`` for the within-analysis null.

        An analysis reporting one focus has exactly one arrangement, so it contributes no
        randomness; an image contributes as many as it has usable voxels. The count is returned
        as a log because a handful of ordinary studies already overflows a float.
        """
        log10_states, contributing = 0.0, 0
        table = getattr(self, "_focus_table_", None)
        if table is not None and len(table):
            _, counts = np.unique(np.asarray(table["id"].values, dtype=object), return_counts=True)
            multi = counts[counts > 1]
            contributing += int(multi.size)
            log10_states += float(np.sum(gammaln(multi + 1.0))) / np.log(10.0)
        for _, _, usable in (getattr(self, "_image_studies_", None) or {}).values():
            n_usable = int(usable.sum())
            if n_usable > 1:
                contributing += 1
                log10_states += float(gammaln(n_usable + 1.0)) / np.log(10.0)
        return log10_states, contributing

    def _null_is_usable(self):
        """Refuse to build the null when the collection admits too few arrangements.

        The within-analysis shuffle has states only where an analysis reported more than one
        focus. A collection of single-peak studies has exactly one arrangement, so every
        permutation reproduces the observed map and every p-value comes back at 1.0 -- which
        looks like a null result rather than like a test that could not be run. Saying so is
        the difference between the two.
        """
        log10_states, contributing = self._null_has_states()
        if log10_states >= _MIN_NULL_STATES_LOG10:
            return True
        # Recomputed rather than cached, so that it cannot go stale against a focus table that
        # changed; only the warning is remembered, because ``fit`` and the description both ask.
        if getattr(self, "_null_refusal_logged_", False):
            return False
        self._null_refusal_logged_ = True
        n_analyses = (
            int(self._focus_table_["id"].nunique()) if len(self._focus_table_) else 0
        ) + len(getattr(self, "_image_studies_", None) or {})
        LGR.warning(
            f"No p-values were computed: the within-analysis null admits about "
            f"1e{log10_states:.1f} arrangements, from {contributing} of {n_analyses} analyses. "
            "An analysis reporting a single focus has one arrangement and contributes no "
            "randomness, so a collection of them cannot be tested for whether effect size is "
            "related to location -- the effect-size maps are still estimated, and 'p' is 1.0 "
            "everywhere to say that nothing was tested."
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
            _, z_null = self._statistic(
                self._permute_magnitudes(rng),
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

        # How the effect-size scale was settled, which decides whether an absolute-scale map
        # can be emitted at all. Defaulted here so the branches below need only raise it.
        self.scale_source_ = "unset"
        self.n_scale_donors_ = 0
        self.scale_interval_ = None

        needs_thresholds = self.selection_model != "none" or self.peak_bias is not None
        if needs_thresholds:
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
        self._geometry_ = None
        self._coverage_ = None
        self._permutation_groups_ = None
        self._null_refusal_logged_ = False
        self._image_studies_ = self._load_image_studies(dataset)
        self._prepare_focus_table(dataset)

        fit, z_values = self._statistic(
            self._focus_table_, self._sample_sizes_, self._thresholds_, self._image_studies_
        )

        if self.null_method == "permute-magnitudes" and self._null_is_usable():
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

        maps = {
            "g": fit["g"].astype(DEFAULT_FLOAT_DTYPE),
            "g_relative": _relative_g(fit["g"], fit["covered"]).astype(DEFAULT_FLOAT_DTYPE),
            "se": np.where(np.isfinite(fit["se"]), fit["se"], 0).astype(DEFAULT_FLOAT_DTYPE),
            "z": z_values.astype(DEFAULT_FLOAT_DTYPE),
            "p": p_values.astype(DEFAULT_FLOAT_DTYPE),
            "logp": _nlogp_to_logp_values(np.log(np.clip(p_values, _LOGP_FLOOR, None))),
            "tau2": fit["tau2"].astype(DEFAULT_FLOAT_DTYPE),
            "n_studies": fit["n_studies"].astype(DEFAULT_FLOAT_DTYPE),
            "n_eff": fit["n_eff"].astype(DEFAULT_FLOAT_DTYPE),
            "dof": np.clip(fit["n_eff"] - 1.0, 0.0, None).astype(DEFAULT_FLOAT_DTYPE),
        }
        if "prevalence" in fit:
            maps["prevalence"] = fit["prevalence"].astype(DEFAULT_FLOAT_DTYPE)
            maps["g_marginal"] = (fit["g"] * fit["prevalence"]).astype(DEFAULT_FLOAT_DTYPE)
            # Zero where there is no usable information, matching how "se" is emitted, so that
            # a reader is not handed an infinity to divide by.
            marginal_se = fit.get("se_marginal")
            if marginal_se is not None:
                maps["se_marginal"] = np.where(np.isfinite(marginal_se), marginal_se, 0).astype(
                    DEFAULT_FLOAT_DTYPE
                )
        if self._scale_is_pinned():
            # Only here is "g" on the Hedges' g scale rather than on its own, so only here is
            # there a second map to emit. It is the same array; the separate name is the claim.
            maps["g_absolute"] = fit["g"].astype(DEFAULT_FLOAT_DTYPE)
        return maps, {}, self._description_text()

    def _scale_is_pinned(self):
        """Report whether the effect-size scale is pinned well enough for an absolute map.

        Only two things pin it: image studies in this collection, or a caller supplying the
        constant outright. Images must number at least ``_MIN_SCALE_DONORS``, so that the
        spread of their individual estimates is measurable and the caller can see how well
        determined the constant is rather than taking one study's word for it.
        """
        if getattr(self, "scale_source_", "unset") == "supplied":
            return True
        return (
            getattr(self, "scale_source_", "unset") == "images"
            and getattr(self, "n_scale_donors_", 0) >= _MIN_SCALE_DONORS
        )

    # -------------------------------------------------------------- correction

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
            f"Family-wise error rate correction was performed with a {scope} permutation "
            f"procedure using {n_iters} iterations, in which the reported foci were reassigned "
            "to each other's locations with the locations themselves held fixed."
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
        if self.null_method == "permute-magnitudes" and self._null_is_usable():
            inference = (
                " Uncorrected p-values were obtained from a permutation null distribution, in "
                "which each analysis's reported effect sizes were reassigned among its own "
                f"reported locations {self.n_iters} times with the locations themselves held "
                "fixed, each voxel being referred to its own null. The test is therefore of "
                "whether the effects reported near a voxel are larger than those the same "
                "studies reported elsewhere, not of whether the foci converge there and not "
                "of whether the effect is zero."
            )
        elif self.null_method == "permute-magnitudes":
            inference = (
                " No null distribution was computed, because the reported foci admit too few "
                "within-analysis arrangements to test, so no p-values are reported."
            )
        else:
            inference = " No null distribution was computed, so no p-values are reported."
        # The peak-height diagnostic goes in the description, not only the log. A run whose
        # reported heights are indistinguishable from noise peaks cannot support a magnitude,
        # and it was measured overestimating by a factor of 10.7 on one collection while
        # logging that fact among a dozen other lines. The description is what ends up in a
        # methods section, so that is where the caveat has to be.
        interval = getattr(self, "scale_interval_", None)
        if interval is None:
            bounds = ""
        else:
            bounds = (
                f" The overall effect-size scale carries a 95% confidence interval of "
                f"{interval[0]:.2f} to {interval[1]:.2f} times the value used, estimated from "
                "the image studies that calibrated it, so the magnitudes should be read as an "
                "order of scale rather than a calibrated value."
            )
        information = getattr(self, "peak_information_", None)
        if information is None:
            diagnostic = ""
        elif information["excess_z"] < _MIN_PEAK_EXCESS_Z:
            diagnostic = (
                " The reported peak heights averaged z = "
                f"{information['observed_mean_z']:.2f} against "
                f"{information['null_peak_mean_z']:.2f} expected for peaks of pure noise at "
                "the same reporting threshold, an excess of "
                f"{information['excess_z']:+.2f}. Their magnitudes therefore carry little "
                "information about the size of the effect, and **the effect-size values should "
                "be read as a relative map only**; the spatial pattern, which is driven by "
                "where the peaks are rather than how large they are, is unaffected."
            )
        else:
            diagnostic = (
                " The reported peak heights averaged z = "
                f"{information['observed_mean_z']:.2f} against "
                f"{information['null_peak_mean_z']:.2f} expected for peaks of pure noise at "
                "the same reporting threshold, an excess of "
                f"{information['excess_z']:+.2f}, so they carry information about the size of "
                "the effect beyond having cleared a threshold."
            )
        return (
            "A coordinate-based effect-size meta-analysis was performed with NiMARE "
            f"{__version__} (RRID:SCR_017398; \\citealt{{Salo2023}}). Each reported peak "
            f"statistic was converted to Hedges' g using the study's sample size and a "
            f"{self.design} design, and peaks were assigned spatial uncertainty with "
            f"{kernel_description}. Voxel-wise pooling used {heterogeneity}.{selection}"
            f"{bias}{inference}{bounds}{diagnostic} "
            f"The input dataset included {n_foci} foci with reported statistics from "
            f"{n_studies} experiments."
        )
