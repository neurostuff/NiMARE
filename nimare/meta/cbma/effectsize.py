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
survived a within-study threshold, so pooling them naively is a spatial winner's curse.
``selection_model="censored"`` replaces the weighted mean with the maximizer of a weighted
*censored* (Tobit) log-likelihood,

.. math::

    \\ell(\\mu; v) = \\sum_k \\Big[ w_k(v)\\,\\log \\phi_{\\sigma_k}(g_k - \\mu)
                    + (1 - w_k(v))\\, \\log P(|\\hat{g}_k| < c_k \\mid \\mu) \\Big],

in which a study that reported nothing near :math:`v` contributes the *probability that it
would have reported nothing* rather than an imputed value. No effect-size images are imputed
at any point; non-reporting enters only through that censoring probability, as in
:footcite:t:`tench2017coordinate`.

References
----------
.. footbibliography::
"""

import logging

import numpy as np
import pandas as pd
from joblib import Memory
from nilearn.maskers import NiftiMasker
from scipy import stats

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.utils import (
    _get_mask_flat_to_masked,
    _kernel_to_sparse_support,
    get_ale_kernel,
    sphere_kernel_offsets,
)
from nimare.transforms import d_to_g, t_to_d, t_to_z, z_to_t
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _add_metadata_to_dataframe,
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
    max_iter : :obj:`int`, default=25
        Maximum Newton iterations for the censored likelihood.
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

    Warnings
    --------
    This estimator is new and its inference has not been validated against a reference
    implementation. The parametric p-values assume the local weighted estimate is normal with
    the reported standard error; with ``selection_model="none"`` they also ignore the
    thresholding that generated the peaks, and so are anticonservative. Prefer
    :meth:`correct_fwe_montecarlo`.

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
        max_iter=25,
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
        self.max_iter = max_iter

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
        """Sparse (offsets, weights) for the spatial-uncertainty kernel, peak-normalized."""
        mask_img = self.masker.mask_img
        if self.fwhm is not None:
            _, kernel = get_ale_kernel(mask_img, fwhm=self.fwhm)
        else:
            _, kernel = get_ale_kernel(mask_img, sample_size=sample_size)
        offsets, values = _kernel_to_sparse_support(kernel / kernel.max())
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
        shape = np.asarray(mask_img.shape, dtype=np.int64)
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
        shape = np.asarray(mask_img.shape, dtype=np.int64)
        mask_flat_to_masked = _get_mask_flat_to_masked(mask_img)

        radius = self.coverage_radius
        if radius is None:
            radius = 2.0 * (self.fwhm if self.fwhm is not None else 10.0)
        offsets = sphere_kernel_offsets(radius, mask_img.header.get_zooms()[:3])

        active_lookup = np.full(n_voxels, -1, dtype=np.int64)
        active_lookup[active] = np.arange(active.size)

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
            local = np.unique(active_lookup[reached])
            local = local[local >= 0]
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
        """EM for one block of voxels. Returns ``(mu, prevalence, se)``, one value per voxel."""
        zero_inflated = self.selection_model == "zero-inflated"

        reports = weights > 0
        # Covered but out of kernel range: the study said something about this region but
        # nothing about this voxel. It informs neither term.
        silent = ~covered

        sigma_reported = np.sqrt(var_obs + tau2)
        sigma_silent = np.sqrt(null_var + tau2)
        sigma_null = np.sqrt(np.broadcast_to(null_var, silent.shape))

        weight_reported = np.where(reports, weights, 0.0)

        # A reporting study's log-likelihood is discounted by the spatial kernel, so a silent
        # study entering at full weight would count for more than a study that actually
        # measured something -- silence would outvote evidence, and the estimate would sit well
        # below the truth however many studies reported. Put a silent study on the same footing
        # as an average reporting study at this voxel instead.
        n_reporting = reports.sum(axis=0)
        reporter_scale = np.divide(
            weight_reported.sum(axis=0),
            n_reporting,
            out=np.ones(weights.shape[1]),
            where=n_reporting > 0,
        )
        weight_silent = silent.astype(float) * reporter_scale

        # Probability a silent study stays silent when it has no effect at all. Fixed across
        # iterations, and close to one whenever the threshold is the usual several sigma.
        prob_silent_null = np.clip(
            stats.norm.cdf(cutoffs / sigma_null) - stats.norm.cdf(-cutoffs / sigma_null),
            1e-12,
            None,
        )
        density_null = stats.norm.pdf(g_obs / np.sqrt(var_obs)) / np.sqrt(var_obs)

        mu = start.copy()
        pi = np.full(mu.shape, 0.5 if zero_inflated else 1.0)
        total_weight = weight_reported.sum(axis=0) + weight_silent.sum(axis=0)

        for _ in range(self.max_iter):
            if zero_inflated:
                density_effect = stats.norm.pdf((g_obs - mu) / sigma_reported) / sigma_reported
                resp_reported = pi * density_effect
                resp_reported /= resp_reported + (1.0 - pi) * density_null + 1e-300

                prob_silent_effect = self._silence_probability(mu, cutoffs, sigma_silent)
                resp_silent = pi * prob_silent_effect
                resp_silent /= resp_silent + (1.0 - pi) * prob_silent_null + 1e-300

                numerator = (weight_reported * resp_reported).sum(axis=0) + (
                    weight_silent * resp_silent
                ).sum(axis=0)
                pi = np.clip(
                    np.divide(
                        numerator,
                        total_weight,
                        out=np.zeros_like(mu),
                        where=total_weight > 0,
                    ),
                    1e-4,
                    1.0 - 1e-4,
                )
            else:
                resp_reported = np.ones_like(g_obs)
                resp_silent = np.ones_like(g_obs)

            score, curvature = self._mu_derivatives(
                mu,
                g_obs=g_obs,
                sigma_reported=sigma_reported,
                sigma_silent=sigma_silent,
                cutoffs=cutoffs,
                weight_reported=weight_reported * resp_reported,
                weight_silent=weight_silent * resp_silent,
            )
            step = np.where(curvature < 0, -score / curvature, 0.0)
            # The likelihood is concave but flat far from the data; cap the step so a voxel
            # with almost no reporting weight cannot run away.
            mu = mu + np.clip(step, -1.0, 1.0)
            if np.max(np.abs(step)) < 1e-6:
                break

        _, curvature = self._mu_derivatives(
            mu,
            g_obs=g_obs,
            sigma_reported=sigma_reported,
            sigma_silent=sigma_silent,
            cutoffs=cutoffs,
            weight_reported=weight_reported * resp_reported,
            weight_silent=weight_silent * resp_silent,
        )
        se = np.full(mu.shape, np.inf)
        informative = curvature < 0
        se[informative] = 1.0 / np.sqrt(-curvature[informative])
        return mu, pi, se

    @staticmethod
    def _silence_probability(mu, cutoffs, sigma):
        """P(|g| < c | mu), the chance a study with effect ``mu`` reports nothing."""
        return np.clip(
            stats.norm.cdf((cutoffs - mu) / sigma) - stats.norm.cdf((-cutoffs - mu) / sigma),
            1e-12,
            None,
        )

    @staticmethod
    def _mu_derivatives(
        mu, *, g_obs, sigma_reported, sigma_silent, cutoffs, weight_reported, weight_silent
    ):
        """First and second derivatives of the weighted log-likelihood with respect to mu."""
        precision = 1.0 / sigma_reported**2
        score = (weight_reported * (g_obs - mu) * precision).sum(axis=0)
        curvature = -(weight_reported * precision).sum(axis=0)

        upper = (cutoffs - mu) / sigma_silent
        lower = (-cutoffs - mu) / sigma_silent
        prob = np.clip(stats.norm.cdf(upper) - stats.norm.cdf(lower), 1e-12, None)
        pdf_upper = stats.norm.pdf(upper)
        pdf_lower = stats.norm.pdf(lower)

        d_prob = -(pdf_upper - pdf_lower) / sigma_silent
        d2_prob = -(upper * pdf_upper - lower * pdf_lower) / sigma_silent**2
        censor_score = d_prob / prob

        score = score + (weight_silent * censor_score).sum(axis=0)
        curvature = curvature + (weight_silent * (d2_prob / prob - censor_score**2)).sum(axis=0)
        return score, curvature

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator."
            )

        table = self._build_focus_table()
        self._focus_table_ = table

        fit = self._pool(table)
        if self.selection_model != "none":
            self._sample_sizes_ = self._all_sample_sizes(dataset)
            self._thresholds_ = self._study_thresholds(table, self._sample_sizes_)
            self._apply_selection_model(fit, table, self._thresholds_, self._sample_sizes_)

        z_values = np.divide(
            fit["g"], fit["se"], out=np.zeros_like(fit["g"]), where=np.isfinite(fit["se"])
        )
        p_values = stats.norm.sf(np.abs(z_values)) * 2.0
        p_values[~fit["covered"]] = 1.0
        z_values[~fit["covered"]] = 0.0

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

    def correct_fwe_montecarlo(self, result, n_iters=1000, n_cores=1, seed=0):
        """Voxel-level FWE correction by relocating foci within the mask.

        Each iteration moves every focus to a uniformly drawn in-mask voxel, carrying its
        effect size and variance with it, and refits. The maximum ``|z|`` over the brain builds
        the null. This tests the same null hypothesis as the convergence-based estimators --
        that reported coordinates fall at random within the mask -- but with a statistic that
        is sensitive to effect-size magnitude rather than coordinate density.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result of a previous :meth:`fit`.
        n_iters : :obj:`int`, default=1000
            Number of permutations.
        n_cores : :obj:`int`, default=1
            Unused; present for interface compatibility with the other CBMA estimators.
        seed : :obj:`int`, default=0
            Seed for the relocation draws.

        Returns
        -------
        maps, tables, description
        """
        del n_cores  # single-threaded for now; kept so FWECorrector can pass it through.

        table = getattr(self, "_focus_table_", None)
        if table is None:
            raise ValueError("correct_fwe_montecarlo requires a fitted estimator.")

        mask_img = self.masker.mask_img
        in_mask_ijk = np.argwhere(mask_img.get_fdata() > 0)
        rng = np.random.default_rng(seed)

        observed_z = np.abs(result.maps["z"])
        null_max = np.empty(n_iters, dtype=float)

        permuted = table.copy()
        for i_iter in range(n_iters):
            draws = rng.integers(0, len(in_mask_ijk), size=len(permuted))
            permuted[["i", "j", "k"]] = in_mask_ijk[draws]
            fit = self._pool(permuted)
            with np.errstate(invalid="ignore", divide="ignore"):
                z_null = np.divide(
                    fit["g"], fit["se"], out=np.zeros_like(fit["g"]), where=np.isfinite(fit["se"])
                )
            null_max[i_iter] = np.max(np.abs(z_null)) if z_null.size else 0.0

        p_corrected = (1 + np.sum(null_max[None, :] >= observed_z[:, None], axis=1)) / (
            1 + n_iters
        )
        z_corrected = stats.norm.isf(np.clip(p_corrected, 1e-16, 1.0) / 2.0) * np.sign(
            result.maps["z"]
        )

        self.null_distributions_ = getattr(self, "null_distributions_", {})
        self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"] = null_max

        maps = {
            "p": p_corrected.astype(DEFAULT_FLOAT_DTYPE),
            "z": z_corrected.astype(DEFAULT_FLOAT_DTYPE),
            "logp": _nlogp_to_logp_values(np.log(np.clip(p_corrected, 1e-300, None))),
        }
        description = (
            "Family-wise error rate correction was performed with a voxel-level Monte Carlo "
            f"procedure using {n_iters} iterations, in which every focus was relocated to a "
            "uniformly drawn voxel within the analysis mask while retaining its effect size."
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
        return (
            "A coordinate-based effect-size meta-analysis was performed with NiMARE "
            f"{__version__} (RRID:SCR_017398; \\citealt{{Salo2023}}). Each reported peak "
            f"statistic was converted to Hedges' g using the study's sample size and a "
            f"{self.design} design, and peaks were assigned spatial uncertainty with "
            f"{kernel_description}. Voxel-wise pooling used {heterogeneity}.{selection} "
            f"The input dataset included {n_foci} foci with reported statistics from "
            f"{n_studies} experiments."
        )
