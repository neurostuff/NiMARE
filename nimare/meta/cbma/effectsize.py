"""Effect-size coordinate-based meta-analysis.

Unlike every other coordinate-based estimator in NiMARE, the estimator in this
module does not test for spatial convergence. It estimates *how large* the
effect is in a region, by meta-analysing the peak statistics reported alongside
the coordinates.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.special import ndtri
from tqdm.auto import tqdm

from nimare.estimator import Estimator
from nimare.meta.cbma._selection import (
    batch_loglikelihood,
    solve_delta,
    solve_delta_batch,
)
from nimare.stats import nlogp_fdr, null_to_p
from nimare.transforms import d_to_g, t_to_d, z_to_nlogp, z_to_t
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _add_metadata_to_dataframe,
    _check_ncores,
    _mask_coverage_to_null_ijk,
    _mask_img_to_bool,
    _nlogp_to_logp_values,
    get_masker,
    get_masker_mask_image,
    mm2vox,
    validate_coordinate_spaces,
    vox2mm,
)

LGR = logging.getLogger(__name__)

#: Statistic columns recognised on a coordinates table, in order of preference.
_STAT_COLUMNS = ("t_stat", "z_stat")

#: Smallest sample size for which Hedges' g has a defined variance: the exact
#: expression divides by ``N - 3``.
_MIN_SAMPLE_SIZE = 4

#: Passes of the model-based-variance fixed point before Aitken extrapolation
#: finishes the job, and the tolerance at which it is already converged.
#: Five passes plus extrapolation reaches ~1e-8 in Hedges' g units, against
#: ~1.5e-7 for eight plain passes -- more accurate, and two solves cheaper.
_VARIANCE_ITERS = 5
_VARIANCE_TOL = 1e-10

#: Newton tolerance on delta, in Hedges' g units.
_NEWTON_TOL = 1e-9

#: Grid points and refinement steps for the tau-squared profile search. The
#: refinement is what carries the accuracy: each evaluation is one batched
#: solve over every cluster, so spending more of them here is cheap.
_TAU2_GRID = 12
_TAU2_REFINE_ITERS = 20

#: Memory budget for one block of the focus-by-cluster distance matrix.
_DISTANCE_CHUNK_BYTES = 64_000_000

#: Largest condensed distance matrix `_cluster_foci` will allocate, in bytes.
#: 1 GB corresponds to roughly 16,000 foci; a single meta-analysis is rarely
#: anywhere near that.
_MAX_DISTANCE_MATRIX_BYTES = 1_000_000_000

#: Study statuses recorded in the ``study_contributions`` table.
_REPORTED, _CENSORED, _BUFFER, _MISSING = "reported", "censored", "buffer", "missing"


def _hedges_g(stat, sample_size, kind):
    """Convert peak statistics to Hedges' g.

    Parameters
    ----------
    stat : :class:`numpy.ndarray`
        Signed peak statistics.
    sample_size : :class:`numpy.ndarray`
        Per-focus sample sizes.
    kind : {"t_stat", "z_stat"}
        Which statistic ``stat`` holds. Z scores are routed through the exact
        Z-to-t conversion first, so both paths share one definition of g.

    Returns
    -------
    :class:`numpy.ndarray`
        Hedges' g for each focus.
    """
    sample_size = np.asarray(sample_size, dtype=float)
    t_values = np.asarray(stat, dtype=float)
    if kind == "z_stat":
        t_values = z_to_t(t_values, sample_size - 1)
    return d_to_g(t_to_d(t_values, sample_size), sample_size)


def _g_variance(delta, sample_size):
    """Return the model-based sampling variance of Hedges' g at ``delta``.

    Parameters
    ----------
    delta : :obj:`float` or :class:`numpy.ndarray`
        Population effect size.
    sample_size : :class:`numpy.ndarray`
        Per-study sample sizes.

    Returns
    -------
    :class:`numpy.ndarray`
        ``Var(g)`` evaluated at ``delta``.

    Notes
    -----
    The usual plug-in variance substitutes each study's *observed* effect size.
    That is unusable here for two reasons: a censored study has no observed
    effect size at all, and a reporting study's observed value is inflated by
    selection, so plugging it in inflates that study's variance and down-weights
    exactly the studies carrying signal. Evaluating the same expression at the
    current ``delta`` keeps every study on a common footing.
    """
    delta, n = np.broadcast_arrays(np.asarray(delta, dtype=float), np.asarray(sample_size, float))
    return np.asarray(d_to_g(delta, n, return_variance=True)[1], dtype=float)


def _dersimonian_laird(g, var):
    """Return the DerSimonian-Laird between-study variance estimate."""
    g, var = np.asarray(g, float), np.asarray(var, float)
    if g.size < 2:
        return 0.0
    weights = 1.0 / var
    mean = np.sum(weights * g) / np.sum(weights)
    q_stat = np.sum(weights * (g - mean) ** 2)
    denominator = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    if denominator <= 0:
        return 0.0
    return float(max(0.0, (q_stat - (g.size - 1)) / denominator))


def _i_squared(g, var, tau2):
    """Return the proportion of total variance attributable to heterogeneity."""
    if g.size < 2:
        return np.nan
    total = var + tau2
    return float(np.clip(tau2 / np.mean(total), 0.0, 1.0)) if np.mean(total) > 0 else np.nan


def _z_to_nlogp(z):
    """Return the natural log of the two-tailed p-value, NaN-safe.

    Kept in log space throughout so that a very large absolute z does not
    collapse to a p-value of zero before the log is taken.
    """
    z = np.asarray(z, dtype=float)
    nlogp = np.full(z.shape, np.nan)
    finite = np.isfinite(z)
    nlogp[finite] = z_to_nlogp(z[finite], tail="two")
    return nlogp


def _nlogp_to_p_and_logp(nlogp):
    """Return the p-value and -log10(p) matching a natural-log p-value."""
    finite = np.isfinite(nlogp)
    p = np.full(nlogp.shape, np.nan)
    p[finite] = np.exp(nlogp[finite])
    logp = np.full(nlogp.shape, np.nan)
    logp[finite] = _nlogp_to_logp_values(nlogp[finite])
    return p, logp


def _fdr_over_clusters(nlogp):
    """Benjamini-Hochberg adjust a vector of natural-log p-values, skipping NaN."""
    adjusted = np.full(nlogp.shape, np.nan)
    finite = np.isfinite(nlogp)
    if np.any(finite):
        adjusted[finite] = nlogp_fdr(nlogp[finite], method="bh")
    return adjusted


def _chunk_bounds(n_clusters, n_foci, budget=_DISTANCE_CHUNK_BYTES):
    """Yield ``(start, stop)`` cluster slices whose distance block fits in memory."""
    per_cluster = max(1, n_foci * np.dtype(np.float64).itemsize * 3)
    size = int(np.clip(budget // per_cluster, 1, n_clusters))
    for start in range(0, n_clusters, size):
        yield start, min(start + size, n_clusters)


def _cluster_foci(xyz, radius):
    """Group foci into spatial clusters by average-linkage agglomerative clustering.

    Parameters
    ----------
    xyz : :class:`numpy.ndarray` of shape (F, 3)
        Focus coordinates, in mm.
    radius : :obj:`float`
        Average-linkage distance cutoff, in mm.

    Returns
    -------
    :class:`numpy.ndarray` of shape (F,)
        Integer cluster label per focus, starting at 1.

    Raises
    ------
    ValueError
        If the condensed distance matrix would exceed
        ``_MAX_DISTANCE_MATRIX_BYTES``.

    Notes
    -----
    One algorithm is used at every size, deliberately. An earlier version
    switched to ``sklearn``'s ``AgglomerativeClustering`` with a
    ``radius_neighbors_graph`` connectivity constraint above a focus count, to
    avoid the O(F^2) distance matrix. That constraint changes the clustering
    outright rather than merely approximating it -- on the same 1,500 foci it
    returned 231 clusters against average linkage's 492, an adjusted Rand index
    of 0.08 -- so crossing the threshold silently changed the result of the
    meta-analysis. (Without the constraint ``sklearn`` and ``scipy`` agree
    exactly, but then neither saves any memory.) Refusing to run is the honest
    behaviour; quietly answering a different question is not.
    """
    n_foci = xyz.shape[0]
    if n_foci < 2:
        return np.ones(n_foci, dtype=int)

    required = n_foci * (n_foci - 1) // 2 * np.dtype(np.float64).itemsize
    if required > _MAX_DISTANCE_MATRIX_BYTES:
        raise ValueError(
            f"Clustering {n_foci} foci needs a {required / 1e9:.1f} GB distance matrix, over "
            f"the {_MAX_DISTANCE_MATRIX_BYTES / 1e9:.1f} GB limit. Increase `radius` to merge "
            "foci sooner, split the analysis (for example by contrast type), or restrict the "
            "collection. NiMARE will not silently substitute a different clustering algorithm, "
            "because that would change the meta-analysis result rather than just its speed."
        )

    return fcluster(linkage(pdist(xyz), method="average"), t=radius, criterion="distance")


def _solve_with_variance(g, sample_size, thresh, reported, tau2, two_sided, likelihood, n_iter=8):
    """Solve one cluster for ``delta``, re-evaluating the model-based variance.

    This is the readable single-cluster reference. Fits go through
    :func:`_solve_batch_with_variance`, which solves every cluster in one
    vectorized pass; the tests assert the two agree.

    Returns
    -------
    delta, se, converged
    """
    weights = 1.0 / np.maximum(_g_variance(g, sample_size), 1e-12)
    if np.any(reported):
        delta = float(np.sum(g[reported] * weights[reported]) / np.sum(weights[reported]))
    else:
        delta = 0.0

    se, converged = np.inf, False
    for _ in range(n_iter):
        s = np.sqrt(np.maximum(_g_variance(delta, sample_size), 1e-12) + tau2)
        new_delta, se, converged = solve_delta(
            g, s, thresh, reported, two_sided=two_sided, likelihood=likelihood
        )
        if not np.isfinite(new_delta):
            return np.nan, np.inf, False
        moved = abs(new_delta - delta)
        delta = new_delta
        if moved < 1e-9:
            break
    return delta, se, converged


def _pack_clusters(clusters):
    """Pad the per-cluster arrays into rectangular ``(M, K)`` blocks.

    Every cluster draws on the same study roster minus whichever studies were
    excluded, so the widths barely differ and padding costs almost nothing.

    Returns
    -------
    :obj:`dict`
        Keys ``g``, ``sample_size``, ``thresh``, ``reported`` and ``include``.
    """
    width = max(cluster["g"].size for cluster in clusters)
    shape = (len(clusters), width)
    packed = {
        "g": np.zeros(shape),
        # Padding sample sizes must stay above the N > 3 floor so that the
        # variance of Hedges' g is finite even in masked-out columns.
        "sample_size": np.full(shape, 10.0),
        "thresh": np.zeros(shape),
        "reported": np.zeros(shape, dtype=bool),
        "include": np.zeros(shape, dtype=bool),
    }
    for row, cluster in enumerate(clusters):
        k = cluster["g"].size
        packed["g"][row, :k] = cluster["g"]
        packed["sample_size"][row, :k] = cluster["sample_size"]
        packed["thresh"][row, :k] = cluster["thresh"]
        packed["reported"][row, :k] = cluster["reported"]
        packed["include"][row, :k] = True
    return packed


def _initial_delta(packed):
    """Inverse-variance mean of the reported effect sizes, per cluster."""
    usable = packed["include"] & packed["reported"]
    weights = np.where(
        usable, 1.0 / np.maximum(_g_variance(packed["g"], packed["sample_size"]), 1e-12), 0.0
    )
    total = np.sum(weights, axis=1)
    return np.divide(
        np.sum(np.where(usable, packed["g"] * weights, 0.0), axis=1),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )


def _solve_batch_with_variance(
    packed, tau2, two_sided, likelihood, init=None, n_iter=_VARIANCE_ITERS
):
    """Solve every cluster at once, re-evaluating the model-based variance.

    Parameters
    ----------
    packed : :obj:`dict`
        Output of :func:`_pack_clusters`.
    tau2 : :obj:`float` or :class:`numpy.ndarray` of shape (M,)
        Between-study variance, shared or per cluster.
    init : :class:`numpy.ndarray` of shape (M,), optional
        Starting values, e.g. the solution at the previous ``tau2``.
    n_iter : :obj:`int`, optional
        Maximum fixed-point passes before extrapolation.

    Returns
    -------
    delta, se, converged : :class:`numpy.ndarray` of shape (M,)

    Notes
    -----
    The sampling variance of Hedges' g depends on ``delta``, so this is a fixed
    point, and it converges only *linearly* -- empirically about one decimal
    digit per pass. Iterating to machine precision would therefore cost ten-odd
    solves. Because the convergence is clean geometric decay, Aitken's delta-
    squared extrapolation recovers the limit from three successive iterates
    instead, and a final pass from the extrapolated point supplies the standard
    error. That reaches better accuracy than plain iteration for roughly half
    the work.
    """
    tau2 = np.broadcast_to(np.asarray(tau2, dtype=float), (packed["g"].shape[0],))
    delta = _initial_delta(packed) if init is None else np.array(init, dtype=float)

    def _pass(current):
        variance = np.maximum(_g_variance(current[:, None], packed["sample_size"]), 1e-12)
        return solve_delta_batch(
            packed["g"],
            np.sqrt(variance + tau2[:, None]),
            packed["thresh"],
            packed["reported"],
            packed["include"],
            two_sided=two_sided,
            likelihood=likelihood,
            tol=_NEWTON_TOL,
        )

    history = []
    se = np.full(delta.shape, np.inf)
    converged = np.zeros(delta.shape, dtype=bool)
    for _ in range(n_iter):
        updated, se, converged = _pass(delta)
        finite = np.isfinite(updated)
        moved = np.max(np.abs(updated[finite] - delta[finite])) if np.any(finite) else 0.0
        delta = np.where(finite, updated, np.nan)
        history.append(delta)
        if moved < _VARIANCE_TOL:
            return delta, se, converged

    # Aitken's delta-squared on the last three iterates. Applied only where the
    # denominator is safely non-zero; elsewhere the plain iterate stands.
    if len(history) >= 3:
        first, second, third = history[-3], history[-2], history[-1]
        denominator = third - 2.0 * second + first
        with np.errstate(invalid="ignore", divide="ignore"):
            extrapolated = third - (third - second) ** 2 / denominator
        usable = np.isfinite(extrapolated) & (np.abs(denominator) > 1e-14)
        delta = np.where(usable, extrapolated, third)
        updated, se, converged = _pass(delta)
        delta = np.where(np.isfinite(updated), updated, delta)

    return delta, se, converged


def _batch_profile_loglik(packed, tau2, two_sided, likelihood, init=None):
    """Return the profile log-likelihood per cluster at a given ``tau2``."""
    delta, _, _ = _solve_batch_with_variance(packed, tau2, two_sided, likelihood, init=init)
    tau2 = np.broadcast_to(np.asarray(tau2, dtype=float), (packed["g"].shape[0],))
    variance = np.maximum(_g_variance(delta[:, None], packed["sample_size"]), 1e-12)
    s = np.sqrt(variance + tau2[:, None])
    values = batch_loglikelihood(
        np.nan_to_num(delta),
        packed["g"],
        s,
        packed["thresh"],
        packed["reported"],
        packed["include"],
        two_sided=two_sided,
        likelihood=likelihood,
    )
    return np.where(np.isfinite(delta), values, -np.inf), delta


def _profile_tau2(packed, two_sided, likelihood, upper, scope):
    """Maximize the profile likelihood over ``tau2``.

    Parameters
    ----------
    packed : :obj:`dict`
        Output of :func:`_pack_clusters`.
    upper : :obj:`float`
        Top of the search range.
    scope : {"global", "cluster"}
        Whether one ``tau2`` is shared across clusters or fitted per cluster.

    Returns
    -------
    :class:`numpy.ndarray` of shape (M,)
        The chosen ``tau2`` for each cluster.

    Notes
    -----
    A grid pass comes first because the maximum sits on the ``tau2 = 0``
    boundary often enough that a purely interior search would miss it. Each grid
    point costs one batched solve for *all* clusters, warm-started from the
    previous point, so the grid is cheap; ``tau2`` only enters through
    ``sqrt(v + tau2)``, and resolving it more finely than this buys nothing.
    """
    n_clusters = packed["g"].shape[0]
    if upper <= 0:
        return np.zeros(n_clusters)

    grid = np.concatenate([[0.0], np.geomspace(upper * 1e-4, upper, _TAU2_GRID - 1)])
    curve = np.empty((grid.size, n_clusters))
    warm = None
    for index, value in enumerate(grid):
        curve[index], warm = _batch_profile_loglik(packed, value, two_sided, likelihood, init=warm)

    if scope == "cluster":
        return grid[np.argmax(curve, axis=0)]

    totals = np.sum(np.where(np.isfinite(curve), curve, -1e300), axis=1)
    best = int(np.argmax(totals))
    if best == 0:
        return np.zeros(n_clusters)

    lo, hi = grid[best - 1], grid[min(best + 1, grid.size - 1)]
    result = minimize_scalar(
        lambda t: -np.sum(_batch_profile_loglik(packed, max(t, 0.0), two_sided, likelihood)[0]),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": max(upper * 1e-3, 1e-8), "maxiter": _TAU2_REFINE_ITERS},
    )
    chosen = float(max(0.0, result.x)) if result.success else float(grid[best])
    return np.full(n_clusters, chosen)


class CoordinateEffectSize(Estimator):
    """Estimate effect size, rather than convergence, from coordinates.

    .. versionadded:: 0.16.0

    Every other coordinate-based estimator in NiMARE tests *H0: foci are
    spatially random*. This one tests *H0: delta = 0 in this region*, and
    reports an interpretable effect size -- Hedges' g -- with a confidence
    interval and a heterogeneity estimate. It is a genuine random-effects
    meta-analysis that happens to run on coordinates.

    The unit of inference is a **spatial cluster of foci**, not a voxel,
    following :footcite:t:`tench2017coordinate` and the censored-likelihood
    treatment of :footcite:t:`costafreda2012parametric`. Nothing is invented at voxels
    where no study reported anything, and the multiplicity burden drops from
    ~200,000 voxels to a few dozen clusters.

    **The idea that makes this work without imputation.** A study that reports
    no peak in a region has not given us *no* information: it has told us its
    statistic there fell below its reporting threshold. That is a left-censored
    observation, so the unobserved values can be integrated out in closed form
    (see :mod:`nimare.meta.cbma._selection`). Treating non-reporting studies as
    zeros biases the estimate toward the null; dropping them biases it away.

    .. warning::
        **The estimate is of the effect size at the reported peak, not the mean
        effect over the region, and it remains biased upward.** Conditioning on
        "cleared the threshold" does not correct for the fact that a reported
        peak is the *maximum* over a search region. For a 10 mm radius and
        typical smoothness that inflation is on the order of ``1.4 / sqrt(N)``
        -- roughly 0.3 in g units at N=20, which can rival the effect being
        estimated. The ``bias_peak`` column of the cluster table reports this
        magnitude per cluster. Modelling it properly needs the random-field
        peak-height distribution :footcite:p:`durnez2016power`; that is not
        implemented here.

    Parameters
    ----------
    radius : :obj:`float`, default=10.0
        Clustering cutoff and region radius, in mm. Foci within this distance
        of a cluster centroid count as reported there.
    buffer : :obj:`float` or None, optional
        Studies whose nearest focus falls between ``radius`` and ``buffer`` mm
        of a centroid are *excluded* from that cluster rather than treated as
        censored -- they plainly did report something nearby, so asserting they
        saw nothing would be wrong. Default is ``2 * radius``.
    min_studies : :obj:`int`, default=4
        Minimum number of reporting studies for a cluster to be fitted.
    statistic : {"auto", "t_stat", "z_stat"}, default="auto"
        Which per-coordinate statistic column to use. ``"auto"`` prefers
        ``t_stat`` and falls back to ``z_stat``.
    tail : {"auto", "one", "two"}, default="auto"
        Whether effects of either sign were reportable. ``"auto"`` picks
        ``"two"`` when both signs appear among the reported statistics and
        ``"one"`` otherwise. Getting this wrong matters: with positive-only
        data and ``tail="two"``, a study with a large negative effect would be
        scored as censored.
    likelihood : {"censored", "conditional"}, default="censored"
        ``"censored"`` is the full-data likelihood and is correct whenever the
        roster of studies is known, which it is here. ``"conditional"`` drops
        non-reporting studies and truncates the reporters instead; use it only
        when non-reporting studies cannot be enumerated.
    threshold : {"infer"}, :obj:`float`, or :obj:`str`, default="infer"
        Each study's reporting threshold. ``"infer"`` uses that study's own
        smallest reported absolute statistic, which is an upper bound on its true
        threshold and therefore biases the estimate *upward*. A float is a
        constant in ``threshold_units``. Any other string names a metadata
        field holding a per-study threshold.
    threshold_units : {"z", "t", "g"}, default="z"
        Units of a numeric or metadata-supplied ``threshold``.
    tau2 : {"profile", "dl", "fixed"}, default="profile"
        Between-study variance. ``"profile"`` maximizes the censored profile
        likelihood. ``"dl"`` applies DerSimonian-Laird to the reported values
        only, which is anti-conservative under heavy censoring. ``"fixed"``
        sets it to zero.
    tau2_scope : {"global", "cluster"}, default="global"
        Whether ``"profile"`` estimates one shared ``tau2`` across clusters or
        one per cluster. With the handful of reporting studies a typical
        cluster has, a per-cluster estimate is mostly noise; ``"global"``
        trades a little bias for a large variance reduction.
    within_study : {"nearest", "max"}, default="nearest"
        Which focus to use when a study reports several in one cluster.
        ``"max"`` takes the largest absolute g, which adds a second layer of
        maximum-selection bias on top of the one already present, so
        ``"nearest"`` (to the centroid) is the default.
    assumed_fwhm : :obj:`float`, default=8.0
        Assumed spatial smoothness, in mm, used only to size the ``bias_peak``
        diagnostic -- it does not enter the estimate.
    n_iters : :obj:`int`, default=0
        Monte Carlo iterations for the false-cluster-discovery null. Zero
        disables it.
    n_cores : :obj:`int`, default=1
        Cores for the Monte Carlo null. ``-1`` uses all available.
    mask : str, :class:`nibabel.Nifti1Image`, or masker, optional
        Mask defining the analysis volume.
    mask_coverage : {"gm", "brain"}, default="gm"
        Voxel set from which Monte Carlo null foci are drawn.

    Notes
    -----
    Available correction methods:
    :meth:`CoordinateEffectSize.correct_fdr_cluster`.

    Studies contributing no coordinates at all cannot currently be represented
    in a NiMARE collection, so the roster of studies used for censoring is the
    set of studies with at least one focus somewhere in the brain.

    Clustering builds a full pairwise distance matrix, so collections beyond
    roughly 16,000 foci are refused with an explanatory error rather than
    silently clustered by a different algorithm. A larger ``radius`` or a split
    analysis is the remedy; a single meta-analysis is rarely near that size.

    References
    ----------
    .. footbibliography::
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        radius=10.0,
        buffer=None,
        min_studies=4,
        statistic="auto",
        tail="auto",
        likelihood="censored",
        threshold="infer",
        threshold_units="z",
        tau2="profile",
        tau2_scope="global",
        within_study="nearest",
        assumed_fwhm=8.0,
        n_iters=0,
        n_cores=1,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        generate_description=True,
        *,
        mask=None,
        mask_coverage="gm",
    ):
        if radius <= 0:
            raise ValueError(f"radius must be positive; got {radius!r}.")
        if buffer is not None and buffer < radius:
            raise ValueError(f"buffer ({buffer!r}) must be at least radius ({radius!r}).")
        if statistic not in ("auto", *_STAT_COLUMNS):
            raise ValueError(f"statistic must be 'auto', 't_stat' or 'z_stat'; got {statistic!r}.")
        if tail not in ("auto", "one", "two"):
            raise ValueError(f"tail must be 'auto', 'one' or 'two'; got {tail!r}.")
        if likelihood not in ("censored", "conditional"):
            raise ValueError(
                f"likelihood must be 'censored' or 'conditional'; got {likelihood!r}."
            )
        if threshold_units not in ("z", "t", "g"):
            raise ValueError(f"threshold_units must be 'z', 't' or 'g'; got {threshold_units!r}.")
        if tau2 not in ("profile", "dl", "fixed"):
            raise ValueError(f"tau2 must be 'profile', 'dl' or 'fixed'; got {tau2!r}.")
        if tau2_scope not in ("global", "cluster"):
            raise ValueError(f"tau2_scope must be 'global' or 'cluster'; got {tau2_scope!r}.")
        if within_study not in ("nearest", "max"):
            raise ValueError(f"within_study must be 'nearest' or 'max'; got {within_study!r}.")
        if assumed_fwhm <= 0:
            raise ValueError(f"assumed_fwhm must be positive; got {assumed_fwhm!r}.")
        if mask_coverage not in ("gm", "brain"):
            raise ValueError(f"mask_coverage must be 'gm' or 'brain'; got {mask_coverage!r}.")

        self.radius = float(radius)
        self.buffer = float(buffer) if buffer is not None else 2.0 * float(radius)
        self.min_studies = int(min_studies)
        self.statistic = statistic
        self.tail = tail
        self.likelihood = likelihood
        self.threshold = threshold
        self.threshold_units = threshold_units
        self.tau2 = tau2
        self.tau2_scope = tau2_scope
        self.within_study = within_study
        self.assumed_fwhm = float(assumed_fwhm)
        self.n_iters = int(n_iters)
        self.n_cores = _check_ncores(n_cores)
        self.mask_coverage = mask_coverage
        self.masker = get_masker(mask, memory=memory, memory_level=memory_level) if mask else None

        super().__init__(
            memory=memory, memory_level=memory_level, generate_description=generate_description
        )

    def _preprocess_input(self, dataset):
        """Attach the mask, matrix indices and sample sizes to the coordinates."""
        validate_coordinate_spaces(self.inputs_["coordinates"])
        masker, mask_img = get_masker_mask_image(
            self.masker,
            dataset=dataset,
            message=(
                "A masker is required for coordinate-based meta-analysis. "
                "Provide a `mask` to the Estimator (e.g., CoordinateEffectSize(mask=...)) or "
                "initialize the Dataset with a `target` and/or `mask` so `dataset.masker` is "
                "defined."
            ),
        )
        self.masker = masker

        coordinates = self.inputs_["coordinates"]
        coordinates[["i", "j", "k"]] = mm2vox(coordinates[["x", "y", "z"]].values, mask_img.affine)
        self.inputs_["coordinates"] = _add_metadata_to_dataframe(
            dataset,
            coordinates,
            metadata_field=("sample_sizes", "sample_size"),
            target_column="sample_size",
            filter_func=np.mean,
        )

    def _resolve_statistic(self, coordinates):
        """Pick the statistic column to meta-analyse, or explain why none works."""
        if self.statistic != "auto":
            if self.statistic not in coordinates.columns:
                raise ValueError(
                    f"statistic={self.statistic!r} was requested, but the coordinates have no "
                    f"{self.statistic!r} column. Available columns: "
                    f"{sorted(coordinates.columns)}."
                )
            return self.statistic

        for column in _STAT_COLUMNS:
            if column in coordinates.columns and coordinates[column].notna().any():
                return column

        raise ValueError(
            "CoordinateEffectSize needs a per-coordinate statistic, but the coordinates have "
            f"neither a 't_stat' nor a 'z_stat' column with usable values (found: "
            f"{sorted(coordinates.columns)}). Datasets converted from Sleuth/BrainMap text "
            "files carry coordinates only; use ALE or MKDA for those, or supply peak "
            "statistics via NIMADS point values."
        )

    def _prepare_foci(self, coordinates):
        """Build the per-focus and per-study frames the model needs."""
        stat_column = self._resolve_statistic(coordinates)
        foci = coordinates[["id", "x", "y", "z", "sample_size"]].copy()
        foci["stat"] = pd.to_numeric(coordinates[stat_column], errors="coerce")
        foci["sample_size"] = pd.to_numeric(foci["sample_size"], errors="coerce")

        too_small = foci.groupby("id")["sample_size"].max() < _MIN_SAMPLE_SIZE
        if too_small.any():
            dropped = sorted(too_small.index[too_small])
            warnings.warn(
                f"Dropping {len(dropped)} study/studies whose sample size is missing or below "
                f"{_MIN_SAMPLE_SIZE}, for which Hedges' g has no defined variance: "
                f"{', '.join(map(str, dropped[:10]))}"
                f"{', ...' if len(dropped) > 10 else ''}.",
                stacklevel=2,
            )
            foci = foci[~foci["id"].isin(dropped)]

        if foci.empty:
            raise ValueError(
                "No studies have both coordinates and a usable sample size "
                f"(>= {_MIN_SAMPLE_SIZE})."
            )

        foci["valid"] = foci["stat"].notna()
        if not foci["valid"].any():
            raise ValueError(
                f"No coordinate has a usable {stat_column!r} value, so no effect size can be "
                "computed."
            )
        if not foci["valid"].all():
            n_missing = int((~foci["valid"]).sum())
            LGR.warning(
                f"{n_missing} of {len(foci)} coordinates have no {stat_column!r} value. Their "
                "studies are excluded from -- not censored in -- the clusters those "
                "coordinates fall in, since the study did report something there."
            )

        foci["g"] = np.nan
        is_valid = foci["valid"].values
        foci.loc[is_valid, "g"] = _hedges_g(
            foci.loc[is_valid, "stat"].values,
            foci.loc[is_valid, "sample_size"].values,
            stat_column,
        )
        foci = foci[~(is_valid & ~np.isfinite(foci["g"].values))]
        return foci, stat_column

    def _resolve_tail(self, foci):
        """Decide whether effects of both signs were reportable."""
        if self.tail != "auto":
            return self.tail == "two"
        g = foci.loc[foci["valid"], "g"].values
        return bool(np.any(g > 0) and np.any(g < 0))

    def _study_thresholds(self, foci, stat_column, dataset):
        """Return each study's reporting threshold, in Hedges' g units."""
        valid = foci[foci["valid"]]
        sample_size = valid.groupby("id")["sample_size"].first()
        smallest_reported = valid.assign(absg=valid["g"].abs()).groupby("id")["absg"].min()

        if isinstance(self.threshold, str) and self.threshold == "infer":
            thresholds = smallest_reported.copy()
        elif isinstance(self.threshold, str):
            values = _add_metadata_to_dataframe(
                dataset,
                valid[["id"]].copy(),
                metadata_field=(self.threshold,),
                target_column="_threshold",
                filter_func=np.mean,
            )
            per_study = values.groupby("id")["_threshold"].first()
            if per_study.isna().any():
                missing = sorted(per_study.index[per_study.isna()])
                raise ValueError(
                    f"threshold={self.threshold!r} names a metadata field, but it is missing "
                    f"for {len(missing)} study/studies: {', '.join(map(str, missing[:10]))}."
                )
            thresholds = self._threshold_to_g(per_study, sample_size)
        else:
            constant = pd.Series(float(self.threshold), index=sample_size.index)
            thresholds = self._threshold_to_g(constant, sample_size)

        thresholds = thresholds.abs()
        clamped = thresholds > smallest_reported
        if clamped.any():
            LGR.warning(
                f"{int(clamped.sum())} study/studies have a supplied threshold above their own "
                "smallest reported effect size, which is impossible; clamping to the smallest "
                "reported value."
            )
            thresholds = thresholds.clip(upper=smallest_reported)

        return pd.DataFrame({"sample_size": sample_size, "thresh": thresholds})

    def _threshold_to_g(self, values, sample_size):
        """Convert thresholds in z, t or g units to Hedges' g."""
        if self.threshold_units == "g":
            return values.abs()
        return pd.Series(
            np.abs(_hedges_g(values.values, sample_size.values, f"{self.threshold_units}_stat")),
            index=values.index,
        )

    def _assemble_clusters(self, foci, studies, centroids):
        """Assign every study a status in every cluster.

        Returns
        -------
        :obj:`list` of :obj:`dict`
            One entry per surviving cluster.

        Notes
        -----
        Vectorized over clusters and studies. The obvious implementation --
        loop over clusters, loop over studies, mask the foci -- spends most of
        its time boxing Python strings into Arrow scalars to compare study ids,
        which dominated the whole fit (99% of runtime at 100 studies). Study ids
        are therefore reduced to integer codes once, and every status is derived
        from two per-study minimum-distance matrices computed by segmented
        reduction.
        """
        roster = studies.index.to_numpy()
        codes = pd.Index(roster).get_indexer(foci["id"].to_numpy())
        all_xyz = foci[["x", "y", "z"]].to_numpy(dtype=float)
        valid = foci["valid"].to_numpy()
        focus_g = np.nan_to_num(foci["g"].to_numpy(dtype=float))
        n_foci, n_studies = all_xyz.shape[0], roster.size

        # Group rows by study so each study's foci are contiguous, which lets
        # `reduceat` take per-study minima in one pass.
        order = np.argsort(codes, kind="stable")
        starts = np.searchsorted(codes[order], np.arange(n_studies))
        row_index = order[:, None]

        clusters = []
        for lo, hi in _chunk_bounds(centroids.shape[0], n_foci):
            block = centroids[lo:hi]
            distance = np.linalg.norm(all_xyz[:, None, :] - block[None, :, :], axis=2)
            valid_distance = np.where(valid[:, None], distance, np.inf)

            dmin_all = np.minimum.reduceat(distance[order], starts, axis=0)
            dmin_valid = np.minimum.reduceat(valid_distance[order], starts, axis=0)

            reported = dmin_valid <= self.radius
            missing = ~reported & (dmin_all <= self.radius)
            # A study's valid foci are a subset of all its foci, so testing every
            # focus against the buffer already covers the valid ones.
            in_buffer = ~reported & ~missing & (dmin_all <= self.buffer)

            if self.within_study == "nearest":
                target, arranged = dmin_valid, valid_distance[order]
            else:
                # Largest |g| among this study's foci inside the region.
                weighted = np.where(
                    valid_distance <= self.radius, np.abs(focus_g)[:, None], -np.inf
                )
                arranged = weighted[order]
                target = np.maximum.reduceat(arranged, starts, axis=0)

            # Lowest original row index attaining the extremum, matching argmin/argmax
            # tie-breaking. `n_foci` is an out-of-range sentinel for rows that do not
            # attain it, so it loses the minimum; clip before indexing.
            attains = arranged == np.repeat(target, np.diff(np.append(starts, n_foci)), axis=0)
            chosen = np.minimum.reduceat(np.where(attains, row_index, n_foci), starts, axis=0)
            chosen_g = np.where(reported, focus_g[np.minimum(chosen, n_foci - 1)], 0.0)

            for offset in range(block.shape[0]):
                included = reported[:, offset] | (
                    ~reported[:, offset] & ~missing[:, offset] & ~in_buffer[:, offset]
                )
                if int(reported[included, offset].sum()) < self.min_studies:
                    continue
                status = np.where(
                    reported[:, offset],
                    _REPORTED,
                    np.where(
                        missing[:, offset],
                        _MISSING,
                        np.where(in_buffer[:, offset], _BUFFER, _CENSORED),
                    ),
                )
                clusters.append(
                    {
                        "label": lo + offset + 1,
                        "centroid": block[offset],
                        "studies": roster[included],
                        "g": chosen_g[included, offset],
                        "sample_size": studies["sample_size"].to_numpy()[included],
                        "thresh": studies["thresh"].to_numpy()[included],
                        "reported": reported[included, offset],
                        "status": status,
                        "roster": roster,
                    }
                )
        return clusters

    def _estimate_tau2(self, packed, clusters, two_sided):
        """Return the between-study variance to use for each cluster."""
        if self.tau2 == "fixed":
            return np.zeros(len(clusters))

        if self.tau2 == "dl":
            return np.array(
                [
                    _dersimonian_laird(
                        cluster["g"][cluster["reported"]],
                        _g_variance(
                            cluster["g"][cluster["reported"]],
                            cluster["sample_size"][cluster["reported"]],
                        ),
                    )
                    for cluster in clusters
                ]
            )

        upper = 4.0 * float(
            np.median(np.concatenate([_g_variance(0.0, c["sample_size"]) for c in clusters]))
        )
        return _profile_tau2(packed, two_sided, self.likelihood, upper, self.tau2_scope)

    def _fit_clusters(self, packed, clusters, two_sided, tau2_values):
        """Fit every cluster and return a row per cluster."""
        delta, se, converged = _solve_batch_with_variance(
            packed, tau2_values, two_sided, self.likelihood
        )

        rows = []
        for index, cluster in enumerate(clusters):
            estimate, error, tau2 = delta[index], se[index], tau2_values[index]
            reported = cluster["reported"]
            observed_g = cluster["g"][reported]
            observed_var = _g_variance(
                estimate if np.isfinite(estimate) else 0.0, cluster["sample_size"]
            )

            usable = np.isfinite(estimate) and np.isfinite(error) and error > 0
            naive_weights = 1.0 / observed_var[reported]
            rows.append(
                {
                    "cluster_id": cluster["label"],
                    "x": cluster["centroid"][0],
                    "y": cluster["centroid"][1],
                    "z": cluster["centroid"][2],
                    "n_studies": int(cluster["studies"].size),
                    "n_reported": int(reported.sum()),
                    "n_censored": int((~reported).sum()),
                    "n_buffer": int(np.sum(cluster["status"] == _BUFFER)),
                    "n_missing_stat": int(np.sum(cluster["status"] == _MISSING)),
                    "g": estimate,
                    "se": error,
                    "ci_low": estimate - 1.959963984540054 * error,
                    "ci_high": estimate + 1.959963984540054 * error,
                    "z_stat": estimate / error if usable else np.nan,
                    "tau2": tau2,
                    "i2": _i_squared(observed_g, observed_var[reported], tau2),
                    "g_naive": float(np.sum(observed_g * naive_weights) / np.sum(naive_weights)),
                    "mean_threshold": float(np.mean(cluster["thresh"])),
                    "bias_peak": self._peak_bias(cluster["sample_size"][reported]),
                    "converged": bool(converged[index]),
                }
            )
        return rows

    def _peak_bias(self, sample_sizes):
        """Estimate the residual maximum-selection ("winner's curse") inflation.

        A reported peak is the largest of roughly ``n_eff`` independent values
        in the region, so it overshoots the regional effect by about
        ``E[max of n_eff standard normals] / sqrt(N)`` in Hedges' g units. This
        is a magnitude, not a correction: nothing is subtracted from ``g``.
        """
        volume = (4.0 / 3.0) * np.pi * self.radius**3
        n_eff = max(1.0, volume / self.assumed_fwhm**3)
        # Blom's approximation to the expected largest of n_eff standard normals.
        expected_max = float(ndtri((n_eff - 0.375) / (n_eff + 0.25)))
        return float(np.mean(expected_max / np.sqrt(sample_sizes)))

    def _paint(self, centroids, rows, n_voxels, world):
        """Broadcast per-cluster values onto the voxels each cluster covers."""
        maps = {
            name: np.full(n_voxels, np.nan, dtype=DEFAULT_FLOAT_DTYPE)
            for name in ("est", "se", "z", "p", "logp", "tau2", "label")
        }
        if not rows:
            return maps

        tree = cKDTree(np.asarray(centroids, dtype=float))
        distance, nearest = tree.query(world)
        inside = distance <= self.radius
        if not np.any(inside):
            return maps

        index = nearest[inside]
        frame = pd.DataFrame(rows)
        z_values = frame["z_stat"].to_numpy(dtype=float)
        p_values, logp_values = _nlogp_to_p_and_logp(_z_to_nlogp(z_values))

        maps["est"][inside] = frame["g"].to_numpy(dtype=float)[index]
        maps["se"][inside] = frame["se"].to_numpy(dtype=float)[index]
        maps["z"][inside] = z_values[index]
        maps["p"][inside] = p_values[index]
        maps["logp"][inside] = logp_values[index]
        maps["tau2"][inside] = frame["tau2"].to_numpy(dtype=float)[index]
        maps["label"][inside] = frame["cluster_id"].to_numpy(dtype=float)[index]
        return maps

    def _contributions_table(self, clusters):
        """Return a long table making the censoring pattern auditable."""
        records = []
        for cluster in clusters:
            lookup = dict(zip(cluster["studies"], zip(cluster["g"], cluster["reported"])))
            for study, status in zip(cluster["roster"], cluster["status"]):
                observed, reported = lookup.get(study, (np.nan, False))
                records.append(
                    {
                        "cluster_id": cluster["label"],
                        "study_id": study,
                        "status": status,
                        "g": observed if reported else np.nan,
                    }
                )
        return pd.DataFrame.from_records(
            records, columns=["cluster_id", "study_id", "status", "g"]
        )

    def _fit(self, dataset):
        """Estimate a regional effect size for every surviving cluster of foci."""
        self.masker = self.masker or dataset.masker

        foci, stat_column = self._prepare_foci(self.inputs_["coordinates"])
        two_sided = self._resolve_tail(foci)
        studies = self._study_thresholds(foci, stat_column, dataset)
        foci = foci[foci["id"].isin(studies.index)]

        valid_xyz = foci.loc[foci["valid"], ["x", "y", "z"]].to_numpy(dtype=float)
        labels = _cluster_foci(valid_xyz, self.radius)
        centroids = np.stack(
            [valid_xyz[labels == label].mean(axis=0) for label in np.unique(labels)]
        )
        clusters = self._assemble_clusters(foci, studies, centroids)

        mask_img = self.masker.mask_img
        world = vox2mm(np.vstack(np.where(_mask_img_to_bool(mask_img))).T, mask_img.affine)
        n_voxels = world.shape[0]

        self.two_sided_ = two_sided
        self.null_distributions_ = {}

        if not clusters:
            warnings.warn(
                f"No cluster of foci had at least min_studies={self.min_studies} reporting "
                "studies, so nothing could be estimated. Try a larger `radius` or a smaller "
                "`min_studies`.",
                stacklevel=2,
            )
            self.clusters_ = []
            empty = self._paint(np.empty((0, 3)), [], n_voxels, world)
            return (
                empty,
                {"clusters": pd.DataFrame(), "study_contributions": pd.DataFrame()},
                (self._description_text()),
            )

        packed = _pack_clusters(clusters)
        tau2_values = self._estimate_tau2(packed, clusters, two_sided)
        rows = self._fit_clusters(packed, clusters, two_sided, tau2_values)

        table = pd.DataFrame(rows)
        nlogp = _z_to_nlogp(table["z_stat"].to_numpy(dtype=float))
        table["p"], _ = _nlogp_to_p_and_logp(nlogp)
        table["p_corr_fdr"], _ = _nlogp_to_p_and_logp(_fdr_over_clusters(nlogp))

        centroid_stack = np.stack([cluster["centroid"] for cluster in clusters])
        maps = self._paint(centroid_stack, rows, n_voxels, world)

        if self.n_iters > 0:
            null_max = self._montecarlo_null(foci, studies, two_sided, tau2_values, mask_img)
            self.null_distributions_["values_desc-maxz_level-cluster"] = null_max
            observed = np.abs(table["z_stat"].to_numpy(dtype=float))
            table["p_desc-mc"] = [
                null_to_p(value, null_max, tail="upper") if np.isfinite(value) else np.nan
                for value in observed
            ]

        self.clusters_ = clusters
        self.cluster_centroids_ = centroid_stack
        return (
            maps,
            {"clusters": table, "study_contributions": self._contributions_table(clusters)},
            self._description_text(),
        )

    def _montecarlo_null(self, foci, studies, two_sided, tau2_values, mask_img):
        """Build a null distribution of the largest absolute z under random coordinates.

        Each study keeps its number of foci and their effect sizes; only the
        locations are redrawn. The resulting null is therefore "coordinates are
        spatially random", matching the false cluster discovery rate of
        :footcite:t:`tench2017coordinate`. It accounts for clusters having been
        discovered from the same data -- it is *not* a test of the effect size
        itself.
        """
        null_xyz = vox2mm(
            _mask_coverage_to_null_ijk(self.masker, mask_coverage=self.mask_coverage),
            mask_img.affine,
        )
        tau2_fixed = float(np.median(tau2_values)) if len(tau2_values) else 0.0

        def _one(seed):
            rng = np.random.default_rng(seed)
            shuffled = foci.copy()
            picks = rng.integers(0, null_xyz.shape[0], size=len(shuffled))
            shuffled[["x", "y", "z"]] = null_xyz[picks]

            valid_xyz = shuffled.loc[shuffled["valid"], ["x", "y", "z"]].to_numpy(dtype=float)
            if valid_xyz.shape[0] < 2:
                return 0.0
            labels = _cluster_foci(valid_xyz, self.radius)
            centroids = np.stack(
                [valid_xyz[labels == label].mean(axis=0) for label in np.unique(labels)]
            )
            null_clusters = self._assemble_clusters(shuffled, studies, centroids)
            if not null_clusters:
                return 0.0

            delta, se, converged = _solve_batch_with_variance(
                _pack_clusters(null_clusters), tau2_fixed, two_sided, self.likelihood
            )
            usable = converged & np.isfinite(delta) & np.isfinite(se) & (se > 0)
            return float(np.max(np.abs(delta[usable] / se[usable]))) if np.any(usable) else 0.0

        seeds = np.random.SeedSequence(0).spawn(self.n_iters)
        values = Parallel(n_jobs=self.n_cores)(
            delayed(_one)(seed)
            for seed in tqdm(seeds, desc="Monte Carlo null", disable=self.n_iters < 50)
        )
        return np.asarray(values, dtype=float)

    def correct_fdr_cluster(self, result, alpha=0.05):
        """Apply Benjamini-Hochberg FDR across clusters rather than voxels.

        .. warning::
            Do not call this directly. Use
            ``FDRCorrector(method="cluster").transform(result)``.

        Voxelwise FDR is wrong for this estimator: every voxel in a cluster
        carries the same p-value, so counting voxels inflates the number of
        tests by several orders of magnitude and destroys power. The tests here
        are the clusters.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            The result to correct.
        alpha : :obj:`float`, default=0.05
            Retained for interface compatibility; BH-adjusted p-values are
            returned regardless, so the caller can threshold at any level.

        Returns
        -------
        maps, tables, description
        """
        table = result.tables.get("clusters")
        if table is None or table.empty:
            return {"p": result.maps["p"].copy()}, {}, "No clusters were available to correct."

        labels = result.maps["label"]
        corrected = np.full(labels.shape, np.nan, dtype=DEFAULT_FLOAT_DTYPE)
        adjusted = table["p_corr_fdr"].to_numpy(dtype=float)
        for cluster_id, value in zip(table["cluster_id"].to_numpy(), adjusted):
            corrected[labels == cluster_id] = value

        description = (
            "False discovery rate correction was performed with the Benjamini-Hochberg "
            "procedure \\citep{benjamini1995controlling}, across clusters rather than voxels, "
            f"treating each of the {len(table)} clusters as one test."
        )
        return {"p": corrected}, {}, description

    def _generate_description(self):
        """Describe the fitted estimator in prose."""
        threshold = (
            "each study's own smallest reported peak statistic"
            if self.threshold == "infer"
            else f"a threshold of {self.threshold} ({self.threshold_units} units)"
        )
        tau2 = {
            "profile": f"profile maximum likelihood ({self.tau2_scope} scope)",
            "dl": "the DerSimonian-Laird estimator applied to reported effect sizes",
            "fixed": "a fixed value of zero (a fixed-effects model)",
        }[self.tau2]
        return (
            "An effect-size coordinate-based meta-analysis was performed with NiMARE's "
            "CoordinateEffectSize estimator, which estimates the magnitude of the effect in a "
            "region rather than the spatial convergence of reported foci "
            "\\citep{tench2017coordinate,costafreda2012parametric}. "
            f"Reported peak statistics were converted to Hedges' g, and foci within {self.radius} "
            "mm were grouped into clusters by average-linkage agglomerative clustering. "
            "Within each cluster, a random-effects model was fitted by maximum likelihood, with "
            "studies reporting no peak in the region treated as left-censored observations "
            f"below {threshold} rather than as missing data or as zeros, so that no imputation "
            "of study images was required. Between-study variance was estimated by "
            f"{tau2}. "
            f"Studies whose nearest focus fell between {self.radius} and {self.buffer} mm of a "
            "cluster centre were excluded from that cluster rather than censored. "
            "Note that the resulting estimate refers to the effect size at the reported peak "
            "and remains biased upward, because a reported peak is the maximum over a search "
            "region."
        )
