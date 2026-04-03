"""CBMA methods from the activation likelihood estimation (ALE) family."""

import gc
import logging
import os
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from numba import jit
from scipy import ndimage
from scipy import sparse as sp_sparse
from tqdm.auto import tqdm

from nimare import _version
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.meta.cbma.utils import collect_csr_ma_maps, require_masked_csr
from nimare.meta.kernel import ALEKernel
from nimare.meta.utils import (
    _calculate_cluster_measures,
    compute_ale_ma,
    get_ale_kernel,
)
from nimare.stats import null_to_p
from nimare.transforms import p_to_z
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _check_ncores,
    mm2vox,
    use_memmap,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


def _csr_row_max(ma_values):
    """Compute row-wise maxima for a CSR matrix without densifying it."""
    ma_values = ma_values.tocsr(copy=False)
    max_values = np.zeros(ma_values.shape[0], dtype=DEFAULT_FLOAT_DTYPE)
    for i_row in range(ma_values.shape[0]):
        start = ma_values.indptr[i_row]
        end = ma_values.indptr[i_row + 1]
        if start != end:
            max_values[i_row] = ma_values.data[start:end].max()
    return max_values


def _compute_ale_summarystat(ma_values):
    """Compute ALE summary statistics from dense arrays or masked CSR matrices."""
    if sp_sparse.isspmatrix(ma_values):
        ma_values = ma_values.tocsr(copy=False)
        log_sums = np.bincount(
            ma_values.indices,
            weights=np.log1p(-ma_values.data),
            minlength=ma_values.shape[1],
        )
        stat_values = 1.0 - np.exp(log_sums)
        return stat_values.astype(DEFAULT_FLOAT_DTYPE, copy=False)

    if isinstance(ma_values, np.ndarray):
        stat_values = 1.0 - np.prod(1.0 - ma_values, axis=0)
        return stat_values

    raise ValueError(f"Unsupported data type '{type(ma_values)}'")


@dataclass
class _ChunkedCSRGroup:
    """Disk-backed study-by-voxel CSR chunks for one MA-map group."""

    chunks: list
    row_offsets: np.ndarray
    shape: tuple


@dataclass
class _PairwiseMAStore:
    """Pairwise ALESubtraction MA-map storage with a common permutation interface."""

    group1: object
    group2: object
    group1_stat: np.ndarray
    group2_stat: np.ndarray
    temp_files: list

    @property
    def n_group1(self):
        return self.group1.shape[0]

    @property
    def n_total(self):
        return self.group1.shape[0] + self.group2.shape[0]

    @property
    def n_voxels(self):
        return self.group1.shape[1]

    def compute_partition_summarystat(self, row_idx):
        """Compute ALE summary statistics for a selected set of study rows."""
        return _compute_partition_ale_summarystat(self.group1, self.group2, row_idx, self.n_group1)

    def close(self):
        """Release memmap-backed arrays and delete temporary files."""
        temp_files = list(self.temp_files)
        _close_csr_memmaps(self.group1)
        _close_csr_memmaps(self.group2)
        self.group1 = None
        self.group2 = None
        self.group1_stat = None
        self.group2_stat = None
        self.temp_files = []
        gc.collect()
        _cleanup_temp_files(temp_files)


@dataclass
class _GroupMAEstimate:
    """Projected CSR footprint for one MA-map group plus an optional reusable sample chunk."""

    total_bytes: float
    bytes_per_study: float
    sample_ma: object
    sample_n_studies: int


def _is_chunked_group(ma_values):
    """Return True when MA values are stored as chunked CSR blocks."""
    return isinstance(ma_values, _ChunkedCSRGroup)


def _accumulate_csr_log_sums(ma_values, log_sums):
    """Accumulate ALE log-sums from a CSR matrix into an existing buffer."""
    if ma_values.nnz:
        log_sums += np.bincount(
            ma_values.indices,
            weights=np.log1p(-ma_values.data),
            minlength=ma_values.shape[1],
        )


def _compute_partition_ale_summarystat(ma_maps1, ma_maps2, row_idx, n_grp1):
    """Compute ALE summary stats for rows selected across two CSR or chunked MA groups."""
    row_idx = np.asarray(row_idx)
    if row_idx.ndim != 1:
        row_idx = row_idx.reshape(-1)
    if not np.issubdtype(row_idx.dtype, np.integer):
        raise TypeError(f"row_idx must contain integers; got dtype {row_idx.dtype}.")
    if ma_maps1.shape[1] != ma_maps2.shape[1]:
        raise ValueError(
            "Group MA maps must share the same number of voxels; "
            f"got {ma_maps1.shape[1]} and {ma_maps2.shape[1]}."
        )
    if ma_maps1.shape[0] != n_grp1:
        raise ValueError(
            "n_grp1 must match the number of rows in group 1 MA maps; "
            f"got n_grp1={n_grp1} and ma_maps1.shape[0]={ma_maps1.shape[0]}."
        )
    n_total_rows = ma_maps1.shape[0] + ma_maps2.shape[0]
    if row_idx.size and (np.any(row_idx < 0) or np.any(row_idx >= n_total_rows)):
        raise IndexError(
            "row_idx contains out-of-bounds study indices for the provided MA groups; "
            f"valid range is [0, {n_total_rows - 1}]."
        )

    n_voxels = ma_maps1.shape[1]
    log_sums = np.zeros(n_voxels, dtype=np.float64)

    grp1_idx = row_idx[row_idx < n_grp1]
    if grp1_idx.size:
        if _is_chunked_group(ma_maps1):
            chunk_ids = np.searchsorted(ma_maps1.row_offsets[1:], grp1_idx, side="right")
            for i_chunk in np.unique(chunk_ids):
                local_idx = grp1_idx[chunk_ids == i_chunk] - ma_maps1.row_offsets[i_chunk]
                grp1_maps = ma_maps1.chunks[i_chunk][local_idx, :]
                _accumulate_csr_log_sums(grp1_maps, log_sums)
        else:
            grp1_maps = require_masked_csr(ma_maps1, source="Group 1 MA maps")[grp1_idx, :]
            _accumulate_csr_log_sums(grp1_maps, log_sums)

    grp2_idx = row_idx[row_idx >= n_grp1] - n_grp1
    if grp2_idx.size:
        if _is_chunked_group(ma_maps2):
            chunk_ids = np.searchsorted(ma_maps2.row_offsets[1:], grp2_idx, side="right")
            for i_chunk in np.unique(chunk_ids):
                local_idx = grp2_idx[chunk_ids == i_chunk] - ma_maps2.row_offsets[i_chunk]
                grp2_maps = ma_maps2.chunks[i_chunk][local_idx, :]
                _accumulate_csr_log_sums(grp2_maps, log_sums)
        else:
            grp2_maps = require_masked_csr(ma_maps2, source="Group 2 MA maps")[grp2_idx, :]
            _accumulate_csr_log_sums(grp2_maps, log_sums)

    stat_values = 1.0 - np.exp(log_sums)
    return stat_values.astype(DEFAULT_FLOAT_DTYPE, copy=False)


@jit(nopython=True, cache=True)
def _study_ma_histogram(study_ma_values, n_zero_voxels, mask_voxel_recip, inv_step_size, n_bins):
    """Bin one study's nonzero ALE values onto the fixed approximate-null grid."""
    exp_hist = np.zeros(n_bins, dtype=np.float64)
    for i_val in range(study_ma_values.shape[0]):
        idx = int(study_ma_values[i_val] * inv_step_size)
        if idx < 0:
            idx = 0
        elif idx >= n_bins:
            idx = n_bins - 1
        exp_hist[idx] += 1.0

    exp_hist[0] += n_zero_voxels
    exp_hist *= mask_voxel_recip
    return exp_hist


@jit(nopython=True, cache=True)
def _update_ale_histogram(
    ale_idx, ale_probs, exp_idx, exp_probs, bin_centers, inv_step_size, n_bins, out
):
    """Combine two nonzero ALE histograms using a reusable output buffer."""
    for i_bin in range(n_bins):
        out[i_bin] = 0.0

    for i_exp in range(exp_idx.shape[0]):
        exp_center = bin_centers[exp_idx[i_exp]]
        exp_prob = exp_probs[i_exp]
        exp_one_minus = 1.0 - exp_center
        for i_ale in range(ale_idx.shape[0]):
            score = 1.0 - exp_one_minus * (1.0 - bin_centers[ale_idx[i_ale]])
            score_idx = int(score * inv_step_size)
            if score_idx < 0:
                score_idx = 0
            elif score_idx >= n_bins:
                score_idx = n_bins - 1
            out[score_idx] += exp_prob * ale_probs[i_ale]

    return out


def _finalize_alediff_tail_counts(left_counts, right_counts, n_iters):
    """Convert ALE subtraction tail counts into p-values and z-map signs."""
    left_tail = left_counts / n_iters
    right_tail = right_counts / n_iters
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / n_iters)
    p_values = 2.0 * np.minimum(left_tail, right_tail)
    p_values = np.maximum(smallest_value, np.minimum(p_values, 1.0 - smallest_value)).astype(
        DEFAULT_FLOAT_DTYPE,
        copy=False,
    )
    diff_signs = np.sign(right_counts.astype(np.int64) - left_counts.astype(np.int64)).astype(
        DEFAULT_FLOAT_DTYPE,
        copy=False,
    )
    return p_values, diff_signs


def _collect_masked_ma_maps(estimator, coords_key="coordinates", maps_key="ma_maps"):
    """Collect ALE-family MA maps in masked CSR form."""
    estimator._study_max_ma_values = None
    return collect_csr_ma_maps(estimator, coords_key=coords_key, maps_key=maps_key)


def _collect_ale_masked_ma_maps(estimator, coords_key="coordinates", maps_key="ma_maps"):
    """Collect ALE MA maps and cache per-study maxima for approximate-null binning."""
    ma_values = _collect_masked_ma_maps(estimator, coords_key=coords_key, maps_key=maps_key)
    estimator._study_max_ma_values = _csr_row_max(ma_values).astype(
        DEFAULT_FLOAT_DTYPE, copy=False
    )
    return ma_values


def _estimate_csr_nbytes(ma_values):
    """Estimate the in-memory footprint of a CSR matrix."""
    ma_values = require_masked_csr(ma_values)
    return ma_values.data.nbytes + ma_values.indices.nbytes + ma_values.indptr.nbytes


def _get_available_memory_bytes():
    """Best-effort estimate of currently available system memory."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None

    if page_size <= 0 or available_pages <= 0:
        return None
    return int(page_size * available_pages)


def _determine_low_memory_chunk_bytes(available_bytes=None):
    """Choose a target per-chunk MA-map budget from available RAM."""
    if available_bytes is None:
        available_bytes = _get_available_memory_bytes()

    if available_bytes is None:
        return 256 * 1024**2

    return max(1 * 1024**2, min(512 * 1024**2, int(available_bytes * 0.1)))


def _copy_array_to_memmap(arr, filename):
    """Copy an array into a disk-backed memmap with the same dtype and shape."""
    arr = np.asarray(arr)
    mapped = np.memmap(filename, dtype=arr.dtype, mode="w+", shape=arr.shape)
    mapped[...] = arr
    return mapped


def _csr_to_memmap(ma_values, prefix):
    """Copy a CSR matrix into disk-backed arrays and return a CSR view plus temp files."""
    ma_values = require_masked_csr(ma_values)

    filenames = []
    for suffix in ("data", "indices", "indptr"):
        fd, filename = tempfile.mkstemp(prefix=f"{prefix}_{suffix}_", suffix=".mmap")
        os.close(fd)
        filenames.append(filename)

    data = _copy_array_to_memmap(ma_values.data, filenames[0])
    indices = _copy_array_to_memmap(ma_values.indices, filenames[1])
    indptr = _copy_array_to_memmap(ma_values.indptr, filenames[2])
    mapped = sp_sparse.csr_matrix(
        (data, indices, indptr),
        shape=ma_values.shape,
        copy=False,
    )
    return mapped, filenames


def _close_memmap_array(arr):
    """Close a numpy memmap backing file when present."""
    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _close_csr_memmaps(ma_values):
    """Close memmap-backed CSR arrays when present."""
    if _is_chunked_group(ma_values):
        for chunk in ma_values.chunks:
            _close_csr_memmaps(chunk)
        return

    if not sp_sparse.isspmatrix(ma_values):
        return

    _close_memmap_array(ma_values.data)
    _close_memmap_array(ma_values.indices)
    _close_memmap_array(ma_values.indptr)


def _cleanup_temp_files(filenames):
    """Remove temporary files created for memmap-backed arrays."""
    for filename in filenames:
        if filename and os.path.isfile(filename):
            for i_try in range(5):
                try:
                    os.remove(filename)
                    break
                except PermissionError:
                    if i_try == 4:
                        raise
                    gc.collect()
                    time.sleep(0.05)


def _iter_study_id_chunks(coordinates, chunk_rows, start_idx=0):
    """Yield coordinate subsets spanning up to ``chunk_rows`` studies each."""
    study_ids = np.unique(coordinates["id"].values)
    for start in range(start_idx, study_ids.size, chunk_rows):
        chunk_ids = study_ids[start : start + chunk_rows]
        yield coordinates[coordinates["id"].isin(chunk_ids)]


class ALE(CBMAEstimator):
    """Activation likelihood estimation.

    .. versionchanged:: 0.2.1

        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.0.12

        - Use a 4D sparse array for modeled activation maps.

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is ALEKernel.
    null_method : {"approximate", "montecarlo"}, optional
        Method by which to determine uncorrected p-values. The available options are

        ======================= =================================================================
        "approximate" (default) Build a histogram of summary-statistic values and their
                                expected frequencies under the assumption of random spatial
                                associated between studies, via a weighted convolution, as
                                described in :footcite:t:`eickhoff2012activation`.

                                This method is much faster, but slightly less accurate, than the
                                "montecarlo" option.
        "montecarlo"            Perform a large number of permutations, in which the coordinates
                                in the studies are randomly drawn from the Estimator's brain mask
                                and the full set of resulting summary-statistic values are
                                incorporated into a null distribution (stored as a histogram for
                                memory reasons).

                                This method is must slower, and is only slightly more accurate.
        ======================= =================================================================

    n_iters : :obj:`int`, default=5000
        Number of iterations to use to define the null distribution.
        This is only used if ``null_method=="montecarlo"``.
        Default is 5000.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    n_cores : :obj:`int`, default=1
        Number of cores to use for parallelization.
        This is only used if ``null_method=="montecarlo"``.
        If <=0, defaults to using all available cores.
        Default is 1.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix ``kernel__`` in the variable name.
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.maskers.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key: coordinates.
        This is an edited version of the dataset's coordinates DataFrame.
    null_distributions_ : :obj:`dict` of :class:`numpy.ndarray`
        Null distributions for the uncorrected summary-statistic-to-p-value conversion and any
        multiple-comparisons correction methods.
        Entries are added to this attribute if and when the corresponding method is applied.

        If ``null_method == "approximate"``:

            -   ``histogram_bins``: Array of bin centers for the null distribution histogram,
                ranging from zero to the maximum possible summary statistic value for the Dataset.
            -   ``histweights_corr-none_method-approximate``: Array of weights for the null
                distribution histogram, with one value for each bin in ``histogram_bins``.

        If ``null_method == "montecarlo"``:

            -   ``histogram_bins``: Array of bin centers for the null distribution histogram,
                ranging from zero to the maximum possible summary statistic value for the Dataset.
            -   ``histweights_corr-none_method-montecarlo``: Array of weights for the null
                distribution histogram, with one value for each bin in ``histogram_bins``.
                These values are derived from the full set of summary statistics from each
                iteration of the Monte Carlo procedure.
            -   ``histweights_level-voxel_corr-fwe_method-montecarlo``: Array of weights for the
                voxel-level FWE-correction null distribution, with one value for each bin in
                ``histogram_bins``. These values are derived from the maximum summary statistic
                from each iteration of the Monte Carlo procedure.

        If :meth:`correct_fwe_montecarlo` is applied:

            -   ``values_level-voxel_corr-fwe_method-montecarlo``: The maximum summary statistic
                value from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-size_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                size from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-mass_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                mass from each Monte Carlo iteration. An array of shape (n_iters,).

    Notes
    -----
    The ALE algorithm was originally developed in :footcite:t:`turkeltaub2002meta`,
    then updated in :footcite:t:`turkeltaub2012minimizing` and
    :footcite:t:`eickhoff2012activation`.

    The ALE algorithm is also implemented as part of the GingerALE app provided by the BrainMap
    organization (https://www.brainmap.org/ale/).

    Available correction methods: :meth:`~nimare.meta.cbma.ale.ALE.correct_fwe_montecarlo`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        kernel_transformer=ALEKernel,
        null_method="approximate",
        n_iters=5000,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        n_cores=1,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, ALEKernel) or kernel_transformer == ALEKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(
            kernel_transformer=kernel_transformer,
            memory=memory,
            memory_level=memory_level,
            **kwargs,
        )
        self.null_method = null_method
        self.n_iters = None if null_method == "approximate" else n_iters or 5000
        self.n_cores = _check_ncores(n_cores)
        self.dataset = None
        self._permutation_parallel_backend = "threading"

    def _generate_description(self):
        """Generate a description of the fitted Estimator.

        Returns
        -------
        str
            Description of the Estimator.
        """
        if self.null_method == "montecarlo":
            null_method_str = (
                "a Monte Carlo-based null distribution, in which dataset coordinates were "
                "randomly drawn from the analysis mask and the full set of ALE values were "
                f"retained, using {self.n_iters} iterations"
            )
        else:
            null_method_str = "an approximate null distribution \\citep{eickhoff2012activation}"

        if (
            hasattr(self.kernel_transformer, "sample_size")  # Only kernels that allow sample sizes
            and (self.kernel_transformer.sample_size is None)
            and (self.kernel_transformer.fwhm is None)
        ):
            # Get the total number of subjects in the inputs.
            n_subjects = (
                self.inputs_["coordinates"].groupby("id")["sample_size"].mean().values.sum()
            )
            sample_size_str = f", with a total of {int(n_subjects)} participants"
        else:
            sample_size_str = ""

        description = (
            "An activation likelihood estimation (ALE) meta-analysis "
            "\\citep{turkeltaub2002meta,turkeltaub2012minimizing,eickhoff2012activation} was "
            f"performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), using a(n) "
            f"{self.kernel_transformer.__class__.__name__.replace('Kernel', '')} kernel. "
            f"{self.kernel_transformer._generate_description()} "
            f"ALE values were converted to p-values using {null_method_str}. "
            f"The input dataset included {self.inputs_['coordinates'].shape[0]} foci from "
            f"{len(self.inputs_['id'])} experiments{sample_size_str}."
        )
        return description

    def _collect_ma_maps(self, coords_key="coordinates", maps_key="ma_maps", return_type="sparse"):
        """Collect ALE MA maps in the masked sparse format used by the estimator."""
        if return_type != "sparse":
            return super()._collect_ma_maps(
                coords_key=coords_key,
                maps_key=maps_key,
                return_type=return_type,
            )

        return _collect_ale_masked_ma_maps(self, coords_key=coords_key, maps_key=maps_key)

    def _compute_summarystat(self, data):
        """Compute ALE summary statistics from input data."""
        if sp_sparse.isspmatrix(data):
            return self._compute_summarystat_est(data)

        return super()._compute_summarystat(data)

    def _compute_summarystat_est(self, ma_values):
        ma_values = require_masked_csr(ma_values) if sp_sparse.isspmatrix(ma_values) else ma_values
        stat_values = _compute_ale_summarystat(ma_values)
        if sp_sparse.isspmatrix(ma_values):
            self.__n_mask_voxels = stat_values.shape[0]
        return stat_values

    def _determine_histogram_bins(self, ma_maps):
        """Determine histogram bins for null distribution methods.

        Parameters
        ----------
        ma_maps : scipy.sparse matrix
            Masked study-by-voxel MA maps.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute: "histogram_bins".
        """
        if not hasattr(self, "null_distributions_"):
            self.null_distributions_ = {}

        if sp_sparse.isspmatrix(ma_maps):
            ma_maps = require_masked_csr(ma_maps)
            max_ma_values = getattr(self, "_study_max_ma_values", None)
            if max_ma_values is None or max_ma_values.shape[0] != ma_maps.shape[0]:
                max_ma_values = _csr_row_max(ma_maps)
        else:
            raise ValueError(f"Unsupported data type '{type(ma_maps)}'")

        # Determine bins for null distribution histogram
        # Remember that numpy histogram bins are bin edges, not centers
        # Assuming values of 0, .001, .002, etc., bins are -.0005-.0005, .0005-.0015, etc.
        INV_STEP_SIZE = 100000
        step_size = 1 / INV_STEP_SIZE
        # round up based on resolution
        max_ma_values = np.ceil(max_ma_values * INV_STEP_SIZE) / INV_STEP_SIZE
        max_poss_ale = self._compute_summarystat(max_ma_values)
        # create bin centers
        hist_bins = np.round(np.arange(0, max_poss_ale + (1.5 * step_size), step_size), 5)
        self.null_distributions_["histogram_bins"] = hist_bins

    def _compute_null_approximate(self, ma_maps):
        """Compute uncorrected ALE null distribution using approximate solution.

        Parameters
        ----------
        ma_maps : scipy.sparse matrix
            Masked study-by-voxel MA maps.

        Notes
        -----
        This method adds two entries to the null_distributions_ dict attribute:

            - "histogram_bins"
            - "histweights_corr-none_method-approximate"
        """
        if sp_sparse.isspmatrix(ma_maps):
            ma_maps = require_masked_csr(ma_maps)
        else:
            raise ValueError(f"Unsupported data type '{type(ma_maps)}'")

        assert "histogram_bins" in self.null_distributions_.keys()

        # Reuse the fixed histogram grid derived earlier in _determine_histogram_bins.
        bin_centers = self.null_distributions_["histogram_bins"].astype(np.float64, copy=False)
        step_size = bin_centers[1] - bin_centers[0]
        inv_step_size = 1 / step_size
        n_bins = bin_centers.shape[0]
        mask_voxel_recip = 1.0 / self.__n_mask_voxels
        n_exp = ma_maps.shape[0]
        data = ma_maps.data
        indptr = ma_maps.indptr

        ale_hist = None
        tmp_hist = np.zeros(n_bins, dtype=np.float64)
        for exp_idx in range(n_exp):
            start = indptr[exp_idx]
            end = indptr[exp_idx + 1]
            study_ma_values = data[start:end]

            n_nonzero_voxels = study_ma_values.shape[0]
            n_zero_voxels = self.__n_mask_voxels - n_nonzero_voxels

            exp_hist = _study_ma_histogram(
                study_ma_values,
                n_zero_voxels,
                mask_voxel_recip,
                inv_step_size,
                n_bins,
            )

            if ale_hist is None:
                ale_hist = exp_hist.copy()
                continue

            ale_idx = np.where(ale_hist > 0)[0]
            exp_hist_idx = np.where(exp_hist > 0)[0]
            _update_ale_histogram(
                ale_idx,
                ale_hist[ale_idx],
                exp_hist_idx,
                exp_hist[exp_hist_idx],
                bin_centers,
                inv_step_size,
                n_bins,
                tmp_hist,
            )
            ale_hist, tmp_hist = tmp_hist, ale_hist

        self.null_distributions_["histweights_corr-none_method-approximate"] = ale_hist


class ALESubtraction(PairwiseCBMAEstimator):
    """ALE subtraction analysis.

    .. versionchanged:: 0.9.0

        - New parameters: ``vfwe_only`` and ``voxel_thresh``
          for montecarlo family wise error correction.

    .. versionchanged:: 0.2.1

        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.0.12

        - Use memmapped array for null distribution and remove ``memory_limit`` parameter.
        - Support parallelization and add progress bar.
        - Add ALE-difference (stat) and -log10(p) (logp) maps to results.
        - Use a 4D sparse array for modeled activation maps.

    .. versionchanged:: 0.0.8

        * [FIX] Assume non-symmetric null distribution.

    .. versionchanged:: 0.0.7

        * [FIX] Assume a zero-centered and symmetric null distribution.

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is ALEKernel.
    n_iters : :obj:`int`, default=5000
        Default is 5000.
    voxel_thresh : :obj:`float`, default=0.001
        Uncorrected voxel-level p-value threshold used for cluster-defining threshold when
        cluster nulls are computed.
    low_memory : {False, True, "auto"}, default="auto"
        Best-effort strategy for reducing resident memory used by study-wise MA maps during
        permutations. When ALESubtraction generates MA maps from coordinates, low-memory mode
        builds each group's CSR MA maps chunk by chunk and stores the chunks as disk-backed
        memmaps before permutation-based resampling. If "auto", only activate this behavior when
        the projected combined MA-map footprint exceeds roughly half of the currently available
        system memory. Precomputed MA maps are used as provided and do not participate in this
        chunked path.
    vfwe_only : :obj:`bool`, default=True
        If True, only compute voxel-level null information. If False, also compute and retain
        cluster size and mass null distributions from the permutation maps.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    n_cores : :obj:`int`, default=1
        Number of processes to use for meta-analysis. If -1, use all available cores.
        Default is 1.

        .. versionadded:: 0.0.12
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix ``kernel__`` in the variable name. Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.maskers.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For ALESubtraction, this includes edited coordinate DataFrames
        for both groups, stored under ``coordinates1`` and ``coordinates2``.

    Notes
    -----
    This method was originally developed in :footcite:t:`laird2005ale` and refined in
    :footcite:t:`eickhoff2012activation`.

    The ALE subtraction algorithm is also implemented as part of the GingerALE app provided by the
    BrainMap organization (https://www.brainmap.org/ale/).

    The voxel-wise null distributions used by this Estimator are very large, so they are not
    retained as Estimator attributes. However, summary distributions (e.g., per-iteration
    maximum statistics for voxel-level FWE correction) are retained.

    Warnings
    --------
    This implementation contains one key difference from the original version.

    In the original version, group 1 > group 2 difference values are only evaluated for voxels
    significant in the group 1 meta-analysis, and group 2 > group 1 difference values are only
    evaluated for voxels significant in the group 2 meta-analysis.

    In NiMARE's implementation, the analysis is run in a two-sided manner for *all* voxels in the
    mask.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        kernel_transformer=ALEKernel,
        n_iters=5000,
        voxel_thresh=0.001,
        low_memory="auto",
        vfwe_only=True,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        n_cores=1,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, ALEKernel) or kernel_transformer == ALEKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(
            kernel_transformer=kernel_transformer,
            memory=memory,
            memory_level=memory_level,
            **kwargs,
        )

        self.dataset1 = None
        self.dataset2 = None
        self.n_iters = n_iters
        self.voxel_thresh = voxel_thresh
        self.low_memory = low_memory
        self.vfwe_only = vfwe_only
        self.n_cores = _check_ncores(n_cores)
        self._permutation_parallel_backend = "threading"
        self._low_memory_fraction = 0.5
        # memory_limit needs to exist to trigger use_memmap decorator, but it will also be used if
        # a Dataset with pre-generated MA maps is provided.
        self.memory_limit = "100mb"

        if self.low_memory not in (False, True, "auto"):
            raise ValueError(
                "low_memory must be False, True, or 'auto'; " f"got {self.low_memory!r}."
            )

        if not self.vfwe_only:
            if self.voxel_thresh is None:
                raise ValueError("voxel_thresh must be provided when vfwe_only is False.")

            # Enforce scalar numeric voxel-wise threshold
            try:
                voxel_thresh_float = float(self.voxel_thresh)
            except (TypeError, ValueError):
                raise TypeError(
                    "voxel_thresh must be a scalar numeric value when vfwe_only is False; "
                    f"got {type(self.voxel_thresh).__name__}."
                )

            if not 0 < voxel_thresh_float < 1:
                raise ValueError(
                    "voxel_thresh must be between 0 and 1 (exclusive) when vfwe_only is False; "
                    f"got {self.voxel_thresh!r}."
                )

    def _generate_description(self):
        if (
            hasattr(self.kernel_transformer, "sample_size")  # Only kernels that allow sample sizes
            and (self.kernel_transformer.sample_size is None)
            and (self.kernel_transformer.fwhm is None)
        ):
            # Get the total number of subjects in the inputs.
            n_subjects = (
                self.inputs_["coordinates1"].groupby("id")["sample_size"].mean().values.sum()
            )
            sample_size_str1 = f", with a total of {int(n_subjects)} participants"
            n_subjects = (
                self.inputs_["coordinates2"].groupby("id")["sample_size"].mean().values.sum()
            )
            sample_size_str2 = f", with a total of {int(n_subjects)} participants"
        else:
            sample_size_str1 = ""
            sample_size_str2 = ""

        description = (
            "An activation likelihood estimation (ALE) subtraction analysis "
            "\\citep{laird2005ale,eickhoff2012activation} was performed with NiMARE "
            f"v{__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), "
            f"using a(n) {self.kernel_transformer.__class__.__name__.replace('Kernel', '')} "
            "kernel. "
            f"{self.kernel_transformer._generate_description()} "
            "The subtraction analysis was implemented according to NiMARE's \\citep{Salo2023} "
            "approach, which differs from the original version. "
            "In this version, ALE-difference scores are calculated between the two datasets, "
            "for all voxels in the mask, rather than for voxels significant in the main effects "
            "analyses of the two datasets. "
            "Next, voxel-wise null distributions of ALE-difference scores were generated via a "
            "randomized group assignment procedure, in which the studies in the two datasets were "
            "randomly reassigned and ALE-difference scores were calculated for the randomized "
            "datasets. "
            f"This randomization procedure was repeated {self.n_iters} times to build the null "
            "distributions. "
            "The significance of the original ALE-difference scores was assessed using a "
            "two-sided statistical test. "
            "The null distributions were assumed to be asymmetric, as ALE-difference scores will "
            "be skewed based on the sample sizes of the two datasets. "
            f"The first input dataset (group1) included {self.inputs_['coordinates1'].shape[0]} "
            f"foci from {len(self.inputs_['id1'])} experiments{sample_size_str1}. "
            f"The second input dataset (group2) included {self.inputs_['coordinates2'].shape[0]} "
            f"foci from {len(self.inputs_['id2'])} experiments{sample_size_str2}. "
        )
        return description

    @use_memmap(LGR, n_files=3)
    def _fit(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.masker = self.masker or dataset1.masker
        self.null_distributions_ = {}
        iter_diff_values = None

        with self._managed_pairwise_ma_store(
            maps_key1="ma_maps1",
            coords_key1="coordinates1",
            maps_key2="ma_maps2",
            coords_key2="coordinates2",
        ) as ma_store:
            diff_ale_values = ma_store.group1_stat - ma_store.group2_stat

            try:
                if not self.vfwe_only:
                    # Cluster-level nulls still require access to all permutation maps.
                    iter_diff_values = np.memmap(
                        self.memmap_filenames[2],
                        dtype=DEFAULT_FLOAT_DTYPE,
                        mode="w+",
                        shape=(self.n_iters, ma_store.n_voxels),
                    )

                iter_abs_max, p_values, diff_signs = self._run_null_permutations(
                    ma_store,
                    n_iters=self.n_iters,
                    n_cores=self.n_cores,
                    diff_ale_values=diff_ale_values,
                    iter_diff_values=iter_diff_values,
                )
                self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"] = (
                    iter_abs_max
                )

                if not self.vfwe_only:
                    if self.voxel_thresh is None:
                        raise ValueError("voxel_thresh must be provided when vfwe_only is False.")

                    ss_thresh, iter_max_sizes, iter_max_masses = self._compute_cluster_nulls(
                        iter_diff_values,
                        voxel_thresh=self.voxel_thresh,
                        n_iters=self.n_iters,
                    )
                    self.null_distributions_[
                        "summary_stat_thresh_level-voxel_corr-fwe_method-montecarlo"
                    ] = ss_thresh
                    self.null_distributions_[
                        "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
                    ] = iter_max_sizes
                    self.null_distributions_[
                        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
                    ] = iter_max_masses
            finally:
                if isinstance(iter_diff_values, np.memmap):
                    LGR.debug(f"Closing memmap at {iter_diff_values.filename}")
                    iter_diff_values._mmap.close()

        z_arr = p_to_z(p_values, tail="two") * diff_signs
        logp_arr = -np.log10(p_values)

        maps = {
            "stat_desc-group1MinusGroup2": diff_ale_values,
            "p_desc-group1MinusGroup2": p_values,
            "z_desc-group1MinusGroup2": z_arr,
            "logp_desc-group1MinusGroup2": logp_arr,
        }
        description = self._description_text()

        return maps, {}, description

    def _compute_summarystat_est(self, ma_values):
        return _compute_ale_summarystat(
            require_masked_csr(ma_values) if sp_sparse.isspmatrix(ma_values) else ma_values
        )

    def _run_permutation(self, i_iter, ma_store):
        """Run a single permutation of the ALESubtraction null distribution procedure.

        Parameters
        ----------
        i_iter : :obj:`int`
            The iteration number.
        ma_store : :class:`_PairwiseMAStore`
            Pairwise MA-map storage containing either in-memory CSR matrices or chunked
            disk-backed CSR groups.

        Returns
        -------
        i_iter : :obj:`int`
            The iteration number.
        iter_diff_values : :obj:`numpy.ndarray` of shape (V,)
            The null ALE-difference scores for one permutation.
        """
        gen = np.random.default_rng(seed=i_iter)
        id_idx = np.arange(ma_store.n_total)
        gen.shuffle(id_idx)
        iter_grp1_ale_values = ma_store.compute_partition_summarystat(id_idx[: ma_store.n_group1])
        iter_grp2_ale_values = ma_store.compute_partition_summarystat(id_idx[ma_store.n_group1 :])
        return i_iter, iter_grp1_ale_values - iter_grp2_ale_values

    def _iterate_permutation_diffs(self, ma_store, n_iters, n_cores):
        """Yield permutation difference maps for one pairwise MA store."""
        parallel_kwargs = {
            "return_as": "generator",
            "n_jobs": _check_ncores(n_cores),
            "backend": self._permutation_parallel_backend,
        }
        return tqdm(
            Parallel(**parallel_kwargs)(
                delayed(self._run_permutation)(i_iter, ma_store) for i_iter in range(n_iters)
            ),
            total=n_iters,
        )

    def _run_null_permutations(
        self, ma_store, n_iters, n_cores, diff_ale_values=None, iter_diff_values=None
    ):
        """Run Monte Carlo null permutations and optionally stream ALE-difference tail counts."""
        iter_abs_max = np.empty(n_iters, dtype=DEFAULT_FLOAT_DTYPE)
        left_counts = right_counts = None

        if diff_ale_values is not None:
            left_counts = np.zeros(ma_store.n_voxels, dtype=np.uint32)
            right_counts = np.zeros(ma_store.n_voxels, dtype=np.uint32)

        for i_iter, iter_diff in self._iterate_permutation_diffs(ma_store, n_iters, n_cores):
            iter_abs_max[i_iter] = np.max(np.abs(iter_diff))
            if diff_ale_values is not None:
                left_counts += iter_diff >= diff_ale_values
                right_counts += iter_diff <= diff_ale_values
            if iter_diff_values is not None:
                iter_diff_values[i_iter, :] = iter_diff

        p_values = diff_signs = None
        if diff_ale_values is not None:
            p_values, diff_signs = _finalize_alediff_tail_counts(
                left_counts, right_counts, n_iters
            )

        return iter_abs_max, p_values, diff_signs

    def _compute_cluster_nulls(self, iter_diff_values, voxel_thresh, n_iters):
        """Compute cluster-forming threshold and cluster null summaries from permutation maps."""
        ss_thresh = np.quantile(np.abs(iter_diff_values), 1 - voxel_thresh)
        conn = ndimage.generate_binary_structure(rank=3, connectivity=1)
        iter_max_sizes = np.zeros(n_iters, dtype=DEFAULT_FLOAT_DTYPE)
        iter_max_masses = np.zeros(n_iters, dtype=DEFAULT_FLOAT_DTYPE)

        for i_iter in range(n_iters):
            iter_map = self.masker.inverse_transform(iter_diff_values[i_iter, :]).get_fdata(
                dtype=DEFAULT_FLOAT_DTYPE
            )
            iter_max_sizes[i_iter], iter_max_masses[i_iter] = _calculate_cluster_measures(
                iter_map, ss_thresh, conn, tail="two"
            )

        return ss_thresh, iter_max_sizes, iter_max_masses

    def _should_use_low_memory(self, projected_nbytes):
        """Determine whether best-effort chunked MA-map storage should be used."""
        if self.low_memory is True:
            return True
        if self.low_memory is False:
            return False

        available_bytes = _get_available_memory_bytes()
        if available_bytes is None:
            return False
        return projected_nbytes >= (available_bytes * self._low_memory_fraction)

    def _estimate_group_ma_bytes(self, coords_key):
        """Estimate total CSR bytes and bytes per study for one MA-map group."""
        coordinates = self.inputs_[coords_key]
        sample_n_studies = min(32, len(np.unique(coordinates["id"].values)))
        sample_df = next(_iter_study_id_chunks(coordinates, chunk_rows=sample_n_studies))
        sample_ma = require_masked_csr(
            self.kernel_transformer.transform(
                sample_df,
                masker=self.masker,
                return_type="sparse",
            ),
            source=f"Generated sample for {coords_key}",
        )
        bytes_per_study = _estimate_csr_nbytes(sample_ma) / max(sample_ma.shape[0], 1)
        total_bytes = bytes_per_study * len(np.unique(coordinates["id"].values))
        return _GroupMAEstimate(
            total_bytes=total_bytes,
            bytes_per_study=bytes_per_study,
            sample_ma=sample_ma,
            sample_n_studies=sample_n_studies,
        )

    def _determine_chunk_rows(self, bytes_per_study, available_bytes=None):
        """Determine how many studies to transform per chunk in low-memory mode."""
        chunk_bytes = _determine_low_memory_chunk_bytes(available_bytes=available_bytes)
        return max(1, int(chunk_bytes / max(bytes_per_study, 1.0)))

    def _collect_chunked_ma_maps(self, coords_key, chunk_rows, prefix, estimate=None):
        """Collect one MA-map group into memmap-backed CSR chunks."""
        temp_files = []
        chunked_maps = []
        row_offsets = [0]
        log_sums = None
        coordinates = self.inputs_[coords_key]
        start_idx = 0

        if estimate is not None and estimate.sample_ma is not None:
            if estimate.sample_n_studies <= chunk_rows:
                start_idx = estimate.sample_n_studies
                initial_chunks = [estimate.sample_ma]
            else:
                initial_chunks = []
        else:
            initial_chunks = []

        chunk_iter = (
            require_masked_csr(
                self.kernel_transformer.transform(
                    chunk_df,
                    masker=self.masker,
                    return_type="sparse",
                ),
                source=f"Generated {coords_key} chunk",
            )
            for chunk_df in _iter_study_id_chunks(coordinates, chunk_rows, start_idx=start_idx)
        )

        for i_chunk, chunk_ma in enumerate(chain(initial_chunks, chunk_iter)):
            if log_sums is None:
                log_sums = np.zeros(chunk_ma.shape[1], dtype=np.float64)

            _accumulate_csr_log_sums(chunk_ma, log_sums)
            chunk_ma, chunk_files = _csr_to_memmap(chunk_ma, prefix=f"{prefix}{i_chunk:04d}")
            temp_files.extend(chunk_files)
            chunked_maps.append(chunk_ma)
            row_offsets.append(row_offsets[-1] + chunk_ma.shape[0])

        if log_sums is None:
            raise ValueError(f"No studies were available for {coords_key}.")

        stat_values = (1.0 - np.exp(log_sums)).astype(DEFAULT_FLOAT_DTYPE, copy=False)
        ma_group = _ChunkedCSRGroup(
            chunks=chunked_maps,
            row_offsets=np.asarray(row_offsets, dtype=np.int64),
            shape=(row_offsets[-1], log_sums.shape[0]),
        )
        return ma_group, stat_values, temp_files

    def _prepare_pairwise_ma_maps(self, maps_key1, coords_key1, maps_key2, coords_key2):
        """Collect pairwise MA maps and optionally spill coordinate-generated maps to disk."""
        temp_files = []

        if maps_key1 in self.inputs_ or maps_key2 in self.inputs_:
            if self.low_memory is not False:
                LGR.info(
                    "ALESubtraction low-memory chunking is only applied when MA maps are "
                    "generated from coordinates; using precomputed MA maps without chunking."
                )
            ma_maps1 = _collect_masked_ma_maps(self, maps_key=maps_key1, coords_key=coords_key1)
            ma_maps2 = _collect_masked_ma_maps(self, maps_key=maps_key2, coords_key=coords_key2)
            grp1_ale_values = self._compute_summarystat_est(ma_maps1)
            grp2_ale_values = self._compute_summarystat_est(ma_maps2)
            return _PairwiseMAStore(
                group1=ma_maps1,
                group2=ma_maps2,
                group1_stat=grp1_ale_values,
                group2_stat=grp2_ale_values,
                temp_files=temp_files,
            )

        grp1_estimate = self._estimate_group_ma_bytes(coords_key1)
        grp2_estimate = self._estimate_group_ma_bytes(coords_key2)
        combined_nbytes = grp1_estimate.total_bytes + grp2_estimate.total_bytes

        if self._should_use_low_memory(combined_nbytes):
            available_bytes = _get_available_memory_bytes()
            chunk_rows1 = self._determine_chunk_rows(
                grp1_estimate.bytes_per_study, available_bytes=available_bytes
            )
            chunk_rows2 = self._determine_chunk_rows(
                grp2_estimate.bytes_per_study, available_bytes=available_bytes
            )
            LGR.info(
                "ALESubtraction low-memory chunked mode activated for permutation MA maps "
                "(projected %.2f GB; chunk_rows=%d/%d).",
                combined_nbytes / float(1024**3),
                chunk_rows1,
                chunk_rows2,
            )
            ma_maps1, grp1_ale_values, group1_files = self._collect_chunked_ma_maps(
                coords_key=coords_key1,
                chunk_rows=chunk_rows1,
                prefix="ALESubtractionGroup1Chunk",
                estimate=grp1_estimate,
            )
            ma_maps2, grp2_ale_values, group2_files = self._collect_chunked_ma_maps(
                coords_key=coords_key2,
                chunk_rows=chunk_rows2,
                prefix="ALESubtractionGroup2Chunk",
                estimate=grp2_estimate,
            )
            temp_files.extend(group1_files)
            temp_files.extend(group2_files)
            return _PairwiseMAStore(
                group1=ma_maps1,
                group2=ma_maps2,
                group1_stat=grp1_ale_values,
                group2_stat=grp2_ale_values,
                temp_files=temp_files,
            )

        ma_maps1 = _collect_masked_ma_maps(self, maps_key=maps_key1, coords_key=coords_key1)
        ma_maps2 = _collect_masked_ma_maps(self, maps_key=maps_key2, coords_key=coords_key2)
        grp1_ale_values = self._compute_summarystat_est(ma_maps1)
        grp2_ale_values = self._compute_summarystat_est(ma_maps2)
        return _PairwiseMAStore(
            group1=ma_maps1,
            group2=ma_maps2,
            group1_stat=grp1_ale_values,
            group2_stat=grp2_ale_values,
            temp_files=temp_files,
        )

    @contextmanager
    def _managed_pairwise_ma_store(self, maps_key1, coords_key1, maps_key2, coords_key2):
        """Yield pairwise MA-map storage and guarantee cleanup on exit."""
        ma_store = self._prepare_pairwise_ma_maps(maps_key1, coords_key1, maps_key2, coords_key2)
        try:
            yield ma_store
        finally:
            ma_store.close()

    def correct_fwe_montecarlo(
        self,
        result,
        voxel_thresh=0.001,
        n_iters=None,
        n_cores=1,
        vfwe_only=False,
    ):
        """Perform FWE correction using the max-value permutation method.

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from an ALE subtraction analysis.
        voxel_thresh : :obj:`float`, default=0.001
            Cluster-defining p-value threshold. Default is 0.001.
        n_iters : :obj:`int`, optional
            Number of iterations to build the voxel-level, cluster-size, and cluster-mass FWE
            null distributions. If None, defaults to the Estimator's ``n_iters``.
        n_cores : :obj:`int`, default=1
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.
        vfwe_only : :obj:`bool`, default=False
            If True, only calculate the voxel-level FWE-corrected maps.

        Returns
        -------
        maps : :obj:`dict`
            Dictionary of corrected map arrays.
        """
        stat_values = result.get_map("stat_desc-group1MinusGroup2", return_type="array")
        z_values = result.get_map("z_desc-group1MinusGroup2", return_type="array")
        sign = np.sign(z_values)
        eps = np.spacing(1)

        if n_iters is None:
            n_iters = self.n_iters
        if voxel_thresh is None:
            voxel_thresh = self.voxel_thresh

        expected = (self.n_iters, self.voxel_thresh, self.vfwe_only)
        requested = (n_iters, voxel_thresh, vfwe_only)

        has_vfwe_null = "values_level-voxel_corr-fwe_method-montecarlo" in self.null_distributions_
        has_cluster_null = all(
            key in self.null_distributions_
            for key in (
                "values_desc-size_level-cluster_corr-fwe_method-montecarlo",
                "values_desc-mass_level-cluster_corr-fwe_method-montecarlo",
                "summary_stat_thresh_level-voxel_corr-fwe_method-montecarlo",
            )
        )

        use_cached = requested == expected and has_vfwe_null and (vfwe_only or has_cluster_null)

        if not use_cached:
            warnings.warn(
                "ALESubtraction FWE correction is recomputing permutations because the "
                "requested parameters do not match the estimator's cached null model "
                "(or cached nulls are missing).\n"
                f"Estimator: n_iters={expected[0]}, voxel_thresh={expected[1]}, "
                f"vfwe_only={expected[2]}\n"
                f"Requested: n_iters={requested[0]}, voxel_thresh={requested[1]}, "
                f"vfwe_only={requested[2]}\n"
                "Recomputing Monte Carlo nulls for correction.",
                UserWarning,
            )

            iter_diff_values = None
            tmp_path = None
            with self._managed_pairwise_ma_store(
                maps_key1="ma_maps1",
                coords_key1="coordinates1",
                maps_key2="ma_maps2",
                coords_key2="coordinates2",
            ) as ma_store:
                try:
                    if not vfwe_only:
                        fd, tmp_path = tempfile.mkstemp(prefix="ALESubtractionFWE", suffix=".mmap")
                        os.close(fd)
                        iter_diff_values = np.memmap(
                            tmp_path,
                            dtype=DEFAULT_FLOAT_DTYPE,
                            mode="w+",
                            shape=(n_iters, stat_values.shape[0]),
                        )

                    vfwe_null, _, _ = self._run_null_permutations(
                        ma_store,
                        n_iters=n_iters,
                        n_cores=n_cores,
                        iter_diff_values=iter_diff_values,
                    )
                    self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"] = (
                        vfwe_null
                    )

                    if not vfwe_only:
                        ss_thresh, iter_max_sizes, iter_max_masses = self._compute_cluster_nulls(
                            iter_diff_values,
                            voxel_thresh=voxel_thresh,
                            n_iters=n_iters,
                        )
                        self.null_distributions_[
                            "summary_stat_thresh_level-voxel_corr-fwe_method-montecarlo"
                        ] = ss_thresh
                        self.null_distributions_[
                            "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
                        ] = iter_max_sizes
                        self.null_distributions_[
                            "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
                        ] = iter_max_masses
                finally:
                    if isinstance(iter_diff_values, np.memmap):
                        iter_diff_values._mmap.close()
                    if tmp_path and os.path.isfile(tmp_path):
                        os.remove(tmp_path)

        vfwe_null = self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
        p_vfwe_vals = null_to_p(np.abs(stat_values), vfwe_null, tail="upper")
        z_vfwe_vals = p_to_z(p_vfwe_vals, tail="two") * sign
        logp_vfwe_vals = -np.log10(p_vfwe_vals)
        logp_vfwe_vals[np.isinf(logp_vfwe_vals)] = -np.log10(eps)

        maps = {
            "p_desc-group1MinusGroup2_level-voxel": p_vfwe_vals,
            "z_desc-group1MinusGroup2_level-voxel": z_vfwe_vals,
            "logp_desc-group1MinusGroup2_level-voxel": logp_vfwe_vals,
        }

        if not vfwe_only:
            csfwe_null = self.null_distributions_[
                "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
            ]
            cmfwe_null = self.null_distributions_[
                "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
            ]
            ss_thresh = self.null_distributions_[
                "summary_stat_thresh_level-voxel_corr-fwe_method-montecarlo"
            ]

            stat_map = self.masker.inverse_transform(stat_values).get_fdata(
                dtype=DEFAULT_FLOAT_DTYPE
            )
            stat_map[np.abs(stat_map) <= ss_thresh] = 0

            conn = ndimage.generate_binary_structure(rank=3, connectivity=1)
            labeled_pos, _ = ndimage.label(stat_map > 0, conn)
            labeled_neg, _ = ndimage.label(stat_map < 0, conn)
            if labeled_pos.size:
                labeled_neg[labeled_neg > 0] += labeled_pos.max()
            labeled = labeled_pos + labeled_neg

            cluster_labels, idx, cluster_sizes = np.unique(
                labeled, return_inverse=True, return_counts=True
            )
            cluster_sizes[0] = 0

            cluster_masses = np.zeros(cluster_labels.shape, dtype=DEFAULT_FLOAT_DTYPE)
            for label in cluster_labels[1:]:
                ss_vals = np.abs(stat_map[labeled == label]) - ss_thresh
                cluster_masses[label] = np.sum(ss_vals)

            p_cmfwe_vals = null_to_p(cluster_masses, cmfwe_null, tail="upper")
            p_cmfwe_map = p_cmfwe_vals[idx].reshape(labeled.shape)
            p_cmfwe_values = np.squeeze(
                self.masker.transform(nib.Nifti1Image(p_cmfwe_map, self.masker.mask_img.affine))
            )

            p_csfwe_vals = null_to_p(cluster_sizes, csfwe_null, tail="upper")
            p_csfwe_map = p_csfwe_vals[idx].reshape(labeled.shape)
            p_csfwe_values = np.squeeze(
                self.masker.transform(nib.Nifti1Image(p_csfwe_map, self.masker.mask_img.affine))
            )

            z_cmfwe_vals = p_to_z(p_cmfwe_values, tail="two") * sign
            logp_cmfwe_vals = -np.log10(p_cmfwe_values)
            logp_cmfwe_vals[np.isinf(logp_cmfwe_vals)] = -np.log10(eps)

            z_csfwe_vals = p_to_z(p_csfwe_values, tail="two") * sign
            logp_csfwe_vals = -np.log10(p_csfwe_values)
            logp_csfwe_vals[np.isinf(logp_csfwe_vals)] = -np.log10(eps)

            maps.update(
                {
                    "p_desc-group1MinusGroup2Mass_level-cluster": p_cmfwe_values,
                    "z_desc-group1MinusGroup2Mass_level-cluster": z_cmfwe_vals,
                    "logp_desc-group1MinusGroup2Mass_level-cluster": logp_cmfwe_vals,
                    "p_desc-group1MinusGroup2Size_level-cluster": p_csfwe_values,
                    "z_desc-group1MinusGroup2Size_level-cluster": z_csfwe_vals,
                    "logp_desc-group1MinusGroup2Size_level-cluster": logp_csfwe_vals,
                }
            )

        if vfwe_only:
            description = (
                "Family-wise error correction was performed using a voxel-level Monte Carlo "
                "procedure for ALE subtraction. "
                "In this procedure, experiments from the two input datasets were randomly "
                "reassigned between groups while preserving the original group sizes, and the "
                "maximum absolute ALE-difference value was retained. "
                f"This procedure was repeated {n_iters} times to build a null distribution of "
                "summary statistics."
            )
        else:
            description = (
                "Family-wise error rate correction was performed using a Monte Carlo procedure "
                "for ALE subtraction. "
                "In this procedure, experiments from the two input datasets were randomly "
                "reassigned between groups while preserving the original group sizes, and "
                "maximum values were retained. "
                f"This procedure was repeated {n_iters} times to build null distributions of "
                "summary statistics, cluster sizes, and cluster masses. "
                "Clusters for cluster-level correction were defined using face-wise connectivity "
                f"and a voxel-level threshold of p < {voxel_thresh} from the uncorrected ALE-"
                "difference null distribution."
            )

        return maps, {}, description


class SCALE(CBMAEstimator):
    r"""Specific coactivation likelihood estimation.

    This method was originally introduced in :footcite:t:`langner2014meta`.

    .. versionchanged:: 0.14.0

        Use direct empirical voxelwise permutation p-values and add voxel-level Monte Carlo
        family-wise error correction.

    .. versionchanged:: 0.14.0

        Stream permutation exceedance counts instead of retaining a full voxelwise permutation
        null matrix in the main SCALE fit path.

    .. versionchanged:: 0.2.1

        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.0.12

        - Remove unused parameters ``voxel_thresh`` and ``memory_limit``.
        - Use memmapped array for null distribution.
        - Use a 4D sparse array for modeled activation maps.

    .. versionchanged:: 0.0.10

        Replace ``ijk`` with ``xyz``. This should be easier for users to collect.

    Parameters
    ----------
    xyz : (N x 3) :obj:`numpy.ndarray`
        Numpy array with XYZ coordinates.
        Voxels are rows and x, y, z (meaning coordinates) values are the three columnns.

        .. versionchanged:: 0.0.12

            This parameter was previously incorrectly labeled as "optional" and indicated that
            it supports tab-delimited files, which it does not (yet).

    n_iters : int, default=5000
        Number of iterations for statistical inference. Default: 5000
    n_cores : int, default=1
        Number of processes to use for meta-analysis. If -1, use all available cores.
        Default: 1
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`~nimare.meta.kernel.ALEKernel`.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix '\kernel__' in the variable name.

    Attributes
    ----------
    masker : :class:`~nilearn.maskers.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key: coordinates.
        This is an edited version of the dataset's coordinates DataFrame.
    null_distributions_ : :obj:`dict` of :class:`numpy.ndarray`
        Null distribution information.
        Entries are added to this attribute if and when the corresponding method is applied.

        .. important::
            The voxel-wise null distributions used by this Estimator are very large, so they are
            not retained as Estimator attributes.

        If :meth:`correct_fwe_montecarlo` is applied:

            -   ``values_level-voxel_corr-fwe_method-montecarlo``: The maximum summary statistic
                value from each SCALE permutation. An array of shape ``(n_iters,)``.

    Notes
    -----
    SCALE uses voxel-specific empirical null distributions derived from the supplied reference
    coordinate pool. NiMARE therefore supports voxel-level Monte Carlo FWE correction for SCALE,
    but does not implement cluster-level Monte Carlo FWE correction.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        xyz,
        n_iters=5000,
        n_cores=1,
        kernel_transformer=ALEKernel,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, ALEKernel) or kernel_transformer == ALEKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(
            kernel_transformer=kernel_transformer,
            memory=memory,
            memory_level=memory_level,
            **kwargs,
        )

        if not isinstance(xyz, np.ndarray):
            raise TypeError(f"Parameter 'xyz' must be a numpy.ndarray, not a {type(xyz)}")
        elif xyz.ndim != 2:
            raise ValueError(f"Parameter 'xyz' must be a 2D array, but has {xyz.ndim} dimensions")
        elif xyz.shape[1] != 3:
            raise ValueError(f"Parameter 'xyz' must have 3 columns, but has shape {xyz.shape}")

        self.xyz = xyz
        self.n_iters = n_iters
        self.n_cores = _check_ncores(n_cores)

    def _generate_description(self):
        if (
            hasattr(self.kernel_transformer, "sample_size")  # Only kernels that allow sample sizes
            and (self.kernel_transformer.sample_size is None)
            and (self.kernel_transformer.fwhm is None)
        ):
            # Get the total number of subjects in the inputs.
            n_subjects = (
                self.inputs_["coordinates"].groupby("id")["sample_size"].mean().values.sum()
            )
            sample_size_str = f", with a total of {int(n_subjects)} participants"
        else:
            sample_size_str = ""

        description = (
            "A specific coactivation likelihood estimation (SCALE) meta-analysis "
            "\\citep{langner2014meta} was performed with NiMARE "
            f"{__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), with "
            f"{self.n_iters} iterations. "
            f"The input dataset included {self.inputs_['coordinates'].shape[0]} foci from "
            f"{len(self.inputs_['id'])} experiments{sample_size_str}."
        )
        return description

    def _fit(self, dataset):
        """Perform specific coactivation likelihood estimation meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        ma_values = _collect_masked_ma_maps(self, coords_key="coordinates", maps_key="ma_maps")

        stat_values = self._compute_summarystat_est(ma_values)

        del ma_values

        iter_df, voxel_ijk, permutation_args, sampled_voxel_idx = self._prepare_permutations(
            self.n_iters
        )
        exceedance_counts = np.zeros(stat_values.shape[0], dtype=np.uint32)

        for iter_values in self._iterate_permuted_stats(
            sampled_voxel_idx, voxel_ijk, iter_df, permutation_args, self.n_cores
        ):
            exceedance_counts += iter_values >= stat_values

        p_values, z_values = self._scale_to_p(stat_values, exceedance_counts)

        logp_values = -np.log10(p_values)
        logp_values[np.isinf(logp_values)] = -np.log10(np.finfo(float).eps)

        # Write out unthresholded value images
        maps = {"stat": stat_values, "logp": logp_values, "z": z_values}
        description = self._description_text()

        return maps, {}, description

    def _compute_summarystat_est(self, data):
        """Generate SCALE ALE summary statistics from contrast data."""
        if isinstance(data, pd.DataFrame):
            ma_values = self.kernel_transformer.transform(
                data, masker=self.masker, return_type="sparse"
            )
        elif isinstance(data, np.ndarray) or sp_sparse.isspmatrix(data):
            ma_values = data
        else:
            raise ValueError(f"Unsupported data type '{type(data)}'")

        return _compute_ale_summarystat(
            require_masked_csr(ma_values) if sp_sparse.isspmatrix(ma_values) else ma_values
        )

    def _prepare_permutations(self, n_iters):
        """Prepare shared SCALE permutation inputs."""
        iter_df = self.inputs_["coordinates"].copy()
        voxel_ijk = mm2vox(self.xyz, self.masker.mask_img.affine).astype(np.int32, copy=False)
        permutation_args = self._prepare_permutation_args(iter_df)
        sampled_voxel_idx = np.random.choice(voxel_ijk.shape[0], size=(iter_df.shape[0], n_iters))
        return iter_df, voxel_ijk, permutation_args, sampled_voxel_idx

    def _prepare_permutation_args(self, coordinates):
        """Prepare static ALE kernel inputs for SCALE permutations."""
        if not isinstance(self.kernel_transformer, ALEKernel):
            return None

        use_dict = True
        kernel = None
        if self.kernel_transformer.sample_size is not None:
            sample_sizes = self.kernel_transformer.sample_size
            use_dict = False
        elif self.kernel_transformer.fwhm is None:
            sample_sizes = coordinates["sample_size"].values
        else:
            sample_sizes = None

        if self.kernel_transformer.fwhm is not None:
            _, kernel = get_ale_kernel(self.masker.mask_img, fwhm=self.kernel_transformer.fwhm)
            use_dict = False

        return {
            "exp_idx": coordinates["id"].values,
            "sample_sizes": sample_sizes,
            "use_dict": use_dict,
            "kernel": kernel,
        }

    def _scale_to_p(self, stat_values, scale_values):
        """Compute p- and z-values from voxelwise exceedance counts.

        Parameters
        ----------
        stat_values : (V) array
            ALE values. Included for API consistency with other estimators.
        scale_values : (V,) array
            Voxelwise exceedance counts from SCALE permutations.

        Returns
        -------
        p_values : (V) array
        z_values : (V) array
        """
        del stat_values
        p_values = scale_values.astype(DEFAULT_FLOAT_DTYPE, copy=False) / self.n_iters
        smallest_value = np.maximum(np.finfo(float).eps, 1.0 / self.n_iters)
        p_values = np.maximum(smallest_value, np.minimum(p_values, 1.0 - smallest_value)).astype(
            DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values

    def _run_permutation(self, sampled_voxel_idx, voxel_ijk, iter_df, permutation_args=None):
        """Run a single random SCALE permutation from sampled voxel-row indices."""
        if permutation_args is not None:
            ma_values, _, _ = compute_ale_ma(
                self.masker.mask_img,
                voxel_ijk[sampled_voxel_idx, :],
                kernel=permutation_args["kernel"],
                exp_idx=permutation_args["exp_idx"],
                sample_sizes=permutation_args["sample_sizes"],
                use_dict=permutation_args["use_dict"],
            )
            return _compute_ale_summarystat(ma_values)

        iter_df = iter_df.copy()
        iter_df[["i", "j", "k"]] = voxel_ijk[sampled_voxel_idx, :]
        stat_values = self._compute_summarystat_est(iter_df)
        return stat_values

    def _iterate_permuted_stats(
        self, sampled_voxel_idx, voxel_ijk, iter_df, permutation_args, n_cores
    ):
        """Yield permuted SCALE statistic maps for a fixed permutation schedule."""
        if n_cores == 1:
            for i_iter in tqdm(
                range(sampled_voxel_idx.shape[1]), total=sampled_voxel_idx.shape[1]
            ):
                yield self._run_permutation(
                    sampled_voxel_idx[:, i_iter],
                    voxel_ijk,
                    iter_df,
                    permutation_args=permutation_args,
                )
            return

        parallel_kwargs = {
            "return_as": "generator",
            "n_jobs": n_cores,
        }
        yield from tqdm(
            Parallel(**parallel_kwargs)(
                delayed(self._run_permutation)(
                    sampled_voxel_idx[:, i_iter],
                    voxel_ijk,
                    iter_df,
                    permutation_args=permutation_args,
                )
                for i_iter in range(sampled_voxel_idx.shape[1])
            ),
            total=sampled_voxel_idx.shape[1],
        )

    def correct_fwe_montecarlo(
        self,
        result,
        voxel_thresh=None,
        n_iters=5000,
        n_cores=1,
        vfwe_only=True,
    ):
        """Perform voxel-level Monte Carlo FWE correction for SCALE.

        Notes
        -----
        This method implements only voxel-level max-statistic correction.
        Cluster-level Monte Carlo FWE correction is not implemented for SCALE because the method
        uses voxel-specific empirical null distributions rather than a single global null model.
        """
        if not vfwe_only:
            raise NotImplementedError(
                "SCALE only supports voxel-level Monte Carlo FWE correction. "
                "Cluster-level FWE is not implemented."
            )
        if voxel_thresh is not None:
            LGR.warning(
                "Ignoring voxel_thresh for SCALE Monte Carlo FWE correction; "
                "only voxel-level max-stat inference is supported."
            )

        stat_values = result.get_map("stat", return_type="array")
        iter_df, voxel_ijk, permutation_args, sampled_voxel_idx = self._prepare_permutations(
            n_iters
        )
        fwe_voxel_max = np.empty(n_iters, dtype=DEFAULT_FLOAT_DTYPE)
        n_cores = _check_ncores(n_cores)

        for i_iter, iter_values in enumerate(
            self._iterate_permuted_stats(
                sampled_voxel_idx, voxel_ijk, iter_df, permutation_args, n_cores
            )
        ):
            fwe_voxel_max[i_iter] = np.max(iter_values)

        p_vfwe_values = null_to_p(stat_values, fwe_voxel_max, tail="upper")
        z_vfwe_values = p_to_z(p_vfwe_values, tail="one")
        logp_vfwe_values = -np.log10(p_vfwe_values)
        logp_vfwe_values[np.isinf(logp_vfwe_values)] = -np.log10(np.finfo(float).eps)

        self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"] = fwe_voxel_max

        maps = {
            "logp_level-voxel": logp_vfwe_values,
            "z_level-voxel": z_vfwe_values,
        }
        description = (
            "Family-wise error correction was performed for SCALE using a voxel-level Monte Carlo "
            "max-statistic procedure. "
            "In this procedure, null datasets are generated by replacing dataset coordinates with "
            "coordinates randomly sampled from the SCALE reference coordinate pool, and the "
            f"maximum ALE summary statistic is retained. This procedure was repeated {n_iters} "
            "times to build a voxel-level FWE null distribution. "
            "Cluster-level Monte Carlo FWE correction is not implemented for SCALE."
        )
        return maps, {}, description
