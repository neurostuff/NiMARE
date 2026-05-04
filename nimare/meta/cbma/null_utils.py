"""Internal ALE null-model helpers."""

import copy
from dataclasses import dataclass

import numpy as np
from numba import jit
from scipy import sparse as sp_sparse

from nimare.meta.cbma.utils import require_masked_csr
from nimare.utils import DEFAULT_FLOAT_DTYPE


@dataclass
class _ChunkedCSRGroup:
    """Disk-backed study-by-voxel CSR chunks for one MA-map group."""

    chunks: list
    row_offsets: np.ndarray
    shape: tuple


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


def _group_row_max(ma_values):
    """Compute row-wise maxima for either a CSR matrix or a chunked CSR group."""
    if sp_sparse.isspmatrix(ma_values):
        return _csr_row_max(require_masked_csr(ma_values))
    if _is_chunked_group(ma_values):
        return np.concatenate([_csr_row_max(chunk) for chunk in ma_values.chunks]).astype(
            DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
    raise ValueError(f"Unsupported MA map container '{type(ma_values)}'.")


def _iter_group_study_values(ma_values):
    """Yield per-study nonzero MA values for a CSR matrix or chunked CSR group."""
    if sp_sparse.isspmatrix(ma_values):
        ma_values = require_masked_csr(ma_values)
        for i_row in range(ma_values.shape[0]):
            start = ma_values.indptr[i_row]
            end = ma_values.indptr[i_row + 1]
            yield ma_values.data[start:end]
        return

    if _is_chunked_group(ma_values):
        for chunk in ma_values.chunks:
            chunk = require_masked_csr(chunk)
            for i_row in range(chunk.shape[0]):
                start = chunk.indptr[i_row]
                end = chunk.indptr[i_row + 1]
                yield chunk.data[start:end]
        return

    raise ValueError(f"Unsupported MA map container '{type(ma_values)}'.")


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


def _compute_group_approximate_null(estimator, ma_maps):
    """Populate ALE approximate-null state from a CSR or chunked MA group."""
    estimator.null_distributions_ = {}
    estimator._study_max_ma_values = _group_row_max(ma_maps)

    inv_step_size = 100000
    step_size = 1 / inv_step_size
    max_ma_values = np.ceil(estimator._study_max_ma_values * inv_step_size) / inv_step_size
    max_poss_ale = estimator._compute_summarystat(max_ma_values)
    hist_bins = np.round(np.arange(0, max_poss_ale + (1.5 * step_size), step_size), 5)
    estimator.null_distributions_["histogram_bins"] = hist_bins

    bin_centers = hist_bins.astype(np.float64, copy=False)
    step_size = bin_centers[1] - bin_centers[0]
    inv_step_size = 1 / step_size
    n_bins = bin_centers.shape[0]
    mask_voxel_recip = 1.0 / estimator._ALE__n_mask_voxels

    ale_hist = None
    tmp_hist = np.zeros(n_bins, dtype=np.float64)
    for study_ma_values in _iter_group_study_values(ma_maps):
        n_nonzero_voxels = study_ma_values.shape[0]
        n_zero_voxels = estimator._ALE__n_mask_voxels - n_nonzero_voxels
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

    estimator.null_distributions_["histweights_corr-none_method-approximate"] = ale_hist


def _ale_approximate_z_from_ma(estimator, ma_maps):
    """Compute ALE summary statistics and approximate-null z values for one MA collection."""
    temp_estimator = copy.deepcopy(estimator)
    temp_estimator.null_distributions_ = {}
    temp_estimator._study_max_ma_values = _csr_row_max(ma_maps).astype(
        DEFAULT_FLOAT_DTYPE,
        copy=False,
    )
    stat_values = temp_estimator._compute_summarystat_est(ma_maps)
    temp_estimator._determine_histogram_bins(ma_maps)
    temp_estimator._compute_null_approximate(ma_maps)
    _, z_values = temp_estimator._summarystat_to_p(stat_values, null_method="approximate")
    return stat_values.astype(DEFAULT_FLOAT_DTYPE, copy=False), z_values.astype(
        DEFAULT_FLOAT_DTYPE,
        copy=False,
    )
