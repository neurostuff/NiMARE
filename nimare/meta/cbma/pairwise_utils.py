"""Internal pairwise helpers."""

import gc
from dataclasses import dataclass

import numpy as np

from nimare.meta.cbma.io_utils import _cleanup_temp_files, _close_csr_memmaps
from nimare.meta.cbma.null_utils import (
    _build_ale_temp_estimator,
    _compute_group_approximate_null,
    _is_chunked_group,
)
from nimare.meta.cbma.utils import require_masked_csr
from nimare.utils import DEFAULT_FLOAT_DTYPE, _p_to_logp_values


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
    return stat_values.astype(np.float64, copy=False)


def _prefix_ale_group_maps(maps, group_label):
    """Rename one-sample ALE maps for storage inside a pairwise result."""
    name_map = {
        "stat": f"stat_desc-{group_label}",
        "p": f"p_desc-{group_label}",
        "z": f"z_desc-{group_label}",
        "logp": f"logp_desc-{group_label}",
        "p_level-voxel": f"p_desc-{group_label}_level-voxel",
        "z_level-voxel": f"z_desc-{group_label}_level-voxel",
        "logp_level-voxel": f"logp_desc-{group_label}_level-voxel",
        "p_desc-size_level-cluster": f"p_desc-{group_label}Size_level-cluster",
        "z_desc-size_level-cluster": f"z_desc-{group_label}Size_level-cluster",
        "logp_desc-size_level-cluster": f"logp_desc-{group_label}Size_level-cluster",
        "p_desc-mass_level-cluster": f"p_desc-{group_label}Mass_level-cluster",
        "z_desc-mass_level-cluster": f"z_desc-{group_label}Mass_level-cluster",
        "logp_desc-mass_level-cluster": f"logp_desc-{group_label}Mass_level-cluster",
    }
    return {name_map[key]: value for key, value in maps.items() if key in name_map}


def _ale_uncorrected_group_maps(pairwise_estimator, ma_maps, group_label, stat_values=None):
    """Compute uncorrected one-sample ALE maps for a pairwise group."""
    temp_estimator = _build_ale_temp_estimator(pairwise_estimator)
    if stat_values is None:
        from nimare.meta.cbma.ale import _compute_ale_summarystat

        stat_values = _compute_ale_summarystat(ma_maps)
    temp_estimator._ALE__n_mask_voxels = stat_values.shape[0]
    _compute_group_approximate_null(temp_estimator, ma_maps)
    p_values, z_values = temp_estimator._summarystat_to_p(stat_values, null_method="approximate")
    maps = {
        "stat": stat_values.astype(np.float64, copy=False),
        "p": p_values.astype(np.float64, copy=False),
        "z": z_values.astype(np.float64, copy=False),
        "logp": _p_to_logp_values(p_values, dtype=DEFAULT_FLOAT_DTYPE),
    }
    return _prefix_ale_group_maps(maps, group_label)


def _resolve_balanced_target_n(dataset1, dataset2, target_n):
    """Infer or validate the matched-size study count for balanced pairwise resampling."""
    max_target_n = min(len(dataset1.ids), len(dataset2.ids))
    resolved_target_n = target_n or max_target_n
    if not 0 < resolved_target_n <= max_target_n:
        raise ValueError(
            "target_n must be between 1 and the smaller group size; " f"got {resolved_target_n}."
        )
    return resolved_target_n


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
