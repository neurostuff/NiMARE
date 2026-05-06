"""Shared utilities for CBMA estimator implementations."""

from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy import sparse as sp_sparse
from scipy.special import comb
from scipy.stats import norm

from nimare.utils import DEFAULT_FLOAT_DTYPE, _mask_img_to_bool


def _threshold_z_clusters(
    z_values, masker, voxel_thresh, cluster_size_threshold=None, mask_arr=None
):
    """Apply cluster thresholding to a z map in masked-array space.

    Parameters
    ----------
    z_values : numpy.ndarray of shape (n_voxels,)
        Masked z-values.
    masker : NiftiMasker
        Fitted masker; used only when ``mask_arr`` is ``None``.
    voxel_thresh : float
        Uncorrected voxel-level p-threshold for defining clusters.
    cluster_size_threshold : int or None
        Minimum cluster size in voxels; ``None`` skips size filtering.
    mask_arr : numpy.ndarray of shape (X, Y, Z), dtype bool, optional
        Pre-computed boolean brain mask.  When supplied the NiBabel
        round-trip (and its ``gc.collect()`` overhead) is avoided.
        Callers in hot loops should precompute this once and pass it in.
    """
    if mask_arr is None:
        mask_arr = _mask_img_to_bool(masker.mask_img)

    # 1D masked → 3D: direct numpy fill avoids inverse_transform → gc.collect
    z_map = np.zeros(mask_arr.shape, dtype=DEFAULT_FLOAT_DTYPE)
    z_map[mask_arr] = z_values

    sig_arr = z_map > norm.ppf(1 - voxel_thresh)
    labels, cluster_count = ndimage.label(sig_arr)
    if cluster_count < 1:
        return np.zeros_like(z_values), 0

    voxel_count_clusters = np.bincount(labels[labels > 0])
    max_clust = int(np.max(voxel_count_clusters)) if voxel_count_clusters.size else 0
    if cluster_size_threshold is not None:
        significant_clusters = voxel_count_clusters > cluster_size_threshold
        sig_clust_labels = np.where(significant_clusters)[0]
        z_map = z_map * np.isin(labels, sig_clust_labels)

    # 3D → 1D masked: direct boolean index avoids masker.transform → gc.collect
    return z_map[mask_arr].astype(DEFAULT_FLOAT_DTYPE, copy=False), max_clust


def resolve_subset_size(policy, total_n, k=None, target_n=None):
    """Resolve the retained subset size implied by a resampling policy."""
    if policy == "leave_1_out":
        return total_n - 1

    if policy == "leave_k_out":
        if k is None:
            raise ValueError("resampling_policy='leave_k_out' requires k.")
        if not 0 < k < total_n:
            raise ValueError(f"k must be between 1 and {total_n - 1}; got {k}.")
        return total_n - int(k)

    if policy == "subsample":
        resolved_target_n = target_n if target_n is not None else total_n
        if not 0 < resolved_target_n <= total_n:
            raise ValueError(f"target_n must be between 1 and {total_n}; got {resolved_target_n}.")
        return resolved_target_n

    raise ValueError(
        "resampling_policy must be one of 'leave_1_out', 'leave_k_out', or 'subsample'."
    )


def generate_subset_schedule(
    total_n,
    target_n,
    n_samples=None,
    random_state=None,
    exhaustive_limit=1000,
):
    """Generate retained-study index subsets without replacement."""
    if not 0 < target_n <= total_n:
        raise ValueError(f"target_n must be between 1 and total_n ({total_n}); got {target_n}.")

    if target_n == total_n - 1:
        return [
            np.delete(np.arange(total_n), i).astype(np.int32, copy=False) for i in range(total_n)
        ]

    max_combinations = int(comb(total_n, target_n, exact=True))
    if n_samples is None:
        if max_combinations > exhaustive_limit:
            raise ValueError(
                "This resampling schedule is too large to enumerate exhaustively. "
                "Set n_resamples to sample a subset of unique schedules."
            )
        n_samples = max_combinations
    else:
        n_samples = min(int(n_samples), max_combinations)

    if n_samples == max_combinations and max_combinations <= exhaustive_limit:
        return [np.asarray(idx, dtype=np.int32) for idx in combinations(range(total_n), target_n)]

    rng = np.random.RandomState(random_state)
    subsets = set()
    while len(subsets) < n_samples:
        subset = tuple(np.sort(rng.choice(total_n, size=target_n, replace=False)))
        subsets.add(subset)
    return [np.asarray(subset, dtype=np.int32) for subset in sorted(subsets)]


def require_masked_csr(ma_values, source="MA maps"):
    """Require CBMA sparse MA maps to be scipy CSR matrices."""
    if not sp_sparse.isspmatrix(ma_values):
        raise ValueError(f"{source} must be a scipy sparse matrix, not {type(ma_values)}.")
    return ma_values.tocsr(copy=False)


def collect_csr_ma_maps(estimator, coords_key="coordinates", maps_key="ma_maps"):
    """Collect study-wise MA maps and normalize them to masked CSR matrices."""
    if maps_key in estimator.inputs_:
        return require_masked_csr(estimator.inputs_[maps_key], source=f"Precomputed {maps_key}")

    return require_masked_csr(
        estimator.kernel_transformer.transform(
            estimator.inputs_[coords_key],
            masker=estimator.masker,
            return_type="sparse",
        ),
        source=f"Generated {maps_key}",
    )
