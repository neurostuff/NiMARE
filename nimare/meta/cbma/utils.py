"""Shared utilities for CBMA estimator implementations."""

from itertools import combinations

import numpy as np
from scipy import sparse as sp_sparse
from scipy.special import comb


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
