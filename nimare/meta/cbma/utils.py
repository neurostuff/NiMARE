"""Shared utilities for CBMA estimator implementations."""

from scipy import sparse as sp_sparse


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
