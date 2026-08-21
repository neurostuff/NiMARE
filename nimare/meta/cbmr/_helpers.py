"""Shared constants and small helpers for the CBMR modules."""

import numpy as np
import scipy.sparse

DEFAULT_GROUP_NAME = "Default"
DEFAULT_INCIDENCE_THRESHOLD = 0.001

# Sentinel distinguishing "inherit the fitted estimator's threshold" from an explicit None,
# which means "keep every fitted voxel".
_INHERIT_INCIDENCE_THRESHOLD = object()


def _uses_cuda(device):
    """Return whether the provided device string targets CUDA."""
    return str(device).startswith("cuda")


def _as_csr_matrix(value):
    """Return a sparse matrix in CSR format."""
    if scipy.sparse.isspmatrix_csr(value):
        return value
    return value.tocsr()


def _csr_row_indices(value):
    """Return row indices for each nonzero entry in a CSR matrix."""
    return np.repeat(np.arange(value.shape[0], dtype=value.indices.dtype), np.diff(value.indptr))


def _is_named_pairwise_contrast(contrast):
    """Return whether a contrast uses tuple shorthand like (A, B)."""
    return (
        isinstance(contrast, tuple)
        and len(contrast) == 2
        and all(isinstance(part, str) for part in contrast)
    )


def _validate_incidence_threshold(incidence_threshold):
    """Validate the empirical incidence threshold used for voxel filtering."""
    if incidence_threshold is None:
        return None
    incidence_threshold = float(incidence_threshold)
    if incidence_threshold < 0 or incidence_threshold >= 1:
        raise ValueError("incidence_threshold must be None or a value in [0, 1).")
    return incidence_threshold


def _normalize_named_pairwise_contrasts(contrasts):
    """Convert tuple shorthand like (A, B) into the legacy string form."""
    if contrasts is None:
        return None
    if isinstance(contrasts, str) or _is_named_pairwise_contrast(contrasts):
        contrasts = [contrasts]

    normalized = []
    for contrast in contrasts:
        if _is_named_pairwise_contrast(contrast):
            normalized.append(f"{contrast[0]}-{contrast[1]}")
        else:
            normalized.append(contrast)
    return normalized
