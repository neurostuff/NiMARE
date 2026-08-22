"""Shared constants and small helpers for the CBMR modules."""

DEFAULT_GROUP_NAME = "Default"
DEFAULT_INCIDENCE_THRESHOLD = 0.001


def _uses_cuda(device):
    """Return whether the provided device string targets CUDA."""
    return str(device).startswith("cuda")


def _validate_incidence_threshold(incidence_threshold):
    """Validate the empirical incidence threshold used for voxel filtering."""
    if incidence_threshold is None:
        return None
    incidence_threshold = float(incidence_threshold)
    if incidence_threshold < 0 or incidence_threshold >= 1:
        raise ValueError("incidence_threshold must be None or a value in [0, 1).")
    return incidence_threshold
