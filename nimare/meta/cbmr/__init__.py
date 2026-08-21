"""Coordinate-based meta-regression methods.

Split from a single 3,400-line module. The names re-exported here are the module's public
surface plus the handful of private helpers that tests and downstream code reach for through
``nimare.meta.cbmr``, so that the split stays invisible to importers.
"""

# Re-exported, not used here: these were module-level names of the old single-file module, and
# tests and downstream code still reach for them through ``nimare.meta.cbmr``.
from nimare.meta.cbmr._helpers import (  # noqa: F401
    DEFAULT_GROUP_NAME,
    DEFAULT_INCIDENCE_THRESHOLD,
    _as_csr_matrix,
    _csr_row_indices,
    _is_named_pairwise_contrast,
    _normalize_named_pairwise_contrasts,
    _uses_cuda,
    _validate_incidence_threshold,
)
from nimare.meta.cbmr.distributions import (
    ClusteredNegativeBinomial,
    Distribution,
    DistributionError,
    NegativeBinomial,
    Poisson,
    resolve_distribution,
)
from nimare.meta.cbmr.estimator import CBMREstimator
from nimare.meta.cbmr.inference import CBMRInference
from nimare.meta.cbmr.model import CBMRModel
from nimare.meta.cbmr.optimizers import fit_voxelwise_cbmr_approximate
from nimare.meta.cbmr.predictor import CBMRPredictor, poisson_log_likelihood
from nimare.meta.cbmr.results import CBMRResult
from nimare.meta.cbmr.terms import Design, FormulaError, Term

__all__ = [
    "CBMREstimator",
    "CBMRModel",
    "CBMRPredictor",
    "ClusteredNegativeBinomial",
    "Design",
    "Distribution",
    "DistributionError",
    "NegativeBinomial",
    "Poisson",
    "FormulaError",
    "Term",
    "poisson_log_likelihood",
    "resolve_distribution",
    "CBMRInference",
    "CBMRResult",
    "DEFAULT_GROUP_NAME",
    "DEFAULT_INCIDENCE_THRESHOLD",
    "fit_voxelwise_cbmr_approximate",
]
