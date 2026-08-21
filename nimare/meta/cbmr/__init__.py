"""Coordinate-based meta-regression methods.

Split from a single 3,400-line module. The names re-exported here are the module's public
surface plus the handful of private helpers that tests and downstream code reach for through
``nimare.meta.cbmr``, so that the split stays invisible to importers.
"""

from nimare.meta.cbmr._helpers import DEFAULT_GROUP_NAME, DEFAULT_INCIDENCE_THRESHOLD
from nimare.meta.cbmr.contrasts import ContrastError, evaluate_hypotheses
from nimare.meta.cbmr.covariance import CovarianceError, sandwich_covariance
from nimare.meta.cbmr.distributions import (
    ClusteredNegativeBinomial,
    Distribution,
    DistributionError,
    NegativeBinomial,
    Poisson,
    resolve_distribution,
)
from nimare.meta.cbmr.estimator import CBMR
from nimare.meta.cbmr.model import CBMRModel
from nimare.meta.cbmr.predictor import CBMRPredictor, poisson_log_likelihood
from nimare.meta.cbmr.results import CBMRResult
from nimare.meta.cbmr.terms import Design, FormulaError, Term

__all__ = [
    "CBMR",
    "CBMRModel",
    "CBMRPredictor",
    "ClusteredNegativeBinomial",
    "Design",
    "ContrastError",
    "CovarianceError",
    "Distribution",
    "DistributionError",
    "NegativeBinomial",
    "Poisson",
    "FormulaError",
    "Term",
    "evaluate_hypotheses",
    "poisson_log_likelihood",
    "resolve_distribution",
    "sandwich_covariance",
    "CBMRResult",
    "DEFAULT_GROUP_NAME",
    "DEFAULT_INCIDENCE_THRESHOLD",
]
