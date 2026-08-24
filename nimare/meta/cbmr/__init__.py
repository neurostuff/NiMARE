"""Coordinate-based meta-regression methods.

Imports are lazy. Torch is an optional extra, but only some of this package needs it: the term
and design layer, the formula parser and the spline basis are plain numpy and patsy. Importing
them eagerly here would make ``from nimare.meta.cbmr.terms import Design`` fail on an install
without torch, for no reason. Resolving names on attribute access keeps the ImportError where it
belongs -- on the things that actually need torch.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    # Torch-free: the formula and design layer, and the spatial basis.
    "Design": (".terms", "Design"),
    "FormulaError": (".terms", "FormulaError"),
    "Term": (".terms", "Term"),
    "b_spline_bases": (".basis", "b_spline_bases"),
    "coef_spline_bases": (".basis", "coef_spline_bases"),
    "DEFAULT_GROUP_NAME": ("._helpers", "DEFAULT_GROUP_NAME"),
    "DEFAULT_INCIDENCE_THRESHOLD": ("._helpers", "DEFAULT_INCIDENCE_THRESHOLD"),
    # Torch-requiring.
    "CBMR": (".estimator", "CBMR"),
    "CBMRModel": (".model", "CBMRModel"),
    "CBMRPredictor": (".predictor", "CBMRPredictor"),
    "CBMRResult": (".results", "CBMRResult"),
    "ClusteredNegativeBinomial": (".distributions", "ClusteredNegativeBinomial"),
    "ContrastError": (".contrasts", "ContrastError"),
    "CovarianceError": (".covariance", "CovarianceError"),
    "Distribution": (".distributions", "Distribution"),
    "DistributionError": (".distributions", "DistributionError"),
    "NegativeBinomial": (".distributions", "NegativeBinomial"),
    "Poisson": (".distributions", "Poisson"),
    "evaluate_hypotheses": (".contrasts", "evaluate_hypotheses"),
    "poisson_log_likelihood": (".predictor", "poisson_log_likelihood"),
    "resolve_distribution": (".distributions", "resolve_distribution"),
    "sandwich_covariance": (".covariance", "sandwich_covariance"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    """Resolve a public name to its defining module on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    """List the lazily exported names alongside anything already imported."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
