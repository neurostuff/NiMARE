"""Coordinate-, image-, and effect-size-based meta-analysis estimators."""

from importlib import import_module

from . import ibma, kernel
from .cbma import ALE, KDA, SCALE, ALESubtraction, MKDAChi2, MKDADensity, ale, mkda
from .ibma import (
    DerSimonianLaird,
    Fishers,
    Hedges,
    PermutedOLS,
    SampleSizeBasedLikelihood,
    Stouffers,
    VarianceBasedLikelihood,
    WeightedLeastSquares,
)
from .kernel import ALEKernel, KDAKernel, MKDAKernel

_OPTIONAL_SUBMODULES = {
    "cbmr": ".cbmr",
    "models": ".models",
}

_OPTIONAL_EXPORTS = {
    "CBMREstimator": (".cbmr", "CBMREstimator"),
    "CBMRInference": (".cbmr", "CBMRInference"),
    "CBMRResult": (".cbmr", "CBMRResult"),
}


def __getattr__(name):
    """Lazily import optional CBMR modules and exports."""
    if name in _OPTIONAL_SUBMODULES:
        module = import_module(_OPTIONAL_SUBMODULES[name], __name__)
        globals()[name] = module
        return module

    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = import_module(module_name, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALE",
    "ALESubtraction",
    "SCALE",
    "MKDADensity",
    "MKDAChi2",
    "KDA",
    "CBMREstimator",
    "CBMRInference",
    "CBMRResult",
    "DerSimonianLaird",
    "Fishers",
    "Hedges",
    "PermutedOLS",
    "SampleSizeBasedLikelihood",
    "Stouffers",
    "VarianceBasedLikelihood",
    "WeightedLeastSquares",
    "MKDAKernel",
    "ALEKernel",
    "KDAKernel",
    "kernel",
    "ibma",
    "cbmr",
    "models",
    "ale",
    "mkda",
]
