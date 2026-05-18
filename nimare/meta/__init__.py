"""Coordinate-, image-, and effect-size-based meta-analysis estimators."""

from importlib import import_module

from . import ibma, kernel
from .cbma import (
    ALE,
    KDA,
    SCALE,
    ALESubtraction,
    BalancedALESubtraction,
    MKDAChi2,
    MKDADensity,
    ale,
    mkda,
)
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
    "spatial_cbmr": ".spatial_cbmr",
}

_OPTIONAL_EXPORTS = {
    "CBMREstimator": (".cbmr", "CBMREstimator"),
    "CBMRInference": (".cbmr", "CBMRInference"),
    "CBMRResult": (".cbmr", "CBMRResult"),
    "SpatialCBMREstimator": (".spatial_cbmr", "SpatialCBMREstimator"),
    "SpatialCBMRInference": (".spatial_cbmr", "SpatialCBMRInference"),
    "SpatialCBMRResult": (".spatial_cbmr", "SpatialCBMRResult"),
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
    "BalancedALESubtraction",
    "SCALE",
    "MKDADensity",
    "MKDAChi2",
    "KDA",
    "CBMREstimator",
    "CBMRInference",
    "CBMRResult",
    "SpatialCBMREstimator",
    "SpatialCBMRInference",
    "SpatialCBMRResult",
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
    "spatial_cbmr",
    "models",
    "ale",
    "mkda",
]
