"""Coordinate-, image-, and effect-size-based meta-analysis estimators."""

from . import cbmr, ibma, kernel, models
from .cbma import ALE, KDA, SCALE, ALESubtraction, MKDAChi2, MKDADensity, ale, mkda
from .cbmr import CBMREstimator, CBMRInference, CBMRResult
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
