"""Coordinate-, image-, and effect-size-based meta-analysis estimators."""

import lazy_loader as lazy

# Estimators are imported on first access: a coordinate-based run never loads PyMARE
# (and through it SymPy) for the image-based estimators, and CBMR's optional torch
# dependency is only needed once CBMR is used.
__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submodules=["cbma", "cbmr", "ibma", "kernel", "utils"],
    submod_attrs={
        "cbma": [
            "ALE",
            "KDA",
            "SCALE",
            "ALESubtraction",
            "BalancedALESubtraction",
            "MKDAChi2",
            "MKDADensity",
            "StudyWeights",
            "ale",
            "mkda",
            "weights",
        ],
        "cbmr": ["CBMR", "CBMRResult"],
        "ibma": [
            "DerSimonianLaird",
            "Fishers",
            "Hedges",
            "PermutedOLS",
            "SampleSizeBasedLikelihood",
            "Stouffers",
            "VarianceBasedLikelihood",
            "WeightedLeastSquares",
        ],
        "kernel": ["ALEKernel", "KDAKernel", "MKDAKernel"],
    },
)
del _

__all__ = [
    "ALE",
    "ALESubtraction",
    "BalancedALESubtraction",
    "SCALE",
    "MKDADensity",
    "MKDAChi2",
    "KDA",
    "CBMR",
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
    "ale",
    "mkda",
    "weights",
    "StudyWeights",
]
