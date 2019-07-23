"""Base classes for meta-analyses.
"""
from .estimators import Estimator, Transformer


class KernelTransformer(Transformer):
    """Base class for modeled activation-generating methods.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.
    """
    pass


class CBMAEstimator(Estimator):
    """Base class for coordinate-based meta-analysis methods.
    """
    pass


class IBMAEstimator(Estimator):
    """Base class for image-based meta-analysis methods.
    """
    pass
