"""
Top-level namespace for meta-analyses.
"""
from abc import ABCMeta, abstractmethod
from six import with_metaclass

from ..base import MetaEstimator


class CBMAEstimator(MetaEstimator):
    """Base class for coordinate-based meta-analysis methods.
    """
    pass


class KernelEstimator(with_metaclass(ABCMeta)):
    """Base class for modeled activation-generating methods.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.
    """
    @abstractmethod
    def transform(self):
        """
        Generate a modeled activation map for each of the contrasts in a
        dataset.
        """
        pass
