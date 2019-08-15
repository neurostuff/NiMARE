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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_file = kwargs.get('mask_file')
        self.mask_regions = kwargs.get('regions', False)
        if self.mask_file is not None:
            self.set_mask(self.mask_file, self.mask_regions)

    def set_mask(self, mask_file, regions=False):
        pass

    def _preprocess_input(self, dataset):
        pass
