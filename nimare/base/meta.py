"""Base classes for meta-analyses.
"""
from os import makedirs
from os.path import exists, join

from nilearn.masking import unmask

from .base import Estimator, Transformer, Result


class KernelTransformer(Transformer):
    """Base class for modeled activation-generating methods.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.
    """
    pass


class MetaEstimator(Estimator):
    """
    Base class for meta-analysis estimators.
    """
    pass


class CBMAEstimator(Estimator):
    """Base class for coordinate-based meta-analysis methods.
    """
    pass


class IBMAEstimator(MetaEstimator):
    """Base class for image-based meta-analysis methods.
    """
    pass


class MetaResult(Result):
    """
    Base class for meta-analytic results.
    Will contain slots for different kinds of results maps (e.g., z-map, p-map)
    """
    def __init__(self, estimator, mask=None, **kwargs):
        self.estimator = estimator
        self.mask = mask
        self.images = {}
        for key, array in kwargs.items():
            self.images[key] = unmask(array, self.mask)

    def save_results(self, output_dir='.', prefix='', prefix_sep='_'):
        """
        Save results to files.

        Parameters
        ----------
        output_dir : :obj:`str`, optional
            Output directory in which to save results. If the directory doesn't
            exist, it will be created. Default is current directory.
        prefix : :obj:`str`, optional
            Prefix to prepent to output file names. Default is none.
        prefix_sep : :obj:`str`, optional
            Separator to add between prefix and default file names. Default is
            _.
        """
        if prefix == '':
            prefix_sep = ''

        if not exists(output_dir):
            makedirs(output_dir)

        for imgtype, img in self.images.items():
            filename = prefix + prefix_sep + imgtype + '.nii.gz'
            outpath = join(output_dir, filename)
            img.to_filename(outpath)
