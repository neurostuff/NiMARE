"""Base classes for meta-analyses.
"""
import os

import nilearn as nl

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


class MetaResult(object):
    """
    Base class for meta-analytic results.
    """
    def __init__(self, estimator, mask, maps=None):
        self.estimator = estimator
        self.mask = mask
        self.maps = maps or {}

    def get_map(self, name, return_type='image'):
        m = self.maps.get(name)
        if not m:
            raise ValueError("No map with name '{}' found.".format(name))
        return nl.masking.unmask(m, self.mask) if return_type == 'image' else m

    def save_maps(self, output_dir='.', prefix='', prefix_sep='_',
                  names=None):
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

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        names = names or list(self.maps.keys())
        maps = {k: self.get_map(k) for k in names}

        for imgtype, img in maps.items():
            filename = prefix + prefix_sep + imgtype + '.nii.gz'
            outpath = os.path.join(output_dir, filename)
            img.to_filename(outpath)
