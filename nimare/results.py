"""
Base classes for datasets.
"""
import os
import copy
import logging

from .utils import get_masker


LGR = logging.getLogger(__name__)


class MetaResult(object):
    """
    Base class for meta-analytic results.
    """
    def __init__(self, estimator, mask, maps=None):
        self.estimator = estimator
        self.masker = get_masker(mask)
        self.maps = maps or {}

    def get_map(self, name, return_type='image'):
        """
        Get stored map as image or array.
        """
        m = self.maps.get(name)
        if m is None:
            raise ValueError("No map with name '{}' found.".format(name))
        return self.masker.inverse_transform(m) if return_type == 'image' else m

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

    def copy(self):
        """
        Returns copy of result object.
        """
        new = MetaResult(self.estimator,
                         self.masker,
                         copy.deepcopy(self.maps))
        return new
