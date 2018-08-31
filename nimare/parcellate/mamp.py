"""
Meta-analytic activation modeling-based parcellation (MAMP).
"""
from .base import Parcellator
from ..due import due, Doi


@due.dcite(Doi('10.1016/j.neuroimage.2015.08.027'),
           description='Introduces the MAMP algorithm.')
class MAMP(Parcellator):
    """
    Meta-analytic activation modeling-based parcellation (MAMP).

    Parameters
    ----------
    text : :obj:`list` of :obj:`str`
        List of texts to use for parcellation.
    mask : :obj:`str` or :obj:`nibabel.Nifti1.Nifti1Image`
        Mask file or image.
    """
    def __init__(self, dataset, ids):
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]
        self.ids = ids

    def fit(self, target_mask, r=5, n_parcels=2, n_iters=10000, n_cores=4):
        """
        Run MAMP parcellation.

        Parameters
        ----------
        target_mask : img_like
            Image with binary mask for region of interest to be parcellated.
        n_parcels : :obj:`int` or array_like of :obj:`int`, optional
            Number of parcels to generate for ROI. If array_like, each parcel
            number will be evaluated and results for all will be returned.
            Default is 2.
        n_iters : :obj:`int`, optional
            Number of iterations to run for each parcel number.
            Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for model fitting.

        Returns
        -------
        results
        """
        pass
