"""
Coactivation-based parcellation
"""
import numpy as np
from nilearn.masking import apply_mask

from .base import Parcellator
from ..meta.cbma.kernel import MKDAKernel
from ..due import due, Doi


@due.dcite(Doi('10.1002/hbm.22138'),
           description='Introduces CBP.')
class CoordCBP(Parcellator):
    """
    Coordinate-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids):
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]
        self.ids = ids

    def fit(self, target_mask, r=5, n_parcels=2, n_iters=10000, n_cores=4):
        """
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
        assert np.array_equal(self.mask.affine, target_mask.affine)
        kernel_estimator = MKDAKernel(self.coordinates, self.mask)
        ma_maps = kernel_estimator.transform(self.ids, r=r)
        ma_data = apply_mask(ma_maps, self.mask)

        target_data = apply_mask(target_mask, self.mask)
        mask_idx = np.where(target_data)[0]
        for i_voxel, idx in enumerate(mask_idx):
            study_idx = np.where(ma_data[:, idx])[0]



class ImCBP(Parcellator):
    """
    Image-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids):
        self.mask = dataset.mask
        self.ids = ids

    def fit(self, target_mask, n_parcels=2):
        """
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
