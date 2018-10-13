"""
Coactivation-based parcellation
"""
import numpy as np
from nilearn.masking import apply_mask, unmask
from scipy.spatial.distance import cdist

from .base import Parcellator
from ..meta.cbma.ale import SCALE
from ..due import due, Doi


@due.dcite(Doi('10.1002/hbm.22138'),
           description='Introduces CBP.')
class CoordCBP(Parcellator):
    """
    Coordinate-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]
        self.ids = ids

    def fit(self, target_mask, method='min_distance', r=5, n_exps=50,
            n_parcels=2, n_iters=10000, meta_estimator=SCALE, **kwargs):
        """
        Run CBP parcellation.

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
        kernel_args = {k: v for k, v in kwargs.items() if
                       k.startswith('kernel__')}
        meta_args = {k.split('meta__')[1]: v for k, v in kwargs.items() if
                     k.startswith('meta__')}

        # Step 1: Build correlation matrix
        target_data = apply_mask(target_mask, self.mask)
        target_map = unmask(target_data, self.mask)
        target_data = target_map.get_data()
        mask_idx = np.vstack(np.where(target_data))
        voxel_arr = np.zeros((mask_idx.shape[1], np.sum(self.mask)))

        ijk = self.coordinates[['i', 'j', 'k']].values
        temp_df = self.coordinates.copy()
        for i_voxel in range(mask_idx.shape[1]):
            voxel = mask_idx[:, i_voxel]
            temp_df['distance'] = cdist(ijk, voxel)

            if method == 'min_studies':
                # number of studies
                temp_df2 = temp_df.groupby('id')[['distance']].min()
                temp_df2 = temp_df2.sort_values(by='distance')
                sel_ids = temp_df2.iloc[:n_exps].index.values
            elif method == 'min_distance':
                # minimum distance
                temp_df2 = temp_df.groupby('id')[['distance']].min()
                sel_ids = temp_df2.loc[temp_df2['distance'] < r].index.values

            voxel_meta = meta_estimator(self.dataset, ids=sel_ids,
                                        **kernel_args)
            voxel_meta.fit(**meta_args)
            voxel_arr[i_voxel, :] = apply_mask(voxel_meta.results['ale'],
                                               self.mask)
        voxel_corr = np.corrcoef(voxel_arr)

        # Step 2: Clustering
        for i_parc in n_parcels:
            data = voxel_corr


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
