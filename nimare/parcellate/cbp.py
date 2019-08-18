"""
Coactivation-based parcellation
"""
import numpy as np
import pandas as pd
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
import scipy.ndimage.measurements as meas
from nilearn.masking import apply_mask, unmask

from .base import Parcellator
from ..meta.cbma.ale import SCALE
from ..due import due
from .. import references


@due.dcite(references.CBP, description='Introduces CBP.')
class CoordCBP(Parcellator):
    """
    Coordinate-based coactivation-based parcellation [1]_.

    Notes
    -----
    Here are the steps:
        1.  For each voxel in the mask, identify studies in dataset
            corresponding to that voxel. Selection criteria can be either
            based on a distance threshold (e.g., all studies with foci
            within 5mm of voxel) or based on a minimum number of studies
            (e.g., the 50 studies reporting foci closest to the voxel).
        2.  For each voxel, perform MACM (meta-analysis) using the
            identified studies.
        3.  Correlate statistical maps between voxel MACMs to generate
            n_voxels X n_voxels correlation matrix.
        4.  Convert correlation coefficients to correlation distance (1 - r)
            values.
        5.  Perform clustering on correlation distance matrix.

    Warnings
    --------
    This method is currently untested.

    References
    ----------
    .. [1] Bzdok, D., Laird, A. R., Zilles, K., Fox, P. T., & Eickhoff, S. B.
        (2013). An investigation of the structural, connectional, and
        functional subspecialization in the human amygdala. Human brain
        mapping, 34(12), 3247-3266. https://doi.org/10.1002/hbm.22138
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]
        self.ids = ids
        self.solutions = None
        self.metrics = None

    def fit(self, target_mask, method='min_distance', r=5, n_exps=50,
            n_parcels=2, meta_estimator=SCALE, **kwargs):
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

        if not isinstance(n_parcels, list):
            n_parcels = [n_parcels]

        # Step 1: Build correlation matrix
        target_data = apply_mask(target_mask, self.mask)
        target_map = unmask(target_data, self.mask)
        target_data = target_map.get_data()
        mask_idx = np.vstack(np.where(target_data))
        n_voxels = mask_idx.shape[1]
        voxel_arr = np.zeros((n_voxels, np.sum(self.mask)))

        ijk = self.coordinates[['i', 'j', 'k']].values
        temp_df = self.coordinates.copy()
        for i_voxel in range(n_voxels):
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

            # Run MACM
            voxel_meta = meta_estimator(self.dataset, ids=sel_ids,
                                        **kernel_args)
            voxel_meta.fit(**meta_args)
            voxel_arr[i_voxel, :] = apply_mask(voxel_meta.results['ale'],
                                               self.mask)

        # Correlate voxel-specific MACMs across voxels in ROI
        voxel_corr = np.corrcoef(voxel_arr)
        corr_dist = 1 - voxel_corr

        # Step 2: Clustering
        labels = np.zeros((n_voxels, len(n_parcels)))
        metric_types = ['contiguous']
        metrics = pd.DataFrame(index=n_parcels, columns=metric_types,
                               data=np.zeros((len(n_parcels),
                                              len(metric_types))))
        for i_parc, n_clusters in enumerate(n_parcels):
            # K-Means clustering
            _, labeled, _ = k_means(
                corr_dist, n_clusters, init='k-means++',
                precompute_distances='auto', n_init=1000, max_iter=1023,
                verbose=False, tol=0.0001, random_state=1, copy_x=True,
                n_jobs=1, algorithm='auto', return_n_iter=False)
            labels[:, i_parc] = labeled

            # Check contiguity of clusters
            # Can nilearn do this?
            temp_mask = np.zeros(target_data.shape)
            for j_voxel in range(n_voxels):
                i, j, k = mask_idx[:, j_voxel]
                temp_mask[i, j, k] = labeled[j_voxel]
            labeled = meas.label(temp_mask, np.ones((3, 3, 3)))[0]
            n_contig = len(np.unique(labeled))
            metrics.loc[n_clusters, 'contiguous'] = int(n_contig > (n_clusters + 1))

        self.solutions = labels
        self.metrics = metrics


class ImCBP(Parcellator):
    """
    Image-based coactivation-based parcellation

    Warnings
    --------
    This method is not yet implemented.
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
