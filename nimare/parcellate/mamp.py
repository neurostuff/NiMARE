"""
Meta-analytic activation modeling-based parcellation (MAMP).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import k_means
import scipy.ndimage.measurements as meas
from nilearn.masking import apply_mask, unmask

from .base import Parcellator
from ..meta.cbma.kernel import ALEKernel
from ..due import due
from .. import references


@due.dcite(references.MAMP, description='Introduces the MAMP algorithm.')
class MAMP(Parcellator):
    """
    Meta-analytic activation modeling-based parcellation (MAMP) [1]_.

    Parameters
    ----------
    text : :obj:`list` of :obj:`str`
        List of texts to use for parcellation.
    mask : :obj:`str` or :obj:`nibabel.Nifti1.Nifti1Image`
        Mask file or image.

    Notes
    -----
    MAMP works similarly to CBP, but skips the step of performing a MACM for
    each voxel. Here are the steps:
        1.  Create an MA map for each study in the dataset.
        2.  Concatenate MA maps across studies to create a 4D dataset.
        3.  Extract values across studies for voxels in mask, resulting in
            n_voxels X n_studies array.
        4.  Correlate "study series" between voxels to generate n_voxels X
            n_voxels correlation matrix.
        5.  Convert correlation coefficients to correlation distance (1 -r)
            values.
        6.  Perform clustering on correlation distance matrix.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Yang, Yong, et al. "Identifying functional subdivisions in the human
        brain using meta-analytic activation modeling-based parcellation."
        Neuroimage 124 (2016): 300-309.
        https://doi.org/10.1016/j.neuroimage.2015.08.027
    """
    def __init__(self, dataset, ids):
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]
        self.ids = ids
        self.solutions = None
        self.metrics = None

    def fit(self, target_mask, n_parcels=2, kernel_estimator=ALEKernel,
            **kwargs):
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
        assert np.array_equal(self.mask.affine, target_mask.affine)
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items() if
                       k.startswith('kernel__')}

        if not isinstance(n_parcels, list):
            n_parcels = [n_parcels]

        k_est = kernel_estimator(self.coordinates, self.mask)
        ma_maps = k_est.transform(self.ids, **kernel_args)

        # Step 1: Build correlation matrix
        target_data = apply_mask(target_mask, self.mask)
        target_map = unmask(target_data, self.mask)
        mask_idx = np.vstack(np.where(target_data))
        n_voxels = mask_idx.shape[1]
        mask_ma_vals = apply_mask(ma_maps, target_map)
        voxel_corr = np.corrcoef(mask_ma_vals)
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
            temp_mask = unmask(labels[:, i_parc], target_map).get_data()
            labeled = meas.label(temp_mask, np.ones((3, 3, 3)))[0]
            n_contig = len(np.unique(labeled))
            metrics.loc[n_clusters, 'contiguous'] = int(n_contig > (n_clusters + 1))

        self.solutions = labels
        self.metrics = metrics
