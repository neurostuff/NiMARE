"""
Coordinate-based meta-analysis estimators
"""
import warnings
import multiprocessing as mp

import numpy as np
import nibabel as nib
from scipy import ndimage
from nilearn.masking import apply_mask, unmask

from .base import CBMAEstimator
from .kernel import MKDAKernel, KDAKernel
from .utils import p_to_z
from ..base import MetaResult
from ...utils import vox2mm
from ...stats import two_way, one_way, fdr
from ...due import due, Doi


@due.dcite(Doi('10.1093/scan/nsm015'), description='Introduces the MKDA algorithm.')
class MKDADensity(CBMAEstimator):
    """
    Multilevel kernel density analysis- Density analysis
    """
    def __init__(self, dataset, ids, kernel_estimator=MKDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()\
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.dataset = dataset

        self.kernel_estimator = kernel_estimator
        self.kernel_arguments = kernel_args
        k_est = kernel_estimator(self.dataset.coordinates, self.dataset.mask)
        ma_maps = k_est.transform(ids, **kernel_args)

        self.ma_maps = ma_maps
        self.ids = ids
        self.voxel_thresh = None
        self.corr = None
        self.n_iters = None
        self.results = None

    def fit(self, voxel_thresh=0.01, q=0.05, corr='FDR', n_iters=1000, n_cores=4):
        null_img = self.dataset.mask
        null_ijk = np.vstack(np.where(null_img.get_data())).T
        self.voxel_thresh = voxel_thresh
        self.corr = corr
        self.n_iters = n_iters

        # Weight each SCM by square root of sample size
        sel_df = self.dataset.coordinates.loc[self.dataset.coordinates['id'].isin(self.ids)]
        ids_df = sel_df.groupby('id').first()
        if 'n' in ids_df.columns and 'inference' not in ids_df.columns:
            ids_n = ids_df.loc[self.ids, 'n'].astype(float).values
            weight_vec = np.sqrt(ids_n)[:, None] / np.sum(np.sqrt(ids_n))
        elif 'n' in ids_df.columns and 'inference' in ids_df.columns:
            ids_n = ids_df.loc[self.ids, 'n'].astype(float).values
            ids_inf = ids_df.loc[self.ids, 'inference'].map({'ffx': 0.75, 'rfx': 1.}).values
            weight_vec = (np.sqrt(ids_n)[:, None] * ids_inf[:, None]) / \
                         np.sum(np.sqrt(ids_n) * ids_inf)
        else:
            weight_vec = np.ones((len(self.ma_maps), 1))

        ma_maps = apply_mask(self.ma_maps, self.dataset.mask)
        ma_maps *= weight_vec
        of_map = np.sum(ma_maps, axis=0)
        of_map = unmask(of_map, self.dataset.mask)

        vthresh_of_map = of_map.get_data().copy()
        vthresh_of_map[vthresh_of_map < voxel_thresh] = 0

        rand_idx = np.random.choice(null_ijk.shape[0],
                                    size=(sel_df.shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = sel_df.copy()

        conn = np.ones((3, 3, 3))

        # Define parameters
        iter_conn = [conn] * n_iters
        iter_wv = [weight_vec] * n_iters
        iter_dfs = [iter_df] * n_iters
        params = zip(iter_ijks, iter_dfs, iter_wv, iter_conn)

        pool = mp.Pool(n_cores)
        perm_results = pool.map(self._perm, params)
        pool.close()
        perm_max_values, perm_clust_sizes = zip(*perm_results)

        percentile = 100 * (1 - q)

        ## Cluster-level FWE
        # Determine size of clusters in [1 - clust_thresh]th percentile (e.g. 95th)
        clust_size_thresh = np.percentile(perm_clust_sizes, percentile)

        cfwe_of_map = np.zeros(of_map.shape)
        labeled_matrix = ndimage.measurements.label(vthresh_of_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        labeled_vector = labeled_matrix.flatten()
        for i, clust_size in enumerate(clust_sizes):
            if clust_size >= clust_size_thresh and i > 0:
                clust_idx = np.where(labeled_matrix == i)
                cfwe_of_map[clust_idx] = vthresh_of_map[clust_idx]
        cfwe_of_map = apply_mask(nib.Nifti1Image(cfwe_of_map, of_map.affine), self.dataset.mask)

        ## Voxel-level FWE
        # Determine ALE values in [1 - clust_thresh]th percentile (e.g. 95th)
        vfwe_thresh = np.percentile(perm_max_values, percentile)
        vfwe_of_map = of_map.get_data().copy()
        vfwe_of_map[vfwe_of_map < vfwe_thresh] = 0.
        vfwe_of_map = apply_mask(nib.Nifti1Image(vfwe_of_map, of_map.affine), self.dataset.mask)

        vthresh_of_map = apply_mask(nib.Nifti1Image(vthresh_of_map, of_map.affine), self.dataset.mask)
        results = MetaResult(vthresh=vthresh_of_map, cfwe=cfwe_of_map,
                             vfwe=vfwe_of_map, mask=self.dataset.mask)
        self.results = results

    def _perm(self, params):
        iter_ijk, iter_df, weight_vec, conn = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        k_est = self.kernel_estimator(iter_df, self.dataset.mask)
        iter_ma_maps = k_est.transform(self.ids, **self.kernel_arguments)
        iter_ma_maps = apply_mask(iter_ma_maps, self.dataset.mask)
        iter_ma_maps *= weight_vec
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        iter_of_map = unmask(iter_of_map, self.dataset.mask)
        vthresh_iter_of_map = iter_of_map.get_data().copy()
        vthresh_iter_of_map[vthresh_iter_of_map < self.voxel_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_of_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        iter_max_cluster = np.max(clust_sizes)
        return iter_max_value, iter_max_cluster


@due.dcite(Doi('10.1093/scan/nsm015'), description='Introduces the MKDA algorithm.')
class MKDAChi2(CBMAEstimator):
    """
    Multilevel kernel density analysis- Chi-square analysis
    """
    def __init__(self, dataset, ids, ids2=None, kernel_estimator=MKDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()\
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.dataset = dataset

        k_est = kernel_estimator(self.dataset.coordinates, self.dataset.mask)
        ma_maps1 = k_est.transform(ids, **kernel_args)

        if ids2 is None:
            ids2 = list(set(self.dataset.ids) - set(ids))
        ma_maps2 = k_est.transform(ids2, **kernel_args)

        self.ma_maps = [ma_maps1, ma_maps2]
        self.ids = [ids, ids2]
        self.voxel_thresh = None
        self.corr = None
        self.n_iters = None
        self.results = None

    def fit(self, voxel_thresh=0.01, q=0.05, corr='FDR', n_iters=10000, prior=0.5):
        self.voxel_thresh = voxel_thresh
        self.corr = corr
        self.n_iters = n_iters

        # Calculate different count variables
        n_selected = len(self.ids[0])
        n_unselected = len(self.ids[1])
        n_mappables = n_selected + n_unselected

        # Transform MA maps to 1d arrays
        ma_maps1 = np.vstack([apply_mask(m, self.dataset.mask) for m in self.ma_maps[0]])
        ma_maps2 = np.vstack([apply_mask(m, self.dataset.mask) for m in self.ma_maps[1]])
        ma_maps_all = np.vstack((ma_maps1, ma_maps2))

        n_selected_active_voxels = np.sum(ma_maps1, axis=0)
        n_unselected_active_voxels = np.sum(ma_maps2, axis=0)

        # Nomenclature for variables below: p = probability, F = feature present, g = given,
        # U = unselected, A = activation. So, e.g., pAgF = p(A|F) = probability of activation
        # in a voxel if we know that the feature is present in a study.
        pF = (n_selected * 1.0) / n_mappables
        pA = np.array(np.sum(ma_maps_all, axis=0) / n_mappables).squeeze()

        # Conditional probabilities
        pAgF = n_selected_active_voxels * 1.0 / n_selected
        pAgU = n_unselected_active_voxels * 1.0 / n_unselected
        pFgA = pAgF * pF / pA

        # Recompute conditionals with uniform prior
        pAgF_prior = prior * pAgF + (1 - prior) * pAgU
        pFgA_prior = pAgF * prior / pAgF_prior

        # One-way chi-square test for consistency of activation
        p_vals = one_way(np.squeeze(n_selected_active_voxels), n_selected)
        p_vals[p_vals < 1e-240] = 1e-240
        z_sign = np.sign(n_selected_active_voxels - np.mean(n_selected_active_voxels)).ravel()
        pAgF_z = p_to_z(p_vals, z_sign)
        fdr_thresh = fdr(p_vals, q)
        pAgF_z_FDR = pAgF_z.copy()
        pAgF_z_FDR[p_vals > fdr_thresh] = 0

        # Two-way chi-square for specificity of activation
        cells = np.squeeze(
            np.array([[n_selected_active_voxels, n_unselected_active_voxels],
                      [n_selected - n_selected_active_voxels,
                       n_unselected - n_unselected_active_voxels]]).T)
        p_vals = two_way(cells)
        p_vals[p_vals < 1e-240] = 1e-240
        z_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(p_vals, z_sign)
        fdr_thresh = fdr(p_vals, q)
        pFgA_z_FDR = pAgF_z.copy()
        pFgA_z_FDR[p_vals > fdr_thresh] = 0

        # Retain any images we may want to save or access later
        images = {
            'pA': pA,
            'pAgF': pAgF,
            'pFgA': pFgA,
            ('pAgF_given_pF=%0.2f' % prior): pAgF_prior,
            ('pFgA_given_pF=%0.2f' % prior): pFgA_prior,
            'consistency_z': pAgF_z,
            'specificity_z': pFgA_z,
            ('pAgF_z_FDR_%s' % q): pAgF_z_FDR,
            ('pFgA_z_FDR_%s' % q): pFgA_z_FDR
        }
        results = MetaResult(mask=self.dataset.mask, **images)
        self.results = results


@due.dcite(Doi('10.1016/S1053-8119(03)00078-8'),
           description='Introduces the KDA algorithm.')
class KDA(CBMAEstimator):
    """
    Kernel density analysis
    """
    def __init__(self, dataset, ids, ids2=None, kernel_estimator=KDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()\
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.dataset = dataset

        k_est = kernel_estimator(self.dataset.coordinates, self.dataset.mask)
        ma_maps1 = k_est.transform(ids, **kernel_args)
        if ids2 is None:
            ids2 = list(set(self.dataset.ids) - set(ids))
        ma_maps2 = k_est.transform(ids2, **kernel_args)
        self.ma_maps = [ma_maps1, ma_maps2]
        self.ids = [ids, ids2]
        self.voxel_thresh = None
        self.corr = None
        self.n_iters = None
        self.images = {}

    def fit(self, sample, voxel_thresh=0.01, corr='FDR', n_iters=10000):
        pass
