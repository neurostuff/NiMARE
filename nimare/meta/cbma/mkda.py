"""
Coordinate-based meta-analysis estimators
"""
import warnings
import multiprocessing as mp

import numpy as np
import nibabel as nib
from scipy import ndimage, special
from nilearn.masking import apply_mask, unmask

from .base import CBMAEstimator
from .kernel import MKDAKernel, KDAKernel
from .utils import p_to_z
from ..base import MetaResult
from ...utils import vox2mm
from ...stats import fdr
from ...due import due, Doi


@due.dcite(Doi('10.1093/scan/nsm015'), description='Introduces MKDA.')
class MKDADensity(CBMAEstimator):
    """
    Multilevel kernel density analysis- Density analysis
    """
    def __init__(self, dataset, ids, kernel_estimator=MKDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]

        self.kernel_estimator = kernel_estimator
        self.kernel_arguments = kernel_args
        self.ids = ids
        self.voxel_thresh = None
        self.clust_thresh = None
        self.n_iters = None
        self.results = None

    def fit(self, voxel_thresh=0.01, q=0.05, n_iters=1000, n_cores=4):
        null_ijk = np.vstack(np.where(self.mask.get_data())).T
        self.voxel_thresh = voxel_thresh
        self.clust_thresh = q
        self.n_iters = n_iters

        k_est = self.kernel_estimator(self.coordinates, self.mask)
        ma_maps = k_est.transform(self.ids, **self.kernel_arguments)

        # Weight each SCM by square root of sample size
        ids_df = self.coordinates.groupby('id').first()
        if 'n' in ids_df.columns and 'inference' not in ids_df.columns:
            ids_n = ids_df.loc[self.ids, 'n'].astype(float).values
            weight_vec = np.sqrt(ids_n)[:, None] / np.sum(np.sqrt(ids_n))
        elif 'n' in ids_df.columns and 'inference' in ids_df.columns:
            ids_n = ids_df.loc[self.ids, 'n'].astype(float).values
            ids_inf = ids_df.loc[self.ids, 'inference'].map({'ffx': 0.75, 'rfx': 1.}).values
            weight_vec = (np.sqrt(ids_n)[:, None] * ids_inf[:, None]) / \
                         np.sum(np.sqrt(ids_n) * ids_inf)
        else:
            weight_vec = np.ones((len(ma_maps), 1))

        ma_maps = apply_mask(ma_maps, self.mask)
        ma_maps *= weight_vec
        of_map = np.sum(ma_maps, axis=0)
        of_map = unmask(of_map, self.mask)

        vthresh_of_map = of_map.get_data().copy()
        vthresh_of_map[vthresh_of_map < voxel_thresh] = 0

        rand_idx = np.random.choice(null_ijk.shape[0],
                                    size=(self.coordinates.shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.coordinates.copy()

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
        for i, clust_size in enumerate(clust_sizes):
            if clust_size >= clust_size_thresh and i > 0:
                clust_idx = np.where(labeled_matrix == i)
                cfwe_of_map[clust_idx] = vthresh_of_map[clust_idx]
        cfwe_of_map = apply_mask(nib.Nifti1Image(cfwe_of_map, of_map.affine),
                                 self.mask)

        ## Voxel-level FWE
        # Determine OF values in [1 - clust_thresh]th percentile (e.g. 95th)
        vfwe_thresh = np.percentile(perm_max_values, percentile)
        vfwe_of_map = of_map.get_data().copy()
        vfwe_of_map[vfwe_of_map < vfwe_thresh] = 0.
        vfwe_of_map = apply_mask(nib.Nifti1Image(vfwe_of_map, of_map.affine),
                                 self.mask)

        vthresh_of_map = apply_mask(nib.Nifti1Image(vthresh_of_map, of_map.affine),
                                    self.mask)
        self.results = MetaResult(vthresh=vthresh_of_map, cfwe=cfwe_of_map,
                                  vfwe=vfwe_of_map, mask=self.mask)

    def _perm(self, params):
        iter_ijk, iter_df, weight_vec, conn = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        k_est = self.kernel_estimator(iter_df, self.mask)
        iter_ma_maps = k_est.transform(self.ids, **self.kernel_arguments)
        iter_ma_maps = apply_mask(iter_ma_maps, self.mask)
        iter_ma_maps *= weight_vec
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        iter_of_map = unmask(iter_of_map, self.mask)
        vthresh_iter_of_map = iter_of_map.get_data().copy()
        vthresh_iter_of_map[vthresh_iter_of_map < self.voxel_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_of_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        if clust_sizes:
            iter_max_cluster = np.max(clust_sizes)
        else:
            iter_max_cluster = 0
        return iter_max_value, iter_max_cluster


@due.dcite(Doi('10.1093/scan/nsm015'), description='Introduces MKDA.')
class MKDAChi2(CBMAEstimator):
    """
    Multilevel kernel density analysis- Chi-square analysis
    """
    def __init__(self, dataset, ids, ids2=None, kernel_estimator=MKDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.mask = dataset.mask

        # Check kernel estimator (must be MKDA)
        k_est = kernel_estimator(dataset.coordinates, self.mask)
        assert isinstance(k_est, MKDAKernel)

        self.kernel_estimator = kernel_estimator
        self.kernel_arguments = kernel_args

        if ids2 is None:
            ids2 = list(set(dataset.coordinates['id'].values) - set(ids))
        all_ids = ids + ids2
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(all_ids)]

        self.ids = ids
        self.ids2 = ids2
        self.voxel_thresh = None
        self.corr = None
        self.n_iters = None
        self.results = None

    def fit(self, voxel_thresh=0.01, q=0.05, corr='FWE', n_iters=5000,
            prior=0.5, n_cores=4):
        self.voxel_thresh = voxel_thresh
        self.corr = corr
        self.n_iters = n_iters

        k_est = self.kernel_estimator(self.coordinates, self.mask)
        ma_maps1 = k_est.transform(self.ids, masked=True, **self.kernel_arguments)
        ma_maps2 = k_est.transform(self.ids2, masked=True, **self.kernel_arguments)

        # Calculate different count variables
        n_selected = len(self.ids)
        n_unselected = len(self.ids2)
        n_mappables = n_selected + n_unselected

        # Transform MA maps to 1d arrays
        ma_maps_all = np.vstack((ma_maps1, ma_maps2))

        n_selected_active_voxels = np.sum(ma_maps1, axis=0)
        n_unselected_active_voxels = np.sum(ma_maps2, axis=0)

        # Nomenclature for variables below: p = probability,
        # F = feature present, g = given, U = unselected, A = activation.
        # So, e.g., pAgF = p(A|F) = probability of activation
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
        pAgF_p_vals = self._one_way(np.squeeze(n_selected_active_voxels), n_selected)
        pAgF_p_vals[pAgF_p_vals < 1e-240] = 1e-240
        z_sign = np.sign(n_selected_active_voxels - np.mean(n_selected_active_voxels)).ravel()
        pAgF_z = p_to_z(pAgF_p_vals, z_sign)

        # Two-way chi-square for specificity of activation
        cells = np.squeeze(
            np.array([[n_selected_active_voxels, n_unselected_active_voxels],
                      [n_selected - n_selected_active_voxels,
                       n_unselected - n_unselected_active_voxels]]).T)
        pFgA_p_vals = self._two_way(cells)
        pFgA_p_vals[pFgA_p_vals < 1e-240] = 1e-240
        z_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(pFgA_p_vals, z_sign)
        images = {
            'pA': pA,
            'pAgF': pAgF,
            'pFgA': pFgA,
            ('pAgF_given_pF=%0.2f' % prior): pAgF_prior,
            ('pFgA_given_pF=%0.2f' % prior): pFgA_prior,
            'consistency_z': pAgF_z,
            'specificity_z': pFgA_z}

        if corr == 'FWE':
            pool = mp.Pool(n_cores)
            iter_dfs = [self.coordinates.copy()] * n_iters
            null_ijk = np.vstack(np.where(self.mask.get_data())).T
            rand_idx = np.random.choice(null_ijk.shape[0],
                                        size=(self.coordinates.shape[0], n_iters))
            rand_ijk = null_ijk[rand_idx, :]
            iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)

            params = zip(iter_dfs, iter_ijks)
            perm_results = pool.map(self._perm, params)
            pool.close()
            pAgF_perm_dist, pFgA_perm_dist = zip(*perm_results)

            # pAgF_FWE
            pAgF_perm_dist = np.vstack(pAgF_perm_dist)
            _, bin_edges = np.apply_along_axis(np.histogram,
                                               arr=np.abs(pAgF_perm_dist),
                                               axis=0, bins=1000,
                                               density=False)
            bin_edges = np.vstack(bin_edges)
            pAgF_p_FWE = np.empty_like(pAgF_z).astype(float)
            for i, (be, tv) in enumerate(zip(bin_edges, pAgF_z)):
                if any(np.abs(tv) > be):
                    pAgF_p_FWE[i] = np.max(np.where(np.abs(tv) > be)[0]) / len(be)
                else:
                    pAgF_p_FWE[i] = 1.
            pAgF_z_FWE = pAgF_z.copy()
            pAgF_z_FWE[pAgF_p_FWE > q] = 0
            images['pAgF_z_FWE_{0}'.format(str(q).split('.')[1])] = pAgF_z_FWE

            # pFgA_FWE
            pFgA_perm_dist = np.vstack(pFgA_perm_dist)
            _, bin_edges = np.apply_along_axis(np.histogram,
                                               arr=np.abs(pFgA_perm_dist),
                                               axis=0, bins=1000,
                                               density=False)
            bin_edges = np.vstack(bin_edges)
            pFgA_p_FWE = np.empty_like(pFgA_z).astype(float)
            for i, (be, tv) in enumerate(zip(bin_edges, pFgA_z)):
                if any(np.abs(tv) > be):
                    pFgA_p_FWE[i] = np.max(np.where(np.abs(tv) > be)[0]) / len(be)
                else:
                    pFgA_p_FWE[i] = 1.
            pFgA_z_FWE = pFgA_z.copy()
            pFgA_z_FWE[pFgA_p_FWE > q] = 0
            images['pFgA_z_FWE_{0}'.format(str(q).split('.')[1])] = pFgA_z_FWE
        elif corr == 'FDR':
            pAgF_fdr_thresh = fdr(pAgF_p_vals, q)
            pAgF_z_FDR = pAgF_z.copy()
            pAgF_z_FDR[pAgF_p_vals > pAgF_fdr_thresh] = 0
            images['pAgF_z_FDR_{0}'.format(str(q).split('.')[1])] = pAgF_z_FDR

            pFgA_fdr_thresh = fdr(pFgA_p_vals, q)
            pFgA_z_FDR = pFgA_z.copy()
            pFgA_z_FDR[pFgA_p_vals > pFgA_fdr_thresh] = 0
            images['pFgA_z_FDR_{0}'.format(str(q).split('.')[1])] = pFgA_z_FDR

        self.results = MetaResult(mask=self.mask, **images)

    def _perm(self, params):
        iter_df, iter_ijk = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk

        k_est = self.kernel_estimator(iter_df, self.mask)
        temp_ma_maps = k_est.transform(self.ids, masked=True, **self.kernel_arguments)
        temp_ma_maps2 = k_est.transform(self.ids2, masked=True, **self.kernel_arguments)

        n_selected = temp_ma_maps.shape[0]
        n_unselected = temp_ma_maps2.shape[0]
        n_selected_active_voxels = np.sum(temp_ma_maps, axis=0)
        n_unselected_active_voxels = np.sum(temp_ma_maps2, axis=0)

        # Conditional probabilities
        pAgF = n_selected_active_voxels * 1.0 / n_selected
        pAgU = n_unselected_active_voxels * 1.0 / n_unselected

        # One-way chi-square test for consistency of activation
        pAgF_p_vals = self._one_way(np.squeeze(n_selected_active_voxels), n_selected)
        pAgF_p_vals[pAgF_p_vals < 1e-240] = 1e-240
        z_sign = np.sign(n_selected_active_voxels - np.mean(n_selected_active_voxels)).ravel()
        pAgF_z = p_to_z(pAgF_p_vals, z_sign)

        # Two-way chi-square for specificity of activation
        cells = np.squeeze(
            np.array([[n_selected_active_voxels, n_unselected_active_voxels],
                      [n_selected - n_selected_active_voxels,
                       n_unselected - n_unselected_active_voxels]]).T)
        pFgA_p_vals = self._two_way(cells)
        pFgA_p_vals[pFgA_p_vals < 1e-240] = 1e-240
        z_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(pFgA_p_vals, z_sign)
        return pAgF_z, pFgA_z

    def _one_way(self, data, n):
        """ One-way chi-square test of independence.
        Takes a 1D array as input and compares activation at each voxel to
        proportion expected under a uniform distribution throughout the array.
        Note that if you're testing activation with this, make sure that only
        valid voxels (e.g., in-mask gray matter voxels) are included in the
        array, or results won't make any sense!
        """
        term = data.astype('float64')
        no_term = n - term
        t_exp = np.mean(term, 0)
        t_exp = np.array([t_exp, ] * data.shape[0])
        nt_exp = n - t_exp
        t_mss = (term - t_exp) ** 2 / t_exp
        nt_mss = (no_term - nt_exp) ** 2 / nt_exp
        chi2 = t_mss + nt_mss
        return special.chdtrc(1, chi2)

    def _two_way(self, cells):
        """ Two-way chi-square test of independence.
        Takes a 3D array as input: N(voxels) x 2 x 2, where the last two
        dimensions are the contingency table for each of N voxels. Returns an
        array of p-values.
        """
        # Mute divide-by-zero warning for bad voxels since we account for that later
        warnings.simplefilter("ignore", RuntimeWarning)

        cells = cells.astype('float64')  # Make sure we don't overflow
        total = np.apply_over_axes(np.sum, cells, [1, 2]).ravel()
        chi_sq = np.zeros(cells.shape, dtype='float64')
        for i in range(2):
            for j in range(2):
                exp = np.sum(cells[:, i, :], 1).ravel() * \
                    np.sum(cells[:, :, j], 1).ravel() / total
                bad_vox = np.where(exp == 0)[0]
                chi_sq[:, i, j] = (cells[:, i, j] - exp) ** 2 / exp
                chi_sq[bad_vox, i, j] = 1.0  # Set p-value for invalid voxels to 1
        chi_sq = np.apply_over_axes(np.sum, chi_sq, [1, 2]).ravel()
        return special.chdtrc(1, chi_sq)


@due.dcite(Doi('10.1016/S1053-8119(03)00078-8'),
           description='Introduces the KDA algorithm.')
@due.dcite(Doi('10.1016/j.neuroimage.2004.03.052'),
           description='Also introduces the KDA algorithm.')
class KDA(CBMAEstimator):
    """
    Kernel density analysis
    """
    def __init__(self, dataset, ids, ids2=None, kernel_estimator=KDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]
        self.kernel_estimator = kernel_estimator
        self.kernel_arguments = kernel_args
        self.ids = ids
        self.clust_thresh = None
        self.n_iters = None
        self.images = {}

    def fit(self, q=0.05, n_iters=10000, n_cores=4):
        null_ijk = np.vstack(np.where(self.mask.get_data())).T
        self.clust_thresh = q
        self.n_iters = n_iters

        k_est = self.kernel_estimator(self.coordinates, self.mask)
        ma_maps = k_est.transform(self.ids, masked=True, **self.kernel_arguments)
        of_map = np.sum(ma_maps, axis=0)

        rand_idx = np.random.choice(null_ijk.shape[0],
                                    size=(self.coordinates.shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.coordinates.copy()

        # Define parameters
        iter_dfs = [iter_df] * n_iters
        params = zip(iter_ijks, iter_dfs)

        pool = mp.Pool(n_cores)
        perm_max_values = pool.map(self._perm, params)
        pool.close()

        percentile = 100 * (1 - q)

        # Determine OF values in [1 - clust_thresh]th percentile (e.g. 95th)
        vfwe_thresh = np.percentile(perm_max_values, percentile)
        vfwe_of_map = of_map.copy()
        vfwe_of_map[vfwe_of_map < vfwe_thresh] = 0.
        self.results = MetaResult(vfwe=vfwe_of_map, mask=self.mask)

    def _perm(self, params):
        iter_ijk, iter_df = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        k_est = self.kernel_estimator(iter_df, self.mask)
        iter_ma_maps = k_est.transform(self.ids, **self.kernel_arguments)
        iter_ma_maps = apply_mask(iter_ma_maps, self.mask)
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        return iter_max_value
