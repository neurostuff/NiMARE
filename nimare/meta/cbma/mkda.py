"""
CBMA methods from the multilevel kernel density analysis (MKDA) family
"""
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm.auto import tqdm
from scipy import ndimage, special
from nilearn.masking import apply_mask, unmask
from statsmodels.sandbox.stats.multicomp import multipletests

from .kernel import MKDAKernel, KDAKernel
from ...base import MetaResult, CBMAEstimator, KernelTransformer
from ...stats import null_to_p, p_to_z, one_way, two_way
from ...due import due, Doi

LGR = logging.getLogger(__name__)


@due.dcite(Doi('10.1093/scan/nsm015'), description='Introduces MKDA.')
class MKDADensity(CBMAEstimator):
    r"""
    Multilevel kernel density analysis- Density analysis [1]_.

    Parameters
    ----------
    voxel_thresh : float, optional
        Uncorrected voxel-level threshold. Default: 0.001
    n_iters : int, optional
        Number of iterations for correction. Default: 10000
    n_cores : int, optional
        Number of processes to use for meta-analysis. If -1, use all
        available cores. Default: -1
    kernel_estimator : :obj:`nimare.meta.cbma.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        MKDAKernel.
    **kwargs
        Keyword arguments. Arguments for the kernel_estimator can be assigned
        here, with the prefix '\kernel__' in the variable name.

    References
    ----------
    .. [1] Wager, Tor D., Martin Lindquist, and Lauren Kaplan. "Meta-analysis
        of functional neuroimaging data: current and future directions." Social
        cognitive and affective neuroscience 2.2 (2007): 150-158.
        https://doi.org/10.1093/scan/nsm015
    """
    def __init__(self, kernel_estimator=MKDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}

        if not issubclass(kernel_estimator, KernelTransformer):
            raise ValueError('Argument "kernel_estimator" must be a '
                             'KernelTransformer')

        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}
        for k in kwargs.keys():
            LGR.warning('Keyword argument "{0}" not recognized'.format(k))

        self.kernel_estimator = kernel_estimator(**kernel_args)

        self.mask = None
        self.dataset = None
        self.results = None

    def _fit(self, dataset):
        """
        Perform MKDA density meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.mask = dataset.mask

        ma_values = self.kernel_estimator.transform(dataset, masked=True)

        # Weight each SCM by square root of sample size
        ids_df = self.dataset.coordinates.groupby('id').first()
        if 'n' in ids_df.columns and 'inference' not in ids_df.columns:
            ids_n = ids_df['n'].astype(float).values
            weight_vec = np.sqrt(ids_n)[:, None] / np.sum(np.sqrt(ids_n))
        elif 'n' in ids_df.columns and 'inference' in ids_df.columns:
            ids_n = ids_df['n'].astype(float).values
            ids_inf = ids_df['inference'].map({'ffx': 0.75,
                                               'rfx': 1.}).values
            weight_vec = ((np.sqrt(ids_n)[:, None] * ids_inf[:, None]) /
                          np.sum(np.sqrt(ids_n) * ids_inf))
        else:
            weight_vec = np.ones((ma_values.shape[0], 1))
        self.weight_vec = weight_vec

        ma_values *= self.weight_vec
        of_values = np.sum(ma_values, axis=0)

        images = {'of': of_values}
        return images

    def _run_fwe_permutation(self, params):
        iter_ijk, iter_df, conn, voxel_thresh = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        iter_ma_maps = self.kernel_estimator.transform(iter_df, mask=self.mask, masked=True)
        iter_ma_maps *= self.weight_vec
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        iter_of_map = unmask(iter_of_map, self.mask)
        vthresh_iter_of_map = iter_of_map.get_data().copy()
        vthresh_iter_of_map[vthresh_iter_of_map < voxel_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_of_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        if clust_sizes:
            iter_max_cluster = np.max(clust_sizes)
        else:
            iter_max_cluster = 0
        return iter_max_value, iter_max_cluster

    def _fwe_correct_permutation(self, result, voxel_thresh=0.01, n_iters=1000,
                                 n_cores=-1):
        of_map = result.get_map('of', return_type='image')
        null_ijk = np.vstack(np.where(self.mask.get_data())).T

        if n_cores <= 0:
            n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            n_cores = mp.cpu_count()

        vthresh_of_map = of_map.get_data().copy()
        vthresh_of_map[vthresh_of_map < voxel_thresh] = 0

        rand_idx = np.random.choice(
            null_ijk.shape[0],
            size=(self.dataset.coordinates.shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.dataset.coordinates.copy()

        conn = np.ones((3, 3, 3))

        # Define parameters
        iter_conn = [conn] * n_iters
        iter_dfs = [iter_df] * n_iters
        iter_voxel_thresh = [voxel_thresh] * n_iters
        params = zip(iter_ijks, iter_dfs, iter_conn, iter_voxel_thresh)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params),
                                         total=n_iters))

        perm_max_values, perm_clust_sizes = zip(*perm_results)

        # Cluster-level FWE
        labeled_matrix, n_clusters = ndimage.measurements.label(vthresh_of_map, conn)
        cfwe_map = np.zeros(self.mask.shape)
        for i_clust in range(1, n_clusters + 1):
            clust_size = np.sum(labeled_matrix == i_clust)
            clust_idx = np.where(labeled_matrix == i_clust)
            cfwe_map[clust_idx] = -np.log(null_to_p(
                clust_size, perm_clust_sizes, 'upper'))
        cfwe_map[np.isinf(cfwe_map)] = -np.log(np.finfo(float).eps)
        cfwe_map = apply_mask(nib.Nifti1Image(cfwe_map, self.mask.affine),
                              self.mask)

        # Voxel-level FWE
        vfwe_map = apply_mask(of_map, self.mask)
        for i_vox, val in enumerate(vfwe_map):
            vfwe_map[i_vox] = -np.log(null_to_p(val, perm_max_values, 'upper'))
        vfwe_map[np.isinf(vfwe_map)] = -np.log(np.finfo(float).eps)
        vthresh_of_map = apply_mask(nib.Nifti1Image(vthresh_of_map,
                                                    of_map.affine),
                                    self.mask)
        images = {'vthresh': vthresh_of_map,
                  'logp_level-cluster': cfwe_map,
                  'logp_level-voxel': vfwe_map}
        return images


@due.dcite(Doi('10.1093/scan/nsm015'), description='Introduces MKDA.')
class MKDAChi2(CBMAEstimator):
    r"""
    Multilevel kernel density analysis- Chi-square analysis [1]_.

    Parameters
    ----------
    voxel_thresh : float, optional
        Uncorrected voxel-level threshold. Default: 0.01
    corr : {'FWE', 'FDR'}, optional
        Type of multiple comparisons correction to employ. Only currently
        supported option are FWE (which derives both cluster- and voxel-
        level corrected results) and FDR (performed directly on p-values).
    n_iters : int, optional
        Number of iterations for correction. Default: 10000
    prior : float, optional
        Uniform prior probability of each feature being active in a map in
        the absence of evidence from the map. Default: 0.5
    n_cores : int, optional
        Number of processes to use for meta-analysis. If -1, use all
        available cores. Default: -1
    kernel_estimator : :obj:`nimare.meta.cbma.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        MKDAKernel.
    **kwargs
        Keyword arguments. Arguments for the kernel_estimator can be assigned
        here, with the prefix '\kernel__' in the variable name.

    References
    ----------
    .. [1] Wager, Tor D., Martin Lindquist, and Lauren Kaplan. "Meta-analysis
        of functional neuroimaging data: current and future directions." Social
        cognitive and affective neuroscience 2.2 (2007): 150-158.
        https://doi.org/10.1093/scan/nsm015
    """
    def __init__(self, voxel_thresh=0.01, corr='FWE', n_iters=5000, prior=0.5,
                 n_cores=-1, kernel_estimator=MKDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}

        if not issubclass(kernel_estimator, KernelTransformer):
            raise ValueError('Argument "kernel_estimator" must be a '
                             'KernelTransformer')

        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}
        for k in kwargs.keys():
            LGR.warning('Keyword argument "{0}" not recognized'.format(k))

        self.kernel_estimator = kernel_estimator(**kernel_args)
        self.voxel_thresh = voxel_thresh
        self.corr = corr
        self.n_iters = n_iters
        self.prior = prior
        self.n_cores = n_cores

        self.mask = None
        self.dataset = None
        self.dataset2 = None
        self.results = None

        if n_cores <= 0:
            self.n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            self.n_cores = mp.cpu_count()
        else:
            self.n_cores = n_cores

    def fit(self, dataset, dataset2):
        """
        Perform MKDA chi2 meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        dataset2 : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.dataset2 = dataset2

        merged_coords = pd.concat((self.dataset.coordinates,
                                   self.dataset2.coordinates))

        ma_maps1 = self.kernel_estimator.transform(self.dataset, mask=self.mask, masked=True)
        ma_maps2 = self.kernel_estimator.transform(self.dataset2, mask=self.mask, masked=True)

        # Calculate different count variables
        eps = np.spacing(1)
        n_selected = ma_maps1.shape[0]
        n_unselected = ma_maps2.shape[0]
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
        pAgF_prior = self.prior * pAgF + (1 - self.prior) * pAgU
        pFgA_prior = pAgF * self.prior / pAgF_prior

        # One-way chi-square test for consistency of activation
        pAgF_chi2_vals = one_way(np.squeeze(n_selected_active_voxels),
                                 n_selected)
        pAgF_p_vals = special.chdtrc(1, pAgF_chi2_vals)
        pAgF_sign = np.sign(n_selected_active_voxels -
                            np.mean(n_selected_active_voxels))
        pAgF_z = p_to_z(pAgF_p_vals, tail='two') * pAgF_sign

        # Two-way chi-square for specificity of activation
        cells = np.squeeze(
            np.array([[n_selected_active_voxels, n_unselected_active_voxels],
                      [n_selected - n_selected_active_voxels,
                       n_unselected - n_unselected_active_voxels]]).T)
        pFgA_chi2_vals = two_way(cells)
        pFgA_p_vals = special.chdtrc(1, pFgA_chi2_vals)
        pFgA_p_vals[pFgA_p_vals < 1e-240] = 1e-240
        pFgA_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(pFgA_p_vals, tail='two') * pFgA_sign
        images = {
            'pA': pA,
            'pAgF': pAgF,
            'pFgA': pFgA,
            ('pAgF_given_pF=%0.2f' % self.prior): pAgF_prior,
            ('pFgA_given_pF=%0.2f' % self.prior): pFgA_prior,
            'consistency_z': pAgF_z,
            'specificity_z': pFgA_z,
            'consistency_chi2': pAgF_chi2_vals,
            'specificity_chi2': pFgA_chi2_vals}

        if self.corr == 'FWE':
            iter_dfs = [merged_coords] * self.n_iters
            null_ijk = np.vstack(np.where(self.mask.get_data())).T
            rand_idx = np.random.choice(null_ijk.shape[0],
                                        size=(merged_coords.shape[0], self.n_iters))
            rand_ijk = null_ijk[rand_idx, :]
            iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)

            params = zip(iter_dfs, iter_ijks)

            if self.n_cores == 1:
                perm_results = []
                for pp in tqdm(params, total=self.n_iters):
                    perm_results.append(self._perm(pp))
            else:
                with mp.Pool(self.n_cores) as p:
                    perm_results = list(tqdm(p.imap(self._perm, params), total=self.n_iters))
            pAgF_null_chi2_dist, pFgA_null_chi2_dist = zip(*perm_results)

            # pAgF_FWE
            pAgF_null_chi2_dist = np.squeeze(pAgF_null_chi2_dist)
            pAgF_p_FWE = np.empty_like(pAgF_chi2_vals).astype(float)
            for voxel in range(pFgA_chi2_vals.shape[0]):
                pAgF_p_FWE[voxel] = null_to_p(pAgF_chi2_vals[voxel],
                                              pAgF_null_chi2_dist,
                                              tail='upper')
            # Crop p-values of 0 or 1 to nearest values that won't evaluate to
            # 0 or 1. Prevents inf z-values.
            pAgF_p_FWE[pAgF_p_FWE < eps] = eps
            pAgF_p_FWE[pAgF_p_FWE > (1. - eps)] = 1. - eps
            pAgF_z_FWE = p_to_z(pAgF_p_FWE, tail='two') * pAgF_sign
            images['consistency_p_FWE'] = pAgF_p_FWE
            images['consistency_z_FWE'] = pAgF_z_FWE

            # pFgA_FWE
            pFgA_null_chi2_dist = np.squeeze(pFgA_null_chi2_dist)
            pFgA_p_FWE = np.empty_like(pFgA_chi2_vals).astype(float)
            for voxel in range(pFgA_chi2_vals.shape[0]):
                pFgA_p_FWE[voxel] = null_to_p(pFgA_chi2_vals[voxel],
                                              pFgA_null_chi2_dist,
                                              tail='upper')
            # Crop p-values of 0 or 1 to nearest values that won't evaluate to
            # 0 or 1. Prevents inf z-values.
            pFgA_p_FWE[pFgA_p_FWE < eps] = eps
            pFgA_p_FWE[pFgA_p_FWE > (1. - eps)] = 1. - eps
            pFgA_z_FWE = p_to_z(pFgA_p_FWE, tail='two') * pFgA_sign
            images['specificity_p_FWE'] = pFgA_p_FWE
            images['specificity_z_FWE'] = pFgA_z_FWE
        elif self.corr == 'FDR':
            _, pAgF_p_FDR, _, _ = multipletests(pAgF_p_vals, alpha=0.05,
                                                method='fdr_bh',
                                                is_sorted=False,
                                                returnsorted=False)
            pAgF_z_FDR = p_to_z(pAgF_p_FDR, tail='two') * pAgF_sign
            images['consistency_z_FDR'] = pAgF_z_FDR

            _, pFgA_p_FDR, _, _ = multipletests(pFgA_p_vals, alpha=0.05,
                                                method='fdr_bh',
                                                is_sorted=False,
                                                returnsorted=False)
            pFgA_z_FDR = p_to_z(pFgA_p_FDR, tail='two') * pFgA_sign
            images['specificity_z_FDR'] = pFgA_z_FDR

        self.results = MetaResult(self, self.mask, maps=images)

    def _perm(self, params):
        iter_df, iter_ijk = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk

        k_est = self.kernel_estimator(iter_df, self.mask)
        temp_ma_maps = k_est.transform(self.ids, masked=True,
                                       **self.kernel_arguments)
        temp_ma_maps2 = k_est.transform(self.ids2, masked=True,
                                        **self.kernel_arguments)

        n_selected = temp_ma_maps.shape[0]
        n_unselected = temp_ma_maps2.shape[0]
        n_selected_active_voxels = np.sum(temp_ma_maps, axis=0)
        n_unselected_active_voxels = np.sum(temp_ma_maps2, axis=0)

        # Conditional probabilities
        # pAgF = n_selected_active_voxels * 1.0 / n_selected
        # pAgU = n_unselected_active_voxels * 1.0 / n_unselected

        # One-way chi-square test for consistency of activation
        pAgF_chi2_vals = one_way(np.squeeze(n_selected_active_voxels),
                                 n_selected)
        iter_pAgF_chi2 = np.max(pAgF_chi2_vals)

        # Two-way chi-square for specificity of activation
        cells = np.squeeze(
            np.array([[n_selected_active_voxels, n_unselected_active_voxels],
                      [n_selected - n_selected_active_voxels,
                       n_unselected - n_unselected_active_voxels]]).T)
        pFgA_chi2_vals = two_way(cells)
        iter_pFgA_chi2 = np.max(pFgA_chi2_vals)
        return iter_pAgF_chi2, iter_pFgA_chi2


@due.dcite(Doi('10.1016/S1053-8119(03)00078-8'),
           description='Introduces the KDA algorithm.')
@due.dcite(Doi('10.1016/j.neuroimage.2004.03.052'),
           description='Also introduces the KDA algorithm.')
class KDA(CBMAEstimator):
    r"""
    Kernel density analysis [1]_.

    Parameters
    ----------
    n_iters : int, optional
        Number of iterations for correction. Default: 10000
    n_cores : int, optional
        Number of processes to use for meta-analysis. If -1, use all
        available cores. Default: -1
    kernel_estimator : :obj:`nimare.meta.cbma.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        KDAKernel.
    **kwargs
        Keyword arguments. Arguments for the kernel_estimator can be assigned
        here, with the prefix '\kernel__' in the variable name.

    References
    ----------
    .. [1] Wager, Tor D., et al. "Valence, gender, and lateralization of
        functional brain anatomy in emotion: a meta-analysis of findings from
        neuroimaging." Neuroimage 19.3 (2003): 513-531.
        https://doi.org/10.1016/S1053-8119(03)00078-8
    .. [2] Wager, Tor D., John Jonides, and Susan Reading. "Neuroimaging
        studies of shifting attention: a meta-analysis." Neuroimage 22.4
        (2004): 1679-1693. https://doi.org/10.1016/j.neuroimage.2004.03.052
    """
    def __init__(self, n_iters=10000, n_cores=-1, kernel_estimator=KDAKernel,
                 **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}

        if not issubclass(kernel_estimator, KernelTransformer):
            raise ValueError('Argument "kernel_estimator" must be a '
                             'KernelTransformer')

        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}
        for k in kwargs.keys():
            LGR.warning('Keyword argument "{0}" not recognized'.format(k))

        self.kernel_estimator = kernel_estimator(**kernel_args)
        self.n_iters = n_iters
        self.n_cores = n_cores

        self.mask = None
        self.dataset = None
        self.results = None

        if n_cores <= 0:
            self.n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            self.n_cores = mp.cpu_count()
        else:
            self.n_cores = n_cores

    def fit(self, dataset):
        """
        Perform KDA meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.mask = dataset.mask

        null_ijk = np.vstack(np.where(self.mask.get_data())).T

        ma_maps = self.kernel_estimator.transform(dataset, masked=True)
        of_map = np.sum(ma_maps, axis=0)

        rand_idx = np.random.choice(
            null_ijk.shape[0],
            size=(self.dataset.coordinates.shape[0], self.n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.dataset.coordinates.copy()

        # Define parameters
        iter_dfs = [iter_df] * self.n_iters
        params = zip(iter_ijks, iter_dfs)

        if self.n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=self.n_iters):
                perm_results.append(self._perm(pp))
        else:
            with mp.Pool(self.n_cores) as p:
                perm_results = list(tqdm(p.imap(self._perm, params), total=self.n_iters))

        perm_max_values = perm_results

        # Voxel-level FWE
        vfwe_map = of_map.copy()
        for i_vox, val in enumerate(of_map):
            vfwe_map[i_vox] = -np.log(null_to_p(val, perm_max_values, 'upper'))
        vfwe_map[np.isinf(vfwe_map)] = -np.log(np.finfo(float).eps)

        images = {'logp_vfwe': vfwe_map}
        self.results = MetaResult(self, self.mask, maps=images)

    def _perm(self, params):
        iter_ijk, iter_df = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        iter_ma_maps = self.kernel_estimator.transform(iter_df, mask=self.mask, masked=True)
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        return iter_max_value
