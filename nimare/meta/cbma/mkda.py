"""
CBMA methods from the multilevel kernel density analysis (MKDA) family
"""
import logging
import multiprocessing as mp

import numpy as np
import nibabel as nib
from tqdm.auto import tqdm
from scipy import ndimage, special
from statsmodels.sandbox.stats.multicomp import multipletests

from .kernel import MKDAKernel, KDAKernel
from ...results import MetaResult
from ...base import CBMAEstimator
from ...stats import null_to_p, one_way, two_way
from ...transforms import p_to_z
from ...due import due
from ... import references

LGR = logging.getLogger(__name__)


@due.dcite(references.MKDA, description='Introduces MKDA.')
class MKDADensity(CBMAEstimator):
    r"""
    Multilevel kernel density analysis- Density analysis.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.cbma.kernel.MKDAKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    Available correction methods: :func:`MKDADensity.correct_fwe_montecarlo`

    References
    ----------
    * Wager, Tor D., Martin Lindquist, and Lauren Kaplan. "Meta-analysis
      of functional neuroimaging data: current and future directions." Social
      cognitive and affective neuroscience 2.2 (2007): 150-158.
      https://doi.org/10.1093/scan/nsm015
    """
    _required_inputs = {
        'coordinates': ('coordinates', None),
    }

    def __init__(self, kernel_transformer=MKDAKernel, **kwargs):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

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
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        ma_values = self.kernel_transformer.transform(
            self.inputs_['coordinates'],
            masker=self.masker,
            return_type='array'
        )

        # Weight each SCM by square root of sample size
        # TODO: Incorporate sample-size and inference metadata extraction and
        # merging into df.
        # This will need to be distinct from the kernel_transformer-based kind
        # done in CBMAEstimator._preprocess_input
        ids_df = self.inputs_['coordinates'].groupby('id').first()
        if 'sample_size' in ids_df.columns and 'inference' not in ids_df.columns:
            ids_n = ids_df['sample_size'].astype(float).values
            weight_vec = np.sqrt(ids_n)[:, None] / np.sum(np.sqrt(ids_n))
        elif 'sample_size' in ids_df.columns and 'inference' in ids_df.columns:
            ids_n = ids_df['sample_size'].astype(float).values
            ids_inf = ids_df['inference'].map({'ffx': 0.75,
                                               'rfx': 1.}).values
            weight_vec = ((np.sqrt(ids_n)[:, None] * ids_inf[:, None]) /
                          np.sum(np.sqrt(ids_n) * ids_inf))
        else:
            weight_vec = np.ones((ma_values.shape[0], 1))
        self.weight_vec = weight_vec

        ma_values *= self.weight_vec
        of_values = np.sum(ma_values, axis=0)

        images = {
            'of': of_values,
        }
        return images

    def _run_fwe_permutation(self, params):
        iter_ijk, iter_df, conn, voxel_thresh = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        iter_ma_maps = self.kernel_transformer.transform(
            iter_df,
            masker=self.masker,
            return_type='array'
        )
        iter_ma_maps *= self.weight_vec
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        iter_of_map = self.masker.inverse_transform(iter_of_map)
        vthresh_iter_of_map = iter_of_map.get_fdata().copy()
        vthresh_iter_of_map[vthresh_iter_of_map < voxel_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_of_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        if clust_sizes:
            iter_max_cluster = np.max(clust_sizes)
        else:
            iter_max_cluster = 0
        return iter_max_value, iter_max_cluster

    def correct_fwe_montecarlo(self, result, voxel_thresh=0.01, n_iters=1000, n_cores=-1):
        """
        Perform FWE correction using the max-value permutation method.
        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        voxel_thresh : :obj:`float`, optional
            Cluster-defining OF-value threshold. Default is 0.01.
        n_iters : :obj:`int`, optional
            Number of iterations to build the vFWE and cFWE null distributions.
            Default is 1000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is -1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'vthresh', 'logp_level-cluster', and 'logp_level-voxel'.

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = MKDADensity()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', voxel_thresh=0.01,
                                     n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        of_map = result.get_map('of', return_type='image')
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T

        if n_cores <= 0:
            n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            n_cores = mp.cpu_count()

        vthresh_of_map = of_map.get_fdata().copy()
        vthresh_of_map[vthresh_of_map < voxel_thresh] = 0

        rand_idx = np.random.choice(
            null_ijk.shape[0],
            size=(self.inputs_['coordinates'].shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.inputs_['coordinates'].copy()

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
        cfwe_map = np.zeros(self.masker.mask_img.shape)
        for i_clust in range(1, n_clusters + 1):
            clust_size = np.sum(labeled_matrix == i_clust)
            clust_idx = np.where(labeled_matrix == i_clust)
            cfwe_map[clust_idx] = -np.log(null_to_p(
                clust_size, perm_clust_sizes, 'upper'))
        cfwe_map[np.isinf(cfwe_map)] = -np.log(np.finfo(float).eps)
        cfwe_map = np.squeeze(
            self.masker.transform(nib.Nifti1Image(cfwe_map, self.masker.mask_img.affine))
        )

        # Voxel-level FWE
        vfwe_map = np.squeeze(self.masker.transform(of_map))
        for i_vox, val in enumerate(vfwe_map):
            vfwe_map[i_vox] = -np.log(null_to_p(val, perm_max_values, 'upper'))
        vfwe_map[np.isinf(vfwe_map)] = -np.log(np.finfo(float).eps)

        images = {
            'logp_level-cluster': cfwe_map,
            'logp_level-voxel': vfwe_map
        }
        return images


@due.dcite(references.MKDA, description='Introduces MKDA.')
class MKDAChi2(CBMAEstimator):
    r"""
    Multilevel kernel density analysis- Chi-square analysis.

    Parameters
    ----------
    prior : float, optional
        Uniform prior probability of each feature being active in a map in
        the absence of evidence from the map. Default: 0.5
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.cbma.kernel.MKDAKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    Available correction methods: :func:`MKDAChi2.correct_fwe_montecarlo`,
    :obj:`MKDAChi2.correct_fdr_bh`

    References
    ----------
    * Wager, Tor D., Martin Lindquist, and Lauren Kaplan. "Meta-analysis
      of functional neuroimaging data: current and future directions." Social
      cognitive and affective neuroscience 2.2 (2007): 150-158.
      https://doi.org/10.1093/scan/nsm015
    """
    _required_inputs = {
        'coordinates': ('coordinates', None),
    }

    def __init__(self, prior=0.5, kernel_transformer=MKDAKernel, **kwargs):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

        self.prior = prior

    def fit(self, dataset1, dataset2):
        """
        Fit CBMAEstimator to datasets.

        Parameters
        ----------
        dataset1/dataset2 : :obj:`nimare.dataset.Dataset`
            Dataset objects to analyze.

        Returns
        -------
        :obj:`nimare.results.MetaResult`
            Results of CBMAEstimator fitting, with the following maps:
            'prob_desc-A', 'prob_desc-AgF', 'prob_desc-FgA',
            'prob_desc-AgF_given_pF=XX', 'prob_desc-FgA_given_pF=XX',
            'z_desc-consistency', 'z_desc-specificity',
            'chi2_desc-consistency', 'chi2_desc-specificity',
            'p_desc-consistency', and 'p_desc-specificity'
        """
        self._validate_input(dataset1)
        self._validate_input(dataset2)
        self._preprocess_input(dataset1)
        # override
        self.inputs_['coordinates1'] = self.inputs_.pop('coordinates')
        self._preprocess_input(dataset2)
        # override
        self.inputs_['coordinates2'] = self.inputs_.pop('coordinates')

        maps = self._fit(dataset1, dataset2)
        self.results = MetaResult(self, dataset1.masker.mask_img, maps)
        return self.results

    def _fit(self, dataset1, dataset2):
        self.masker = self.masker or dataset1.masker
        self.null_distributions_ = {}

        ma_maps1 = self.kernel_transformer.transform(
            self.inputs_['coordinates1'],
            masker=self.masker,
            return_type='array'
        )
        ma_maps2 = self.kernel_transformer.transform(
            self.inputs_['coordinates2'],
            masker=self.masker,
            return_type='array'
        )

        # Calculate different count variables
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
            'prob_desc-A': pA,
            'prob_desc-AgF': pAgF,
            'prob_desc-FgA': pFgA,
            ('prob_desc-AgF_given_pF=%0.2f' % self.prior): pAgF_prior,
            ('prob_desc-FgA_given_pF=%0.2f' % self.prior): pFgA_prior,
            'z_desc-consistency': pAgF_z,
            'z_desc-specificity': pFgA_z,
            'chi2_desc-consistency': pAgF_chi2_vals,
            'chi2_desc-specificity': pFgA_chi2_vals,
            'p_desc-consistency': pAgF_p_vals,
            'p_desc-specificity': pFgA_p_vals,
        }
        return images

    def _run_fwe_permutation(self, params):
        iter_df1, iter_df2, iter_ijk1, iter_ijk2 = params
        iter_ijk1 = np.squeeze(iter_ijk1)
        iter_ijk2 = np.squeeze(iter_ijk2)
        iter_df1[['i', 'j', 'k']] = iter_ijk1
        iter_df2[['i', 'j', 'k']] = iter_ijk2

        temp_ma_maps1 = self.kernel_transformer.transform(
            iter_df1,
            self.masker,
            return_type='array'
        )
        temp_ma_maps2 = self.kernel_transformer.transform(
            iter_df2,
            self.masker,
            return_type='array'
        )

        n_selected = temp_ma_maps1.shape[0]
        n_unselected = temp_ma_maps2.shape[0]
        n_selected_active_voxels = np.sum(temp_ma_maps1, axis=0)
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

    def correct_fwe_montecarlo(self, result, n_iters=5000, n_cores=-1):
        """
        Perform FWE correction using the max-value permutation method.
        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        n_iters : :obj:`int`, optional
            Number of iterations to build the vFWE null distribution.
            Default is 5000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is -1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'p_desc-consistency_level-voxel',
            'z_desc-consistency_level-voxel', 'p_desc-specificity_level-voxel',
            and 'p_desc-specificity_level-voxel'.

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = MKDAChi2()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T
        pAgF_chi2_vals = result.get_map('chi2_desc-consistency', return_type='array')
        pFgA_chi2_vals = result.get_map('chi2_desc-specificity', return_type='array')
        pAgF_z_vals = result.get_map('z_desc-consistency', return_type='array')
        pFgA_z_vals = result.get_map('z_desc-specificity', return_type='array')
        pAgF_sign = np.sign(pAgF_z_vals)
        pFgA_sign = np.sign(pFgA_z_vals)

        if n_cores <= 0:
            n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            n_cores = mp.cpu_count()

        iter_df1 = self.inputs_['coordinates1'].copy()
        iter_df2 = self.inputs_['coordinates2'].copy()
        iter_dfs1 = [iter_df1] * n_iters
        iter_dfs2 = [iter_df2] * n_iters
        rand_idx1 = np.random.choice(null_ijk.shape[0],
                                     size=(iter_df1.shape[0], n_iters))
        rand_ijk1 = null_ijk[rand_idx1, :]
        iter_ijks1 = np.split(rand_ijk1, rand_ijk1.shape[1], axis=1)
        rand_idx2 = np.random.choice(null_ijk.shape[0],
                                     size=(iter_df2.shape[0], n_iters))
        rand_ijk2 = null_ijk[rand_idx2, :]
        iter_ijks2 = np.split(rand_ijk2, rand_ijk2.shape[1], axis=1)
        eps = np.spacing(1)

        params = zip(iter_dfs1, iter_dfs2, iter_ijks1, iter_ijks2)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params),
                                         total=n_iters))
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

        images = {
            'p_desc-consistency_level-voxel': pAgF_p_FWE,
            'z_desc-consistency_level-voxel': pAgF_z_FWE,
            'p_desc-specificity_level-voxel': pFgA_p_FWE,
            'z_desc-specificity_level-voxel': pFgA_z_FWE,
        }
        return images

    def correct_fdr_bh(self, result, alpha=0.05):
        """
        Perform FDR correction using the Benjamini-Hochberg method.
        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        alpha : :obj:`float`, optional
            Alpha. Default is 0.05.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'consistency_z_FDR' and 'specificity_z_FDR'.

        See Also
        --------
        nimare.correct.FDRCorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = MKDAChi2()
        >>> result = meta.fit(dset)
        >>> corrector = FDRCorrector(method='bh', alpha=0.05)
        >>> cresult = corrector.transform(result)
        """
        pAgF_p_vals = result.get_map('p_desc-consistency', return_type='array')
        pFgA_p_vals = result.get_map('p_desc-specificity', return_type='array')
        pAgF_z_vals = result.get_map('z_desc-consistency', return_type='array')
        pFgA_z_vals = result.get_map('z_desc-specificity', return_type='array')
        pAgF_sign = np.sign(pAgF_z_vals)
        pFgA_sign = np.sign(pFgA_z_vals)
        _, pAgF_p_FDR, _, _ = multipletests(pAgF_p_vals, alpha=alpha,
                                            method='fdr_bh',
                                            is_sorted=False,
                                            returnsorted=False)
        pAgF_z_FDR = p_to_z(pAgF_p_FDR, tail='two') * pAgF_sign

        _, pFgA_p_FDR, _, _ = multipletests(pFgA_p_vals, alpha=alpha,
                                            method='fdr_bh',
                                            is_sorted=False,
                                            returnsorted=False)
        pFgA_z_FDR = p_to_z(pFgA_p_FDR, tail='two') * pFgA_sign

        images = {
            'z_desc-consistency_level-voxel': pAgF_z_FDR,
            'z_desc-specificity_level-voxel': pFgA_z_FDR,
        }
        return images


@due.dcite(references.KDA1, description='Introduces the KDA algorithm.')
@due.dcite(references.KDA2, description='Also introduces the KDA algorithm.')
class KDA(CBMAEstimator):
    r"""
    Kernel density analysis.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.cbma.kernel.KDAKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    Kernel density analysis was first introduced in [1]_ and [2]_.

    Available correction methods: :func:`KDA.correct_fwe_montecarlo`

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
    _required_inputs = {
        'coordinates': ('coordinates', None),
    }

    def __init__(self, kernel_transformer=KDAKernel, **kwargs):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

        self.dataset = None
        self.results = None

    def _fit(self, dataset):
        """
        Perform KDA meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        ma_maps = self.kernel_transformer.transform(
            self.inputs_['coordinates'],
            masker=self.masker,
            return_type='array'
        )
        of_values = np.sum(ma_maps, axis=0)
        images = {
            'of': of_values
        }
        return images

    def _run_fwe_permutation(self, params):
        iter_ijk, iter_df = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        iter_ma_maps = self.kernel_transformer.transform(
            iter_df,
            masker=self.masker,
            return_type='array'
        )
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        return iter_max_value

    def correct_fwe_montecarlo(self, result, n_iters=10000, n_cores=-1):
        """
        Perform FWE correction using the max-value permutation method.
        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        n_iters : :obj:`int`, optional
            Number of iterations to build the vFWE null distribution.
            Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is -1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'logp_level-voxel'.

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = KDA()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        of_values = result.get_map('of', return_type='array')
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T

        if n_cores <= 0:
            n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            n_cores = mp.cpu_count()

        rand_idx = np.random.choice(
            null_ijk.shape[0],
            size=(self.inputs_['coordinates'].shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.inputs_['coordinates'].copy()

        # Define parameters
        iter_dfs = [iter_df] * n_iters
        params = zip(iter_ijks, iter_dfs)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params), total=n_iters))

        perm_max_values = perm_results

        # Voxel-level FWE
        vfwe_map = of_values.copy()
        for i_vox, val in enumerate(of_values):
            vfwe_map[i_vox] = -np.log(null_to_p(val, perm_max_values, 'upper'))
        vfwe_map[np.isinf(vfwe_map)] = -np.log(np.finfo(float).eps)

        images = {
            'logp_level-voxel': vfwe_map
        }
        return images
