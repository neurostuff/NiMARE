"""
CBMA methods from the activation likelihood estimation (ALE) family
"""
import logging
import multiprocessing as mp
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from nilearn.masking import apply_mask, unmask

from .kernel import ALEKernel, KernelTransformer
from ...results import MetaResult
from .base import CBMAEstimator
from ...due import due
from ... import references
from ...stats import null_to_p, p_to_z
from ...utils import round2

LGR = logging.getLogger(__name__)


@due.dcite(references.ALE1, description='Introduces ALE.')
@due.dcite(references.ALE2,
           description='Modifies ALE algorithm to eliminate within-experiment '
                       'effects and generate MA maps based on subject group '
                       'instead of experiment.')
@due.dcite(references.ALE3,
           description='Modifies ALE algorithm to allow FWE correction and to '
                       'more quickly and accurately generate the null '
                       'distribution for significance testing.')
class ALE(CBMAEstimator):
    r"""
    Activation likelihood estimation

    Parameters
    ----------
    kernel_estimator : :obj:`nimare.meta.cbma.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    **kwargs
        Keyword arguments. Arguments for the kernel_estimator can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    The ALE algorithm was originally developed in [1]_, then updated in [2]_
    and [3]_.

    References
    ----------
    .. [1] Turkeltaub, Peter E., et al. "Meta-analysis of the functional
        neuroanatomy of single-word reading: method and validation."
        Neuroimage 16.3 (2002): 765-780.
    .. [2] Turkeltaub, Peter E., et al. "Minimizing within‐experiment and
        within‐group effects in activation likelihood estimation
        meta‐analyses." Human brain mapping 33.1 (2012): 1-13.
    .. [3] Eickhoff, Simon B., et al. "Activation likelihood estimation
        meta-analysis revisited." Neuroimage 59.3 (2012): 2349-2361.
    """
    def __init__(self, kernel_estimator=ALEKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}
        for k in kwargs.keys():
            LGR.warning('Keyword argument "{0}" not recognized'.format(k))

        if not issubclass(kernel_estimator, KernelTransformer):
            raise ValueError('Argument "kernel_estimator" must be a '
                             'KernelTransformer')

        self.mask = None
        self.dataset = None

        self.kernel_estimator = kernel_estimator(**kernel_args)
        self.results = None
        self.null_distributions = {}

    def _fit(self, dataset):
        self.dataset = dataset
        self.mask = dataset.masker.mask_img

        ma_maps = self.kernel_estimator.transform(self.dataset, mask=self.mask, masked=False)
        ale_values = self._compute_ale(ma_maps)
        self._compute_null(ma_maps)
        p_values, z_values = self._ale_to_p(ale_values)

        images = {
            'ale': ale_values,
            'p': p_values,
            'z': z_values,
        }
        return images

    def _compute_ale(self, data):
        """
        Generate ALE-value array from MA values.
        Returns masked array of ALE values.
        """
        if isinstance(data, pd.DataFrame):
            ma_values = self.kernel_estimator.transform(data, mask=self.mask, masked=True)
        elif isinstance(data, list):
            ma_values = apply_mask(data, self.mask)
        elif isinstance(data, np.ndarray):
            ma_values = data.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(data)))

        ale_values = np.ones(ma_values.shape[1])
        for i in range(ma_values.shape[0]):
            ale_values *= (1. - ma_values[i, :])
        ale_values = 1 - ale_values
        return ale_values

    def _compute_null(self, ma_maps):
        """
        Compute uncorrected ALE-value null distribution from MA values.
        """
        if isinstance(ma_maps, list):
            ma_values = apply_mask(ma_maps, self.mask)
        elif isinstance(ma_maps, np.ndarray):
            ma_values = ma_maps.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(ma_maps)))

        # Determine histogram bins for ALE-value null distribution
        max_poss_ale = 1.
        for i in range(ma_values.shape[0]):
            max_poss_ale *= (1 - np.max(ma_values[i, :]))
        max_poss_ale = 1 - max_poss_ale

        self.null_distributions['histogram_bins'] = np.round(
            np.arange(0, max_poss_ale + 0.001, 0.0001), 4)

        ma_hists = np.zeros((ma_values.shape[0],
                             self.null_distributions['histogram_bins'].shape[0]))
        for i in range(ma_values.shape[0]):
            # Remember that histogram uses bin edges (not centers), so it
            # returns a 1xhist_bins-1 array
            n_zeros = len(np.where(ma_values[i, :] == 0)[0])
            reduced_ma_values = ma_values[i, ma_values[i, :] > 0]
            ma_hists[i, 0] = n_zeros
            ma_hists[i, 1:] = np.histogram(a=reduced_ma_values,
                                           bins=self.null_distributions['histogram_bins'],
                                           density=False)[0]

        # Inverse of step size in histBins (0.0001) = 10000
        step = 1 / np.mean(np.diff(self.null_distributions['histogram_bins']))

        # Null distribution to convert ALE to p-values.
        ale_hist = ma_hists[0, :]
        for i_exp in range(1, ma_hists.shape[0]):
            temp_hist = np.copy(ale_hist)
            ma_hist = np.copy(ma_hists[i_exp, :])

            # Find histogram bins with nonzero values for each histogram.
            ale_idx = np.where(temp_hist > 0)[0]
            exp_idx = np.where(ma_hist > 0)[0]

            # Normalize histograms.
            temp_hist /= np.sum(temp_hist)
            ma_hist /= np.sum(ma_hist)

            # Perform weighted convolution of histograms.
            ale_hist = np.zeros(self.null_distributions['histogram_bins'].shape[0])
            for j_idx in exp_idx:
                # Compute probabilities of observing each ALE value in histBins
                # by randomly combining maps represented by maHist and aleHist.
                # Add observed probabilities to corresponding bins in ALE
                # histogram.
                probabilities = ma_hist[j_idx] * temp_hist[ale_idx]
                ale_scores = 1 - (1 - self.null_distributions['histogram_bins'][j_idx]) *\
                    (1 - self.null_distributions['histogram_bins'][ale_idx])
                score_idx = np.floor(ale_scores * step).astype(int)
                np.add.at(ale_hist, score_idx, probabilities)

        # Convert aleHist into null distribution. The value in each bin
        # represents the probability of finding an ALE value (stored in
        # histBins) of that value or lower.
        null_distribution = ale_hist / np.sum(ale_hist)
        null_distribution = np.cumsum(null_distribution[::-1])[::-1]
        null_distribution /= np.max(null_distribution)
        self.null_distributions['histogram_weights'] = null_distribution

    def _ale_to_p(self, ale_values):
        """
        Compute p- and z-values.
        """
        step = 1 / np.mean(np.diff(self.null_distributions['histogram_bins']))

        # Determine p- and z-values from ALE values and null distribution.
        p_values = np.ones(ale_values.shape)

        idx = np.where(ale_values > 0)[0]
        ale_bins = round2(ale_values[idx] * step)
        p_values[idx] = self.null_distributions['histogram_weights'][ale_bins]
        z_values = p_to_z(p_values, tail='one')
        return p_values, z_values

    def _run_fwe_permutation(self, params):
        """
        Run a single random permutation of a dataset. Does the shared work
        between vFWE and cFWE.
        """
        iter_df, iter_ijk, conn, z_thresh = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        ale_values = self._compute_ale(iter_df)
        _, z_values = self._ale_to_p(ale_values)
        iter_max_value = np.max(ale_values)

        # Begin cluster-extent thresholding by thresholding matrix at cluster-
        # defining voxel-level threshold
        iter_z_map = unmask(z_values, self.mask)
        vthresh_iter_z_map = iter_z_map.get_data()
        vthresh_iter_z_map[vthresh_iter_z_map < z_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_z_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        if len(clust_sizes) == 1:
            iter_max_cluster = 0
        else:
            clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
            iter_max_cluster = np.max(clust_sizes)
        return iter_max_value, iter_max_cluster

    def _fwe_correct_permutation(self, result, voxel_thresh=0.001, n_cores=-1,
                                 n_iters=10000):
        """
        Perform FWE correction using the max-value permutation method.
        """
        z_values = result.get_map('z', return_type='array')
        ale_values = result.get_map('ale', return_type='array')
        null_ijk = np.vstack(np.where(self.mask.get_data())).T

        if n_cores <= 0:
            n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            n_cores = mp.cpu_count()

        # Begin cluster-extent thresholding by thresholding matrix at cluster-
        # defining voxel-level threshold
        z_thresh = p_to_z(voxel_thresh, tail='one')
        vthresh_z_values = z_values.copy()
        vthresh_z_values[np.abs(vthresh_z_values) < z_thresh] = 0

        # Find number of voxels per cluster (includes 0, which is empty space in
        # the matrix)
        conn = np.zeros((3, 3, 3), int)
        conn[:, :, 1] = 1
        conn[:, 1, :] = 1
        conn[1, :, :] = 1

        # Multiple comparisons correction
        iter_df = self.dataset.coordinates.copy()
        rand_idx = np.random.choice(null_ijk.shape[0],
                                    size=(iter_df.shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)

        # Define parameters
        iter_conns = [conn] * n_iters
        iter_dfs = [iter_df] * n_iters
        iter_thresh = [z_thresh] * n_iters
        params = zip(iter_dfs, iter_ijks, iter_conns, iter_thresh)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params),
                                         total=n_iters))

        (self.null_distributions['fwe_level-voxel'],
         self.null_distributions['fwe_level-cluster']) = zip(*perm_results)

        # Cluster-level FWE
        vthresh_z_map = unmask(vthresh_z_values, self.mask).get_data()
        labeled_matrix, n_clusters = ndimage.measurements.label(vthresh_z_map, conn)
        logp_cfwe_map = np.zeros(self.mask.shape)
        for i_clust in range(1, n_clusters + 1):
            clust_size = np.sum(labeled_matrix == i_clust)
            clust_idx = np.where(labeled_matrix == i_clust)
            logp_cfwe_map[clust_idx] = -np.log(null_to_p(
                clust_size, self.null_distributions['fwe_level-cluster'], 'upper'))
        logp_cfwe_map[np.isinf(logp_cfwe_map)] = -np.log(np.finfo(float).eps)
        logp_cfwe_map = apply_mask(nib.Nifti1Image(logp_cfwe_map, self.mask.affine),
                                   self.mask)

        # Voxel-level FWE
        p_fwe_values = np.zeros(ale_values.shape)
        for voxel in range(ale_values.shape[0]):
            p_fwe_values[voxel] = null_to_p(
                ale_values[voxel], self.null_distributions['fwe_level-voxel'],
                tail='upper')

        z_fwe_values = p_to_z(p_fwe_values, tail='one')

        # Write out unthresholded value images
        images = {
            'z_vthresh': vthresh_z_values,
            'p_level-voxel': p_fwe_values,
            'z_level-voxel': z_fwe_values,
            'logp_level-cluster': logp_cfwe_map,
        }
        return images


class ALESubtraction(CBMAEstimator):
    """
    ALE subtraction analysis.

    Parameters
    ----------
    n_iters : :obj:`int`, optional
        Default is 10000.

    Notes
    -----
    This method was originally developed in [1]_ and refined in [2]_.

    References
    ----------
    .. [1] Laird, Angela R., et al. "ALE meta‐analysis: Controlling the
        false discovery rate and performing statistical contrasts." Human
        brain mapping 25.1 (2005): 155-164.
        https://doi.org/10.1002/hbm.20136
    .. [2] Eickhoff, Simon B., et al. "Activation likelihood estimation
        meta-analysis revisited." Neuroimage 59.3 (2012): 2349-2361.
        https://doi.org/10.1016/j.neuroimage.2011.09.017
    """
    def __init__(self, n_iters=10000):
        self.n_iters = n_iters

    def fit(self, ale1, ale2, image1=None, image2=None, ma_maps1=None,
            ma_maps2=None):
        """
        Run a subtraction analysis comparing two groups of experiments from
        separate meta-analyses.

        Parameters
        ----------
        ale1, ale2 : :obj:`nimare.meta.cbma.ALE`
            Fitted ALE models for datasets to compare.
        image1, image2 : img_like or array_like
            Cluster-level FWE-corrected z-statistic maps associated with the
            respective models.
        ma_maps1 : (E x V) array_like or None, optional
            Experiments by voxels array of modeled activation
            values. If not provided, MA maps will be generated from dataset1.
        ma_maps2 : (E x V) array_like or None, optional
            Experiments by voxels array of modeled activation
            values. If not provided, MA maps will be generated from dataset2.

        Returns
        -------
        :obj:`nimare.base.base.MetaResult`
            Results of ALE subtraction analysis.
        """
        maps = self._fit(ale1, ale2, image1, image2, ma_maps1, ma_maps2)
        self.results = MetaResult(self, ale1.mask, maps)
        return self.results

    def _fit(self, ale1, ale2, image1=None, image2=None, ma_maps1=None,
             ma_maps2=None):
        assert np.array_equal(ale1.dataset.masker.mask_img.affine,
                              ale2.dataset.masker.mask_img.affine)
        self.mask = ale1.dataset.masker.mask_img

        if image1 is None:
            LGR.info('Performing subtraction analysis with cluster-level '
                     'FWE-corrected maps thresholded at p < 0.05.')
            image1 = ale1.results.get_map(
                'logp_level-cluster_corr-FWE_method-permutation',
                return_type='array') >= np.log(0.05)

        if image2 is None:
            image2 = ale2.results.get_map(
                'logp_level-cluster_corr-FWE_method-permutation',
                return_type='array') >= np.log(0.05)

        if not isinstance(image1, np.ndarray):
            image1 = apply_mask(image1, self.mask)
            image2 = apply_mask(image2, self.mask)
        grp1_voxel = image1 > 0
        grp2_voxel = image2 > 0

        if ma_maps1 is None:
            ma_maps1 = ale1.kernel_estimator.transform(ale1.dataset, masked=False)

        if ma_maps2 is None:
            ma_maps2 = ale2.kernel_estimator.transform(ale2.dataset, masked=False)

        n_grp1 = len(ma_maps1)
        ma_maps = ma_maps1 + ma_maps2

        id_idx = np.arange(len(ma_maps))

        # Get MA values for both samples.
        ma_arr = apply_mask(ma_maps, self.mask)

        # Get ALE values for first group.
        grp1_ma_arr = ma_arr[:n_grp1, :]
        grp1_ale_values = np.ones(ma_arr.shape[1])
        for i_exp in range(grp1_ma_arr.shape[0]):
            grp1_ale_values *= (1. - grp1_ma_arr[i_exp, :])
        grp1_ale_values = 1 - grp1_ale_values

        # Get ALE values for first group.
        grp2_ma_arr = ma_arr[n_grp1:, :]
        grp2_ale_values = np.ones(ma_arr.shape[1])
        for i_exp in range(grp2_ma_arr.shape[0]):
            grp2_ale_values *= (1. - grp2_ma_arr[i_exp, :])
        grp2_ale_values = 1 - grp2_ale_values

        # A > B contrast
        grp1_p_arr = np.ones(np.sum(grp1_voxel))
        grp1_z_map = np.zeros(image1.shape[0])
        grp1_z_map[:] = np.nan
        if np.sum(grp1_voxel) > 0:
            diff_ale_values = grp1_ale_values - grp2_ale_values
            diff_ale_values = diff_ale_values[grp1_voxel]

            red_ma_arr = ma_arr[:, grp1_voxel]
            iter_diff_values = np.zeros((self.n_iters, np.sum(grp1_voxel)))

            for i_iter in range(self.n_iters):
                np.random.shuffle(id_idx)
                iter_grp1_ale_values = np.ones(np.sum(grp1_voxel))
                for j_exp in id_idx[:n_grp1]:
                    iter_grp1_ale_values *= (1. - red_ma_arr[j_exp, :])
                iter_grp1_ale_values = 1 - iter_grp1_ale_values

                iter_grp2_ale_values = np.ones(np.sum(grp1_voxel))
                for j_exp in id_idx[n_grp1:]:
                    iter_grp2_ale_values *= (1. - red_ma_arr[j_exp, :])
                iter_grp2_ale_values = 1 - iter_grp2_ale_values

                iter_diff_values[i_iter, :] = iter_grp1_ale_values - iter_grp2_ale_values

            for voxel in range(np.sum(grp1_voxel)):
                # TODO: Check that upper is appropriate
                grp1_p_arr[voxel] = null_to_p(diff_ale_values[voxel],
                                              iter_diff_values[:, voxel],
                                              tail='upper')
            grp1_z_arr = p_to_z(grp1_p_arr, tail='one')
            # Unmask
            grp1_z_map = np.zeros(image1.shape[0])
            grp1_z_map[:] = np.nan
            grp1_z_map[grp1_voxel] = grp1_z_arr

        # B > A contrast
        grp2_p_arr = np.ones(np.sum(grp2_voxel))
        grp2_z_map = np.zeros(image2.shape[0])
        grp2_z_map[:] = np.nan
        if np.sum(grp2_voxel) > 0:
            # Get MA values for second sample only for voxels significant in
            # second sample's meta-analysis.
            diff_ale_values = grp2_ale_values - grp1_ale_values
            diff_ale_values = diff_ale_values[grp2_voxel]

            red_ma_arr = ma_arr[:, grp2_voxel]
            iter_diff_values = np.zeros((self.n_iters, np.sum(grp2_voxel)))

            for i_iter in range(self.n_iters):
                np.random.shuffle(id_idx)
                iter_grp1_ale_values = np.ones(np.sum(grp2_voxel))
                for j_exp in id_idx[:n_grp1]:
                    iter_grp1_ale_values *= (1. - red_ma_arr[j_exp, :])
                iter_grp1_ale_values = 1 - iter_grp1_ale_values

                iter_grp2_ale_values = np.ones(np.sum(grp2_voxel))
                for j_exp in id_idx[n_grp1:]:
                    iter_grp2_ale_values *= (1. - red_ma_arr[j_exp, :])
                iter_grp2_ale_values = 1 - iter_grp2_ale_values

                iter_diff_values[i_iter, :] = iter_grp2_ale_values - iter_grp1_ale_values

            for voxel in range(np.sum(grp2_voxel)):
                # TODO: Check that upper is appropriate
                grp2_p_arr[voxel] = null_to_p(diff_ale_values[voxel],
                                              iter_diff_values[:, voxel],
                                              tail='upper')
            grp2_z_arr = p_to_z(grp2_p_arr, tail='one')
            # Unmask
            grp2_z_map = np.zeros(grp2_voxel.shape[0])
            grp2_z_map[:] = np.nan
            grp2_z_map[grp2_voxel] = grp2_z_arr

        # Fill in output map
        diff_z_map = np.zeros(image1.shape[0])
        diff_z_map[grp2_voxel] = -1 * grp2_z_map[grp2_voxel]
        # could overwrite some values. not a problem.
        diff_z_map[grp1_voxel] = grp1_z_map[grp1_voxel]

        images = {'grp1-grp2_z': diff_z_map}
        self.results = MetaResult(self, self.mask, maps=images)


@due.dcite(references.SCALE,
           description='Introduces the specific co-activation likelihood '
                       'estimation (SCALE) algorithm.')
class SCALE(CBMAEstimator):
    r"""
    Specific coactivation likelihood estimation [1]_.

    Parameters
    ----------
    voxel_thresh : float, optional
        Uncorrected voxel-level threshold. Default: 0.001
    n_iters : int, optional
        Number of iterations for correction. Default: 10000
    n_cores : int, optional
        Number of processes to use for meta-analysis. If -1, use all
        available cores. Default: -1
    ijk : :obj:`str` or (N x 3) array_like
        Tab-delimited file of coordinates from database or numpy array with ijk
        coordinates. Voxels are rows and i, j, k (meaning matrix-space) values
        are the three columnns.
    kernel_estimator : :obj:`nimare.meta.cbma.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    **kwargs
        Keyword arguments. Arguments for the kernel_estimator can be assigned
        here, with the prefix '\kernel__' in the variable name.

    References
    ----------
    .. [1] Langner, Robert, et al. "Meta-analytic connectivity modeling
        revisited: controlling for activation base rates." NeuroImage 99
        (2014): 559-570. https://doi.org/10.1016/j.neuroimage.2014.06.007
    """
    def __init__(self, voxel_thresh=0.001, n_iters=10000, n_cores=-1, ijk=None,
                 kernel_estimator=ALEKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}

        if not issubclass(kernel_estimator, KernelTransformer):
            raise ValueError('Argument "kernel_estimator" must be a '
                             'KernelTransformer')

        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}
        for k in kwargs.keys():
            LGR.warning('Keyword argument "{0}" not recognized'.format(k))

        self.mask = None
        self.dataset = None

        self.kernel_estimator = kernel_estimator(**kernel_args)
        self.voxel_thresh = voxel_thresh
        self.ijk = ijk
        self.n_iters = n_iters
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

    def _fit(self, dataset):
        """
        Perform specific coactivation likelihood estimation meta-analysis
        on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.mask = dataset.masker.mask_img

        ma_maps = self.kernel_estimator.transform(self.dataset, masked=False)

        max_poss_ale = 1.
        for ma_map in ma_maps:
            max_poss_ale *= (1 - np.max(ma_map.get_data()))
        max_poss_ale = 1 - max_poss_ale

        hist_bins = np.round(np.arange(0, max_poss_ale + 0.001, 0.0001), 4)

        ale_values = self._compute_ale(df=None, ma_maps=ma_maps)

        iter_df = self.dataset.coordinates.copy()
        rand_idx = np.random.choice(self.ijk.shape[0],
                                    size=(iter_df.shape[0], self.n_iters))
        rand_ijk = self.ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)

        # Define parameters
        iter_dfs = [iter_df] * self.n_iters
        params = zip(iter_dfs, iter_ijks)

        if self.n_cores == 1:
            perm_scale_values = []
            for pp in tqdm(params, total=self.n_iters):
                perm_scale_values.append(self._run_permutation(pp))
        else:
            with mp.Pool(self.n_cores) as p:
                perm_scale_values = list(tqdm(p.imap(self._run_permutation, params), total=self.n_iters))

        perm_scale_values = np.stack(perm_scale_values)

        p_values, z_values = self._scale_to_p(ale_values, perm_scale_values,
                                              hist_bins)

        # Begin cluster-extent thresholding by thresholding matrix at cluster-
        # defining voxel-level threshold
        z_thresh = p_to_z(self.voxel_thresh, tail='one')
        vthresh_z_values = z_values.copy()
        vthresh_z_values[vthresh_z_values < z_thresh] = 0

        # Write out unthresholded value images
        images = {'ale': ale_values,
                  'p': p_values,
                  'z': z_values,
                  'z_vthresh': vthresh_z_values}
        return images

    def _compute_ale(self, df=None, ma_maps=None):
        """
        Generate ALE-value array and null distribution from list of contrasts.
        For ALEs on the original dataset, computes the null distribution.
        For permutation ALEs and all SCALEs, just computes ALE values.
        Returns masked array of ALE values and 1XnBins null distribution.
        """
        if df is not None:
            ma_maps = self.kernel_estimator.transform(df, self.mask, masked=True)
            ma_values = ma_maps
        else:
            assert ma_maps is not None
            ma_values = apply_mask(ma_maps, self.mask)

        ale_values = np.ones(ma_values.shape[1])
        for i in range(ma_values.shape[0]):
            ale_values *= (1. - ma_values[i, :])

        ale_values = 1 - ale_values
        return ale_values

    def _scale_to_p(self, ale_values, scale_values, hist_bins):
        """
        Compute p- and z-values.
        """
        step = 1 / np.mean(np.diff(hist_bins))

        scale_zeros = scale_values == 0
        n_zeros = np.sum(scale_zeros, axis=0)
        scale_values[scale_values == 0] = np.nan
        scale_hists = np.zeros(((len(hist_bins),) + n_zeros.shape))
        scale_hists[0, :] = n_zeros
        scale_hists[1:, :] = np.apply_along_axis(self._make_hist, 0,
                                                 scale_values,
                                                 hist_bins=hist_bins)

        # Convert voxel-wise histograms to voxel-wise null distributions.
        null_distribution = scale_hists / np.sum(scale_hists, axis=0)
        null_distribution = np.cumsum(null_distribution[::-1, :], axis=0)[::-1, :]
        null_distribution /= np.max(null_distribution, axis=0)

        # Get the hist_bins associated with each voxel's ale value, in order to
        # get the p-value from the associated bin in the null distribution.
        n_bins = len(hist_bins)
        ale_bins = round2(ale_values * step).astype(int)
        ale_bins[ale_bins > n_bins] = n_bins

        # Get p-values by getting the ale_bin-th value in null_distribution
        # per voxel.
        p_values = np.empty_like(ale_bins).astype(float)
        for i, (x, y) in enumerate(zip(null_distribution.transpose(), ale_bins)):
            p_values[i] = x[y]

        z_values = p_to_z(p_values, tail='one')
        return p_values, z_values

    def _make_hist(self, oned_arr, hist_bins):
        """
        Make a histogram from a 1d array and hist_bins. Meant to be applied
        along an axis to a 2d array.
        """
        hist_ = np.histogram(a=oned_arr, bins=hist_bins,
                             range=(np.min(hist_bins), np.max(hist_bins)),
                             density=False)[0]
        return hist_

    def _run_permutation(self, params):
        """
        Run a single random SCALE permutation of a dataset.
        """
        iter_df, iter_ijk = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        ale_values = self._compute_ale(iter_df)
        return ale_values
