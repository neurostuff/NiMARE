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

from .kernel import ALEKernel
from ...results import MetaResult
from ...base import CBMAEstimator
from ...due import due
from ... import references
from ...stats import null_to_p
from ...transforms import p_to_z
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
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key:
        coordinates. This is an edited version of the dataset's coordinates
        DataFrame.
    null_distributions_ : :obj:`dict` or :class:`numpy.ndarray`
        Null distributions for ALE and any multiple-comparisons correction
        methods. Entries are added to this attribute if and when the
        corresponding method is fit.

    Notes
    -----
    The ALE algorithm was originally developed in [1]_, then updated in [2]_
    and [3]_.

    Available correction methods: :func:`ALE.correct_fwe_montecarlo`

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
    _required_inputs = {
        'coordinates': ('coordinates', None),
    }

    def __init__(self, kernel_transformer=ALEKernel, **kwargs):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)
        self.dataset = None
        self.results = None

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        ma_maps = self.kernel_transformer.transform(
            self.inputs_['coordinates'],
            masker=self.masker,
            return_type='image'
        )
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
            ma_values = self.kernel_transformer.transform(
                data, masker=self.masker, return_type='array'
            )
        elif isinstance(data, list):
            ma_values = self.masker.transform(data)
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
            ma_values = self.masker.transform(ma_maps)
        elif isinstance(ma_maps, np.ndarray):
            ma_values = ma_maps.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(ma_maps)))

        # Determine histogram bins for ALE-value null distribution
        max_poss_ale = 1.
        for i in range(ma_values.shape[0]):
            max_poss_ale *= (1 - np.max(ma_values[i, :]))
        max_poss_ale = 1 - max_poss_ale

        self.null_distributions_['histogram_bins'] = np.round(
            np.arange(0, max_poss_ale + 0.001, 0.0001), 4)

        ma_hists = np.zeros((ma_values.shape[0],
                             self.null_distributions_['histogram_bins'].shape[0]))
        for i in range(ma_values.shape[0]):
            # Remember that histogram uses bin edges (not centers), so it
            # returns a 1xhist_bins-1 array
            n_zeros = len(np.where(ma_values[i, :] == 0)[0])
            reduced_ma_values = ma_values[i, ma_values[i, :] > 0]
            ma_hists[i, 0] = n_zeros
            ma_hists[i, 1:] = np.histogram(a=reduced_ma_values,
                                           bins=self.null_distributions_['histogram_bins'],
                                           density=False)[0]

        # Inverse of step size in histBins (0.0001) = 10000
        step = 1 / np.mean(np.diff(self.null_distributions_['histogram_bins']))

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
            ale_hist = np.zeros(self.null_distributions_['histogram_bins'].shape[0])
            for j_idx in exp_idx:
                # Compute probabilities of observing each ALE value in histBins
                # by randomly combining maps represented by maHist and aleHist.
                # Add observed probabilities to corresponding bins in ALE
                # histogram.
                probabilities = ma_hist[j_idx] * temp_hist[ale_idx]
                ale_scores = 1 - (1 - self.null_distributions_['histogram_bins'][j_idx]) *\
                    (1 - self.null_distributions_['histogram_bins'][ale_idx])
                score_idx = np.floor(ale_scores * step).astype(int)
                np.add.at(ale_hist, score_idx, probabilities)

        # Convert aleHist into null distribution. The value in each bin
        # represents the probability of finding an ALE value (stored in
        # histBins) of that value or lower.
        null_distribution = ale_hist / np.sum(ale_hist)
        null_distribution = np.cumsum(null_distribution[::-1])[::-1]
        null_distribution /= np.max(null_distribution)
        self.null_distributions_['histogram_weights'] = null_distribution

    def _ale_to_p(self, ale_values):
        """
        Compute p- and z-values.
        """
        step = 1 / np.mean(np.diff(self.null_distributions_['histogram_bins']))

        # Determine p- and z-values from ALE values and null distribution.
        p_values = np.ones(ale_values.shape)

        idx = np.where(ale_values > 0)[0]
        ale_bins = round2(ale_values[idx] * step)
        p_values[idx] = self.null_distributions_['histogram_weights'][ale_bins]
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
        iter_z_map = self.masker.inverse_transform(z_values)
        vthresh_iter_z_map = iter_z_map.get_fdata()
        vthresh_iter_z_map[vthresh_iter_z_map < z_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_z_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        if len(clust_sizes) == 1:
            iter_max_cluster = 0
        else:
            clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
            iter_max_cluster = np.max(clust_sizes)
        return iter_max_value, iter_max_cluster

    def correct_fwe_montecarlo(self, result, voxel_thresh=0.001, n_iters=10000, n_cores=-1):
        """
        Perform FWE correction using the max-value permutation method.
        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from an ALE meta-analysis.
        voxel_thresh : :obj:`float`, optional
            Cluster-defining uncorrected p-value threshold. Default is 0.001.
        n_iters : :obj:`int`, optional
            Number of iterations to build vFWE and cFWE null distributions.
            Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is -1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'z_vthresh', 'p_level-voxel', 'z_level-voxel', and
            'logp_level-cluster'.

        Notes
        -----
        This method also adds the following arrays to the CBMAEstimator's null
        distributions attribute (``null_distributions_``):
        'fwe_level-voxel_method-montecarlo' and
        'fwe_level-cluster_method-montecarlo'.

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = ALE()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', voxel_thresh=0.001,
                                     n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        z_values = result.get_map('z', return_type='array')
        ale_values = result.get_map('ale', return_type='array')
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T

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
        iter_df = self.inputs_['coordinates'].copy()
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

        (self.null_distributions_['fwe_level-voxel_method-montecarlo'],
         self.null_distributions_['fwe_level-cluster_method-montecarlo']) = zip(*perm_results)

        # Cluster-level FWE
        vthresh_z_map = self.masker.inverse_transform(vthresh_z_values).get_fdata()
        labeled_matrix, n_clusters = ndimage.measurements.label(vthresh_z_map, conn)
        p_cfwe_map = np.ones(self.masker.mask_img.shape)
        for i_clust in range(1, n_clusters + 1):
            clust_size = np.sum(labeled_matrix == i_clust)
            clust_idx = np.where(labeled_matrix == i_clust)
            p_cfwe_map[clust_idx] = null_to_p(
                clust_size,
                self.null_distributions_['fwe_level-cluster_method-montecarlo'],
                'upper'
            )
        p_cfwe_values = np.squeeze(self.masker.transform(
            nib.Nifti1Image(p_cfwe_map, self.masker.mask_img.affine)
        ))
        logp_cfwe_values = -np.log(p_cfwe_values)
        logp_cfwe_values[np.isinf(logp_cfwe_values)] = -np.log(np.finfo(float).eps)
        z_cfwe_values = p_to_z(p_cfwe_values, tail='one')

        # Voxel-level FWE
        p_vfwe_values = np.ones(ale_values.shape)
        for voxel in range(ale_values.shape[0]):
            p_vfwe_values[voxel] = null_to_p(
                ale_values[voxel], self.null_distributions_['fwe_level-voxel_method-montecarlo'],
                tail='upper')

        z_vfwe_values = p_to_z(p_vfwe_values, tail='one')
        logp_vfwe_values = -np.log(p_vfwe_values)
        logp_vfwe_values[np.isinf(logp_vfwe_values)] = -np.log(np.finfo(float).eps)

        # Write out unthresholded value images
        images = {
            'logp_level-voxel': logp_vfwe_values,
            'z_level-voxel': z_vfwe_values,
            'logp_level-cluster': logp_cfwe_values,
            'z_level-cluster': z_cfwe_values,
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

    Warning
    -------
    This implementation contains one key difference from the original version.
    In the original version, group 1 > group 2 difference values are only
    evaluated for voxels significant in the group 1 meta-analysis, and group 2
    > group 1 difference values are only evaluated for voxels significant in
    the group 2 meta-analysis. In NiMARE's implementation, the analysis is run
    in a two-sided manner for *all* voxels in the mask.

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
    _required_inputs = {
        'coordinates': ('coordinates', None),
    }

    def __init__(self, n_iters=10000):
        self.meta1 = None
        self.meta2 = None
        self.results = None
        self.n_iters = n_iters

    def fit(self, meta1, meta2):
        """
        Run a subtraction analysis comparing two groups of experiments from
        separate meta-analyses.

        Parameters
        ----------
        meta1/meta2 : :obj:`nimare.meta.cbma.ale.ALE`
            Fitted ALE Estimators for datasets to compare.
            These Estimators do not require multiple comparisons correction.

        Returns
        -------
        :obj:`nimare.results.MetaResult`
            Results of ALE subtraction analysis, with one map:
            'z_desc-group1MinusGroup2'.
        """
        maps = self._fit(meta1, meta2)
        self.results = MetaResult(self, meta1.dataset.masker, maps)
        return self.results

    def _fit(self, meta1, meta2):
        assert np.array_equal(meta1.dataset.masker.mask_img.affine,
                              meta2.dataset.masker.mask_img.affine)
        self.masker = meta1.dataset.masker

        ma_maps1 = meta1.kernel_transformer.transform(
            meta1.inputs_['coordinates'],
            masker=self.masker,
            return_type='image'
        )

        ma_maps2 = meta2.kernel_transformer.transform(
            meta2.inputs_['coordinates'],
            masker=self.masker,
            return_type='image'
        )

        n_grp1 = len(ma_maps1)
        ma_maps = ma_maps1 + ma_maps2

        id_idx = np.arange(len(ma_maps))

        # Get MA values for both samples.
        ma_arr = self.masker.transform(ma_maps)
        n_voxels = ma_arr.shape[1]

        # Get ALE values for first group.
        grp1_ma_arr = ma_arr[:n_grp1, :]
        grp1_ale_values = np.ones(n_voxels)
        for i_exp in range(grp1_ma_arr.shape[0]):
            grp1_ale_values *= (1. - grp1_ma_arr[i_exp, :])
        grp1_ale_values = 1 - grp1_ale_values

        # Get ALE values for second group.
        grp2_ma_arr = ma_arr[n_grp1:, :]
        grp2_ale_values = np.ones(n_voxels)
        for i_exp in range(grp2_ma_arr.shape[0]):
            grp2_ale_values *= (1. - grp2_ma_arr[i_exp, :])
        grp2_ale_values = 1 - grp2_ale_values

        p_arr = np.ones(n_voxels)

        diff_ale_values = grp1_ale_values - grp2_ale_values

        iter_diff_values = np.zeros((self.n_iters, n_voxels))

        for i_iter in range(self.n_iters):
            np.random.shuffle(id_idx)
            iter_grp1_ale_values = np.ones(n_voxels)
            for j_exp in id_idx[:n_grp1]:
                iter_grp1_ale_values *= (1. - ma_arr[j_exp, :])
            iter_grp1_ale_values = 1 - iter_grp1_ale_values

            iter_grp2_ale_values = np.ones(n_voxels)
            for j_exp in id_idx[n_grp1:]:
                iter_grp2_ale_values *= (1. - ma_arr[j_exp, :])
            iter_grp2_ale_values = 1 - iter_grp2_ale_values

            iter_diff_values[i_iter, :] = iter_grp1_ale_values - iter_grp2_ale_values

        for voxel in range(n_voxels):
            p_arr[voxel] = null_to_p(diff_ale_values[voxel],
                                     iter_diff_values[:, voxel],
                                     tail='two')
        diff_signs = np.sign(diff_ale_values - np.median(iter_diff_values, axis=0))
        z_arr = p_to_z(p_arr, tail='two') * diff_signs

        images = {
            'z_desc-group1MinusGroup2': z_arr
        }
        return images


@due.dcite(references.SCALE,
           description='Introduces the specific co-activation likelihood '
                       'estimation (SCALE) algorithm.')
class SCALE(CBMAEstimator):
    r"""
    Specific coactivation likelihood estimation.

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
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.cbma.kernel.ALEKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    References
    ----------
    * Langner, Robert, et al. "Meta-analytic connectivity modeling
      revisited: controlling for activation base rates." NeuroImage 99
      (2014): 559-570. https://doi.org/10.1016/j.neuroimage.2014.06.007
    """
    _required_inputs = {
        'coordinates': ('coordinates', None),
    }

    def __init__(self, voxel_thresh=0.001, n_iters=10000, n_cores=-1, ijk=None,
                 kernel_transformer=ALEKernel, **kwargs):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

        self.dataset = None
        self.results = None
        self.voxel_thresh = voxel_thresh
        self.ijk = ijk
        self.n_iters = n_iters

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
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        ma_maps = self.kernel_transformer.transform(
            self.inputs_['coordinates'],
            masker=self.masker,
            return_type='image'
        )

        max_poss_ale = 1.
        for ma_map in ma_maps:
            max_poss_ale *= (1 - np.max(ma_map.get_fdata()))
        max_poss_ale = 1 - max_poss_ale

        self.null_distributions_['histogram_bins'] = np.round(
            np.arange(0, max_poss_ale + 0.001, 0.0001),
            4
        )

        ale_values = self._compute_ale(ma_maps)

        iter_df = self.inputs_['coordinates'].copy()
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
                perm_scale_values = list(tqdm(p.imap(self._run_permutation, params),
                                              total=self.n_iters))

        perm_scale_values = np.stack(perm_scale_values)

        p_values, z_values = self._scale_to_p(ale_values, perm_scale_values)
        logp_values = -np.log(p_values)
        logp_values[np.isinf(logp_values)] = -np.log(np.finfo(float).eps)

        # Write out unthresholded value images
        images = {
            'ale': ale_values,
            'logp': logp_values,
            'z': z_values,
        }
        return images

    def _compute_ale(self, data):
        """
        Generate ALE-value array and null distribution from list of contrasts.
        For ALEs on the original dataset, computes the null distribution.
        For permutation ALEs and all SCALEs, just computes ALE values.
        Returns masked array of ALE values and 1XnBins null distribution.
        """
        if isinstance(data, pd.DataFrame):
            ma_values = self.kernel_transformer.transform(
                data, masker=self.masker, return_type='array'
            )
        elif isinstance(data, list):
            ma_values = self.masker.transform(data)
        elif isinstance(data, np.ndarray):
            ma_values = data.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(data)))

        ale_values = np.ones(ma_values.shape[1])
        for i in range(ma_values.shape[0]):
            ale_values *= (1. - ma_values[i, :])

        ale_values = 1 - ale_values
        return ale_values

    def _scale_to_p(self, ale_values, scale_values):
        """
        Compute p- and z-values.
        """
        step = 1 / np.mean(np.diff(self.null_distributions_['histogram_bins']))

        scale_zeros = scale_values == 0
        n_zeros = np.sum(scale_zeros, axis=0)
        scale_values[scale_values == 0] = np.nan
        scale_hists = np.zeros(
            ((len(self.null_distributions_['histogram_bins']),) + n_zeros.shape)
        )
        scale_hists[0, :] = n_zeros
        scale_hists[1:, :] = np.apply_along_axis(
            self._make_hist,
            0,
            scale_values
        )

        # Convert voxel-wise histograms to voxel-wise null distributions.
        null_distribution = scale_hists / np.sum(scale_hists, axis=0)
        null_distribution = np.cumsum(null_distribution[::-1, :], axis=0)[::-1, :]
        null_distribution /= np.max(null_distribution, axis=0)

        # Get the hist bins associated with each voxel's ale value, in order to
        # get the p-value from the associated bin in the null distribution.
        n_bins = len(self.null_distributions_['histogram_bins'])
        ale_bins = round2(ale_values * step).astype(int)
        ale_bins[ale_bins > n_bins] = n_bins

        # Get p-values by getting the ale_bin-th value in null_distribution
        # per voxel.
        p_values = np.empty_like(ale_bins).astype(float)
        for i, (x, y) in enumerate(zip(null_distribution.transpose(), ale_bins)):
            p_values[i] = x[y]

        z_values = p_to_z(p_values, tail='one')
        return p_values, z_values

    def _make_hist(self, oned_arr):
        """
        Make a histogram from a 1d array and histogram bins. Meant to be applied
        along an axis to a 2d array.
        """
        hist_ = np.histogram(
            a=oned_arr,
            bins=self.null_distributions_['histogram_bins'],
            range=(np.min(self.null_distributions_['histogram_bins']),
                   np.max(self.null_distributions_['histogram_bins'])),
            density=False
        )[0]
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
