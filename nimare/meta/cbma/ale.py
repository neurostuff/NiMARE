"""
Coordinate-based meta-analysis estimators
"""
from __future__ import print_function
import os
import copy
import warnings
from time import time
import multiprocessing as mp

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.special import ndtri
from nilearn.masking import apply_mask, unmask

from .base import CBMAEstimator
from .kernel import ALEKernel
from .utils import _make_hist, compute_ma
from ..base import MetaResult
from ...due import due, Doi
from ...utils import (save_nifti, read_nifti,
                      round2, thresh_str, get_resource_path, cite_mni152)

@due.dcite(Doi('10.1002/hbm.20718'),
           description='Introduces the ALE algorithm.')
@due.dcite(Doi('10.1002/hbm.21186'),
           description='Modifies ALE algorithm to eliminate within-experiment '
                       'effects and generate MA maps based on subject group '
                       'instead of experiment.')
@due.dcite(Doi('10.1016/j.neuroimage.2011.09.017'),
           description='Modifies ALE algorithm to allow FWE correction and to '
                       'more quickly and accurately generate the null '
                       'distribution for significance testing.')
class ALE(CBMAEstimator):
    """
    Activation likelihood estimation
    """
    def __init__(self, dataset, ids, kernel_estimator=ALEKernel, **kwargs):
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
        self.clust_thresh = None
        self.corr = None
        self.n_iters = None
        self.results = None

    def fit(self, voxel_thresh=0.001, q=0.05, corr='FWE', n_iters=10000, n_cores=4):
        """
        """
        null_ijk = np.vstack(np.where(self.dataset.mask.get_data())).T
        self.voxel_thresh = voxel_thresh
        self.clust_thresh = q
        self.corr = corr
        self.n_iters = n_iters

        sel_df = self.dataset.coordinates.loc[self.dataset.coordinates['id'].isin(self.ids)]

        max_poss_ale = 1.
        for ma_map in self.ma_maps:
            max_poss_ale *= (1 - np.max(ma_map.get_data()))

        max_poss_ale = 1 - max_poss_ale
        hist_bins = np.round(np.arange(0, max_poss_ale+0.001, 0.0001), 4)

        ale_values, null_distribution = self._compute_ale(df=None, hist_bins=hist_bins)
        p_values, z_values = self._ale_to_p(ale_values, hist_bins, null_distribution)

        # Begin cluster-extent thresholding by thresholding matrix at cluster-
        # defining voxel-level threshold
        z_thresh = ndtri(1 - self.voxel_thresh)
        vthresh_z_values = z_values.copy()
        vthresh_z_values[vthresh_z_values < z_thresh] = 0

        # Find number of voxels per cluster (includes 0, which is empty space in
        # the matrix)
        conn = np.zeros((3, 3, 3), int)
        conn[:, :, 1] = 1
        conn[:, 1, :] = 1
        conn[1, :, :] = 1

        ## Multiple comparisons correction
        iter_df = sel_df.copy()
        rand_idx = np.random.choice(null_ijk.shape[0],
                                    size=(sel_df.shape[0], n_iters))
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)

        # Define parameters
        iter_conns = [conn] * n_iters
        iter_dfs = [iter_df] * n_iters
        iter_null_dists = [null_distribution] * n_iters
        iter_hist_bins = [hist_bins] * n_iters
        params = zip(iter_dfs, iter_ijks, iter_null_dists, iter_hist_bins, iter_conns)
        pool = mp.Pool(n_cores)
        perm_results = pool.map(self._perm, params)
        pool.close()
        perm_max_values, perm_clust_sizes = zip(*perm_results)

        percentile = 100 * (1 - self.clust_thresh)

        ## Cluster-level FWE
        # Determine size of clusters in [1 - clust_thresh]th percentile (e.g. 95th)
        vthresh_z_map = unmask(vthresh_z_values, self.dataset.mask).get_data()
        labeled_matrix = ndimage.measurements.label(vthresh_z_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        clust_size_thresh = np.percentile(perm_clust_sizes, percentile)
        z_map = unmask(z_values, self.dataset.mask).get_data()
        cfwe_map = np.zeros(self.dataset.mask.shape)
        for i, clust_size in enumerate(clust_sizes):
            if clust_size >= clust_size_thresh and i > 0:
                clust_idx = np.where(labeled_matrix == i)
                cfwe_map[clust_idx] = z_map[clust_idx]
        cfwe_map = apply_mask(nib.Nifti1Image(cfwe_map, self.dataset.mask.affine),
                              self.dataset.mask)

        ## Voxel-level FWE
        # Determine ALE values in [1 - clust_thresh]th percentile (e.g. 95th)
        ale_value_thresh = np.percentile(perm_max_values, percentile)
        sig_idx = ale_values >= ale_value_thresh
        vfwe_map = z_values * sig_idx

        # Write out unthresholded value images
        images = {'ale': ale_values,
                  'p': p_values,
                  'z': z_values,
                  'vthresh': vthresh_z_values,
                  'vfwe': vfwe_map,
                  'cfwe': cfwe_map}
        self.results = MetaResult(mask=self.dataset.mask, **images)

    def _compute_ale(self, df=None, hist_bins=None):
        """
        Generate ALE-value array and null distribution from list of contrasts.
        For ALEs on the original dataset, computes the null distribution.
        For permutation ALEs and all SCALEs, just computes ALE values.
        Returns masked array of ALE values and 1XnBins null distribution.
        """
        if hist_bins is not None:
            ma_hists = np.zeros((len(self.ma_maps), hist_bins.shape[0]))
        else:
            ma_hists = None

        if df is not None:
            k_est = self.kernel_estimator(df, self.dataset.mask)
            ma_maps = k_est.transform(self.ids, **self.kernel_arguments)
        else:
            ma_maps = self.ma_maps

        ma_values = apply_mask(ma_maps, self.dataset.mask)
        ale_values = np.ones(ma_values.shape[1])
        for i in range(ma_values.shape[0]):
            # Remember that histogram uses bin edges (not centers), so it returns
            # a 1xhist_bins-1 array
            if hist_bins is not None:
                n_zeros = len(np.where(ma_values[i, :] == 0)[0])
                reduced_ma_values = ma_values[i, ma_values[i, :] > 0]
                ma_hists[i, 0] = n_zeros
                ma_hists[i, 1:] = np.histogram(a=reduced_ma_values, bins=hist_bins,
                                               density=False)[0]
            ale_values *= (1. - ma_values[i, :])

        ale_values = 1 - ale_values

        if hist_bins is not None:
            null_distribution = self._compute_null(hist_bins, ma_hists)
        else:
            null_distribution = None

        return ale_values, null_distribution

    def _compute_null(self, hist_bins, ma_hists):
        """
        Compute ALE null distribution.
        """
        # Inverse of step size in histBins (0.0001) = 10000
        step = 1 / np.mean(np.diff(hist_bins))

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
            ale_hist = np.zeros(hist_bins.shape[0])
            for j_idx in exp_idx:
                # Compute probabilities of observing each ALE value in histBins
                # by randomly combining maps represented by maHist and aleHist.
                # Add observed probabilities to corresponding bins in ALE
                # histogram.
                probabilities = ma_hist[j_idx] * temp_hist[ale_idx]
                ale_scores = 1 - (1 - hist_bins[j_idx]) * (1 - hist_bins[ale_idx])
                score_idx = np.floor(ale_scores * step).astype(int)
                np.add.at(ale_hist, score_idx, probabilities)

        # Convert aleHist into null distribution. The value in each bin
        # represents the probability of finding an ALE value (stored in
        # histBins) of that value or lower.
        last_used = np.where(ale_hist > 0)[0][-1]
        null_distribution = ale_hist[:last_used+1] / np.sum(ale_hist)
        null_distribution = np.cumsum(null_distribution[::-1])[::-1]
        null_distribution /= np.max(null_distribution)
        return null_distribution

    def _ale_to_p(self, ale_values, hist_bins, null_distribution):
        """
        Compute p- and z-values.
        """
        eps = np.spacing(1)  # pylint: disable=no-member
        step = 1 / np.mean(np.diff(hist_bins))

        # Determine p- and z-values from ALE values and null distribution.
        p_values = np.ones(ale_values.shape)

        idx = np.where(ale_values > 0)[0]
        ale_bins = round2(ale_values[idx] * step)
        p_values[idx] = null_distribution[ale_bins]

        z_values = ndtri(1 - p_values)
        z_values[p_values < eps] = ndtri(1 - eps) + (ale_values[p_values < eps] * 2)
        z_values[z_values < 0] = 0

        return p_values, z_values

    def _perm(self, params):
        """
        Run a single random permutation of a dataset. Does the shared work between
        vFWE and cFWE.
        """
        iter_df, iter_ijk, null_dist, hist_bins, conn = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[['i', 'j', 'k']] = iter_ijk
        ale_values, _ = self._compute_ale(iter_df, hist_bins=None)
        _, z_values = self._ale_to_p(ale_values, hist_bins, null_dist)
        iter_max_value = np.max(ale_values)

        # Begin cluster-extent thresholding by thresholding matrix at cluster-
        # defining voxel-level threshold
        z_thresh = ndtri(1 - self.voxel_thresh)

        iter_z_map = unmask(z_values, self.dataset.mask)
        vthresh_iter_z_map = iter_z_map.get_data()
        vthresh_iter_z_map[vthresh_iter_z_map < self.voxel_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_z_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        iter_max_cluster = np.max(clust_sizes)
        return iter_max_value, iter_max_cluster


@due.dcite(Doi('10.1016/j.neuroimage.2014.06.007'),
           description='Introduces the specific co-activation likelihood '
                       'estimation (SCALE) algorithm.')
class SCALE(CBMAEstimator):
    """
    Specific coactivation likelihood estimation
    """
    def __init__(self, dataset, database_file=None, n_iters=2500,
                 voxel_thresh=0.001, verbose=True, n_cores=4):
        self.dataset = dataset
        self.database_file = database_file
        self.n_iters = n_iters
        self.voxel_thresh = voxel_thresh
        self.verbose = verbose
        self.n_cores = n_cores

    def fit(self, sample):
        """
        Perform specific coactivation likelihood estimation[1]_ meta-analysis on dataset.

        Parameters
        ----------
        dataset : ale.Dataset
            Dataset to analyze.
        voxel_thresh : float
            Uncorrected voxel-level threshold.
        n_iters : int
            Number of iterations for correction. Default 2500
        verbose : bool
            If True, prints out status updates.
        prefix : str
            String prepended to default output filenames. May include path.
        database_file : str
            Tab-delimited file of coordinates from database. Voxels are rows and
            i, j, k (meaning matrix-space) values are the three columnns.

        Examples
        --------

        References
        ----------
        .. [1] Langner, R., Rottschy, C., Laird, A. R., Fox, P. T., &
               Eickhoff, S. B. (2014). Meta-analytic connectivity modeling
               revisited: controlling for activation base rates.
               NeuroImage, 99, 559-570.
        """
        database_coords = self.dataset.get_coords()
        self.scale(self.dataset, database_file=database_coords)


    def scale(self, dataset, n_cores=1, voxel_thresh=0.001, n_iters=2500, verbose=True,
              prefix='', database_file='grey_matter_ijk.txt',
              template_file='Grey10.nii.gz'):
        """
        Perform specific coactivation likelihood estimation[1]_ meta-analysis on dataset.

        Parameters
        ----------
        dataset : ale.Dataset
            Dataset to analyze.
        voxel_thresh : float
            Uncorrected voxel-level threshold.
        n_iters : int
            Number of iterations for correction. Default 2500
        verbose : bool
            If True, prints out status updates.
        prefix : str
            String prepended to default output filenames. May include path.
        database_file : str
            Tab-delimited file of coordinates from database. Voxels are rows and
            i, j, k (meaning matrix-space) values are the three columnns.

        Examples
        --------

        References
        ----------
        .. [1] Langner, R., Rottschy, C., Laird, A. R., Fox, P. T., &
               Eickhoff, S. B. (2014). Meta-analytic connectivity modeling
               revisited: controlling for activation base rates.
               NeuroImage, 99, 559-570.
        """
        name = dataset.name
        contrasts = dataset.contrasts

        # Cite MNI152 paper if default template is used
        if template_file == 'Grey10.nii.gz':
            cite_mni152()

        # Check paths for input files
        if not os.path.dirname(template_file):
            template_file = os.path.join(get_resource_path(), template_file)

        if not os.path.dirname(database_file):
            database_file = os.path.join(get_resource_path(), database_file)

        max_cores = mp.cpu_count()
        if not 1 <= n_cores <= max_cores:
            print('Desired number of cores ({0}) outside range of acceptable values. '
                  'Setting number of cores to max ({1}).'.format(n_cores, max_cores))
            n_cores = max_cores

        if prefix == '':
            prefix = name

        if os.path.basename(prefix) == '':
            prefix_sep = ''
        else:
            prefix_sep = '_'

        max_poss_ale = 1
        for con in contrasts:
            max_poss_ale *= (1 - np.max(con.kernel))

        max_poss_ale = 1 - max_poss_ale
        hist_bins = np.round(np.arange(0, max_poss_ale+0.001, 0.0001), 4)

        # Compute ALE values
        template_data, affine = read_nifti(template_file)
        dims = template_data.shape
        template_arr = template_data.flatten()
        prior = np.where(template_arr != 0)[0]
        shape = dims + np.array([30, 30, 30])

        database_ijk = np.loadtxt(database_file, dtype=int)
        if len(database_ijk.shape) != 2 or database_ijk.shape[-1] != 3:
            raise Exception('Database coordinates not in voxelsX3 shape.')
        elif np.any(database_ijk < 0):
            raise Exception('Negative value(s) detected. Database coordinates must '
                            'be in matrix space.')
        elif not np.all(np.equal(np.mod(database_ijk, 1), 0)):  # pylint: disable=no-member
            raise Exception('Float(s) detected. Database coordinates must all be '
                            'integers.')

        ale_values, _ = _compute_ale(contrasts, dims, shape, prior)

        n_foci = np.sum([con.ijk.shape[0] for con in contrasts])
        np.random.seed(0)  # pylint: disable=no-member
        rand_idx = np.random.choice(database_ijk.shape[0], size=(n_foci, n_iters))  # pylint: disable=no-member
        rand_ijk = database_ijk[rand_idx, :]

        # Define parameters
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        shapes = [shape] * n_iters
        exp_list = [contrasts] * n_iters
        priors = [prior] * n_iters
        dims_arrs = [dims] * n_iters
        iter_nums = range(1, n_iters+1)
        start = [time()] * n_iters

        params = zip(exp_list, iter_ijks, priors, dims_arrs, shapes, start, iter_nums)
        pool = mp.Pool(n_cores)
        scale_values = pool.map(self._perm_scale, params)
        pool.close()
        scale_values = np.stack(scale_values)

        p_values, z_values = self._scale_to_p(ale_values, scale_values, hist_bins)

        # Write out unthresholded value images
        out_images = {'ale': ale_values,
                      'p': p_values,
                      'z': z_values,}

        for (f, vals) in out_images.items():
            mat = np.zeros(dims).ravel()
            mat[prior] = vals
            mat = mat.reshape(dims)

            thresh_suffix = '{0}.nii.gz'.format(f)
            filename = prefix + prefix_sep + thresh_suffix
            save_nifti(mat, filename, affine)
            if verbose:
                print('File {0} saved.'.format(filename))

        # Perform voxel-level thresholding
        z_thresh = ndtri(1-voxel_thresh)
        sig_idx = np.where(z_values > z_thresh)[0]

        out_vector = np.zeros(prior.shape)
        out_vector[sig_idx] = z_values[sig_idx]

        out_matrix = np.zeros(dims).ravel()
        out_matrix[prior] = out_vector
        out_matrix = out_matrix.reshape(dims)

        thresh_suffix = 'thresh_{0}unc.nii.gz'.format(thresh_str(voxel_thresh))
        filename = prefix + prefix_sep + thresh_suffix
        save_nifti(out_matrix, filename, affine)
        if verbose:
            print('File {0} saved.'.format(filename))

    def _scale_to_p(self, ale_values, scale_values, hist_bins):
        """
        Compute p- and z-values.
        """
        eps = np.spacing(1)  # pylint: disable=no-member
        step = 1 / np.mean(np.diff(hist_bins))

        scale_zeros = scale_values == 0
        n_zeros = np.sum(scale_zeros, axis=0)
        scale_values[scale_values == 0] = np.nan
        scale_hists = np.zeros(((len(hist_bins),) + n_zeros.shape))
        scale_hists[0, :] = n_zeros
        scale_hists[1:, :] = np.apply_along_axis(_make_hist, 0, scale_values,
                                                 hist_bins=hist_bins)

        # Convert voxel-wise histograms to voxel-wise null distributions.
        null_distribution = scale_hists / np.sum(scale_hists, axis=0)
        null_distribution = np.cumsum(null_distribution[::-1, :], axis=0)[::-1, :]
        null_distribution /= np.max(null_distribution, axis=0)

        # Get the hist_bins associated with each voxel's ale value, in order to get
        # the p-value from the associated bin in the null distribution.
        n_bins = len(hist_bins)
        ale_bins = round2(ale_values * step).astype(int)
        ale_bins[ale_bins > n_bins] = n_bins

        # Get p-values by getting the ale_bin-th value in null_distribution
        # per voxel.
        p_values = np.empty_like(ale_bins).astype(float)
        for i, (x, y) in enumerate(zip(null_distribution.transpose(), ale_bins)):
            p_values[i] = x[y]

        z_values = ndtri(1-p_values)
        z_values[p_values < eps] = ndtri(1-eps) + (ale_values[p_values < eps] * 2)
        z_values[z_values < 0] = 0

        return p_values, z_values

    def _perm_scale(self, params):
        """
        Run a single random SCALE permutation of a dataset.
        """
        cons, ijk, prior, dims, shape, start, iter_num = params
        ijk = np.squeeze(ijk)

        # Assign random GM coordinates to contrasts
        start_idx = 0
        for con in cons:
            n_peaks = con.ijk.shape[0]
            end_idx = start_idx + n_peaks
            con.ijk = ijk[start_idx:end_idx]
            start_idx = end_idx
        ale_values, _ = _compute_ale(cons, dims, shape, prior)

        if iter_num % 250 == 0:
            elapsed = (time() - start) / 60.
            print('Iteration {0} completed after {1} mins.'.format(iter_num, elapsed))
        return ale_values
