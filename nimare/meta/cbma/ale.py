"""CBMA methods from the activation likelihood estimation (ALE) family."""
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ... import references
from ...due import due
from ...stats import null_to_p, nullhist_to_p
from ...transforms import p_to_z
from ...utils import use_memmap
from ..kernel import ALEKernel
from .base import CBMAEstimator, PairwiseCBMAEstimator

LGR = logging.getLogger(__name__)


@due.dcite(references.ALE1, description="Introduces ALE.")
@due.dcite(
    references.ALE2,
    description="Modifies ALE algorithm to eliminate within-experiment "
    "effects and generate MA maps based on subject group "
    "instead of experiment.",
)
@due.dcite(
    references.ALE3,
    description="Modifies ALE algorithm to allow FWE correction and to "
    "more quickly and accurately generate the null "
    "distribution for significance testing.",
)
class ALE(CBMAEstimator):
    r"""Activation likelihood estimation.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    null_method : {"approximate", "montecarlo"}, optional
        Method by which to determine uncorrected p-values.
    n_iters : int, optional
        Number of iterations to use to define the null distribution.
        This is only used if ``null_method=="montecarlo"``.
        Default is 10000.
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

    def __init__(
        self,
        kernel_transformer=ALEKernel,
        null_method="approximate",
        n_iters=10000,
        n_cores=1,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, ALEKernel) or kernel_transformer == ALEKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)
        self.null_method = null_method
        self.n_iters = n_iters
        self.n_cores = n_cores
        self.dataset = None
        self.results = None

    def _compute_summarystat(self, ma_values):
        stat_values = 1.0 - np.prod(1.0 - ma_values, axis=0)
        return stat_values

    def _determine_histogram_bins(self, ma_maps):
        """Determine histogram bins for null distribution methods.

        Parameters
        ----------
        ma_maps

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute: "histogram_bins".
        """
        if isinstance(ma_maps, list):
            ma_values = self.masker.transform(ma_maps)
        elif isinstance(ma_maps, np.ndarray):
            ma_values = ma_maps.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(ma_maps)))

        # Determine bins for null distribution histogram
        # Remember that numpy histogram bins are bin edges, not centers
        # Assuming values of 0, .001, .002, etc., bins are -.0005-.0005, .0005-.0015, etc.
        INV_STEP_SIZE = 100000
        step_size = 1 / INV_STEP_SIZE
        max_ma_values = np.max(ma_values, axis=1)
        # round up based on resolution
        max_ma_values = np.ceil(max_ma_values * INV_STEP_SIZE) / INV_STEP_SIZE
        max_poss_ale = self.compute_summarystat(max_ma_values)
        # create bin centers
        hist_bins = np.round(np.arange(0, max_poss_ale + (1.5 * step_size), step_size), 5)
        self.null_distributions_["histogram_bins"] = hist_bins

    def _compute_null_approximate(self, ma_maps):
        """Compute uncorrected ALE null distribution using approximate solution.

        Parameters
        ----------
        ma_maps : list of imgs or numpy.ndarray
            MA maps.

        Notes
        -----
        This method adds two entries to the null_distributions_ dict attribute:
        "histogram_bins" and "histogram_weights".
        """
        if isinstance(ma_maps, list):
            ma_values = self.masker.transform(ma_maps)
        elif isinstance(ma_maps, np.ndarray):
            ma_values = ma_maps.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(ma_maps)))

        assert "histogram_bins" in self.null_distributions_.keys()

        def just_histogram(*args, **kwargs):
            """Collect the first output (weights) from numpy histogram."""
            return np.histogram(*args, **kwargs)[0].astype(float)

        # Derive bin edges from histogram bin centers for numpy histogram function
        bin_centers = self.null_distributions_["histogram_bins"]
        step_size = bin_centers[1] - bin_centers[0]
        inv_step_size = 1 / step_size
        bin_edges = bin_centers - (step_size / 2)
        bin_edges = np.append(bin_centers, bin_centers[-1] + step_size)

        ma_hists = np.apply_along_axis(just_histogram, 1, ma_values, bins=bin_edges, density=False)

        # Normalize MA histograms to get probabilities
        ma_hists /= ma_hists.sum(1)[:, None]

        ale_hist = ma_hists[0, :].copy()

        for i_exp in range(1, ma_hists.shape[0]):

            exp_hist = ma_hists[i_exp, :]

            # Find histogram bins with nonzero values for each histogram.
            ale_idx = np.where(ale_hist > 0)[0]
            exp_idx = np.where(exp_hist > 0)[0]

            # Compute output MA values, ale_hist indices, and probabilities
            ale_scores = (
                1 - np.outer((1 - bin_centers[exp_idx]), (1 - bin_centers[ale_idx])).ravel()
            )
            score_idx = np.floor(ale_scores * inv_step_size).astype(int)
            probabilities = np.outer(exp_hist[exp_idx], ale_hist[ale_idx]).ravel()

            # Reset histogram and set probabilities.
            # Use at() instead of setting values directly (ale_hist[score_idx] = probabilities)
            # because there can be redundant values in score_idx.
            ale_hist = np.zeros(ale_hist.shape)
            np.add.at(ale_hist, score_idx, probabilities)

        self.null_distributions_["histweights_corr-none_method-approximate"] = ale_hist


class ALESubtraction(PairwiseCBMAEstimator):
    r"""
    ALE subtraction analysis.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is ALEKernel.
    n_iters : :obj:`int`, optional
        Default is 10000.
    low_memory : :obj:`bool`, optional
        If True, use memory-mapped files for large arrays to reduce memory usage.
        If False, do everything in memory.
        Default is False.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.
        Another optional argument is ``mask``.

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

    def __init__(self, kernel_transformer=ALEKernel, n_iters=10000, low_memory=False, **kwargs):
        if not (isinstance(kernel_transformer, ALEKernel) or kernel_transformer == ALEKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

        self.dataset1 = None
        self.dataset2 = None
        self.results = None
        self.n_iters = n_iters
        self.low_memory = low_memory

    @use_memmap(LGR, n_files=3)
    def _fit(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.masker = self.masker or dataset1.masker

        ma_maps1 = self._collect_ma_maps(
            maps_key="ma_maps1",
            coords_key="coordinates1",
            fname_idx=0,
        )
        ma_maps2 = self._collect_ma_maps(
            maps_key="ma_maps2",
            coords_key="coordinates2",
            fname_idx=1,
        )

        n_grp1 = ma_maps1.shape[0]
        ma_arr = np.vstack((ma_maps1, ma_maps2))
        id_idx = np.arange(ma_arr.shape[0])
        n_voxels = ma_arr.shape[1]

        # Get ALE values for first group.
        grp1_ma_arr = ma_arr[:n_grp1, :]
        grp1_ale_values = 1.0 - np.prod(1.0 - grp1_ma_arr, axis=0)

        # Get ALE values for second group.
        grp2_ma_arr = ma_arr[n_grp1:, :]
        grp2_ale_values = 1.0 - np.prod(1.0 - grp2_ma_arr, axis=0)

        diff_ale_values = grp1_ale_values - grp2_ale_values

        # Calculate null distribution for each voxel based on group-assignment randomization
        if self.low_memory:
            # Use a memmapped 4D array
            iter_diff_values = np.memmap(
                self.memmap_filenames[2],
                dtype=ma_arr.dtype,
                mode="w+",
                shape=(self.n_iters, n_voxels),
            )
        else:
            iter_diff_values = np.zeros((self.n_iters, n_voxels), dtype=ma_arr.dtype)

        for i_iter in range(self.n_iters):
            np.random.shuffle(id_idx)
            iter_grp1_ale_values = 1.0 - np.prod(1.0 - ma_arr[id_idx[:n_grp1], :], axis=0)
            iter_grp2_ale_values = 1.0 - np.prod(1.0 - ma_arr[id_idx[n_grp1:], :], axis=0)
            iter_diff_values[i_iter, :] = iter_grp1_ale_values - iter_grp2_ale_values
            del iter_grp1_ale_values, iter_grp2_ale_values
            if self.low_memory:
                # Write changes to disk
                iter_diff_values.flush()

        # Determine p-values based on voxel-wise null distributions
        # In cases with differently-sized groups,
        # the ALE-difference values will be biased and skewed,
        # but the null distributions will be too, so symmetric should be False.
        p_arr = np.ones(n_voxels)
        for voxel in range(n_voxels):
            p_arr[voxel] = null_to_p(
                diff_ale_values[voxel], iter_diff_values[:, voxel], tail="two", symmetric=False
            )
        diff_signs = np.sign(diff_ale_values - np.median(iter_diff_values, axis=0))

        del iter_diff_values

        z_arr = p_to_z(p_arr, tail="two") * diff_signs

        images = {
            "z_desc-group1MinusGroup2": z_arr,
            "p_desc-group1MinusGroup2": p_arr,
        }
        return images


@due.dcite(
    references.SCALE,
    description=("Introduces the specific co-activation likelihood estimation (SCALE) algorithm."),
)
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
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.kernel.ALEKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    References
    ----------
    * Langner, Robert, et al. "Meta-analytic connectivity modeling
      revisited: controlling for activation base rates." NeuroImage 99
      (2014): 559-570. https://doi.org/10.1016/j.neuroimage.2014.06.007
    """

    def __init__(
        self,
        voxel_thresh=0.001,
        n_iters=10000,
        n_cores=-1,
        ijk=None,
        kernel_transformer=ALEKernel,
        low_memory=False,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, ALEKernel) or kernel_transformer == ALEKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

        self.voxel_thresh = voxel_thresh
        self.ijk = ijk
        self.n_iters = n_iters
        self.n_cores = self._check_ncores(n_cores)
        self.low_memory = low_memory

    @use_memmap(LGR, n_files=2)
    def _fit(self, dataset):
        """Perform specific coactivation likelihood estimation meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        ma_values = self._collect_ma_maps(
            coords_key="coordinates",
            maps_key="ma_maps",
            fname_idx=0,
        )

        # Determine bins for null distribution histogram
        max_ma_values = np.max(ma_values, axis=1)
        max_poss_ale = self._compute_summarystat(max_ma_values)
        self.null_distributions_["histogram_bins"] = np.round(
            np.arange(0, max_poss_ale + 0.001, 0.0001), 4
        )

        stat_values = self._compute_summarystat(ma_values)

        iter_df = self.inputs_["coordinates"].copy()
        rand_idx = np.random.choice(self.ijk.shape[0], size=(iter_df.shape[0], self.n_iters))
        rand_ijk = self.ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)

        # Define parameters
        iter_dfs = [iter_df] * self.n_iters
        params = zip(iter_dfs, iter_ijks)

        if self.n_cores == 1:
            if self.low_memory:
                perm_scale_values = np.memmap(
                    self.memmap_filenames[1],
                    dtype=stat_values.dtype,
                    mode="w+",
                    shape=(self.n_iters, stat_values.shape[0]),
                )
            else:
                perm_scale_values = np.zeros(
                    (self.n_iters, stat_values.shape[0]), dtype=stat_values.dtype
                )
            for i_iter, pp in enumerate(tqdm(params, total=self.n_iters)):
                perm_scale_values[i_iter, :] = self._run_permutation(pp)
                if self.low_memory:
                    # Write changes to disk
                    perm_scale_values.flush()
        else:
            with mp.Pool(self.n_cores) as p:
                perm_scale_values = list(
                    tqdm(p.imap(self._run_permutation, params), total=self.n_iters)
                )
            perm_scale_values = np.stack(perm_scale_values)

        p_values, z_values = self._scale_to_p(stat_values, perm_scale_values)

        del perm_scale_values

        logp_values = -np.log10(p_values)
        logp_values[np.isinf(logp_values)] = -np.log10(np.finfo(float).eps)

        # Write out unthresholded value images
        images = {"stat": stat_values, "logp": logp_values, "z": z_values}
        return images

    def _compute_summarystat(self, data):
        """Generate ALE-value array and null distribution from list of contrasts.

        For ALEs on the original dataset, computes the null distribution.
        For permutation ALEs and all SCALEs, just computes ALE values.
        Returns masked array of ALE values and 1XnBins null distribution.
        """
        if isinstance(data, pd.DataFrame):
            ma_values = self.kernel_transformer.transform(
                data, masker=self.masker, return_type="array"
            )
        elif isinstance(data, list):
            ma_values = self.masker.transform(data)
        elif isinstance(data, np.ndarray):
            ma_values = data.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(data)))

        stat_values = 1.0 - np.prod(1.0 - ma_values, axis=0)
        return stat_values

    def _scale_to_p(self, stat_values, scale_values):
        """
        Compute p- and z-values.

        Parameters
        ----------
        stat_values : (V) array
            ALE values.
        scale_values : (I x V) array
            Permutation ALE values.

        Returns
        -------
        p_values : (V) array
        z_values : (V) array

        Notes
        -----
        This method also uses the "histogram_bins" element in the null_distributions_ attribute.
        """
        scale_zeros = scale_values == 0
        n_zeros = np.sum(scale_zeros, axis=0)
        scale_values[scale_values == 0] = np.nan
        scale_hists = np.zeros(
            ((len(self.null_distributions_["histogram_bins"]),) + n_zeros.shape)
        )
        scale_hists[0, :] = n_zeros
        scale_hists[1:, :] = np.apply_along_axis(self._make_hist, 0, scale_values)

        p_values = nullhist_to_p(
            stat_values, scale_hists, self.null_distributions_["histogram_bins"]
        )
        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values

    def _make_hist(self, oned_arr):
        """Make a histogram from a 1d array and histogram bins.

        Meant to be applied along an axis to a 2d array.
        """
        hist_ = np.histogram(
            a=oned_arr,
            bins=self.null_distributions_["histogram_bins"],
            range=(
                np.min(self.null_distributions_["histogram_bins"]),
                np.max(self.null_distributions_["histogram_bins"]),
            ),
            density=False,
        )[0]
        return hist_

    def _run_permutation(self, params):
        """Run a single random SCALE permutation of a dataset."""
        iter_df, iter_ijk = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[["i", "j", "k"]] = iter_ijk
        stat_values = self._compute_summarystat(iter_df)
        return stat_values
