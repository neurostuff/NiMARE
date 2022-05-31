"""CBMA methods from the activation likelihood estimation (ALE) family."""
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from nimare import references
from nimare.due import due
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.meta.kernel import ALEKernel
from nimare.stats import null_to_p, nullhist_to_p
from nimare.transforms import p_to_z
from nimare.utils import _check_ncores, tqdm_joblib, use_memmap

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
    """Activation likelihood estimation.

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is ALEKernel.
    null_method : {"approximate", "montecarlo"}, optional
        Method by which to determine uncorrected p-values. The available options are

        ======================= =================================================================
        "approximate" (default) Build a histogram of summary-statistic values and their
                                expected frequencies under the assumption of random spatial
                                associated between studies, via a weighted convolution.

                                This method is much faster, but slightly less accurate.
        "montecarlo"            Perform a large number of permutations, in which the coordinates
                                in the studies are randomly drawn from the Estimator's brain mask
                                and the full set of resulting summary-statistic values are
                                incorporated into a null distribution (stored as a histogram for
                                memory reasons).

                                This method is must slower, and is only slightly more accurate.
        ======================= =================================================================

    n_iters : :obj:`int`, optional
        Number of iterations to use to define the null distribution.
        This is only used if ``null_method=="montecarlo"``.
        Default is 10000.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        This is only used if ``null_method=="montecarlo"``.
        If <=0, defaults to using all available cores.
        Default is 1.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix ``kernel__`` in the variable name.
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key: coordinates.
        This is an edited version of the dataset's coordinates DataFrame.
    null_distributions_ : :obj:`dict` of :class:`numpy.ndarray`
        Null distributions for the uncorrected summary-statistic-to-p-value conversion and any
        multiple-comparisons correction methods.
        Entries are added to this attribute if and when the corresponding method is applied.

        If ``null_method == "approximate"``:

            -   ``histogram_bins``: Array of bin centers for the null distribution histogram,
                ranging from zero to the maximum possible summary statistic value for the Dataset.
            -   ``histweights_corr-none_method-approximate``: Array of weights for the null
                distribution histogram, with one value for each bin in ``histogram_bins``.

        If ``null_method == "montecarlo"``:

            -   ``histogram_bins``: Array of bin centers for the null distribution histogram,
                ranging from zero to the maximum possible summary statistic value for the Dataset.
            -   ``histweights_corr-none_method-montecarlo``: Array of weights for the null
                distribution histogram, with one value for each bin in ``histogram_bins``.
                These values are derived from the full set of summary statistics from each
                iteration of the Monte Carlo procedure.
            -   ``histweights_level-voxel_corr-fwe_method-montecarlo``: Array of weights for the
                voxel-level FWE-correction null distribution, with one value for each bin in
                ``histogram_bins``. These values are derived from the maximum summary statistic
                from each iteration of the Monte Carlo procedure.

        If :meth:`correct_fwe_montecarlo` is applied:

            -   ``values_level-voxel_corr-fwe_method-montecarlo``: The maximum summary statistic
                value from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-size_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                size from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-mass_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                mass from each Monte Carlo iteration. An array of shape (n_iters,).

    Notes
    -----
    The ALE algorithm was originally developed in :footcite:t:`turkeltaub2002meta`,
    then updated in :footcite:t:`turkeltaub2012minimizing` and
    :footcite:t:`eickhoff2012activation`.

    The ALE algorithm is also implemented as part of the GingerALE app provided by the BrainMap
    organization (https://www.brainmap.org/ale/).

    Available correction methods: :meth:`~nimare.meta.cbma.ale.ALE.correct_fwe_montecarlo`.

    References
    ----------
    .. footbibliography::
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
        self.n_cores = _check_ncores(n_cores)
        self.dataset = None

    def _compute_summarystat_est(self, ma_values):
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
            raise ValueError(f"Unsupported data type '{type(ma_maps)}'")

        # Determine bins for null distribution histogram
        # Remember that numpy histogram bins are bin edges, not centers
        # Assuming values of 0, .001, .002, etc., bins are -.0005-.0005, .0005-.0015, etc.
        INV_STEP_SIZE = 100000
        step_size = 1 / INV_STEP_SIZE
        max_ma_values = np.max(ma_values, axis=1)
        # round up based on resolution
        max_ma_values = np.ceil(max_ma_values * INV_STEP_SIZE) / INV_STEP_SIZE
        max_poss_ale = self._compute_summarystat(max_ma_values)
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

            - "histogram_bins"
            - "histweights_corr-none_method-approximate"
        """
        if isinstance(ma_maps, list):
            ma_values = self.masker.transform(ma_maps)
        elif isinstance(ma_maps, np.ndarray):
            ma_values = ma_maps.copy()
        else:
            raise ValueError(f"Unsupported data type '{type(ma_maps)}'")

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
    """ALE subtraction analysis.

    .. versionchanged:: 0.0.12

        - Use memmapped array for null distribution and remove ``memory_limit`` parameter.
        - Support parallelization and add progress bar.
        - Add ALE-difference (stat) and -log10(p) (logp) maps to results.

    .. versionchanged:: 0.0.8

        * [FIX] Assume non-symmetric null distribution.

    .. versionchanged:: 0.0.7

        * [FIX] Assume a zero-centered and symmetric null distribution.

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is ALEKernel.
    n_iters : :obj:`int`, optional
        Default is 10000.
    n_cores : :obj:`int`, optional
        Number of processes to use for meta-analysis. If -1, use all available cores.
        Default is 1.

        .. versionadded:: 0.0.12
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix ``kernel__`` in the variable name. Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key: coordinates.
        This is an edited version of the dataset's coordinates DataFrame.

    Notes
    -----
    This method was originally developed in :footcite:t:`laird2005ale` and refined in
    :footcite:t:`eickhoff2012activation`.

    The ALE subtraction algorithm is also implemented as part of the GingerALE app provided by the
    BrainMap organization (https://www.brainmap.org/ale/).

    The voxel-wise null distributions used by this Estimator are very large, so they are not
    retained as Estimator attributes.

    Warnings
    --------
    This implementation contains one key difference from the original version.

    In the original version, group 1 > group 2 difference values are only evaluated for voxels
    significant in the group 1 meta-analysis, and group 2 > group 1 difference values are only
    evaluated for voxels significant in the group 2 meta-analysis.

    In NiMARE's implementation, the analysis is run in a two-sided manner for *all* voxels in the
    mask.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, kernel_transformer=ALEKernel, n_iters=10000, n_cores=1, **kwargs):
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
        self.n_iters = n_iters
        self.n_cores = _check_ncores(n_cores)
        # memory_limit needs to exist to trigger use_memmap decorator, but it will also be used if
        # a Dataset with pre-generated MA maps is provided.
        self.memory_limit = "100mb"

    @use_memmap(LGR, n_files=3)
    def _fit(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.masker = self.masker or dataset1.masker

        ma_maps1 = self._collect_ma_maps(
            maps_key="ma_maps1",
            coords_key="coordinates1",
        )
        ma_maps2 = self._collect_ma_maps(
            maps_key="ma_maps2",
            coords_key="coordinates2",
        )

        n_grp1, n_voxels = ma_maps1.shape

        # Get ALE values for the two groups and difference scores
        grp1_ale_values = self._compute_summarystat_est(ma_maps1)
        grp2_ale_values = self._compute_summarystat_est(ma_maps2)
        diff_ale_values = grp1_ale_values - grp2_ale_values
        del grp1_ale_values, grp2_ale_values

        # Combine the MA maps into a single array to draw from for null distribution
        ma_arr = np.vstack((ma_maps1, ma_maps2))

        del ma_maps1, ma_maps2

        # Calculate null distribution for each voxel based on group-assignment randomization
        # Use a memmapped 2D array
        iter_diff_values = np.memmap(
            self.memmap_filenames[2],
            dtype=ma_arr.dtype,
            mode="w+",
            shape=(self.n_iters, n_voxels),
        )

        with tqdm_joblib(tqdm(total=self.n_iters)):
            Parallel(n_jobs=self.n_cores)(
                delayed(self._run_permutation)(i_iter, n_grp1, ma_arr, iter_diff_values)
                for i_iter in range(self.n_iters)
            )

        # Determine p-values based on voxel-wise null distributions
        # I know that joblib probably preserves order of outputs, but I'm paranoid, so we track
        # the iteration as well and sort the resulting p-value array based on that.
        with tqdm_joblib(tqdm(total=n_voxels)):
            p_values, voxel_idx = zip(
                *Parallel(n_jobs=self.n_cores)(
                    delayed(self._alediff_to_p_voxel)(
                        i_voxel,
                        diff_ale_values[i_voxel],
                        iter_diff_values[:, i_voxel],
                    )
                    for i_voxel in range(n_voxels)
                )
            )
        # Convert to an array and sort the p-values array based on the voxel index.
        p_values = np.array(p_values)[np.array(voxel_idx)]

        diff_signs = np.sign(diff_ale_values - np.median(iter_diff_values, axis=0))

        if isinstance(iter_diff_values, np.memmap):
            LGR.debug(f"Closing memmap at {iter_diff_values.filename}")
            iter_diff_values._mmap.close()

        del iter_diff_values

        z_arr = p_to_z(p_values, tail="two") * diff_signs
        logp_arr = -np.log10(p_values)

        images = {
            "stat_desc-group1MinusGroup2": diff_ale_values,
            "p_desc-group1MinusGroup2": p_values,
            "z_desc-group1MinusGroup2": z_arr,
            "logp_desc-group1MinusGroup2": logp_arr,
        }
        return images

    def _compute_summarystat_est(self, ma_values):
        stat_values = 1.0 - np.prod(1.0 - ma_values, axis=0)
        return stat_values

    def _run_permutation(self, i_iter, n_grp1, ma_arr, iter_diff_values):
        """Run a single permutations of the ALESubtraction null distribution procedure.

        This method writes out a single row to the memmapped array in ``iter_diff_values``.

        Parameters
        ----------
        i_iter : :obj:`int`
            The iteration number.
        n_grp1 : :obj:`int`
            The number of experiments in the first group (of two, total).
        ma_arr : :obj:`numpy.ndarray` of shape (E, V)
            The voxel-wise (V) modeled activation values for all experiments E.
        iter_diff_values : :obj:`numpy.memmap` of shape (I, V)
            The null distribution of ALE-difference scores, with one row per iteration (I)
            and one column per voxel (V).
        """
        gen = np.random.default_rng(seed=i_iter)
        id_idx = np.arange(ma_arr.shape[0])
        gen.shuffle(id_idx)
        iter_grp1_ale_values = 1.0 - np.prod(1.0 - ma_arr[id_idx[:n_grp1], :], axis=0)
        iter_grp2_ale_values = 1.0 - np.prod(1.0 - ma_arr[id_idx[n_grp1:], :], axis=0)
        iter_diff_values[i_iter, :] = iter_grp1_ale_values - iter_grp2_ale_values

    def _alediff_to_p_voxel(self, i_voxel, stat_value, voxel_null):
        """Compute one voxel's p-value from its specific null distribution.

        Notes
        -----
        In cases with differently-sized groups, the ALE-difference values will be biased and
        skewed, but the null distributions will be too, so symmetric should be False.
        """
        p_value = null_to_p(stat_value, voxel_null, tail="two", symmetric=False)
        return p_value, i_voxel

    def correct_fwe_montecarlo(self):
        """Perform Monte Carlo-based FWE correction.

        Warnings
        --------
        This method is not implemented for this class.
        """
        raise NotImplementedError(
            f"The {type(self)} class does not support `correct_fwe_montecarlo`."
        )


@due.dcite(
    references.SCALE,
    description=("Introduces the specific co-activation likelihood estimation (SCALE) algorithm."),
)
class SCALE(CBMAEstimator):
    r"""Specific coactivation likelihood estimation.

    This method was originally introduced in :footcite:t:`langner2014meta`.

    .. versionchanged:: 0.0.12

        - Remove unused parameters ``voxel_thresh`` and ``memory_limit``.
        - Use memmapped array for null distribution.

    .. versionchanged:: 0.0.10

        Replace ``ijk`` with ``xyz``. This should be easier for users to collect.

    Parameters
    ----------
    xyz : (N x 3) :obj:`numpy.ndarray`
        Numpy array with XYZ coordinates.
        Voxels are rows and x, y, z (meaning coordinates) values are the three columnns.

        .. versionchanged:: 0.0.12

            This parameter was previously incorrectly labeled as "optional" and indicated that
            it supports tab-delimited files, which it does not (yet).

    n_iters : int, optional
        Number of iterations for statistical inference. Default: 10000
    n_cores : int, optional
        Number of processes to use for meta-analysis. If -1, use all available cores.
        Default: 1
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`~nimare.meta.kernel.ALEKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix '\kernel__' in the variable name.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key: coordinates.
        This is an edited version of the dataset's coordinates DataFrame.
    null_distributions_ : :obj:`dict` of :class:`numpy.ndarray`
        Null distribution information.
        Entries are added to this attribute if and when the corresponding method is applied.

        .. important::
            The voxel-wise null distributions used by this Estimator are very large, so they are
            not retained as Estimator attributes.

        If :meth:`fit` is applied:

            -   ``histogram_bins``: Array of bin centers for the null distribution histogram,
                ranging from zero to the maximum possible summary statistic value for the Dataset.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        xyz,
        n_iters=10000,
        n_cores=1,
        kernel_transformer=ALEKernel,
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

        if not isinstance(xyz, np.ndarray):
            raise TypeError(f"Parameter 'xyz' must be a numpy.ndarray, not a {type(xyz)}")
        elif xyz.ndim != 2:
            raise ValueError(f"Parameter 'xyz' must be a 2D array, but has {xyz.ndim} dimensions")
        elif xyz.shape[1] != 3:
            raise ValueError(f"Parameter 'xyz' must have 3 columns, but has shape {xyz.shape}")

        self.xyz = xyz
        self.n_iters = n_iters
        self.n_cores = _check_ncores(n_cores)
        # memory_limit needs to exist to trigger use_memmap decorator, but it will also be used if
        # a Dataset with pre-generated MA maps is provided.
        self.memory_limit = "100mb"

    @use_memmap(LGR, n_files=2)
    def _fit(self, dataset):
        """Perform specific coactivation likelihood estimation meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        ma_values = self._collect_ma_maps(
            coords_key="coordinates",
            maps_key="ma_maps",
        )

        # Determine bins for null distribution histogram
        max_ma_values = np.max(ma_values, axis=1)
        max_poss_ale = self._compute_summarystat_est(max_ma_values)
        self.null_distributions_["histogram_bins"] = np.round(
            np.arange(0, max_poss_ale + 0.001, 0.0001), 4
        )

        stat_values = self._compute_summarystat_est(ma_values)

        iter_df = self.inputs_["coordinates"].copy()
        rand_idx = np.random.choice(self.xyz.shape[0], size=(iter_df.shape[0], self.n_iters))
        rand_xyz = self.xyz[rand_idx, :]
        iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)

        perm_scale_values = np.memmap(
            self.memmap_filenames[1],
            dtype=stat_values.dtype,
            mode="w+",
            shape=(self.n_iters, stat_values.shape[0]),
        )
        with tqdm_joblib(tqdm(total=self.n_iters)):
            Parallel(n_jobs=self.n_cores)(
                delayed(self._run_permutation)(
                    i_iter, iter_xyzs[i_iter], iter_df, perm_scale_values
                )
                for i_iter in range(self.n_iters)
            )

        p_values, z_values = self._scale_to_p(stat_values, perm_scale_values)

        if isinstance(perm_scale_values, np.memmap):
            LGR.debug(f"Closing memmap at {perm_scale_values.filename}")
            perm_scale_values._mmap.close()

        del perm_scale_values

        logp_values = -np.log10(p_values)
        logp_values[np.isinf(logp_values)] = -np.log10(np.finfo(float).eps)

        # Write out unthresholded value images
        images = {"stat": stat_values, "logp": logp_values, "z": z_values}
        return images

    def _compute_summarystat_est(self, data):
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
            raise ValueError(f"Unsupported data type '{type(data)}'")

        stat_values = 1.0 - np.prod(1.0 - ma_values, axis=0)
        return stat_values

    def _scale_to_p(self, stat_values, scale_values):
        """Compute p- and z-values.

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
        n_voxels = stat_values.shape[0]

        # I know that joblib probably preserves order of outputs, but I'm paranoid, so we track
        # the iteration as well and sort the resulting p-value array based on that.
        with tqdm_joblib(tqdm(total=n_voxels)):
            p_values, voxel_idx = zip(
                *Parallel(n_jobs=self.n_cores)(
                    delayed(self._scale_to_p_voxel)(
                        i_voxel, stat_values[i_voxel], scale_values[:, i_voxel]
                    )
                    for i_voxel in range(n_voxels)
                )
            )
        # Convert to an array and sort the p-values array based on the voxel index.
        p_values = np.array(p_values)[np.array(voxel_idx)]

        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values

    def _scale_to_p_voxel(self, i_voxel, stat_value, voxel_null):
        """Compute one voxel's p-value from its specific null distribution."""
        scale_zeros = voxel_null == 0
        n_zeros = np.sum(scale_zeros)
        voxel_null[scale_zeros] = np.nan
        scale_hist = np.empty(len(self.null_distributions_["histogram_bins"]))
        scale_hist[0] = n_zeros

        scale_hist[1:] = np.histogram(
            a=voxel_null,
            bins=self.null_distributions_["histogram_bins"],
            range=(
                np.min(self.null_distributions_["histogram_bins"]),
                np.max(self.null_distributions_["histogram_bins"]),
            ),
            density=False,
        )[0]

        p_value = nullhist_to_p(
            stat_value,
            scale_hist,
            self.null_distributions_["histogram_bins"],
        )
        return p_value, i_voxel

    def _run_permutation(self, i_row, iter_xyz, iter_df, perm_scale_values):
        """Run a single random SCALE permutation of a dataset."""
        iter_xyz = np.squeeze(iter_xyz)
        iter_df[["x", "y", "z"]] = iter_xyz
        stat_values = self._compute_summarystat_est(iter_df)
        perm_scale_values[i_row, :] = stat_values

    def correct_fwe_montecarlo(self):
        """Perform Monte Carlo-based FWE correction.

        Warnings
        --------
        This method is not implemented for this class.
        """
        raise NotImplementedError(
            f"The {type(self)} class does not support `correct_fwe_montecarlo`."
        )
