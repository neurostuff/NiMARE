"""CBMA methods from the ALE and MKDA families."""

import logging
from abc import abstractmethod

import nibabel as nib
import numpy as np
import pandas as pd
import sparse
from joblib import Memory, Parallel, delayed
from nilearn.input_data import NiftiMasker
from scipy import ndimage
from tqdm.auto import tqdm

from nimare.estimator import Estimator
from nimare.meta.kernel import KernelTransformer
from nimare.meta.utils import _calculate_cluster_measures, _get_last_bin
from nimare.results import MetaResult
from nimare.stats import null_to_p, nullhist_to_p
from nimare.transforms import p_to_z
from nimare.utils import (
    _add_metadata_to_dataframe,
    _check_ncores,
    _check_type,
    get_masker,
    mm2vox,
    vox2mm,
)

LGR = logging.getLogger(__name__)


class CBMAEstimator(Estimator):
    """Base class for coordinate-based meta-analysis methods.

    .. versionchanged:: 0.0.12

        * Remove *low_memory* option
        * CBMA-specific elements of ``Estimator`` excised and moved into ``CBMAEstimator``.
        * Generic kwargs and args converted to named kwargs. All remaining kwargs are for kernels.
        * Use a 4D sparse array for modeled activation maps.

    .. versionchanged:: 0.0.8

        * [REF] Use saved MA maps, when available.
        * [REF] Add *low_memory* option.

    .. versionadded:: 0.0.3

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    *args
        Optional arguments to the :obj:`~nimare.base.Estimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`~nimare.base.Estimator`
        __init__ (called automatically).
    """

    # The standard required inputs are just coordinates.
    # An individual CBMAEstimator may override this.
    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        kernel_transformer,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        *,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            mask = get_masker(mask, memory=memory, memory_level=memory_level)
        self.masker = mask

        # Identify any kwargs
        kernel_args = {k: v for k, v in kwargs.items() if k.startswith("kernel__")}

        # Flag any extraneous kwargs
        other_kwargs = dict(set(kwargs.items()) - set(kernel_args.items()))
        if other_kwargs:
            LGR.warning(f"Unused keyword arguments found: {tuple(other_kwargs.items())}")

        # Get kernel transformer
        kernel_args = {k.split("kernel__")[1]: v for k, v in kernel_args.items()}
        if "memory" not in kernel_args.keys() and "memory_level" not in kernel_args.keys():
            kernel_args.update(memory=memory, memory_level=memory_level)
        kernel_transformer = _check_type(kernel_transformer, KernelTransformer, **kernel_args)
        self.kernel_transformer = kernel_transformer

        super().__init__(memory=memory, memory_level=memory_level)

    def _preprocess_input(self, dataset):
        """Mask required input images using either the Dataset's mask or the Estimator's.

        Also, insert required metadata into coordinates DataFrame.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            In this method, the Dataset is used to (1) select the appropriate mask image,
            and (2) extract sample size metadata and place it into the coordinates input.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            This attribute (created by ``_collect_inputs()``) is updated in this method.
            Specifically, (1) an "ma_maps" key may be added if pre-generated MA maps are available,
            (2) IJK coordinates will be added based on the mask image's affine,
            and (3) sample sizes may be added to the "coordinates" key, as needed.
        """
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "coordinates":
                # Calculate IJK matrix indices for target mask
                # Mask space is assumed to be the same as the Dataset's space
                # These indices are used directly by any KernelTransformer
                xyz = self.inputs_["coordinates"][["x", "y", "z"]].values
                ijk = mm2vox(xyz, mask_img.affine)
                self.inputs_["coordinates"][["i", "j", "k"]] = ijk

        # All extra (non-ijk) parameters for a kernel should be overrideable as
        # parameters to __init__, so we can access them with get_params()
        kt_args = list(self.kernel_transformer.get_params().keys())

        # Integrate "sample_size" from metadata into DataFrame so that
        # kernel_transformer can access it.
        if "sample_size" in kt_args:
            self.inputs_["coordinates"] = _add_metadata_to_dataframe(
                dataset,
                self.inputs_["coordinates"],
                metadata_field="sample_sizes",
                target_column="sample_size",
                filter_func=np.mean,
            )

    def _fit(self, dataset):
        """Perform coordinate-based meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator."
            )

        self.null_distributions_ = {}

        ma_values = self._collect_ma_maps(
            coords_key="coordinates",
            maps_key="ma_maps",
        )

        # Infer a weight vector, when applicable. Primarily used only for MKDADensity.
        self.weight_vec_ = self._compute_weights(ma_values)

        stat_values = self._compute_summarystat(ma_values)

        # Determine null distributions for summary stat (OF) to p conversion
        self._determine_histogram_bins(ma_values)
        if self.null_method.startswith("approximate"):
            self._compute_null_approximate(ma_values)

        elif self.null_method == "montecarlo":
            self._compute_null_montecarlo(n_iters=self.n_iters, n_cores=self.n_cores)

        else:
            # A hidden option only used for internal validation/testing
            self._compute_null_reduced_montecarlo(ma_values, n_iters=self.n_iters)

        p_values, z_values = self._summarystat_to_p(stat_values, null_method=self.null_method)

        maps = {"stat": stat_values, "p": p_values, "z": z_values}
        description = self._generate_description()
        return maps, {}, description

    def _compute_weights(self, ma_values):
        """Perform optional weight computation routine.

        Takes an array of meta-analysis values as input and returns an array
        of the same shape, weighted as desired.
        Can be ignored by algorithms that don't support weighting.
        """
        return None

    def _collect_ma_maps(self, coords_key="coordinates", maps_key="ma_maps", return_type="sparse"):
        """Collect modeled activation maps from Estimator inputs.

        Parameters
        ----------
        coords_key : :obj:`str`, optional
            Key to ``Estimator.inputs_`` dictionary containing coordinates DataFrame.
            This key should **always** be present.
            Default is "coordinates".
        maps_key : :obj:`str`, optional
            Key to ``Estimator.inputs_`` dictionary containing list of MA map files.
            This key should only be present if the kernel transformer was already fitted to the
            input Dataset.
            Default is "ma_maps".

        Returns
        -------
        ma_maps : :obj:`sparse._coo.core.COO`
            Return a 4D sparse array of shape
            (n_studies, mask.shape) with MA maps.
        """
        LGR.debug(f"Generating MA maps from coordinates ({coords_key}).")

        ma_maps = self.kernel_transformer.transform(
            self.inputs_[coords_key],
            masker=self.masker,
            return_type=return_type,
        )

        return ma_maps

    def _compute_summarystat(self, data):
        """Compute summary statistics from data.

        The actual summary statistic varies across Estimators, and is implemented in
        ``_compute_summarystat_est``.
        For ALE and SCALE, the values are known as ALE values.
        For (M)KDA, they are "OF" scores.

        Parameters
        ----------
        data : array, sparse._coo.core.COO, pandas.DataFrame, or list of img_like
            Data from which to estimate summary statistics.
            The data can be:
            (1) a 1d contrast-len or 2d contrast-by-voxel array of MA values,
            (2) a 4d sparse array of MA maps,
            (3) a DataFrame containing coordinates to produce MA values,
            or (4) a list of imgs containing MA values.

        Returns
        -------
        stat_values : 1d array
            Summary statistic values. One value per voxel.
        """
        if isinstance(data, pd.DataFrame):
            ma_values = self.kernel_transformer.transform(
                data, masker=self.masker, return_type="sparse"
            )
        elif isinstance(data, list):
            ma_values = self.masker.transform(data)
        elif isinstance(data, (np.ndarray, sparse._coo.core.COO)):
            ma_values = data
        else:
            raise ValueError(f"Unsupported data type '{type(data)}'")

        # Apply weights before returning
        return self._compute_summarystat_est(ma_values)

    @abstractmethod
    def _compute_summarystat_est(self, ma_values):
        """Compute summary statistic according to estimator-specific method.

        Must be overriden by subclasses.
        Input and output are both numpy arrays; the output must
        aggregate over the 0th dimension of the input.
        (i.e., if the input has K dimensions, the output has K - 1 dimensions.)
        """
        pass

    def _summarystat_to_p(self, stat_values, null_method="approximate"):
        """Compute p- and z-values from summary statistics (e.g., ALE scores).

        Uses either histograms from "approximate" null or null distribution from "montecarlo" null.

        Parameters
        ----------
        stat_values : 1D array_like
            Array of summary statistic values from estimator.
        null_method : {"approximate", "montecarlo"}, optional
            Whether to use approximate null or montecarlo null.
            Default is "approximate".

        Returns
        -------
        p_values, z_values : 1D array
            P- and Z-values for statistic values.
            Same shape as stat_values.
        """
        if null_method.startswith("approximate"):
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histweights_corr-none_method-approximate" in self.null_distributions_.keys()

            p_values = nullhist_to_p(
                stat_values,
                self.null_distributions_["histweights_corr-none_method-approximate"],
                self.null_distributions_["histogram_bins"],
            )

        elif null_method == "montecarlo":
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histweights_corr-none_method-montecarlo" in self.null_distributions_.keys()

            p_values = nullhist_to_p(
                stat_values,
                self.null_distributions_["histweights_corr-none_method-montecarlo"],
                self.null_distributions_["histogram_bins"],
            )

        elif null_method == "reduced_montecarlo":
            assert "values_corr-none_method-reducedMontecarlo" in self.null_distributions_.keys()

            p_values = null_to_p(
                stat_values,
                self.null_distributions_["values_corr-none_method-reducedMontecarlo"],
                tail="upper",
            )

        else:
            raise ValueError("Argument 'null_method' must be one of: 'approximate', 'montecarlo'.")

        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values

    def _p_to_summarystat(self, p, null_method=None):
        """Compute a summary statistic threshold that corresponds to the provided p-value.

        Uses either histograms from approximate null or null distribution from montecarlo null.

        Parameters
        ----------
        p : :obj:`float`
            The p-value that corresponds to the summary statistic threshold.
        null_method : {None, "approximate", "montecarlo"}, optional
            Whether to use approximate null or montecarlo null. If None, defaults to using
            whichever method was set at initialization.

        Returns
        -------
        ss : float
            A float giving the summary statistic value corresponding to the passed p.
        """
        if null_method is None:
            null_method = self.null_method

        if null_method.startswith("approximate"):
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histweights_corr-none_method-approximate" in self.null_distributions_.keys()

            # Convert unnormalized histogram weights to null distribution
            histogram_weights = self.null_distributions_[
                "histweights_corr-none_method-approximate"
            ]
            null_distribution = histogram_weights / np.sum(histogram_weights)
            null_distribution = np.cumsum(null_distribution[::-1])[::-1]
            null_distribution /= np.max(null_distribution)
            null_distribution = np.squeeze(null_distribution)

            # Desired bin is the first one _before_ the target p-value (for uniformity
            # with the montecarlo null).
            ss_idx = np.maximum(0, np.where(null_distribution <= p)[0][0] - 1)
            ss = self.null_distributions_["histogram_bins"][ss_idx]

        elif null_method == "montecarlo":
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histweights_corr-none_method-montecarlo" in self.null_distributions_.keys()

            hist_weights = self.null_distributions_["histweights_corr-none_method-montecarlo"]
            # Desired bin is the first one _before_ the target p-value (for uniformity
            # with the montecarlo null).
            ss_idx = np.maximum(0, np.where(hist_weights <= p)[0][0] - 1)
            ss = self.null_distributions_["histogram_bins"][ss_idx]

        elif null_method == "reduced_montecarlo":
            assert "values_corr-none_method-reducedMontecarlo" in self.null_distributions_.keys()

            null_dist = np.sort(
                self.null_distributions_["values_corr-none_method-reducedMontecarlo"]
            )
            n_vals = len(null_dist)
            ss_idx = np.floor(p * n_vals).astype(int)
            ss = null_dist[-ss_idx]

        else:
            raise ValueError("Argument 'null_method' must be one of: 'approximate', 'montecarlo'.")

        return ss

    def _compute_null_reduced_montecarlo(self, ma_maps, n_iters=5000):
        """Compute uncorrected null distribution using the reduced montecarlo method.

        This method is much faster than the full montecarlo approach, but is still slower than the
        approximate method. Given that its resolution is roughly the same as the approximate
        method, we recommend against using this method.

        Parameters
        ----------
        ma_maps : (C x V) array
            Contrast by voxel array of MA values, after weighting with weight_vec.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute:
        "values_corr-none_method-reducedMontecarlo".

        Warnings
        --------
        This method is only retained for testing and algorithm development.
        """
        if isinstance(ma_maps, sparse._coo.core.COO):
            masker = self.dataset.masker if not self.masker else self.masker
            mask = masker.mask_img
            mask_data = mask.get_fdata().astype(bool)

            ma_maps = ma_maps.todense()
            ma_maps = ma_maps[:, mask_data]

        n_studies, n_voxels = ma_maps.shape
        null_ijk = np.random.choice(np.arange(n_voxels), (n_iters, n_studies))
        iter_ma_values = ma_maps[np.arange(n_studies), tuple(null_ijk)].T
        null_dist = self._compute_summarystat(iter_ma_values)
        self.null_distributions_["values_corr-none_method-reducedMontecarlo"] = null_dist

    def _compute_null_montecarlo_permutation(self, iter_xyz, iter_df):
        """Run a single Monte Carlo permutation of a dataset.

        Does the shared work between uncorrected stat-to-p conversion and vFWE.

        Parameters
        ----------
        params : tuple
            A tuple containing 2 elements, respectively providing (1) the permuted
            coordinates and (2) the original coordinate DataFrame.

        Returns
        -------
        counts : 1D array_like
            Weights associated with the attribute `null_distributions_["histogram_bins"]`.
        """
        # Not sure if joblib will automatically use a copy of the object, but I'll make a copy to
        # be safe.
        iter_df = iter_df.copy()

        iter_xyz = np.squeeze(iter_xyz)
        iter_df[["x", "y", "z"]] = iter_xyz

        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="sparse"
        )
        iter_ss_map = self._compute_summarystat(iter_ma_maps)

        del iter_ma_maps

        # Get bin edges for histogram
        bin_centers = self.null_distributions_["histogram_bins"]
        step_size = bin_centers[1] - bin_centers[0]
        bin_edges = bin_centers - (step_size / 2)
        bin_edges = np.append(bin_centers, bin_centers[-1] + step_size)

        counts, _ = np.histogram(iter_ss_map, bins=bin_edges, density=False)
        return counts

    def _compute_null_montecarlo(self, n_iters, n_cores):
        """Compute uncorrected null distribution using Monte Carlo method.

        Parameters
        ----------
        n_iters : int
            Number of permutations.
        n_cores : int
            Number of cores to use.

        Notes
        -----
        This method adds two entries to the null_distributions_ dict attribute:
        "histweights_corr-none_method-montecarlo" and
        "histweights_level-voxel_corr-fwe_method-montecarlo".
        """
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T

        n_cores = _check_ncores(n_cores)

        rand_idx = np.random.choice(
            null_ijk.shape[0],
            size=(self.inputs_["coordinates"].shape[0], n_iters),
        )
        rand_ijk = null_ijk[rand_idx, :]
        rand_xyz = vox2mm(rand_ijk, self.masker.mask_img.affine)
        iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)
        iter_df = self.inputs_["coordinates"].copy()

        perm_histograms = [
            r
            for r in tqdm(
                Parallel(return_as="generator", n_jobs=n_cores)(
                    delayed(self._compute_null_montecarlo_permutation)(
                        iter_xyzs[i_iter], iter_df=iter_df
                    )
                    for i_iter in range(n_iters)
                ),
                total=n_iters,
            )
        ]

        perm_histograms = np.vstack(perm_histograms)
        self.null_distributions_["histweights_corr-none_method-montecarlo"] = np.sum(
            perm_histograms, axis=0
        )

        fwe_voxel_max = np.apply_along_axis(_get_last_bin, 1, perm_histograms)
        histweights = np.zeros(perm_histograms.shape[1], dtype=perm_histograms.dtype)
        for perm in fwe_voxel_max:
            histweights[perm] += 1

        self.null_distributions_["histweights_level-voxel_corr-fwe_method-montecarlo"] = (
            histweights
        )

    def _correct_fwe_montecarlo_permutation(
        self,
        iter_xyz,
        iter_df,
        conn,
        voxel_thresh,
        vfwe_only,
    ):
        """Run a single Monte Carlo permutation of a dataset.

        Does the shared work between vFWE and cFWE.

        Parameters
        ----------
        iter_xyz : :obj:`numpy.ndarray` of shape (C, 3)
            The permuted coordinates. One row for each peak.
            Columns correspond to x, y, and z coordinates.
        iter_df : :obj:`pandas.DataFrame`
            The coordinates DataFrame, to be filled with the permuted coordinates in ``iter_xyz``
            before permutation MA maps are generated.
        conn : :obj:`numpy.ndarray` of shape (3, 3, 3)
            The 3D structuring array for labeling clusters.
        voxel_thresh : :obj:`float`
            Uncorrected summary statistic threshold for defining clusters.
        vfwe_only : :obj:`bool`
            If True, only calculate the voxel-level FWE-corrected maps.

        Returns
        -------
        (iter_max value, iter_max_cluster, iter_max_mass)
            A 3-tuple of floats giving the maximum voxel-wise value, maximum cluster size,
            and maximum cluster mass for the permuted dataset.
            If ``vfwe_only`` is True, the latter two values will be None.
        """
        iter_df = iter_df.copy()

        iter_xyz = np.squeeze(iter_xyz)
        iter_df[["x", "y", "z"]] = iter_xyz

        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="sparse"
        )
        iter_ss_map = self._compute_summarystat(iter_ma_maps)

        del iter_ma_maps

        # Voxel-level inference
        iter_max_value = np.max(iter_ss_map)

        if vfwe_only:
            iter_max_size, iter_max_mass = None, None
        else:
            # Cluster-level inference
            iter_ss_map = self.masker.inverse_transform(iter_ss_map).get_fdata()
            iter_max_size, iter_max_mass = _calculate_cluster_measures(
                iter_ss_map, voxel_thresh, conn, tail="upper"
            )
        return iter_max_value, iter_max_size, iter_max_mass

    def correct_fwe_montecarlo(
        self,
        result,
        voxel_thresh=0.001,
        n_iters=5000,
        n_cores=1,
        vfwe_only=False,
    ):
        """Perform FWE correction using the max-value permutation method.

        Only call this method from within a Corrector.

        .. versionchanged:: 0.0.13

            Change cluster neighborhood from faces+edges to faces, to match Nilearn.

        .. versionchanged:: 0.0.12

            * Fix the ``vfwe_only`` option.

        .. versionchanged:: 0.0.11

            * Rename ``*_level-cluster`` maps to ``*_desc-size_level-cluster``.
            * Add new ``*_desc-mass_level-cluster`` maps that use cluster mass-based inference.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from a CBMA meta-analysis.
        voxel_thresh : :obj:`float`, default=0.001
            Cluster-defining p-value threshold. Default is 0.001.
        n_iters : :obj:`int`, default=5000
            Number of iterations to build the voxel-level, cluster-size, and cluster-mass FWE
            null distributions. Default is 5000.
        n_cores : :obj:`int`, default=1
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.
        vfwe_only : :obj:`bool`, default=False
            If True, only calculate the voxel-level FWE-corrected maps. Voxel-level correction
            can be performed very quickly if the Estimator's ``null_method`` was "montecarlo".
            Default is False.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method:

            -   ``logp_desc-size_level-cluster``: Cluster-level FWE-corrected ``-log10(p)`` map
                based on cluster size. This was previously simply called "logp_level-cluster".
                This array is **not** generated if ``vfwe_only`` is ``True``.
            -   ``logp_desc-mass_level-cluster``: Cluster-level FWE-corrected ``-log10(p)`` map
                based on cluster mass. According to :footcite:t:`bullmore1999global` and
                :footcite:t:`zhang2009cluster`, cluster mass-based inference is more powerful than
                cluster size.
                This array is **not** generated if ``vfwe_only`` is ``True``.
            -   ``logp_level-voxel``: Voxel-level FWE-corrected ``-log10(p)`` map.
                Voxel-level correction is generally more conservative than cluster-level
                correction, so it is only recommended for very large meta-analyses
                (i.e., hundreds of studies), per :footcite:t:`eickhoff2016behavior`.
        description_ : :obj:`str`
            A text description of the correction procedure.

        Notes
        -----
        If ``vfwe_only`` is ``False``, this method adds three new keys to the
        ``null_distributions_`` attribute:

            -   ``values_level-voxel_corr-fwe_method-montecarlo``: The maximum summary statistic
                value from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-size_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                size from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-mass_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                mass from each Monte Carlo iteration. An array of shape (n_iters,).

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.

        References
        ----------
        .. footbibliography::

        Examples
        --------
        >>> meta = MKDADensity()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', voxel_thresh=0.01,
                                     n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        stat_values = result.get_map("stat", return_type="array")

        if vfwe_only and (self.null_method == "montecarlo"):
            LGR.info("Using precalculated histogram for voxel-level FWE correction.")

            # Determine p- and z-values from stat values and null distribution.
            p_vfwe_values = nullhist_to_p(
                stat_values,
                self.null_distributions_["histweights_level-voxel_corr-fwe_method-montecarlo"],
                self.null_distributions_["histogram_bins"],
            )

        else:
            if vfwe_only:
                LGR.warn(
                    "In order to run this method with the 'vfwe_only' option, "
                    "the Estimator must use the 'montecarlo' null_method. "
                    "Running permutations from scratch."
                )

            null_xyz = vox2mm(
                np.vstack(np.where(self.masker.mask_img.get_fdata())).T,
                self.masker.mask_img.affine,
            )

            n_cores = _check_ncores(n_cores)

            # Identify summary statistic corresponding to intensity threshold
            ss_thresh = self._p_to_summarystat(voxel_thresh)

            rand_idx = np.random.choice(
                null_xyz.shape[0],
                size=(self.inputs_["coordinates"].shape[0], n_iters),
            )
            rand_xyz = null_xyz[rand_idx, :]
            iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)
            iter_df = self.inputs_["coordinates"].copy()

            # Define connectivity matrix for cluster labeling
            conn = ndimage.generate_binary_structure(rank=3, connectivity=1)

            perm_results = [
                r
                for r in tqdm(
                    Parallel(return_as="generator", n_jobs=n_cores)(
                        delayed(self._correct_fwe_montecarlo_permutation)(
                            iter_xyzs[i_iter],
                            iter_df=iter_df,
                            conn=conn,
                            voxel_thresh=ss_thresh,
                            vfwe_only=vfwe_only,
                        )
                        for i_iter in range(n_iters)
                    ),
                    total=n_iters,
                )
            ]

            fwe_voxel_max, fwe_cluster_size_max, fwe_cluster_mass_max = zip(*perm_results)

            if not vfwe_only:
                # Cluster-level FWE
                # Extract the summary statistics in voxel-wise (3D) form, threshold, and
                # cluster-label
                thresh_stat_values = self.masker.inverse_transform(stat_values).get_fdata()
                thresh_stat_values[thresh_stat_values <= ss_thresh] = 0
                labeled_matrix, _ = ndimage.label(thresh_stat_values, conn)

                cluster_labels, idx, cluster_sizes = np.unique(
                    labeled_matrix,
                    return_inverse=True,
                    return_counts=True,
                )
                assert cluster_labels[0] == 0

                # Cluster mass-based inference
                cluster_masses = np.zeros(cluster_labels.shape)
                for i_val in cluster_labels:
                    if i_val == 0:
                        cluster_masses[i_val] = 0

                    cluster_mass = np.sum(thresh_stat_values[labeled_matrix == i_val] - ss_thresh)
                    cluster_masses[i_val] = cluster_mass

                p_cmfwe_vals = null_to_p(cluster_masses, fwe_cluster_mass_max, "upper")
                p_cmfwe_map = p_cmfwe_vals[np.reshape(idx, labeled_matrix.shape)]

                p_cmfwe_values = np.squeeze(
                    self.masker.transform(
                        nib.Nifti1Image(p_cmfwe_map, self.masker.mask_img.affine)
                    )
                )
                logp_cmfwe_values = -np.log10(p_cmfwe_values)
                logp_cmfwe_values[np.isinf(logp_cmfwe_values)] = -np.log10(np.finfo(float).eps)
                z_cmfwe_values = p_to_z(p_cmfwe_values, tail="one")

                # Cluster size-based inference
                cluster_sizes[0] = 0  # replace background's "cluster size" with zeros
                p_csfwe_vals = null_to_p(cluster_sizes, fwe_cluster_size_max, "upper")
                p_csfwe_map = p_csfwe_vals[np.reshape(idx, labeled_matrix.shape)]

                p_csfwe_values = np.squeeze(
                    self.masker.transform(
                        nib.Nifti1Image(p_csfwe_map, self.masker.mask_img.affine)
                    )
                )
                logp_csfwe_values = -np.log10(p_csfwe_values)
                logp_csfwe_values[np.isinf(logp_csfwe_values)] = -np.log10(np.finfo(float).eps)
                z_csfwe_values = p_to_z(p_csfwe_values, tail="one")

                self.null_distributions_[
                    "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
                ] = fwe_cluster_size_max
                self.null_distributions_[
                    "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
                ] = fwe_cluster_mass_max

            # Voxel-level FWE
            LGR.info("Using null distribution for voxel-level FWE correction.")
            p_vfwe_values = null_to_p(stat_values, fwe_voxel_max, tail="upper")
            self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"] = (
                fwe_voxel_max
            )

        z_vfwe_values = p_to_z(p_vfwe_values, tail="one")
        logp_vfwe_values = -np.log10(p_vfwe_values)
        logp_vfwe_values[np.isinf(logp_vfwe_values)] = -np.log10(np.finfo(float).eps)

        if vfwe_only:
            # Return unthresholded value images
            maps = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
            }

        else:
            # Return unthresholded value images
            maps = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
                "logp_desc-size_level-cluster": logp_csfwe_values,
                "z_desc-size_level-cluster": z_csfwe_values,
                "logp_desc-mass_level-cluster": logp_cmfwe_values,
                "z_desc-mass_level-cluster": z_cmfwe_values,
            }

        if vfwe_only:
            description = (
                "Family-wise error correction was performed using a voxel-level Monte Carlo "
                "procedure. "
                "In this procedure, null datasets are generated in which dataset coordinates are "
                "substituted with coordinates randomly drawn from the meta-analysis mask, and "
                "the maximum summary statistic is retained. "
                f"This procedure was repeated {n_iters} times to build a null distribution of "
                "summary statistics."
            )
        else:
            description = (
                "Family-wise error rate correction was performed using a Monte Carlo procedure. "
                "In this procedure, null datasets are generated in which dataset coordinates are "
                "substituted with coordinates randomly drawn from the meta-analysis mask, and "
                "maximum values are retained. "
                f"This procedure was repeated {n_iters} times to build null distributions of "
                "summary statistics, cluster sizes, and cluster masses. "
                "Clusters for cluster-level correction were defined using edge-wise connectivity "
                f"and a voxel-level threshold of p < {voxel_thresh} from the uncorrected null "
                "distribution."
            )

        return maps, {}, description


class PairwiseCBMAEstimator(CBMAEstimator):
    """Base class for pairwise coordinate-based meta-analysis methods.

    .. versionchanged:: 0.0.12

        - Use a 4D sparse array for modeled activation maps.

    .. versionchanged:: 0.0.8

        * [REF] Use saved MA maps, when available.

    .. versionadded:: 0.0.3

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    *args
        Optional arguments to the :obj:`~nimare.base.Estimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`~nimare.base.Estimator`
        __init__ (called automatically).
    """

    def _compute_summarystat_est(self, ma_values):
        """Calculate the Estimator's summary statistic.

        This method is only included because CBMAEstimator has it as an abstract method.
        PairwiseCBMAEstimators are not constructed uniformly enough for this structure to work
        consistently.
        """
        raise NotImplementedError

    def fit(self, dataset1, dataset2, drop_invalid=True):
        """Fit Estimator to two Datasets.

        Parameters
        ----------
        dataset1/dataset2 : :obj:`~nimare.dataset.Dataset`
            Dataset objects to analyze.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Estimator fitting.

        Notes
        -----
        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Estimators' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.
        """
        # Reproduce fit() for dataset1 to collect and process inputs.
        self._collect_inputs(dataset1, drop_invalid=drop_invalid)
        self._preprocess_input(dataset1)
        if "ma_maps" in self.inputs_.keys():
            # Grab pre-generated MA maps
            self.inputs_["ma_maps1"] = self.inputs_.pop("ma_maps")

        self.inputs_["id1"] = self.inputs_.pop("id")
        self.inputs_["coordinates1"] = self.inputs_.pop("coordinates")

        # Reproduce fit() for dataset2 to collect and process inputs.
        self._collect_inputs(dataset2, drop_invalid=drop_invalid)
        self._preprocess_input(dataset2)
        if "ma_maps" in self.inputs_.keys():
            # Grab pre-generated MA maps
            self.inputs_["ma_maps2"] = self.inputs_.pop("ma_maps")

        self.inputs_["id2"] = self.inputs_.pop("id")
        self.inputs_["coordinates2"] = self.inputs_.pop("coordinates")

        # Now run the Estimator-specific _fit() method.
        maps, tables, description = self._cache(self._fit, func_memory_level=1)(dataset1, dataset2)

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset1.masker

        return MetaResult(self, mask=masker, maps=maps, tables=tables, description=description)
