"""CBMA methods from the ALE and MKDA families."""
import logging

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage
from tqdm.auto import tqdm

from ...base import MetaEstimator
from ...results import MetaResult
from ...stats import null_to_p, nullhist_to_p
from ...transforms import p_to_z
from ...utils import (
    _add_metadata_to_dataframe,
    _check_type,
    _safe_transform,
    tqdm_joblib,
    use_memmap,
    vox2mm,
)
from ..kernel import KernelTransformer

LGR = logging.getLogger(__name__)


class CBMAEstimator(MetaEstimator):
    """Base class for coordinate-based meta-analysis methods.

    .. versionchanged:: 0.0.8

        * [REF] Use saved MA maps, when available.
        * [REF] Add *low_memory* option.

    .. versionadded:: 0.0.3

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    *args
        Optional arguments to the :obj:`~nimare.base.MetaEstimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`~nimare.base.MetaEstimator`
        __init__ (called automatically).
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(self, kernel_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get kernel transformer
        kernel_args = {
            k.split("kernel__")[1]: v for k, v in kwargs.items() if k.startswith("kernel__")
        }
        kernel_transformer = _check_type(kernel_transformer, KernelTransformer, **kernel_args)
        self.kernel_transformer = kernel_transformer

    @use_memmap(LGR, n_files=1)
    def _fit(self, dataset):
        """
        Perform coordinate-based meta-analysis on dataset.

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
            fname_idx=0,
        )

        self.weight_vec_ = self._compute_weights(ma_values)

        stat_values = self.compute_summarystat(ma_values)

        # Determine null distributions for summary stat (OF) to p conversion
        self._determine_histogram_bins(ma_values)
        if self.null_method.startswith("approximate"):
            self._compute_null_approximate(ma_values)

        elif self.null_method == "montecarlo":
            self._compute_null_montecarlo(n_iters=self.n_iters, n_cores=self.n_cores)

        else:
            # A hidden option only used for internal validation/testing
            self._compute_null_reduced_montecarlo(ma_values, n_iters=self.n_iters)

        # Only should occur when MA maps have been pre-generated in the Dataset and a memory_limit
        # is set. The memmap must be closed.
        if isinstance(ma_values, np.memmap):
            LGR.debug(f"Closing memmap at {ma_values.filename}")
            ma_values._mmap.close()

        p_values, z_values = self._summarystat_to_p(stat_values, null_method=self.null_method)

        images = {"stat": stat_values, "p": p_values, "z": z_values}
        return images

    def _compute_weights(self, ma_values):
        """Perform optional weight computation routine.

        Takes an array of meta-analysis values as input and returns an array
        of the same shape, weighted as desired.
        Can be ignored by algorithms that don't support weighting.
        """
        return None

    def _preprocess_input(self, dataset):
        """Mask required input images using either the dataset's mask or the estimator's.

        Also, insert required metadata into coordinates DataFrame.
        """
        super()._preprocess_input(dataset)

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

    def _collect_ma_maps(self, coords_key="coordinates", maps_key="ma_maps", fname_idx=0):
        """Collect modeled activation maps from Estimator inputs.

        Parameters
        ----------
        coords_key : :obj:`str`, optional
            Key to Estimator.inputs_ dictionary containing coordinates DataFrame.
            This key should **always** be present.
        maps_key : :obj:`str`, optional
            Key to Estimator.inputs_ dictionary containing list of MA map files.
            This key should only be present if the kernel transformer was already fitted to the
            input Dataset.
        fname_idx : :obj:`int`, optional
            When the Estimator is set with ``memory_limit`` as a string,
            there is a ``memmap_filenames`` attribute that is a list of filenames or Nones.
            This parameter specifies which item in that list should be used for a memory-mapped
            array. Default is 0.

        Returns
        -------
        ma_maps : :obj:`numpy.ndarray` or :obj:`numpy.memmap`
            2D numpy array of shape (n_studies, n_voxels) with MA values.
            This will be a memmap if MA maps have been pre-generated.
        """
        if maps_key in self.inputs_.keys():
            LGR.debug(f"Loading pre-generated MA maps ({maps_key}).")
            if self.memory_limit:
                # perform transform on chunks of the input maps
                ma_maps = _safe_transform(
                    self.inputs_[maps_key],
                    masker=self.masker,
                    memory_limit=self.memory_limit,
                    memfile=self.memmap_filenames[fname_idx],
                )
            else:
                ma_maps = self.masker.transform(self.inputs_[maps_key])
        else:
            LGR.debug(f"Generating MA maps from coordinates ({coords_key}).")
            ma_maps = self.kernel_transformer.transform(
                self.inputs_[coords_key],
                masker=self.masker,
                return_type="array",
            )
        return ma_maps

    def compute_summarystat(self, data):
        """Compute summary statistics from data.

        The actual summary statistic varies across Estimators.
        For ALE and SCALE, the values are known as ALE values.
        For (M)KDA, they are "OF" scores.

        Parameters
        ----------
        data : array, pandas.DataFrame, or list of img_like
            Data from which to estimate summary statistics.
            The data can be:
            (1) a 1d contrast-len or 2d contrast-by-voxel array of MA values,
            (2) a DataFrame containing coordinates to produce MA values,
            or (3) a list of imgs containing MA values.

        Returns
        -------
        stat_values : 1d array
            Summary statistic values. One value per voxel.
        """
        if isinstance(data, pd.DataFrame):
            ma_values = self.kernel_transformer.transform(
                data, masker=self.masker, return_type="array"
            )
        elif isinstance(data, list):
            ma_values = self.masker.transform(data)
        elif isinstance(data, np.ndarray):
            ma_values = data
        elif not isinstance(data, np.ndarray):
            raise ValueError(f"Unsupported data type '{type(data)}'")

        # Apply weights before returning
        return self._compute_summarystat(ma_values)

    def _compute_summarystat(self, ma_values):
        """Compute summary statistic according to estimator-specific method.

        Must be overriden by subclasses.
        Input and output are both numpy arrays; the output must
        aggregate over the 0th dimension of the input.
        (i.e., if the input has K dimensions, the output has K - 1 dimensions.)
        """
        pass

    def _summarystat_to_p(self, stat_values, null_method="approximate"):
        """Compute p- and z-values from summary statistics (e.g., ALE scores).

        Uses either histograms from approximate null or null distribution from montecarlo null.

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
        p : The p-value that corresponds to the summary statistic threshold
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

            # Desired bin is the first one _before_ the target p-value (for consistency
            # with the montecarlo null).
            ss_idx = np.maximum(0, np.where(null_distribution <= p)[0][0] - 1)
            ss = self.null_distributions_["histogram_bins"][ss_idx]

        elif null_method == "montecarlo":
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histweights_corr-none_method-montecarlo" in self.null_distributions_.keys()

            hist_weights = self.null_distributions_["histweights_corr-none_method-montecarlo"]
            # Desired bin is the first one _before_ the target p-value (for consistency
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

    def _compute_null_reduced_montecarlo(self, ma_maps, n_iters=10000):
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

        Warning
        -------
        This method is only retained for testing and algorithm development.
        """
        n_studies, n_voxels = ma_maps.shape
        null_ijk = np.random.choice(np.arange(n_voxels), (n_iters, n_studies))
        iter_ma_values = ma_maps[np.arange(n_studies), tuple(null_ijk)].T
        null_dist = self.compute_summarystat(iter_ma_values)
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
        # Not sure if partial will automatically use a copy of the object, but I'll make a copy to
        # be safe.
        iter_df = iter_df.copy()

        iter_xyz = np.squeeze(iter_xyz)
        iter_df[["x", "y", "z"]] = iter_xyz

        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="array"
        )
        iter_ss_map = self.compute_summarystat(iter_ma_maps)

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

        n_cores = self._check_ncores(n_cores)

        rand_idx = np.random.choice(
            null_ijk.shape[0],
            size=(self.inputs_["coordinates"].shape[0], n_iters),
        )
        rand_ijk = null_ijk[rand_idx, :]
        rand_xyz = vox2mm(rand_ijk, self.masker.mask_img.affine)
        iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)
        iter_df = self.inputs_["coordinates"].copy()

        with tqdm_joblib(tqdm(total=n_iters)):
            perm_histograms = Parallel(n_jobs=n_cores)(
                delayed(self._compute_null_montecarlo_permutation)(
                    iter_xyzs[i_iter], iter_df=iter_df
                )
                for i_iter in range(n_iters)
            )

        perm_histograms = np.vstack(perm_histograms)
        self.null_distributions_["histweights_corr-none_method-montecarlo"] = np.sum(
            perm_histograms, axis=0
        )

        def get_last_bin(arr1d):
            """Index the last location in a 1D array with a non-zero value."""
            if np.any(arr1d):
                last_bin = np.where(arr1d)[0][-1]
            else:
                last_bin = 0
            return last_bin

        fwe_voxel_max = np.apply_along_axis(get_last_bin, 1, perm_histograms)
        self.null_distributions_[
            "histweights_level-voxel_corr-fwe_method-montecarlo"
        ] = fwe_voxel_max

    def _correct_fwe_montecarlo_permutation(self, iter_xyz, iter_df, conn, voxel_thresh):
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

        Returns
        -------
        (iter_max value, iter_max_cluster, iter_max_mass)
            A 3-tuple of floats giving the maximum voxel-wise value, maximum cluster size,
            and maximum cluster mass for the permuted dataset.
        """
        iter_df = iter_df.copy()

        iter_xyz = np.squeeze(iter_xyz)
        iter_df[["x", "y", "z"]] = iter_xyz

        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="array"
        )
        iter_ss_map = self.compute_summarystat(iter_ma_maps)

        del iter_ma_maps

        # Voxel-level inference
        iter_max_value = np.max(iter_ss_map)

        # Cluster-level inference
        iter_ss_map = self.masker.inverse_transform(iter_ss_map).get_fdata().copy()
        iter_ss_map[iter_ss_map <= voxel_thresh] = 0
        labeled_matrix = ndimage.measurements.label(iter_ss_map, conn)[0]
        clust_vals, clust_sizes = np.unique(labeled_matrix, return_counts=True)
        assert clust_vals[0] == 0

        # Cluster mass-based inference
        iter_max_mass = 0
        for unique_val in clust_vals[1:]:
            ss_vals = iter_ss_map[labeled_matrix == unique_val] - voxel_thresh
            iter_max_mass = np.maximum(iter_max_mass, np.sum(ss_vals))

        del iter_ss_map, labeled_matrix

        # Cluster size-based inference
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        if clust_sizes.size:
            iter_max_cluster = np.max(clust_sizes)
        else:
            iter_max_cluster = 0

        return iter_max_value, iter_max_cluster, iter_max_mass

    def correct_fwe_montecarlo(
        self, result, voxel_thresh=0.001, n_iters=10000, n_cores=1, vfwe_only=False
    ):
        """Perform FWE correction using the max-value permutation method.

        Only call this method from within a Corrector.

        .. versionchanged:: 0.0.11

            * Rename ``*_level-cluster`` maps to ``*_desc-size_level-cluster``.
            * Add new ``*_desc-mass_level-cluster`` maps that use cluster mass-based inference.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from a CBMA meta-analysis.
        voxel_thresh : :obj:`float`, optional
            Cluster-defining p-value threshold. Default is 0.001.
        n_iters : :obj:`int`, optional
            Number of iterations to build the voxel-level, cluster-size, and cluster-mass FWE
            null distributions. Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.
        vfwe_only : :obj:`bool`, optional
            If True, only calculate the voxel-level FWE-corrected maps. Voxel-level correction
            can be performed very quickly if the Estimator's ``null_method`` was "montecarlo".
            If this is set to True and the original null method was not montecarlo, an exception
            will be raised.
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
                based on cluster mass. According to [1]_ and [2]_, cluster mass-based inference is
                more powerful than cluster size.
                This array is **not** generated if ``vfwe_only`` is ``True``.
            -   ``logp_level-voxel``: Voxel-level FWE-corrected ``-log10(p)`` map.
                Voxel-level correction is generally more conservative than cluster-level
                correction, so it is only recommended for very large meta-analyses
                (i.e., hundreds of studies), per [3]_.

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
        .. [1] Bullmore, E. T., Suckling, J., Overmeyer, S., Rabe-Hesketh, S., Taylor, E., &
               Brammer, M. J. (1999). Global, voxel, and cluster tests, by theory and permutation,
               for a difference between two groups of structural MR images of the brain.
               IEEE transactions on medical imaging, 18(1), 32-42. doi: 10.1109/42.750253
        .. [2] Zhang, H., Nichols, T. E., & Johnson, T. D. (2009).
               Cluster mass inference via random field theory. Neuroimage, 44(1), 51-61.
               doi: 10.1016/j.neuroimage.2008.08.017
        .. [3] Eickhoff, S. B., Nichols, T. E., Laird, A. R., Hoffstaedter, F., Amunts, K.,
               Fox, P. T., ... & Eickhoff, C. R. (2016).
               Behavior, sensitivity, and power of activation likelihood estimation characterized
               by massive empirical simulation. Neuroimage, 137, 70-85.
               doi: 10.1016/j.neuroimage.2016.04.072

        Examples
        --------
        >>> meta = MKDADensity()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', voxel_thresh=0.01,
                                     n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        stat_values = result.get_map("stat", return_type="array")
        if vfwe_only:
            assert self.null_method == "montecarlo"

            LGR.info("Using precalculated histogram for voxel-level FWE correction.")

            # Determine p- and z-values from stat values and null distribution.
            p_vfwe_values = nullhist_to_p(
                stat_values,
                self.null_distributions_["histweights_level-voxel_corr-fwe_method-montecarlo"],
                self.null_distributions_["histogram_bins"],
            )

        else:
            null_xyz = vox2mm(
                np.vstack(np.where(self.masker.mask_img.get_fdata())).T,
                self.masker.mask_img.affine,
            )

            n_cores = self._check_ncores(n_cores)

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
            conn = np.zeros((3, 3, 3), int)
            conn[:, :, 1] = 1
            conn[:, 1, :] = 1
            conn[1, :, :] = 1

            with tqdm_joblib(tqdm(total=n_iters)):
                perm_results = Parallel(n_jobs=n_cores)(
                    delayed(self._correct_fwe_montecarlo_permutation)(
                        iter_xyzs[i_iter], iter_df=iter_df, conn=conn, voxel_thresh=ss_thresh
                    )
                    for i_iter in range(n_iters)
                )

            fwe_voxel_max, fwe_cluster_size_max, fwe_cluster_mass_max = zip(*perm_results)

            # Cluster-level FWE
            # Extract the summary statistics in voxel-wise (3D) form, threshold, and cluster-label
            thresh_stat_values = self.masker.inverse_transform(stat_values).get_fdata()
            thresh_stat_values[thresh_stat_values <= ss_thresh] = 0
            labeled_matrix, _ = ndimage.measurements.label(thresh_stat_values, conn)

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
                self.masker.transform(nib.Nifti1Image(p_cmfwe_map, self.masker.mask_img.affine))
            )
            logp_cmfwe_values = -np.log10(p_cmfwe_values)
            logp_cmfwe_values[np.isinf(logp_cmfwe_values)] = -np.log10(np.finfo(float).eps)
            z_cmfwe_values = p_to_z(p_cmfwe_values, tail="one")

            # Cluster size-based inference
            cluster_sizes[0] = 0  # replace background's "cluster size" with zeros
            p_csfwe_vals = null_to_p(cluster_sizes, fwe_cluster_size_max, "upper")
            p_csfwe_map = p_csfwe_vals[np.reshape(idx, labeled_matrix.shape)]

            p_csfwe_values = np.squeeze(
                self.masker.transform(nib.Nifti1Image(p_csfwe_map, self.masker.mask_img.affine))
            )
            logp_csfwe_values = -np.log10(p_csfwe_values)
            logp_csfwe_values[np.isinf(logp_csfwe_values)] = -np.log10(np.finfo(float).eps)
            z_csfwe_values = p_to_z(p_csfwe_values, tail="one")

            # Voxel-level FWE
            LGR.info("Using null distribution for voxel-level FWE correction.")
            p_vfwe_values = null_to_p(stat_values, fwe_voxel_max, tail="upper")
            self.null_distributions_[
                "values_level-voxel_corr-fwe_method-montecarlo"
            ] = fwe_voxel_max
            self.null_distributions_[
                "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
            ] = fwe_cluster_size_max
            self.null_distributions_[
                "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
            ] = fwe_cluster_mass_max

        z_vfwe_values = p_to_z(p_vfwe_values, tail="one")
        logp_vfwe_values = -np.log10(p_vfwe_values)
        logp_vfwe_values[np.isinf(logp_vfwe_values)] = -np.log10(np.finfo(float).eps)

        if vfwe_only:
            # Return unthresholded value images
            images = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
            }

        else:
            # Return unthresholded value images
            images = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
                "logp_desc-size_level-cluster": logp_csfwe_values,
                "z_desc-size_level-cluster": z_csfwe_values,
                "logp_desc-mass_level-cluster": logp_cmfwe_values,
                "z_desc-mass_level-cluster": z_cmfwe_values,
            }

        return images


class PairwiseCBMAEstimator(CBMAEstimator):
    """Base class for pairwise coordinate-based meta-analysis methods.

    .. versionchanged:: 0.0.8

        * [REF] Use saved MA maps, when available.

    .. versionadded:: 0.0.3

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    *args
        Optional arguments to the :obj:`~nimare.base.MetaEstimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`~nimare.base.MetaEstimator`
        __init__ (called automatically).
    """

    def fit(self, dataset1, dataset2, drop_invalid=True):
        """
        Fit Estimator to two Datasets.

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
        # grab and override
        self._validate_input(dataset1, drop_invalid=drop_invalid)
        self._preprocess_input(dataset1)
        if "ma_maps" in self.inputs_.keys():
            # Grab pre-generated MA maps
            self.inputs_["ma_maps1"] = self.inputs_.pop("ma_maps")

        self.inputs_["coordinates1"] = self.inputs_.pop("coordinates")

        # grab and override
        self._validate_input(dataset2, drop_invalid=drop_invalid)
        self._preprocess_input(dataset2)
        if "ma_maps" in self.inputs_.keys():
            # Grab pre-generated MA maps
            self.inputs_["ma_maps2"] = self.inputs_.pop("ma_maps")

        self.inputs_["coordinates2"] = self.inputs_.pop("coordinates")

        maps = self._fit(dataset1, dataset2)

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset1.masker
        self.results = MetaResult(self, masker, maps)
        return self.results
