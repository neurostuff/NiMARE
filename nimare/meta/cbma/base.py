"""CBMA methods from the ALE and MKDA families."""
import logging
import multiprocessing as mp

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm.auto import tqdm

from ...base import MetaEstimator
from ...results import MetaResult
from ...stats import null_to_p, nullhist_to_p
from ...transforms import p_to_z
from ...utils import add_metadata_to_dataframe, check_type, use_memmap
from ..kernel import KernelTransformer

LGR = logging.getLogger(__name__)


class CBMAEstimator(MetaEstimator):
    """Base class for coordinate-based meta-analysis methods.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    *args
        Optional arguments to the :obj:`nimare.base.MetaEstimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`nimare.base.MetaEstimator`
        __init__ (called automatically).
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(self, kernel_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get kernel transformer
        kernel_args = {
            k.split("kernel__")[1]: v for k, v in kwargs.items() if k.startswith("kernel__")
        }
        kernel_transformer = check_type(kernel_transformer, KernelTransformer, **kernel_args)
        self.kernel_transformer = kernel_transformer

    @use_memmap(LGR, n_files=1)
    def _fit(self, dataset):
        """
        Perform coordinate-based meta-analysis on dataset.

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
            self.inputs_["coordinates"] = add_metadata_to_dataframe(
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
            When the Estimator is set with ``low_memory = True``, there is a ``memmap_filenames``
            attribute that is a list of filenames or Nones. This parameter specifies which item
            in that list should be used for a memory-mapped array.

        Returns
        -------
        ma_maps : :obj:`numpy.ndarray`
            2D numpy array of shape (n_studies, n_voxels) with MA values.
        """
        if maps_key in self.inputs_.keys():
            LGR.debug(f"Loading pre-generated MA maps ({maps_key}).")
            if self.low_memory:
                temp = self.masker.transform(self.inputs_[maps_key][0])
                unmasked_shape = (len(self.inputs_[maps_key]), temp.size)
                ma_maps = np.memmap(
                    self.memmap_filenames[fname_idx],
                    dtype=temp.dtype,
                    mode="w+",
                    shape=unmasked_shape,
                )
                for i, f in enumerate(self.inputs_[maps_key]):
                    ma_maps[i, :] = self.masker.transform(f)
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
        """Compute OF scores from data.

        Parameters
        ----------
        data : array, pandas.DataFrame, or list of img_like
            Data from which to estimate ALE scores.
            The data can be:
            (1) a 1d contrast-len or 2d contrast-by-voxel array of MA values,
            (2) a DataFrame containing coordinates to produce MA values,
            or (3) a list of imgs containing MA values.

        Returns
        -------
        stat_values : 1d array
            OF values. One value per voxel.
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
            raise ValueError('Unsupported data type "{}"'.format(type(data)))

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
            Contrast by voxel array of MA values, after weighting with
            weight_vec.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute:
        "values_corr-none_method-reducedMontecarlo".
        """
        n_studies, n_voxels = ma_maps.shape
        null_ijk = np.random.choice(np.arange(n_voxels), (n_iters, n_studies))
        iter_ma_values = ma_maps[np.arange(n_studies), tuple(null_ijk)].T
        null_dist = self.compute_summarystat(iter_ma_values)
        self.null_distributions_["values_corr-none_method-reducedMontecarlo"] = null_dist

    def _compute_null_montecarlo_permutation(self, params):
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
        iter_ijk, iter_df = params

        iter_ijk = np.squeeze(iter_ijk)
        iter_df[["i", "j", "k"]] = iter_ijk

        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="array"
        )
        iter_ss_map = self.compute_summarystat(iter_ma_maps)

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
            null_ijk.shape[0], size=(self.inputs_["coordinates"].shape[0], n_iters)
        )
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.inputs_["coordinates"].copy()
        iter_dfs = [iter_df] * n_iters

        params = zip(iter_ijks, iter_dfs)
        if n_cores == 1:
            perm_histograms = []
            for pp in tqdm(params, total=n_iters):
                perm_histograms.append(self._compute_null_montecarlo_permutation(pp))

        else:
            with mp.Pool(n_cores) as p:
                perm_histograms = list(
                    tqdm(p.imap(self._compute_null_montecarlo_permutation, params), total=n_iters)
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

    def _correct_fwe_montecarlo_permutation(self, params):
        """Run a single Monte Carlo permutation of a dataset.

        Does the shared work between vFWE and cFWE.

        Parameters
        ----------
        params : tuple
            A tuple containing 4 elements, respectively providing (1) the permuted
            coordinates; (2) the original coordinate DataFrame; (3) a 3d structuring
            array passed to ndimage.label; and (4) the voxel-wise intensity threshold.

        Returns
        -------
        (iter_max value, iter_max_cluster)
            A 2-tuple of floats giving the maximum voxel-wise value, and maximum
            cluster size for the permuted dataset.
        """
        iter_ijk, iter_df, conn, voxel_thresh = params

        iter_ijk = np.squeeze(iter_ijk)
        iter_df[["i", "j", "k"]] = iter_ijk

        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="array"
        )
        iter_ss_map = self.compute_summarystat(iter_ma_maps)
        iter_max_value = np.max(iter_ss_map)
        iter_ss_map = self.masker.inverse_transform(iter_ss_map).get_fdata().copy()
        iter_ss_map[iter_ss_map <= voxel_thresh] = 0

        labeled_matrix = ndimage.measurements.label(iter_ss_map, conn)[0]
        _, clust_sizes = np.unique(labeled_matrix, return_counts=True)
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        if clust_sizes.size:
            iter_max_cluster = np.max(clust_sizes)
        else:
            iter_max_cluster = 0
        return iter_max_value, iter_max_cluster

    def correct_fwe_montecarlo(
        self, result, voxel_thresh=0.001, n_iters=10000, n_cores=-1, vfwe_only=False
    ):
        """Perform FWE correction using the max-value permutation method.

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        voxel_thresh : :obj:`float`, optional
            Cluster-defining p-value threshold. Default is 0.001.
        n_iters : :obj:`int`, optional
            Number of iterations to build the vFWE and cFWE null distributions.
            Default is 10000.
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
            null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T

            n_cores = self._check_ncores(n_cores)

            # Identify summary statistic corresponding to intensity threshold
            ss_thresh = self._p_to_summarystat(voxel_thresh)

            rand_idx = np.random.choice(
                null_ijk.shape[0], size=(self.inputs_["coordinates"].shape[0], n_iters)
            )
            rand_ijk = null_ijk[rand_idx, :]
            iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
            iter_df = self.inputs_["coordinates"].copy()
            iter_dfs = [iter_df] * n_iters

            # Find number of voxels per cluster (includes 0, which is empty space in
            # the matrix)
            conn = np.zeros((3, 3, 3), int)
            conn[:, :, 1] = 1
            conn[:, 1, :] = 1
            conn[1, :, :] = 1

            # Define parameters
            iter_conn = [conn] * n_iters
            iter_ss_thresh = [ss_thresh] * n_iters
            params = zip(iter_ijks, iter_dfs, iter_conn, iter_ss_thresh)

            if n_cores == 1:
                perm_results = []
                for pp in tqdm(params, total=n_iters):
                    perm_results.append(self._correct_fwe_montecarlo_permutation(pp))

            else:
                with mp.Pool(n_cores) as p:
                    perm_results = list(
                        tqdm(
                            p.imap(self._correct_fwe_montecarlo_permutation, params), total=n_iters
                        )
                    )

            fwe_voxel_max, fwe_clust_max = zip(*perm_results)

            # Cluster-level FWE
            thresh_stat_values = self.masker.inverse_transform(stat_values).get_fdata()
            thresh_stat_values[thresh_stat_values <= ss_thresh] = 0
            labeled_matrix, _ = ndimage.measurements.label(thresh_stat_values, conn)

            _, idx, sizes = np.unique(labeled_matrix, return_inverse=True, return_counts=True)
            # first cluster has value 0 (i.e., all non-zero voxels in brain), so replace
            # with 0, which gives us a p-value of 1.
            sizes[0] = 0
            p_vals = null_to_p(sizes, fwe_clust_max, "upper")
            p_cfwe_map = p_vals[np.reshape(idx, labeled_matrix.shape)]

            p_cfwe_values = np.squeeze(
                self.masker.transform(nib.Nifti1Image(p_cfwe_map, self.masker.mask_img.affine))
            )
            logp_cfwe_values = -np.log10(p_cfwe_values)
            logp_cfwe_values[np.isinf(logp_cfwe_values)] = -np.log10(np.finfo(float).eps)
            z_cfwe_values = p_to_z(p_cfwe_values, tail="one")

            # Voxel-level FWE
            LGR.info("Using null distribution for voxel-level FWE correction.")
            p_vfwe_values = null_to_p(stat_values, fwe_voxel_max, tail="upper")
            self.null_distributions_[
                "values_level-voxel_corr-fwe_method-montecarlo"
            ] = fwe_voxel_max
            self.null_distributions_[
                "values_level-cluster_corr-fwe_method-montecarlo"
            ] = fwe_clust_max

        z_vfwe_values = p_to_z(p_vfwe_values, tail="one")
        logp_vfwe_values = -np.log10(p_vfwe_values)
        logp_vfwe_values[np.isinf(logp_vfwe_values)] = -np.log10(np.finfo(float).eps)

        if vfwe_only:
            # Write out unthresholded value images
            images = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
            }

        else:
            # Write out unthresholded value images
            images = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
                "logp_level-cluster": logp_cfwe_values,
                "z_level-cluster": z_cfwe_values,
            }

        return images


class PairwiseCBMAEstimator(CBMAEstimator):
    """Base class for pairwise coordinate-based meta-analysis methods.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    *args
        Optional arguments to the :obj:`nimare.base.MetaEstimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`nimare.base.MetaEstimator`
        __init__ (called automatically).
    """

    def fit(self, dataset1, dataset2):
        """
        Fit Estimator to two Datasets.

        Parameters
        ----------
        dataset1/dataset2 : :obj:`nimare.dataset.Dataset`
            Dataset objects to analyze.

        Returns
        -------
        :obj:`nimare.results.MetaResult`
            Results of Estimator fitting.

        Notes
        -----
        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Estimators' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.
        """
        self._validate_input(dataset1)
        self._validate_input(dataset2)

        # grab and override
        self._preprocess_input(dataset1)
        if "ma_maps" in self.inputs_.keys():
            # Grab pre-generated MA maps
            self.inputs_["ma_maps1"] = self.inputs_.pop("ma_maps")

        self.inputs_["coordinates1"] = self.inputs_.pop("coordinates")

        # grab and override
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
