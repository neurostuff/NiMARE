"""CBMA methods from the ALE and MKDA families."""
import logging
import multiprocessing as mp
import inspect

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm.auto import tqdm

from ...base import MetaEstimator
from ...results import MetaResult
from ...stats import null_to_p, nullhist_to_p
from ...transforms import p_to_z

LGR = logging.getLogger(__name__)


class CBMAEstimator(MetaEstimator):
    """Base class for coordinate-based meta-analysis methods.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
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

        # Allow both instances and classes for the kernel transformer input.
        from ..kernel import KernelTransformer

        if not issubclass(type(kernel_transformer), KernelTransformer) and not issubclass(
            kernel_transformer, KernelTransformer
        ):
            raise ValueError(
                'Argument "kernel_transformer" must be a kind of ' "KernelTransformer"
            )
        elif not inspect.isclass(kernel_transformer) and kernel_args:
            LGR.warning(
                'Argument "kernel_transformer" has already been '
                "initialized, so kernel arguments will be ignored: "
                "{}".format(", ".join(kernel_args.keys()))
            )
        elif inspect.isclass(kernel_transformer):
            kernel_transformer = kernel_transformer(**kernel_args)
        self.kernel_transformer = kernel_transformer

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

        ma_values = self.kernel_transformer.transform(
            self.inputs_["coordinates"], masker=self.masker, return_type="array"
        )

        self.weight_vec_ = self._compute_weights(ma_values)

        stat_values = self.compute_summarystat(ma_values)

        # Determine null distributions for summary stat (OF) to p conversion
        if self.null_method.startswith("analytic"):
            self._compute_null_analytic(ma_values)
        else:
            self._compute_null_empirical(ma_values, n_iters=self.n_iters)
        p_values, z_values = self._summarystat_to_p(stat_values, null_method=self.null_method)

        images = {"stat": stat_values, "p": p_values, "z": z_values}
        return images

    def _compute_weights(self, ma_values):
        """Optional weight computation routine. Takes an array of meta-analysis
        values as input and returns an array of the same shape, weighted as
        desired. Can be ignored by algorithms that don't support weighting."""
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
            if "sample_sizes" in dataset.get_metadata():
                # Extract sample sizes and make DataFrame
                sample_sizes = dataset.get_metadata(field="sample_sizes", ids=dataset.ids)
                # we need an extra layer of lists
                sample_sizes = [[ss] for ss in sample_sizes]
                sample_sizes = pd.DataFrame(
                    index=dataset.ids, data=sample_sizes, columns=["sample_sizes"]
                )
                sample_sizes["sample_size"] = sample_sizes["sample_sizes"].apply(np.mean)
                # Merge sample sizes df into coordinates df
                self.inputs_["coordinates"] = self.inputs_["coordinates"].merge(
                    right=sample_sizes,
                    left_on="id",
                    right_index=True,
                    sort=False,
                    validate="many_to_one",
                    suffixes=(False, False),
                    how="left",
                )
            else:
                LGR.warning(
                    'Metadata field "sample_sizes" not found. '
                    "Set a constant sample size as a kernel transformer "
                    "argument, if possible."
                )

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
        """Core summary statistic computation logic. Must be overriden by
        subclasses. Input and output are both numpy arrays; the output must
        aggregate over the 0th dimension of the input. (I.e., if the input
        has K dimensions, the output has K - 1 dimensions.)"""
        pass

    def _compute_null_empirical(self, ma_maps, n_iters=10000):
        """Compute uncorrected null distribution using empirical method.

        Parameters
        ----------
        ma_maps : (C x V) array
            Contrast by voxel array of MA values, after weighting with
            weight_vec.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute:
        "empirical_null".
        """
        n_studies, n_voxels = ma_maps.shape
        null_ijk = np.random.choice(np.arange(n_voxels), (n_iters, n_studies))
        iter_ma_values = ma_maps[np.arange(n_studies), tuple(null_ijk)].T
        null_dist = self.compute_summarystat(iter_ma_values)
        self.null_distributions_["empirical_null"] = null_dist

    def _summarystat_to_p(self, stat_values, null_method="analytic"):
        """Compute p- and z-values from summary statistics (e.g., ALE scores).

        Uses either histograms from analytic null or null distribution from empirical null.

        Parameters
        ----------
        stat_values : 1D array_like
            Array of summary statistic values from estimator.
        null_method : {"analytic", "empirical"}, optional
            Whether to use analytic null or empirical null.
            Default is "analytic".

        Returns
        -------
        p_values, z_values : 1D array
            P- and Z-values for statistic values.
            Same shape as stat_values.
        """
        if null_method.startswith("analytic"):
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histogram_weights" in self.null_distributions_.keys()

            p_values = nullhist_to_p(
                stat_values,
                self.null_distributions_["histogram_weights"],
                self.null_distributions_["histogram_bins"],
            )
        elif null_method == "empirical":
            assert "empirical_null" in self.null_distributions_.keys()
            p_values = null_to_p(
                stat_values, self.null_distributions_["empirical_null"], tail="upper"
            )
        else:
            raise ValueError("Argument 'null_method' must be one of: 'analytic', 'empirical'.")

        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values

    def _p_to_summarystat(self, p, null_method=None):
        """Compute a summary statistic threshold that corresponds to the provided p-value.

        Uses either histograms from analytic null or null distribution from empirical null.

        Parameters
        ----------
        p : The p-value that corresponds to the summary statistic threshold
        null_method : {None, "analytic", "empirical"}, optional
            Whether to use analytic null or empirical null. If None, defaults to using
            whichever method was set at initialization.

        Returns
        -------
        ss : float
            A float giving the summary statistic value corresponding to the passed p.
        """
        if null_method is None:
            null_method = self.null_method

        if null_method.startswith("analytic"):
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histogram_weights" in self.null_distributions_.keys()

            hist_weights = self.null_distributions_["histogram_weights"]
            # Desired bin is the first one _before_ the target p-value (for consistency
            # with the empirical null).
            ss_idx = np.maximum(0, np.where(hist_weights <= p)[0][0] - 1)
            ss = self.null_distributions_["histogram_bins"][ss_idx]

        elif null_method == "empirical":
            assert "empirical_null" in self.null_distributions_.keys()
            null_dist = np.sort(self.null_distributions_["empirical_null"])
            n_vals = len(null_dist)
            ss_idx = np.floor(p * n_vals).astype(int)
            ss = null_dist[-ss_idx]
        else:
            raise ValueError("Argument 'null_method' must be one of: 'analytic', 'empirical'.")

        return ss

    def _run_fwe_permutation(self, params):
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
        u, clust_sizes = np.unique(labeled_matrix, return_counts=True)
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        if clust_sizes.size:
            iter_max_cluster = np.max(clust_sizes)
        else:
            iter_max_cluster = 0
        return iter_max_value, iter_max_cluster

    def correct_fwe_montecarlo(self, result, voxel_thresh=0.001, n_iters=10000, n_cores=-1):
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

        # Find number of voxels per cluster (includes 0, which is empty space in
        # the matrix)
        conn = np.zeros((3, 3, 3), int)
        conn[:, :, 1] = 1
        conn[:, 1, :] = 1
        conn[1, :, :] = 1

        # Define parameters
        iter_conn = [conn] * n_iters
        iter_dfs = [iter_df] * n_iters
        iter_ss_thresh = [ss_thresh] * n_iters
        params = zip(iter_ijks, iter_dfs, iter_conn, iter_ss_thresh)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params), total=n_iters))

        fwe_voxel_max, fwe_clust_max = zip(*perm_results)

        # Cluster-level FWE
        thresh_stat_values = self.masker.inverse_transform(stat_values).get_fdata()
        thresh_stat_values[thresh_stat_values <= ss_thresh] = 0
        labeled_matrix, n_clusters = ndimage.measurements.label(thresh_stat_values, conn)

        u, idx, sizes = np.unique(labeled_matrix, return_inverse=True, return_counts=True)
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
        p_vfwe_values = null_to_p(stat_values, fwe_voxel_max, tail="upper")

        self.null_distributions_["fwe_level-voxel_method-montecarlo"] = fwe_voxel_max
        self.null_distributions_["fwe_level-cluster_method-montecarlo"] = fwe_clust_max

        z_vfwe_values = p_to_z(p_vfwe_values, tail="one")
        logp_vfwe_values = -np.log10(p_vfwe_values)
        logp_vfwe_values[np.isinf(logp_vfwe_values)] = -np.log10(np.finfo(float).eps)

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
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
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
        self.inputs_["coordinates1"] = self.inputs_.pop("coordinates")
        # grab and override
        self._preprocess_input(dataset2)
        self.inputs_["coordinates2"] = self.inputs_.pop("coordinates")

        maps = self._fit(dataset1, dataset2)

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset1.masker
        self.results = MetaResult(self, masker, maps)
        return self.results
