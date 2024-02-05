"""CBMA methods from the multilevel kernel density analysis (MKDA) family."""

import logging

import nibabel as nib
import numpy as np
import sparse
from joblib import Memory, Parallel, delayed
from pymare.stats import fdr
from scipy import ndimage
from scipy.stats import chi2
from tqdm.auto import tqdm

from nimare import _version
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.meta.kernel import KDAKernel, MKDAKernel
from nimare.meta.utils import _calculate_cluster_measures
from nimare.stats import null_to_p, one_way, two_way
from nimare.transforms import p_to_z
from nimare.utils import _check_ncores, vox2mm

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


class MKDADensity(CBMAEstimator):
    r"""Multilevel kernel density analysis- Density analysis.

    The MKDA density method was originally introduced in :footcite:t:`wager2007meta`.

    .. versionchanged:: 0.2.1

        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.0.12

        - Use a 4D sparse array for modeled activation maps.

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`~nimare.meta.kernel.MKDAKernel`.
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

    n_iters : int, default=5000
        Number of iterations to use to define the null distribution.
        This is only used if ``null_method=="montecarlo"``.
        Default is 5000.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        This is only used if ``null_method=="montecarlo"``.
        If <=0, defaults to using all available cores.
        Default is 1.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

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

            -   ``histogram_means``: Array of mean value per experiment.
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
    The MKDA density algorithm is also implemented in MATLAB at
    https://github.com/canlab/Canlab_MKDA_MetaAnalysis.

    Available correction methods: :func:`MKDADensity.correct_fwe_montecarlo`

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        kernel_transformer=MKDAKernel,
        null_method="approximate",
        n_iters=5000,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        n_cores=1,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, MKDAKernel) or kernel_transformer == MKDAKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(
            kernel_transformer=kernel_transformer,
            memory=memory,
            memory_level=memory_level,
            **kwargs,
        )
        self.null_method = null_method
        self.n_iters = None if null_method == "approximate" else n_iters or 5000
        self.n_cores = _check_ncores(n_cores)
        self.dataset = None

    def _generate_description(self):
        """Generate a description of the fitted Estimator.

        Returns
        -------
        str
            Description of the Estimator.
        """
        if self.null_method == "montecarlo":
            null_method_str = (
                "a Monte Carlo-based null distribution, in which dataset coordinates were "
                "randomly drawn from the analysis mask and the full set of ALE values were "
                f"retained, using {self.n_iters} iterations"
            )
        else:
            null_method_str = "an approximate null distribution"

        description = (
            "A multilevel kernel density (MKDA) meta-analysis \\citep{wager2007meta} was "
            "performed was performed with NiMARE "
            f"{__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), using a(n) "
            f"{self.kernel_transformer.__class__.__name__.replace('Kernel', '')} kernel. "
            f"{self.kernel_transformer._generate_description()} "
            f"Summary statistics (OF values) were converted to p-values using {null_method_str}. "
            f"The input dataset included {self.inputs_['coordinates'].shape[0]} foci from "
            f"{len(self.inputs_['id'])} experiments."
        )
        return description

    def _compute_weights(self, ma_values):
        """Determine experiment-wise weights per the conventional MKDA approach."""
        # TODO: Incorporate sample-size and inference metadata extraction and
        # merging into df.
        # This will need to be distinct from the kernel_transformer-based kind
        # done in CBMAEstimator._preprocess_input
        ids_df = self.inputs_["coordinates"].groupby("id").first()

        n_exp = len(ids_df)

        # Default to unit weighting for missing inference or sample size
        if "inference" not in ids_df.columns:
            ids_df["inference"] = "rfx"
        if "sample_size" not in ids_df.columns:
            ids_df["sample_size"] = 1.0

        n = ids_df["sample_size"].astype(float).values
        inf = ids_df["inference"].map({"ffx": 0.75, "rfx": 1.0}).values

        weight_vec = n_exp * ((np.sqrt(n) * inf) / np.sum(np.sqrt(n) * inf))
        weight_vec = weight_vec[:, None]

        assert weight_vec.shape[0] == ma_values.shape[0]
        return weight_vec

    def _compute_summarystat_est(self, ma_values):
        ma_values = ma_values.reshape((ma_values.shape[0], -1))
        stat_values = ma_values.T.dot(self.weight_vec_)

        if isinstance(ma_values, sparse._coo.core.COO):
            # NOTE: This may not work correctly with a non-NiftiMasker.
            mask_data = self.masker.mask_img.get_fdata().astype(bool)

            stat_values = stat_values[mask_data.reshape(-1)].ravel()
            # This is used by _compute_null_approximate
            self.__n_mask_voxels = stat_values.shape[0]
        else:
            # np.array type is used by _compute_null_reduced_montecarlo
            stat_values = stat_values.ravel()

        return stat_values

    def _determine_histogram_bins(self, ma_maps):
        """Determine histogram bins for null distribution methods.

        Parameters
        ----------
        ma_maps : :obj:`sparse._coo.core.COO`
            MA maps.
            The ma_maps can be a 4d sparse array of MA maps,

        Notes
        -----
        This method adds two entries to the null_distributions_ dict attribute: "histogram_bins",
        and "histogram_means" only if ``null_method == "approximate"``.
        """
        if not isinstance(ma_maps, sparse._coo.core.COO):
            raise ValueError(f"Unsupported data type '{type(ma_maps)}'")

        n_exp = ma_maps.shape[0]
        prop_active = np.zeros(n_exp)
        data = ma_maps.data
        coords = ma_maps.coords
        for exp_idx in range(n_exp):
            # The first column of coords is the fourth dimension of the dense array
            study_ma_values = data[coords[0, :] == exp_idx]

            n_nonzero_voxels = study_ma_values.shape[0]
            n_zero_voxels = self.__n_mask_voxels - n_nonzero_voxels

            prop_active[exp_idx] = np.mean(np.hstack([study_ma_values, np.zeros(n_zero_voxels)]))

        self.null_distributions_["histogram_bins"] = np.arange(len(prop_active) + 1, step=1)

        if self.null_method.startswith("approximate"):
            # To speed things up in _compute_null_approximate, we save the means too,
            self.null_distributions_["histogram_means"] = prop_active

    def _compute_null_approximate(self, ma_maps):
        """Compute uncorrected null distribution using approximate solution.

        Parameters
        ----------
        ma_maps
            Modeled activation maps. Unused for this estimator.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute: "histogram_weights".
        """
        assert "histogram_means" in self.null_distributions_.keys()

        # MKDA maps are binary, so we only have k + 1 bins in the final
        # histogram, where k is the number of studies. We can analytically
        # compute the null distribution by convolution.
        # prop_active contains the mean value per experiment
        prop_active = self.null_distributions_["histogram_means"]

        ss_hist = 1.0
        for exp_prop in prop_active:
            ss_hist = np.convolve(ss_hist, [1 - exp_prop, exp_prop])

        self.null_distributions_["histweights_corr-none_method-approximate"] = ss_hist


class MKDAChi2(PairwiseCBMAEstimator):
    r"""Multilevel kernel density analysis- Chi-square analysis.

    The MKDA chi-square method was originally introduced in :footcite:t:`wager2007meta`.

    .. versionchanged:: 0.2.1

        - Make `prior` parameter default to None, which controls if posterior probabilities
            pFgA, pAgF_prior and pFgA_prior are calculated. This is useful because probability
            maps are difficult to interpret and for speeding up the algorithm.
        - Rename ``consistency`` to ``uniformity`` and ``specificity`` to ``association`` to match
          Neurosynth's terminology
        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.0.12

        - Use a 4D sparse array for modeled activation maps.

    .. versionchanged:: 0.0.8

        * [REF] Use saved MA maps, when available.

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`~nimare.meta.kernel.MKDAKernel`.
    prior : float, default=0.5
        Uniform prior probability of each feature being active in a map in
        the absence of evidence from the map. Default: 0.5
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key: coordinates.
        This is an edited version of the dataset's coordinates DataFrame.
    null_distributions_ : :obj:`dict` of :class:`numpy.ndarray`
        Null distributions for any multiple-comparisons correction methods.

        .. important::
            MKDAChi2 does not retain uncorrected summary-statistic-to-p null distributions,
            since the summary statistic in this case is the chi-squared value, which has an
            established null distribution.

        Entries are added to this attribute if and when the corresponding method is applied.

        If :meth:`correct_fwe_montecarlo` is applied:

            -   ``values_desc-pAgF_level-voxel_corr-fwe_method-montecarlo``: The maximum
                chi-squared value from the p(A|F) one-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pAgFsize_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster size value from the p(A|F) one-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pAgFmass_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster mass value from the p(A|F) one-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pFgA_level-voxel_corr-fwe_method-montecarlo``: The maximum
                chi-squared value from the p(F|A) two-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pFgAsize_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster size value from the p(F|A) two-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pFgAmass_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster mass value from the p(F|A) two-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).

    Notes
    -----
    The MKDA Chi-square algorithm was originally implemented as part of the Neurosynth Python
    library (https://github.com/neurosynth/neurosynth).

    Available correction methods: :meth:`MKDAChi2.correct_fwe_montecarlo`,
    :meth:`MKDAChi2.correct_fdr_indep`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        kernel_transformer=MKDAKernel(),
        prior=0.5,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, MKDAKernel) or kernel_transformer == MKDAKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(
            kernel_transformer=kernel_transformer,
            memory=memory,
            memory_level=memory_level,
            **kwargs,
        )

        self.prior = prior

    def _generate_description(self):
        description = (
            "A multilevel kernel density chi-squared analysis \\citep{wager2007meta} was "
            "performed according to the same procedure as implemented in Neurosynth with NiMARE "
            f"{__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), "
            f"using a(n) {self.kernel_transformer.__class__.__name__.replace('Kernel', '')} "
            "kernel. "
            f"{self.kernel_transformer._generate_description()} "
            "This analysis calculated several measures. "
            "The first dataset was evaluated for uniformity of activation via a one-way "
            "chi-square test. "
            f"The first input dataset included {self.inputs_['coordinates1'].shape[0]} foci from "
            f"{len(self.inputs_['id1'])} experiments. "
            f"The second input dataset included {self.inputs_['coordinates2'].shape[0]} foci from "
            f"{len(self.inputs_['id2'])} experiments."
        )

        return description

    def _fit(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.masker = self.masker or dataset1.masker
        self.null_distributions_ = {}

        # Generate MA maps and calculate count variables for first dataset
        n_selected_active_voxels = self._collect_ma_maps(
            maps_key="ma_maps1",
            coords_key="coordinates1",
            return_type="summary_array",
        )

        n_selected = self.dataset1.coordinates["id"].unique().shape[0]

        # Generate MA maps and calculate count variables for second dataset
        n_unselected_active_voxels = self._collect_ma_maps(
            maps_key="ma_maps2",
            coords_key="coordinates2",
            return_type="summary_array",
        )
        n_unselected = self.dataset2.coordinates["id"].unique().shape[0]

        n_mappables = n_selected + n_unselected

        # Nomenclature for variables below: p = probability,
        # F = feature present, g = given, U = unselected, A = activation.
        # So, e.g., pAgF = p(A|F) = probability of activation
        # in a voxel if we know that the feature is present in a study.
        pF = n_selected / n_mappables
        pA = np.array(
            (n_selected_active_voxels + n_unselected_active_voxels) / n_mappables
        ).squeeze()

        del n_mappables

        pAgF = n_selected_active_voxels / n_selected
        pAgU = n_unselected_active_voxels / n_unselected
        pFgA = pAgF * pF / pA

        del pF

        if self.prior:
            # Recompute conditionals with uniform prior
            pAgF_prior = self.prior * pAgF + (1 - self.prior) * pAgU
            pFgA_prior = pAgF * self.prior / pAgF_prior

        # One-way chi-square test for uniformity of activation
        pAgF_chi2_vals = one_way(np.squeeze(n_selected_active_voxels), n_selected)
        pAgF_p_vals = chi2.sf(pAgF_chi2_vals, 1)
        pAgF_sign = np.sign(n_selected_active_voxels - np.mean(n_selected_active_voxels))
        pAgF_z = p_to_z(pAgF_p_vals, tail="two") * pAgF_sign

        del pAgF_sign

        # Two-way chi-square for association of activation
        cells = np.squeeze(
            np.array(
                [
                    [n_selected_active_voxels, n_unselected_active_voxels],
                    [
                        n_selected - n_selected_active_voxels,
                        n_unselected - n_unselected_active_voxels,
                    ],
                ]
            ).T
        )

        del n_selected, n_unselected

        pFgA_chi2_vals = two_way(cells)

        del n_selected_active_voxels, n_unselected_active_voxels

        eps = np.spacing(1)
        pFgA_p_vals = chi2.sf(pFgA_chi2_vals, 1)
        pFgA_p_vals[pFgA_p_vals < eps] = eps
        pFgA_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(pFgA_p_vals, tail="two") * pFgA_sign

        del pFgA_sign, pAgU

        maps = {
            "z_desc-uniformity": pAgF_z,
            "z_desc-association": pFgA_z,
            "chi2_desc-uniformity": pAgF_chi2_vals,
            "chi2_desc-association": pFgA_chi2_vals,
            "p_desc-uniformity": pAgF_p_vals,
            "p_desc-association": pFgA_p_vals,
            "prob_desc-A": pA,
            "prob_desc-AgF": pAgF,
            "prob_desc-FgA": pFgA,
        }

        if self.prior:
            maps["prob_desc-AgF_prior"] = pAgF_prior
            maps["prob_desc-FgA_prior"] = pFgA_prior

        description = self._generate_description()
        return maps, {}, description

    def _run_fwe_permutation(self, iter_xyz1, iter_xyz2, iter_df1, iter_df2, conn, voxel_thresh):
        """Run a single permutation of the Monte Carlo FWE correction procedure.

        Parameters
        ----------
        iter_xyz1, iter_xyz2 : :obj:`numpy.ndarray`
            Random coordinates for the permutation.
        iter_df1, iter_df2 : :obj:`pandas.DataFrame`
            DataFrames with as many rows as there are coordinates in each of the two datasets,
            to be filled in with random coordinates for the permutation.
        conn : :obj:`numpy.ndarray` of shape (3, 3, 3)
            Connectivity matrix for defining clusters.
        voxel_thresh : :obj:`float`
            Uncorrected summary-statistic thresholded for defining clusters.

        Returns
        -------
        pAgF_max_chi2_value : :obj:`float`
            Forward inference maximum chi-squared value, for voxel-level FWE correction.
        pAgF_max_size : :obj:`float`
            Forward inference maximum cluster size, for cluster-level FWE correction.
        pAgF_max_mass : :obj:`float`
            Forward inference maximum cluster mass, for cluster-level FWE correction.
        pFgA_max_chi2_value : :obj:`float`
            Reverse inference maximum chi-squared value, for voxel-level FWE correction.
        pFgA_max_size : :obj:`float`
            Reverse inference maximum cluster size, for cluster-level FWE correction.
        pFgA_max_mass : :obj:`float`
            Reverse inference maximum cluster mass, for cluster-level FWE correction.
        """
        # Not sure if joblib will automatically use a copy of the object, but I'll make a copy to
        # be safe.
        iter_df1 = iter_df1.copy()
        iter_df2 = iter_df2.copy()

        iter_xyz1 = np.squeeze(iter_xyz1)
        iter_xyz2 = np.squeeze(iter_xyz2)
        iter_df1[["x", "y", "z"]] = iter_xyz1
        iter_df2[["x", "y", "z"]] = iter_xyz2

        # Generate MA maps and calculate count variables for first dataset
        n_selected_active_voxels = self.kernel_transformer.transform(
            iter_df1, self.masker, return_type="summary_array"
        )

        n_selected = self.dataset1.coordinates["id"].unique().shape[0]

        # Generate MA maps and calculate count variables for second dataset
        n_unselected_active_voxels = self.kernel_transformer.transform(
            iter_df2, self.masker, return_type="summary_array"
        )
        n_unselected = self.dataset2.coordinates["id"].unique().shape[0]

        # Currently unused conditional probabilities
        # pAgF = n_selected_active_voxels / n_selected
        # pAgU = n_unselected_active_voxels / n_unselected

        # One-way chi-square test for uniformity of activation
        pAgF_chi2_vals = one_way(np.squeeze(n_selected_active_voxels), n_selected)

        # Voxel-level inference
        pAgF_max_chi2_value = np.max(np.abs(pAgF_chi2_vals))

        # Cluster-level inference
        pAgF_chi2_map = self.masker.inverse_transform(pAgF_chi2_vals).get_fdata()
        pAgF_max_size, pAgF_max_mass = _calculate_cluster_measures(
            pAgF_chi2_map, voxel_thresh, conn, tail="two"
        )

        # Two-way chi-square for association of activation
        cells = np.squeeze(
            np.array(
                [
                    [n_selected_active_voxels, n_unselected_active_voxels],
                    [
                        n_selected - n_selected_active_voxels,
                        n_unselected - n_unselected_active_voxels,
                    ],
                ]
            ).T
        )
        pFgA_chi2_vals = two_way(cells)

        # Voxel-level inference
        pFgA_max_chi2_value = np.max(np.abs(pFgA_chi2_vals))

        # Cluster-level inference
        pFgA_chi2_map = self.masker.inverse_transform(pFgA_chi2_vals).get_fdata()
        pFgA_max_size, pFgA_max_mass = _calculate_cluster_measures(
            pFgA_chi2_map, voxel_thresh, conn, tail="two"
        )

        return (
            pAgF_max_chi2_value,
            pAgF_max_size,
            pAgF_max_mass,
            pFgA_max_chi2_value,
            pFgA_max_size,
            pFgA_max_mass,
        )

    def _apply_correction(self, stat_values, voxel_thresh, vfwe_null, csfwe_null, cmfwe_null):
        """Apply different kinds of FWE correction to statistical value matrix.

        .. versionchanged:: 0.0.13

            Change cluster neighborhood from faces+edges to faces, to match Nilearn.

        Parameters
        ----------
        stat_values : :obj:`numpy.ndarray`
            1D array of summary-statistic values.
        voxel_thresh : :obj:`float`
            Summary statistic threshold for defining clusters.
        vfwe_null, csfwe_null, cmfwe_null : :obj:`numpy.ndarray`
            Null distributions for FWE correction.

        Returns
        -------
        p_vfwe_values, p_csfwe_values, p_cmfwe_values : :obj:`numpy.ndarray`
            1D arrays of FWE-corrected p-values.
        """
        eps = np.spacing(1)

        # Define connectivity matrix for cluster labeling
        conn = ndimage.generate_binary_structure(rank=3, connectivity=1)

        # Voxel-level FWE
        p_vfwe_values = null_to_p(np.abs(stat_values), vfwe_null, tail="upper")

        # Crop p-values of 0 or 1 to nearest values that won't evaluate to 0 or 1.
        # Prevents inf z-values.
        p_vfwe_values[p_vfwe_values < eps] = eps
        p_vfwe_values[p_vfwe_values > (1.0 - eps)] = 1.0 - eps

        # Cluster-level FWE
        # Extract the summary statistics in voxel-wise (3D) form, threshold, and cluster-label
        stat_map_thresh = self.masker.inverse_transform(stat_values).get_fdata()

        stat_map_thresh[np.abs(stat_map_thresh) <= voxel_thresh] = 0

        # Label positive and negative clusters separately
        labeled_matrix = np.empty(stat_map_thresh.shape, int)
        labeled_matrix, _ = ndimage.label(stat_map_thresh > 0, conn)
        n_positive_clusters = np.max(labeled_matrix)
        temp_labeled_matrix, _ = ndimage.label(stat_map_thresh < 0, conn)
        temp_labeled_matrix[temp_labeled_matrix > 0] += n_positive_clusters
        labeled_matrix = labeled_matrix + temp_labeled_matrix
        del temp_labeled_matrix

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

            cluster_mass = np.sum(np.abs(stat_map_thresh[labeled_matrix == i_val]) - voxel_thresh)
            cluster_masses[i_val] = cluster_mass

        p_cmfwe_vals = null_to_p(cluster_masses, cmfwe_null, tail="upper")
        p_cmfwe_map = p_cmfwe_vals[np.reshape(idx, labeled_matrix.shape)]

        p_cmfwe_values = np.squeeze(
            self.masker.transform(nib.Nifti1Image(p_cmfwe_map, self.masker.mask_img.affine))
        )

        # Cluster size-based inference
        cluster_sizes[0] = 0  # replace background's "cluster size" with zeros
        p_csfwe_vals = null_to_p(cluster_sizes, csfwe_null, tail="upper")
        p_csfwe_map = p_csfwe_vals[np.reshape(idx, labeled_matrix.shape)]
        p_csfwe_values = np.squeeze(
            self.masker.transform(nib.Nifti1Image(p_csfwe_map, self.masker.mask_img.affine))
        )

        return p_vfwe_values, p_csfwe_values, p_cmfwe_values

    def correct_fwe_montecarlo(self, result, voxel_thresh=0.001, n_iters=1000, n_cores=1):
        """Perform FWE correction using the max-value permutation method.

        Only call this method from within a Corrector.

        .. versionchanged:: 0.0.13

            Change cluster neighborhood from faces+edges to faces, to match Nilearn.

        .. versionchanged:: 0.0.12

            Include cluster level-corrected results in Monte Carlo null method.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        voxel_thresh : :obj:`float`, default=0.001
            Voxel-level threshold. Default is 0.001.
        n_iters : :obj:`int`, default=1000
            Number of iterations to build the vFWE null distribution.
            Default is 1000.
        n_cores : :obj:`int`, default=1
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.

        Returns
        -------
        maps : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked maps generated by
            the correction procedure. The following arrays are generated by
            this method:

            -   ``p_desc-uniformity_level-voxel``: Voxel-level FWE-corrected p-values from the
                uniformity/forward inference analysis.
            -   ``z_desc-uniformity_level-voxel``: Voxel-level FWE-corrected z-values from the
                uniformity/forward inference analysis.
            -   ``logp_desc-uniformity_level-voxel``: Voxel-level FWE-corrected -log10 p-values
                from the uniformity/forward inference analysis.
            -   ``p_desc-uniformityMass_level-cluster``: Cluster-level FWE-corrected p-values
                from the uniformity/forward inference analysis, using cluster mass.
            -   ``z_desc-uniformityMass_level-cluster``: Cluster-level FWE-corrected z-values
                from the uniformity/forward inference analysis, using cluster mass.
            -   ``logp_desc-uniformityMass_level-cluster``: Cluster-level FWE-corrected -log10
                p-values from the uniformity/forward inference analysis, using cluster mass.
            -   ``p_desc-uniformitySize_level-cluster``: Cluster-level FWE-corrected p-values
                from the uniformity/forward inference analysis, using cluster size.
            -   ``z_desc-uniformitySize_level-cluster``: Cluster-level FWE-corrected z-values
                from the uniformity/forward inference analysis, using cluster size.
            -   ``logp_desc-uniformitySize_level-cluster``: Cluster-level FWE-corrected -log10
                p-values from the uniformity/forward inference analysis, using cluster size.
            -   ``p_desc-association_level-voxel``: Voxel-level FWE-corrected p-values from the
                association/reverse inference analysis.
            -   ``z_desc-association_level-voxel``: Voxel-level FWE-corrected z-values from the
                association/reverse inference analysis.
            -   ``logp_desc-association_level-voxel``: Voxel-level FWE-corrected -log10 p-values
                from the association/reverse inference analysis.
            -   ``p_desc-associationMass_level-cluster``: Cluster-level FWE-corrected p-values
                from the association/reverse inference analysis, using cluster mass.
            -   ``z_desc-associationMass_level-cluster``: Cluster-level FWE-corrected z-values
                from the association/reverse inference analysis, using cluster mass.
            -   ``logp_desc-associationMass_level-cluster``: Cluster-level FWE-corrected -log10
                p-values from the association/reverse inference analysis, using cluster mass.
            -   ``p_desc-associationSize_level-cluster``: Cluster-level FWE-corrected p-values
                from the association/reverse inference analysis, using cluster size.
            -   ``z_desc-associationSize_level-cluster``: Cluster-level FWE-corrected z-values
                from the association/reverse inference analysis, using cluster size.
            -   ``logp_desc-associationSize_level-cluster``: Cluster-level FWE-corrected -log10
                p-values from the association/reverse inference analysis, using cluster size.

        Notes
        -----
        This method adds six new keys to the ``null_distributions_`` attribute:

            -   ``values_desc-pAgF_level-voxel_corr-fwe_method-montecarlo``: The maximum
                chi-squared value from the p(A|F) one-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pAgFsize_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster size value from the p(A|F) one-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pAgFmass_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster mass value from the p(A|F) one-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pFgA_level-voxel_corr-fwe_method-montecarlo``: The maximum
                chi-squared value from the p(F|A) two-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pFgAsize_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster size value from the p(F|A) two-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).
            -   ``values_desc-pFgAmass_level-cluster_corr-fwe_method-montecarlo``: The maximum
                cluster mass value from the p(F|A) two-way chi-squared test from each Monte Carlo
                iteration. An array of shape (n_iters,).

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
        null_xyz = vox2mm(
            np.vstack(np.where(self.masker.mask_img.get_fdata())).T,
            self.masker.mask_img.affine,
        )
        pAgF_chi2_vals = result.get_map("chi2_desc-uniformity", return_type="array")
        pFgA_chi2_vals = result.get_map("chi2_desc-association", return_type="array")
        pAgF_z_vals = result.get_map("z_desc-uniformity", return_type="array")
        pFgA_z_vals = result.get_map("z_desc-association", return_type="array")
        pAgF_sign = np.sign(pAgF_z_vals)
        pFgA_sign = np.sign(pFgA_z_vals)

        n_cores = _check_ncores(n_cores)

        iter_df1 = self.inputs_["coordinates1"]
        iter_df2 = self.inputs_["coordinates2"]
        rand_idx1 = np.random.choice(null_xyz.shape[0], size=(iter_df1.shape[0], n_iters))
        rand_xyz1 = null_xyz[rand_idx1, :]
        iter_xyzs1 = np.split(rand_xyz1, rand_xyz1.shape[1], axis=1)
        rand_idx2 = np.random.choice(null_xyz.shape[0], size=(iter_df2.shape[0], n_iters))
        rand_xyz2 = null_xyz[rand_idx2, :]
        iter_xyzs2 = np.split(rand_xyz2, rand_xyz2.shape[1], axis=1)
        eps = np.spacing(1)

        # Identify summary statistic corresponding to intensity threshold
        ss_thresh = chi2.isf(voxel_thresh, 1)

        # Define connectivity matrix for cluster labeling
        conn = ndimage.generate_binary_structure(rank=3, connectivity=1)

        perm_results = [
            r
            for r in tqdm(
                Parallel(return_as="generator", n_jobs=n_cores)(
                    delayed(self._run_fwe_permutation)(
                        iter_xyz1=iter_xyzs1[i_iter],
                        iter_xyz2=iter_xyzs2[i_iter],
                        iter_df1=iter_df1,
                        iter_df2=iter_df2,
                        conn=conn,
                        voxel_thresh=ss_thresh,
                    )
                    for i_iter in range(n_iters)
                ),
                total=n_iters,
            )
        ]

        del rand_idx1, rand_xyz1, iter_xyzs1
        del rand_idx2, rand_xyz2, iter_xyzs2

        (
            pAgF_vfwe_null,
            pAgF_csfwe_null,
            pAgF_cmfwe_null,
            pFgA_vfwe_null,
            pFgA_csfwe_null,
            pFgA_cmfwe_null,
        ) = zip(*perm_results)

        del perm_results

        # pAgF_FWE
        pAgF_p_vfwe_vals, pAgF_p_csfwe_vals, pAgF_p_cmfwe_vals = self._apply_correction(
            pAgF_chi2_vals,
            ss_thresh,
            vfwe_null=pAgF_vfwe_null,
            csfwe_null=pAgF_csfwe_null,
            cmfwe_null=pAgF_cmfwe_null,
        )

        self.null_distributions_["values_desc-pAgF_level-voxel_corr-fwe_method-montecarlo"] = (
            pAgF_vfwe_null
        )
        self.null_distributions_[
            "values_desc-pAgFsize_level-cluster_corr-fwe_method-montecarlo"
        ] = pAgF_csfwe_null
        self.null_distributions_[
            "values_desc-pAgFmass_level-cluster_corr-fwe_method-montecarlo"
        ] = pAgF_cmfwe_null

        del pAgF_vfwe_null, pAgF_csfwe_null, pAgF_cmfwe_null

        # pFgA_FWE
        pFgA_p_vfwe_vals, pFgA_p_csfwe_vals, pFgA_p_cmfwe_vals = self._apply_correction(
            pFgA_chi2_vals,
            ss_thresh,
            vfwe_null=pFgA_vfwe_null,
            csfwe_null=pFgA_csfwe_null,
            cmfwe_null=pFgA_cmfwe_null,
        )

        self.null_distributions_["values_desc-pFgA_level-voxel_corr-fwe_method-montecarlo"] = (
            pFgA_vfwe_null
        )
        self.null_distributions_[
            "values_desc-pFgAsize_level-cluster_corr-fwe_method-montecarlo"
        ] = pFgA_csfwe_null
        self.null_distributions_[
            "values_desc-pFgAmass_level-cluster_corr-fwe_method-montecarlo"
        ] = pFgA_cmfwe_null

        del pFgA_vfwe_null, pFgA_csfwe_null, pFgA_cmfwe_null

        # Convert p-values
        # pAgF
        pAgF_z_vfwe_vals = p_to_z(pAgF_p_vfwe_vals, tail="two") * pAgF_sign
        pAgF_logp_vfwe_vals = -np.log10(pAgF_p_vfwe_vals)
        pAgF_logp_vfwe_vals[np.isinf(pAgF_logp_vfwe_vals)] = -np.log10(eps)
        pAgF_z_cmfwe_vals = p_to_z(pAgF_p_cmfwe_vals, tail="two") * pAgF_sign
        pAgF_logp_cmfwe_vals = -np.log10(pAgF_p_cmfwe_vals)
        pAgF_logp_cmfwe_vals[np.isinf(pAgF_logp_cmfwe_vals)] = -np.log10(eps)
        pAgF_z_csfwe_vals = p_to_z(pAgF_p_csfwe_vals, tail="two") * pAgF_sign
        pAgF_logp_csfwe_vals = -np.log10(pAgF_p_csfwe_vals)
        pAgF_logp_csfwe_vals[np.isinf(pAgF_logp_csfwe_vals)] = -np.log10(eps)

        # pFgA
        pFgA_z_vfwe_vals = p_to_z(pFgA_p_vfwe_vals, tail="two") * pFgA_sign
        pFgA_logp_vfwe_vals = -np.log10(pFgA_p_vfwe_vals)
        pFgA_logp_vfwe_vals[np.isinf(pFgA_logp_vfwe_vals)] = -np.log10(eps)
        pFgA_z_cmfwe_vals = p_to_z(pFgA_p_cmfwe_vals, tail="two") * pFgA_sign
        pFgA_logp_cmfwe_vals = -np.log10(pFgA_p_cmfwe_vals)
        pFgA_logp_cmfwe_vals[np.isinf(pFgA_logp_cmfwe_vals)] = -np.log10(eps)
        pFgA_z_csfwe_vals = p_to_z(pFgA_p_csfwe_vals, tail="two") * pFgA_sign
        pFgA_logp_csfwe_vals = -np.log10(pFgA_p_csfwe_vals)
        pFgA_logp_csfwe_vals[np.isinf(pFgA_logp_csfwe_vals)] = -np.log10(eps)

        maps = {
            # uniformity analysis
            "p_desc-uniformity_level-voxel": pAgF_p_vfwe_vals,
            "z_desc-uniformity_level-voxel": pAgF_z_vfwe_vals,
            "logp_desc-uniformity_level-voxel": pAgF_logp_vfwe_vals,
            "p_desc-uniformityMass_level-cluster": pAgF_p_cmfwe_vals,
            "z_desc-uniformityMass_level-cluster": pAgF_z_cmfwe_vals,
            "logp_desc-uniformityMass_level-cluster": pAgF_logp_cmfwe_vals,
            "p_desc-uniformitySize_level-cluster": pAgF_p_csfwe_vals,
            "z_desc-uniformitySize_level-cluster": pAgF_z_csfwe_vals,
            "logp_desc-uniformitySize_level-cluster": pAgF_logp_csfwe_vals,
            # association analysis
            "p_desc-association_level-voxel": pFgA_p_vfwe_vals,
            "z_desc-association_level-voxel": pFgA_z_vfwe_vals,
            "logp_desc-association_level-voxel": pFgA_logp_vfwe_vals,
            "p_desc-associationMass_level-cluster": pFgA_p_cmfwe_vals,
            "z_desc-associationMass_level-cluster": pFgA_z_cmfwe_vals,
            "logp_desc-associationMass_level-cluster": pFgA_logp_cmfwe_vals,
            "p_desc-associationSize_level-cluster": pFgA_p_csfwe_vals,
            "z_desc-associationSize_level-cluster": pFgA_z_csfwe_vals,
            "logp_desc-associationSize_level-cluster": pFgA_logp_csfwe_vals,
        }

        description = ""

        return maps, {}, description

    def correct_fdr_indep(self, result, alpha=0.05):
        """Perform FDR correction using the Benjamini-Hochberg method.

        Only call this method from within a Corrector.

        .. versionchanged:: 0.0.12

            Renamed from ``correct_fdr_bh`` to ``correct_fdr_indep``.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        alpha : :obj:`float`, optional
            Alpha. Default is 0.05.

        Returns
        -------
        maps : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked maps generated by
            the correction procedure. The following arrays are generated by
            this method: 'z_desc-uniformity_level-voxel' and 'z_desc-association_level-voxel'.

        See Also
        --------
        nimare.correct.FDRCorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = MKDAChi2()
        >>> result = meta.fit(dset)
        >>> corrector = FDRCorrector(method='indep', alpha=0.05)
        >>> cresult = corrector.transform(result)
        """
        pAgF_p_vals = result.get_map("p_desc-uniformity", return_type="array")
        pFgA_p_vals = result.get_map("p_desc-association", return_type="array")
        pAgF_z_vals = result.get_map("z_desc-uniformity", return_type="array")
        pFgA_z_vals = result.get_map("z_desc-association", return_type="array")
        pAgF_sign = np.sign(pAgF_z_vals)
        pFgA_sign = np.sign(pFgA_z_vals)
        pAgF_p_FDR = fdr(pAgF_p_vals, q=alpha, method="bh")
        pAgF_z_FDR = p_to_z(pAgF_p_FDR, tail="two") * pAgF_sign

        pFgA_p_FDR = fdr(pFgA_p_vals, q=alpha, method="bh")
        pFgA_z_FDR = p_to_z(pFgA_p_FDR, tail="two") * pFgA_sign

        maps = {
            "z_desc-uniformity_level-voxel": pAgF_z_FDR,
            "z_desc-association_level-voxel": pFgA_z_FDR,
        }

        description = ""

        return maps, {}, description


class KDA(CBMAEstimator):
    r"""Kernel density analysis.

    .. versionchanged:: 0.2.1

        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.0.12

        - Use a 4D sparse array for modeled activation maps.

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`~nimare.meta.kernel.KDAKernel`.
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

    n_iters : int, default=5000
        Number of iterations to use to define the null distribution.
        This is only used if ``null_method=="montecarlo"``.
        Default is 5000.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    n_cores : :obj:`int`, default=1
        Number of cores to use for parallelization.
        This is only used if ``null_method=="montecarlo"``.
        If <=0, defaults to using all available cores.
        Default is 1.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

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
    Kernel density analysis was first introduced in :footcite:t:`wager2003valence` and
    :footcite:t:`wager2004neuroimaging`.

    Available correction methods: :func:`KDA.correct_fwe_montecarlo`

    Warnings
    --------
    The KDA algorithm has been replaced in the literature with the MKDA algorithm.
    As such, this estimator should almost never be used, outside of systematic
    comparisons between algorithms.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        kernel_transformer=KDAKernel,
        null_method="approximate",
        n_iters=5000,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        n_cores=1,
        **kwargs,
    ):
        LGR.warning(
            "The KDA algorithm has been replaced in the literature with the MKDA algorithm. "
            "As such, this estimator should almost never be used, outside of systematic "
            "comparisons between algorithms."
        )

        if not (isinstance(kernel_transformer, KDAKernel) or kernel_transformer == KDAKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(
            kernel_transformer=kernel_transformer,
            memory=memory,
            memory_level=memory_level,
            **kwargs,
        )
        self.null_method = null_method
        self.n_iters = None if null_method == "approximate" else n_iters or 5000
        self.n_cores = _check_ncores(n_cores)
        self.dataset = None

    def _generate_description(self):
        """Generate a description of the fitted Estimator.

        Returns
        -------
        str
            Description of the Estimator.
        """
        if self.null_method == "montecarlo":
            null_method_str = (
                "a Monte Carlo-based null distribution, in which dataset coordinates were "
                "randomly drawn from the analysis mask and the full set of ALE values were "
                f"retained, using {self.n_iters} iterations"
            )
        else:
            null_method_str = "an approximate null distribution"

        description = (
            "A kernel density (KDA) meta-analysis \\citep{wager2007meta} was "
            "performed was performed with NiMARE "
            f"{__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), "
            f"using a(n) {self.kernel_transformer.__class__.__name__.replace('Kernel', '')} "
            "kernel. "
            f"{self.kernel_transformer._generate_description()} "
            f"Summary statistics (OF values) were converted to p-values using {null_method_str}. "
            f"The input dataset included {self.inputs_['coordinates'].shape[0]} foci from "
            f"{len(self.inputs_['id'])} experiments."
        )
        return description

    def _compute_summarystat_est(self, ma_values):
        """Compute OF scores from data.

        Parameters
        ----------
        ma_maps : :obj:`numpy.ndarray` or :obj:`sparse._coo.core.COO`
            MA maps.
            The ma_maps can be:
            (1) a 1d contrast-len or 2d contrast-by-voxel array of MA values,
            or (2) a 4d sparse array of MA maps,

        Returns
        -------
        stat_values : 1d array
            OF values. One value per voxel.
        """
        # OF is just a sum of MA values.
        if isinstance(ma_values, sparse._coo.core.COO):
            # NOTE: This may not work correctly with a non-NiftiMasker.
            mask_data = self.masker.mask_img.get_fdata().astype(bool)

            stat_values = ma_values.sum(axis=0)

            stat_values = stat_values.todense().reshape(-1)
            stat_values = stat_values[mask_data.reshape(-1)]

            # This is used by _compute_null_approximate
            self.__n_mask_voxels = stat_values.shape[0]
        else:
            # np.array type is used by _determine_histogram_bins to calculate max_poss_value
            stat_values = np.sum(ma_values, axis=0)

        return stat_values

    def _determine_histogram_bins(self, ma_maps):
        """Determine histogram bins for null distribution methods.

        Parameters
        ----------
        ma_maps : :obj:`sparse._coo.core.COO`
            MA maps.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute: "histogram_bins".
        """
        if not isinstance(ma_maps, sparse._coo.core.COO):
            raise ValueError(f"Unsupported data type '{type(ma_maps)}'")

        # assumes that groupby results in same order as MA maps
        n_foci_per_study = self.inputs_["coordinates"].groupby("id").size().values

        # Determine bins for null distribution histogram
        if hasattr(self.kernel_transformer, "value"):
            # Binary-sphere kernels (KDA & MKDA)
            # The maximum possible MA value for each study is the weighting factor (generally 1)
            # times the number of foci in the study.
            # We grab the weighting factor from the kernel transformer.
            step_size = self.kernel_transformer.value  # typically 1
            max_ma_values = step_size * n_foci_per_study
            max_poss_value = self._compute_summarystat_est(max_ma_values)
        else:
            # Continuous-sphere kernels (ALE)
            LGR.info(
                "A non-binary kernel has been detected. Parameters for the null distribution "
                "will be guesstimated."
            )

            N_BINS = 100000
            # The maximum possible MA value is the max value from each MA map,
            # unlike the case with a summation-based kernel.
            # Need to convert to dense because np.ceil is too slow with sparse
            max_ma_values = ma_maps.max(axis=[1, 2, 3]).todense()

            # round up based on resolution
            # hardcoding 1000 here because figuring out what to round to was difficult.
            max_ma_values = np.ceil(max_ma_values * 1000) / 1000
            max_poss_value = self._compute_summarystat(max_ma_values)

            # create bin centers
            hist_bins = np.linspace(0, max_poss_value, N_BINS - 1)
            step_size = hist_bins[1] - hist_bins[0]

        # Weighting is not supported yet, so I'm going to build my bins around the min MA value.
        # The histogram bins are bin *centers*, not edges.
        hist_bins = np.arange(0, max_poss_value + (step_size * 1.5), step_size)
        self.null_distributions_["histogram_bins"] = hist_bins

    def _compute_null_approximate(self, ma_maps):
        """Compute uncorrected null distribution using approximate solution.

        Parameters
        ----------
        ma_maps : :obj:`sparse._coo.core.COO`
            MA maps.

        Notes
        -----
        This method adds two entries to the null_distributions_ dict attribute:
        "histogram_bins" and "histogram_weights".
        """
        if not isinstance(ma_maps, sparse._coo.core.COO):
            raise ValueError(f"Unsupported data type '{type(ma_maps)}'")

        # Derive bin edges from histogram bin centers for numpy histogram function
        bin_centers = self.null_distributions_["histogram_bins"]
        step_size = bin_centers[1] - bin_centers[0]
        inv_step_size = 1 / step_size
        bin_edges = bin_centers - (step_size / 2)
        bin_edges = np.append(bin_centers, bin_centers[-1] + step_size)

        n_exp = ma_maps.shape[0]
        n_bins = bin_centers.shape[0]
        ma_hists = np.zeros((n_exp, n_bins))
        data = ma_maps.data
        coords = ma_maps.coords
        for exp_idx in range(n_exp):
            # The first column of coords is the fourth dimension of the dense array
            study_ma_values = data[coords[0, :] == exp_idx]

            n_nonzero_voxels = study_ma_values.shape[0]
            n_zero_voxels = self.__n_mask_voxels - n_nonzero_voxels

            ma_hists[exp_idx, :] = np.histogram(study_ma_values, bins=bin_edges, density=False)[
                0
            ].astype(float)
            ma_hists[exp_idx, 0] += n_zero_voxels

        # Normalize MA histograms to get probabilities
        ma_hists /= ma_hists.sum(1)[:, None]

        # Null distribution to convert summary statistics to p-values.
        stat_hist = ma_hists[0, :].copy()

        for i_exp in range(1, ma_hists.shape[0]):
            exp_hist = ma_hists[i_exp, :]

            # Find histogram bins with nonzero values for each histogram.
            stat_idx = np.where(stat_hist > 0)[0]
            exp_idx = np.where(exp_hist > 0)[0]

            # Compute output MA values, stat_hist indices, and probabilities
            stat_scores = np.add.outer(bin_centers[exp_idx], bin_centers[stat_idx]).ravel()
            score_idx = np.floor(stat_scores * inv_step_size).astype(int)
            probabilities = np.outer(exp_hist[exp_idx], stat_hist[stat_idx]).ravel()

            # Reset histogram and set probabilities. Use at() because there can
            # be redundant values in score_idx.
            stat_hist = np.zeros(stat_hist.shape)
            np.add.at(stat_hist, score_idx, probabilities)

        self.null_distributions_["histweights_corr-none_method-approximate"] = stat_hist
