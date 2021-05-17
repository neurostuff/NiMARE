"""CBMA methods from the multilevel kernel density analysis (MKDA) family."""
import logging
import multiprocessing as mp

import numpy as np
from scipy import special
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm.auto import tqdm

from ... import references
from ...due import due
from ...stats import null_to_p, one_way, two_way
from ...transforms import p_to_z
from ...utils import use_memmap
from ..kernel import KDAKernel, MKDAKernel
from .base import CBMAEstimator, PairwiseCBMAEstimator

LGR = logging.getLogger(__name__)


@due.dcite(references.MKDA, description="Introduces MKDA.")
class MKDADensity(CBMAEstimator):
    r"""
    Multilevel kernel density analysis- Density analysis.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.kernel.MKDAKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    Available correction methods: :func:`MKDADensity.correct_fwe_montecarlo`

    References
    ----------
    * Wager, Tor D., Martin Lindquist, and Lauren Kaplan. "Meta-analysis
      of functional neuroimaging data: current and future directions." Social
      cognitive and affective neuroscience 2.2 (2007): 150-158.
      https://doi.org/10.1093/scan/nsm015
    """

    def __init__(
        self,
        kernel_transformer=MKDAKernel,
        null_method="approximate",
        n_iters=10000,
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
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)
        self.null_method = null_method
        self.n_iters = n_iters
        self.n_cores = n_cores
        self.dataset = None
        self.results = None

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

    def _compute_summarystat(self, ma_values):
        # Note: .dot should be faster, but causes multiprocessing to stall
        # on some (Mac) architectures. If this is ever resolved, we can
        # replace with the commented line.
        # return ma_values.T.dot(self.weight_vec_).ravel()
        weighted_ma_vals = ma_values * self.weight_vec_
        return weighted_ma_vals.sum(0)

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

        prop_active = ma_values.mean(1)
        self.null_distributions_["histogram_bins"] = np.arange(len(prop_active) + 1, step=1)

    def _compute_null_approximate(self, ma_maps):
        """Compute uncorrected null distribution using approximate solution.

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

        # MKDA maps are binary, so we only have k + 1 bins in the final
        # histogram, where k is the number of studies. We can analytically
        # compute the null distribution by convolution.
        prop_active = ma_values.mean(1)
        ss_hist = 1.0
        for exp_prop in prop_active:
            ss_hist = np.convolve(ss_hist, [1 - exp_prop, exp_prop])
        self.null_distributions_["histogram_bins"] = np.arange(len(prop_active) + 1, step=1)
        self.null_distributions_["histweights_corr-none_method-approximate"] = ss_hist


@due.dcite(references.MKDA, description="Introduces MKDA.")
class MKDAChi2(PairwiseCBMAEstimator):
    r"""
    Multilevel kernel density analysis- Chi-square analysis.

    Parameters
    ----------
    prior : float, optional
        Uniform prior probability of each feature being active in a map in
        the absence of evidence from the map. Default: 0.5
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.kernel.MKDAKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    Available correction methods: :func:`MKDAChi2.correct_fwe_montecarlo`,
    :obj:`MKDAChi2.correct_fdr_bh`

    References
    ----------
    * Wager, Tor D., Martin Lindquist, and Lauren Kaplan. "Meta-analysis
      of functional neuroimaging data: current and future directions." Social
      cognitive and affective neuroscience 2.2 (2007): 150-158.
      https://doi.org/10.1093/scan/nsm015
    """

    def __init__(self, kernel_transformer=MKDAKernel, prior=0.5, **kwargs):
        if not (isinstance(kernel_transformer, MKDAKernel) or kernel_transformer == MKDAKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

        self.prior = prior

    @use_memmap(LGR, n_files=2)
    def _fit(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.masker = self.masker or dataset1.masker
        self.null_distributions_ = {}

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

        # Calculate different count variables
        n_selected = ma_maps1.shape[0]
        n_unselected = ma_maps2.shape[0]
        n_mappables = n_selected + n_unselected
        n_selected_active_voxels = np.sum(ma_maps1, axis=0)
        n_unselected_active_voxels = np.sum(ma_maps2, axis=0)

        # Transform MA maps to 1d arrays
        ma_maps_all = np.vstack((ma_maps1, ma_maps2))
        del ma_maps1, ma_maps2

        # Nomenclature for variables below: p = probability,
        # F = feature present, g = given, U = unselected, A = activation.
        # So, e.g., pAgF = p(A|F) = probability of activation
        # in a voxel if we know that the feature is present in a study.
        pF = (n_selected * 1.0) / n_mappables
        pA = np.array(np.sum(ma_maps_all, axis=0) / n_mappables).squeeze()

        # Conditional probabilities
        pAgF = n_selected_active_voxels * 1.0 / n_selected
        pAgU = n_unselected_active_voxels * 1.0 / n_unselected
        pFgA = pAgF * pF / pA

        # Recompute conditionals with uniform prior
        pAgF_prior = self.prior * pAgF + (1 - self.prior) * pAgU
        pFgA_prior = pAgF * self.prior / pAgF_prior

        # One-way chi-square test for consistency of activation
        pAgF_chi2_vals = one_way(np.squeeze(n_selected_active_voxels), n_selected)
        pAgF_p_vals = special.chdtrc(1, pAgF_chi2_vals)
        pAgF_sign = np.sign(n_selected_active_voxels - np.mean(n_selected_active_voxels))
        pAgF_z = p_to_z(pAgF_p_vals, tail="two") * pAgF_sign

        # Two-way chi-square for specificity of activation
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
        pFgA_p_vals = special.chdtrc(1, pFgA_chi2_vals)
        pFgA_p_vals[pFgA_p_vals < 1e-240] = 1e-240
        pFgA_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(pFgA_p_vals, tail="two") * pFgA_sign
        images = {
            "prob_desc-A": pA,
            "prob_desc-AgF": pAgF,
            "prob_desc-FgA": pFgA,
            ("prob_desc-AgF_given_pF=%0.2f" % self.prior): pAgF_prior,
            ("prob_desc-FgA_given_pF=%0.2f" % self.prior): pFgA_prior,
            "z_desc-consistency": pAgF_z,
            "z_desc-specificity": pFgA_z,
            "chi2_desc-consistency": pAgF_chi2_vals,
            "chi2_desc-specificity": pFgA_chi2_vals,
            "p_desc-consistency": pAgF_p_vals,
            "p_desc-specificity": pFgA_p_vals,
        }
        return images

    def _run_fwe_permutation(self, params):
        iter_df1, iter_df2, iter_ijk1, iter_ijk2 = params
        iter_ijk1 = np.squeeze(iter_ijk1)
        iter_ijk2 = np.squeeze(iter_ijk2)
        iter_df1[["i", "j", "k"]] = iter_ijk1
        iter_df2[["i", "j", "k"]] = iter_ijk2

        temp_ma_maps1 = self.kernel_transformer.transform(
            iter_df1, self.masker, return_type="array"
        )
        temp_ma_maps2 = self.kernel_transformer.transform(
            iter_df2, self.masker, return_type="array"
        )

        n_selected = temp_ma_maps1.shape[0]
        n_unselected = temp_ma_maps2.shape[0]
        n_selected_active_voxels = np.sum(temp_ma_maps1, axis=0)
        n_unselected_active_voxels = np.sum(temp_ma_maps2, axis=0)

        # Currently unused conditional probabilities
        # pAgF = n_selected_active_voxels * 1.0 / n_selected
        # pAgU = n_unselected_active_voxels * 1.0 / n_unselected

        # One-way chi-square test for consistency of activation
        pAgF_chi2_vals = one_way(np.squeeze(n_selected_active_voxels), n_selected)
        iter_pAgF_chi2 = np.max(pAgF_chi2_vals)

        # Two-way chi-square for specificity of activation
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
        iter_pFgA_chi2 = np.max(pFgA_chi2_vals)
        return iter_pAgF_chi2, iter_pFgA_chi2

    def correct_fwe_montecarlo(self, result, n_iters=5000, n_cores=-1):
        """Perform FWE correction using the max-value permutation method.

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        n_iters : :obj:`int`, optional
            Number of iterations to build the vFWE null distribution.
            Default is 5000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is -1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'p_desc-consistency_level-voxel',
            'z_desc-consistency_level-voxel', 'p_desc-specificity_level-voxel',
            and 'p_desc-specificity_level-voxel'.

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
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T
        pAgF_chi2_vals = result.get_map("chi2_desc-consistency", return_type="array")
        pFgA_chi2_vals = result.get_map("chi2_desc-specificity", return_type="array")
        pAgF_z_vals = result.get_map("z_desc-consistency", return_type="array")
        pFgA_z_vals = result.get_map("z_desc-specificity", return_type="array")
        pAgF_sign = np.sign(pAgF_z_vals)
        pFgA_sign = np.sign(pFgA_z_vals)

        n_cores = self._check_ncores(n_cores)

        iter_df1 = self.inputs_["coordinates1"].copy()
        iter_df2 = self.inputs_["coordinates2"].copy()
        iter_dfs1 = [iter_df1] * n_iters
        iter_dfs2 = [iter_df2] * n_iters
        rand_idx1 = np.random.choice(null_ijk.shape[0], size=(iter_df1.shape[0], n_iters))
        rand_ijk1 = null_ijk[rand_idx1, :]
        iter_ijks1 = np.split(rand_ijk1, rand_ijk1.shape[1], axis=1)
        rand_idx2 = np.random.choice(null_ijk.shape[0], size=(iter_df2.shape[0], n_iters))
        rand_ijk2 = null_ijk[rand_idx2, :]
        iter_ijks2 = np.split(rand_ijk2, rand_ijk2.shape[1], axis=1)
        eps = np.spacing(1)

        params = zip(iter_dfs1, iter_dfs2, iter_ijks1, iter_ijks2)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params), total=n_iters))
        pAgF_null_chi2_dist, pFgA_null_chi2_dist = zip(*perm_results)

        # pAgF_FWE
        pAgF_null_chi2_dist = np.squeeze(pAgF_null_chi2_dist)
        pAgF_p_FWE = np.empty_like(pAgF_chi2_vals).astype(float)
        for voxel in range(pFgA_chi2_vals.shape[0]):
            pAgF_p_FWE[voxel] = null_to_p(pAgF_chi2_vals[voxel], pAgF_null_chi2_dist, tail="upper")
        # Crop p-values of 0 or 1 to nearest values that won't evaluate to
        # 0 or 1. Prevents inf z-values.
        pAgF_p_FWE[pAgF_p_FWE < eps] = eps
        pAgF_p_FWE[pAgF_p_FWE > (1.0 - eps)] = 1.0 - eps
        pAgF_z_FWE = p_to_z(pAgF_p_FWE, tail="two") * pAgF_sign

        # pFgA_FWE
        pFgA_null_chi2_dist = np.squeeze(pFgA_null_chi2_dist)
        pFgA_p_FWE = np.empty_like(pFgA_chi2_vals).astype(float)
        for voxel in range(pFgA_chi2_vals.shape[0]):
            pFgA_p_FWE[voxel] = null_to_p(pFgA_chi2_vals[voxel], pFgA_null_chi2_dist, tail="upper")
        # Crop p-values of 0 or 1 to nearest values that won't evaluate to
        # 0 or 1. Prevents inf z-values.
        pFgA_p_FWE[pFgA_p_FWE < eps] = eps
        pFgA_p_FWE[pFgA_p_FWE > (1.0 - eps)] = 1.0 - eps
        pFgA_z_FWE = p_to_z(pFgA_p_FWE, tail="two") * pFgA_sign

        images = {
            "p_desc-consistency_level-voxel": pAgF_p_FWE,
            "z_desc-consistency_level-voxel": pAgF_z_FWE,
            "p_desc-specificity_level-voxel": pFgA_p_FWE,
            "z_desc-specificity_level-voxel": pFgA_z_FWE,
        }
        return images

    def correct_fdr_bh(self, result, alpha=0.05):
        """Perform FDR correction using the Benjamini-Hochberg method.

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        alpha : :obj:`float`, optional
            Alpha. Default is 0.05.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'consistency_z_FDR' and 'specificity_z_FDR'.

        See Also
        --------
        nimare.correct.FDRCorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = MKDAChi2()
        >>> result = meta.fit(dset)
        >>> corrector = FDRCorrector(method='bh', alpha=0.05)
        >>> cresult = corrector.transform(result)
        """
        pAgF_p_vals = result.get_map("p_desc-consistency", return_type="array")
        pFgA_p_vals = result.get_map("p_desc-specificity", return_type="array")
        pAgF_z_vals = result.get_map("z_desc-consistency", return_type="array")
        pFgA_z_vals = result.get_map("z_desc-specificity", return_type="array")
        pAgF_sign = np.sign(pAgF_z_vals)
        pFgA_sign = np.sign(pFgA_z_vals)
        _, pAgF_p_FDR, _, _ = multipletests(
            pAgF_p_vals, alpha=alpha, method="fdr_bh", is_sorted=False, returnsorted=False
        )
        pAgF_z_FDR = p_to_z(pAgF_p_FDR, tail="two") * pAgF_sign

        _, pFgA_p_FDR, _, _ = multipletests(
            pFgA_p_vals, alpha=alpha, method="fdr_bh", is_sorted=False, returnsorted=False
        )
        pFgA_z_FDR = p_to_z(pFgA_p_FDR, tail="two") * pFgA_sign

        images = {
            "z_desc-consistency_level-voxel": pAgF_z_FDR,
            "z_desc-specificity_level-voxel": pFgA_z_FDR,
        }
        return images


@due.dcite(references.KDA1, description="Introduces the KDA algorithm.")
@due.dcite(references.KDA2, description="Also introduces the KDA algorithm.")
class KDA(CBMAEstimator):
    r"""
    Kernel density analysis.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.kernel.KDAKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    Kernel density analysis was first introduced in [1]_ and [2]_.

    Available correction methods: :func:`KDA.correct_fwe_montecarlo`

    Warning
    -------
    The KDA algorithm has been replaced in the literature with the MKDA algorithm.
    As such, this estimator should almost never be used, outside of systematic
    comparisons between algorithms.

    References
    ----------
    .. [1] Wager, Tor D., et al. "Valence, gender, and lateralization of
        functional brain anatomy in emotion: a meta-analysis of findings from
        neuroimaging." Neuroimage 19.3 (2003): 513-531.
        https://doi.org/10.1016/S1053-8119(03)00078-8
    .. [2] Wager, Tor D., John Jonides, and Susan Reading. "Neuroimaging
        studies of shifting attention: a meta-analysis." Neuroimage 22.4
        (2004): 1679-1693. https://doi.org/10.1016/j.neuroimage.2004.03.052
    """

    def __init__(
        self,
        kernel_transformer=KDAKernel,
        null_method="approximate",
        n_iters=10000,
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
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)
        self.null_method = null_method
        self.n_iters = n_iters
        self.n_cores = n_cores
        self.dataset = None
        self.results = None

    def _compute_summarystat(self, ma_values):
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
        # OF is just a sum of MA values.
        stat_values = np.sum(ma_values, axis=0)
        return stat_values

    def _determine_histogram_bins(self, ma_maps):
        """Determine histogram bins for null distribution methods.

        Parameters
        ----------
        ma_maps
            Modeled activation maps. Unused for this estimator.

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
            max_poss_value = self._compute_summarystat(max_ma_values)
        else:
            # Continuous-sphere kernels (ALE)
            LGR.info(
                "A non-binary kernel has been detected. Parameters for the null distribution "
                "will be guesstimated."
            )

            N_BINS = 100000
            # The maximum possible MA value is the max value from each MA map,
            # unlike the case with a summation-based kernel.
            max_ma_values = np.max(ma_values, axis=1)
            # round up based on resolution
            # hardcoding 1000 here because figuring out what to round to was difficult.
            max_ma_values = np.ceil(max_ma_values * 1000) / 1000
            max_poss_value = self.compute_summarystat(max_ma_values)

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
