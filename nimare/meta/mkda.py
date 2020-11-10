"""
CBMA methods from the multilevel kernel density analysis (MKDA) family
"""
import logging
import multiprocessing as mp

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, special
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm.auto import tqdm

from .. import references
from ..base import CBMAEstimator, PairwiseCBMAEstimator
from ..due import due
from ..stats import null_to_p, one_way, two_way
from ..transforms import p_to_z
from ..utils import round2
from .kernel import KDAKernel, MKDAKernel

LGR = logging.getLogger(__name__)


@due.dcite(references.MKDA, description="Introduces MKDA.")
class MKDADensity(CBMAEstimator):
    r"""
    Multilevel kernel density analysis- Density analysis.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
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
    _required_inputs = {
        "coordinates": ("coordinates", None),
    }

    def __init__(
        self, kernel_transformer=MKDAKernel, null_method="empirical", n_iters=10000, **kwargs
    ):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)
        self.null_method = null_method
        self.n_iters = n_iters
        self.dataset = None
        self.results = None

    def _fit(self, dataset):
        """
        Perform MKDA density meta-analysis on dataset.

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

        # Weight each SCM by square root of sample size
        # TODO: Incorporate sample-size and inference metadata extraction and
        # merging into df.
        # This will need to be distinct from the kernel_transformer-based kind
        # done in CBMAEstimator._preprocess_input
        ids_df = self.inputs_["coordinates"].groupby("id").first()
        if "sample_size" in ids_df.columns and "inference" not in ids_df.columns:
            ids_n = ids_df["sample_size"].astype(float).values
            weight_vec = np.sqrt(ids_n)[:, None] / np.sum(np.sqrt(ids_n))
        elif "sample_size" in ids_df.columns and "inference" in ids_df.columns:
            ids_n = ids_df["sample_size"].astype(float).values
            ids_inf = ids_df["inference"].map({"ffx": 0.75, "rfx": 1.0}).values
            weight_vec = (np.sqrt(ids_n)[:, None] * ids_inf[:, None]) / np.sum(
                np.sqrt(ids_n) * ids_inf
            )
        else:
            weight_vec = np.ones((ma_values.shape[0], 1))
        self.weight_vec_ = weight_vec  # C x 1 array
        assert self.weight_vec_.shape[0] == ma_values.shape[0]
        ma_values = ma_values * self.weight_vec_
        stat_values = self._compute_summarystat(ma_values)

        # Determine null distributions for summary stat (OF) to p conversion
        if self.null_method == "analytic":
            self._compute_null_analytic(ma_values)
        else:
            self._compute_null_empirical(ma_values, n_iters=self.n_iters)
        p_values, z_values = self._summarystat_to_p(stat_values, null_method=self.null_method)

        images = {
            "stat": stat_values,
            "p": p_values,
            "z": z_values,
        }
        return images

    def _compute_summarystat(self, data):
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
            ma_values = data.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(data)))

        # OF is just a sum of MA values.
        stat_values = np.sum(ma_values, axis=0)
        return stat_values

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
        null_distribution = np.zeros(n_iters)
        for i_iter in range(n_iters):
            # One random MA value per study
            null_ijk = np.random.choice(np.arange(n_voxels), n_studies)
            iter_ma_values = ma_maps[np.arange(n_studies), null_ijk]
            # Calculate summary statistic
            iter_ss_value = self._compute_summarystat(iter_ma_values)
            # Retain value in null distribution
            null_distribution[i_iter] = iter_ss_value
        self.null_distributions_["empirical_null"] = null_distribution

    def _compute_null_analytic(self, ma_maps):
        """Compute uncorrected null distribution using analytic solution.

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

        # Determine bins for null distribution histogram
        max_ma_values = np.max(ma_values, axis=1)
        max_poss_value = self._compute_summarystat(max_ma_values)
        # Set up histogram with bins from 0 to max value + one bin
        N_BINS = 10000
        bins_max = max_poss_value + (max_poss_value / (N_BINS - 1))  # one extra bin
        self.null_distributions_["histogram_bins"] = np.linspace(0, bins_max, num=N_BINS)

        ma_hists = np.zeros(
            (ma_values.shape[0], self.null_distributions_["histogram_bins"].shape[0])
        )
        for i_exp in range(ma_values.shape[0]):
            # Remember that histogram uses bin edges (not centers), so it
            # returns a 1xhist_bins-1 array
            n_zeros = len(np.where(ma_values[i_exp, :] == 0)[0])
            reduced_ma_values = ma_values[i_exp, ma_values[i_exp, :] > 0]
            ma_hists[i_exp, 0] = n_zeros
            ma_hists[i_exp, 1:] = np.histogram(
                a=reduced_ma_values, bins=self.null_distributions_["histogram_bins"], density=False
            )[0]

        # Inverse of step size in histBins (0.0001) = 10000
        step = 1 / np.mean(np.diff(self.null_distributions_["histogram_bins"]))

        # Null distribution to convert ALE to p-values.
        stat_hist = ma_hists[0, :]
        for i_exp in range(1, ma_hists.shape[0]):
            temp_hist = np.copy(stat_hist)
            ma_hist = np.copy(ma_hists[i_exp, :])

            # Find histogram bins with nonzero values for each histogram.
            ale_idx = np.where(temp_hist > 0)[0]
            exp_idx = np.where(ma_hist > 0)[0]

            # Normalize histograms.
            temp_hist /= np.sum(temp_hist)
            ma_hist /= np.sum(ma_hist)

            # Perform weighted convolution of histograms.
            stat_hist = np.zeros(self.null_distributions_["histogram_bins"].shape[0])
            for j_idx in exp_idx:
                # Compute probabilities of observing each ALE value in histBins
                # by randomly combining maps represented by maHist and aleHist.
                # Add observed probabilities to corresponding bins in ALE
                # histogram.
                probabilities = ma_hist[j_idx] * temp_hist[ale_idx]
                ale_scores = 1 - (1 - self.null_distributions_["histogram_bins"][j_idx]) * (
                    1 - self.null_distributions_["histogram_bins"][ale_idx]
                )
                score_idx = np.floor(ale_scores * step).astype(int)
                np.add.at(stat_hist, score_idx, probabilities)

        # Convert aleHist into null distribution. The value in each bin
        # represents the probability of finding an ALE value (stored in
        # histBins) of that value or lower.
        null_distribution = stat_hist / np.sum(stat_hist)
        null_distribution = np.cumsum(null_distribution[::-1])[::-1]
        null_distribution /= np.max(null_distribution)
        self.null_distributions_["histogram_weights"] = null_distribution

    def _summarystat_to_p(self, stat_values, null_method="analytic"):
        """
        Compute p- and z-values from summary statistics (e.g., ALE scores) and
        either histograms from analytic null or null distribution from
        empirical null.

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
        p_values = np.ones(stat_values.shape)

        if null_method == "analytic":
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histogram_weights" in self.null_distributions_.keys()

            step = 1 / np.mean(np.diff(self.null_distributions_["histogram_bins"]))

            # Determine p- and z-values from stat values and null distribution.
            idx = np.where(stat_values > 0)[0]
            stat_bins = round2(stat_values[idx] * step)
            p_values[idx] = self.null_distributions_["histogram_weights"][stat_bins]
        elif null_method == "empirical":
            assert "empirical_null" in self.null_distributions_.keys()

            for i_voxel in range(stat_values.shape[0]):
                p_values[i_voxel] = null_to_p(
                    stat_values[i_voxel],
                    self.null_distributions_["empirical_null"],
                    tail="upper",
                )
        else:
            raise ValueError("Argument 'null_method' must be one of: 'analytic', 'empirical'.")

        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values

    def _run_fwe_permutation(self, params):
        """
        Run a single Monte Carlo permutation of a dataset. Does the shared work
        between vFWE and cFWE.
        """
        iter_ijk, iter_df, conn, voxel_thresh = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[["i", "j", "k"]] = iter_ijk
        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="array"
        )
        iter_ma_maps = iter_ma_maps * self.weight_vec_
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        iter_of_map = self.masker.inverse_transform(iter_of_map)
        vthresh_iter_of_map = iter_of_map.get_fdata().copy()
        vthresh_iter_of_map[vthresh_iter_of_map < voxel_thresh] = 0

        labeled_matrix = ndimage.measurements.label(vthresh_iter_of_map, conn)[0]
        clust_sizes = [np.sum(labeled_matrix == val) for val in np.unique(labeled_matrix)]
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        if clust_sizes:
            iter_max_cluster = np.max(clust_sizes)
        else:
            iter_max_cluster = 0
        return iter_max_value, iter_max_cluster

    def correct_fwe_montecarlo(self, result, voxel_thresh=0.001, n_iters=10000, n_cores=-1):
        """
        Perform FWE correction using the max-value permutation method.
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
        z_values = result.get_map("z", return_type="array")
        stat_values = result.get_map("stat", return_type="array")
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T

        n_cores = self._check_ncores(n_cores)

        # Begin cluster-extent thresholding by thresholding matrix at cluster-
        # defining voxel-level threshold
        z_thresh = p_to_z(voxel_thresh, tail="one")
        vthresh_z_values = z_values.copy()
        vthresh_z_values[np.abs(vthresh_z_values) < z_thresh] = 0

        rand_idx = np.random.choice(
            null_ijk.shape[0], size=(self.inputs_["coordinates"].shape[0], n_iters)
        )
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.inputs_["coordinates"].copy()

        conn = np.ones((3, 3, 3))

        # Define parameters
        iter_conn = [conn] * n_iters
        iter_dfs = [iter_df] * n_iters
        iter_voxel_thresh = [voxel_thresh] * n_iters
        params = zip(iter_ijks, iter_dfs, iter_conn, iter_voxel_thresh)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params), total=n_iters))

        (
            self.null_distributions_["fwe_level-voxel_method-montecarlo"],
            self.null_distributions_["fwe_level-cluster_method-montecarlo"],
        ) = zip(*perm_results)

        # Cluster-level FWE
        vthresh_z_map = self.masker.inverse_transform(vthresh_z_values).get_fdata()
        labeled_matrix, n_clusters = ndimage.measurements.label(vthresh_z_map, conn)
        p_cfwe_map = np.ones(self.masker.mask_img.shape)
        for i_clust in range(1, n_clusters + 1):
            clust_size = np.sum(labeled_matrix == i_clust)
            clust_idx = np.where(labeled_matrix == i_clust)
            p_cfwe_map[clust_idx] = null_to_p(
                clust_size,
                self.null_distributions_["fwe_level-cluster_method-montecarlo"],
                "upper",
            )
        p_cfwe_map[np.isinf(p_cfwe_map)] = -np.log10(np.finfo(float).eps)
        p_cfwe_values = np.squeeze(
            self.masker.transform(nib.Nifti1Image(p_cfwe_map, self.masker.mask_img.affine))
        )
        logp_cfwe_values = -np.log10(p_cfwe_values)
        logp_cfwe_values[np.isinf(logp_cfwe_values)] = -np.log10(np.finfo(float).eps)
        z_cfwe_values = p_to_z(p_cfwe_values, tail="one")

        # Voxel-level FWE
        p_vfwe_values = np.ones(stat_values.shape)
        for voxel in range(stat_values.shape[0]):
            p_vfwe_values[voxel] = null_to_p(
                stat_values[voxel],
                self.null_distributions_["fwe_level-voxel_method-montecarlo"],
                tail="upper",
            )

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


@due.dcite(references.MKDA, description="Introduces MKDA.")
class MKDAChi2(PairwiseCBMAEstimator):
    r"""
    Multilevel kernel density analysis- Chi-square analysis.

    Parameters
    ----------
    prior : float, optional
        Uniform prior probability of each feature being active in a map in
        the absence of evidence from the map. Default: 0.5
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
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
    _required_inputs = {
        "coordinates": ("coordinates", None),
    }

    def __init__(self, kernel_transformer=MKDAKernel, prior=0.5, **kwargs):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)

        self.prior = prior

    def _fit(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.masker = self.masker or dataset1.masker
        self.null_distributions_ = {}

        ma_maps1 = self.kernel_transformer.transform(
            self.inputs_["coordinates1"], masker=self.masker, return_type="array"
        )
        ma_maps2 = self.kernel_transformer.transform(
            self.inputs_["coordinates2"], masker=self.masker, return_type="array"
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
        """
        Perform FWE correction using the max-value permutation method.
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
        """
        Perform FDR correction using the Benjamini-Hochberg method.
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
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        :class:`nimare.meta.kernel.KDAKernel`.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned
        here, with the prefix '\kernel__' in the variable name.

    Notes
    -----
    Kernel density analysis was first introduced in [1]_ and [2]_.

    Available correction methods: :func:`KDA.correct_fwe_montecarlo`

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
    _required_inputs = {
        "coordinates": ("coordinates", None),
    }

    def __init__(
        self, kernel_transformer=KDAKernel, null_method="empirical", n_iters=10000, **kwargs
    ):
        # Add kernel transformer attribute and process keyword arguments
        super().__init__(kernel_transformer=kernel_transformer, **kwargs)
        self.null_method = null_method
        self.n_iters = n_iters
        self.dataset = None
        self.results = None

    def _fit(self, dataset):
        """
        Perform KDA meta-analysis on dataset.

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
        stat_values = self._compute_summarystat(ma_values)

        # Determine null distributions for summary stat (OF) to p conversion
        if self.null_method == "analytic":
            # assumes that groupby results in same order as MA maps
            n_foci_per_study = self.inputs_["coordinates"].groupby("id").size().values
            self._compute_null_analytic(ma_values, n_foci_per_study)
        else:
            self._compute_null_empirical(ma_values, n_iters=self.n_iters)
        p_values, z_values = self._summarystat_to_p(stat_values, null_method=self.null_method)

        images = {
            "stat": stat_values,
            "p": p_values,
            "z": z_values,
        }
        return images

    def _compute_summarystat(self, data):
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
            ma_values = data.copy()
        else:
            raise ValueError('Unsupported data type "{}"'.format(type(data)))

        # OF is just a sum of MA values.
        stat_values = np.sum(ma_values, axis=0)
        return stat_values

    def _compute_null_empirical(self, ma_maps, n_iters=10000):
        """Compute uncorrected null distribution using empirical method.

        Parameters
        ----------
        ma_maps : (C x V) array
            Contrast by voxel array of MA values.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute:
        "empirical_null".
        """
        n_studies, n_voxels = ma_maps.shape
        null_distribution = np.zeros(n_iters)
        for i_iter in range(n_iters):
            # One random MA value per study
            null_ijk = np.random.choice(np.arange(n_voxels), n_studies)
            iter_ma_values = ma_maps[np.arange(n_studies), null_ijk]
            # Calculate summary statistic
            iter_ss_value = self._compute_summarystat(iter_ma_values)
            # Retain value in null distribution
            null_distribution[i_iter] = iter_ss_value
        self.null_distributions_["empirical_null"] = null_distribution

    def _compute_null_analytic(self, ma_maps, n_foci_per_study):
        """Compute uncorrected null distribution using analytic solution.

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

        # Determine bins for null distribution histogram
        # The maximum possible MA value for each study is the weighting factor (generally 1)
        # times the number of foci in the study.
        # To get the weighting factor, we find the minimum value in each MA map, ignoring zeros.
        n_studies = ma_values.shape[0]
        min_ma_values = np.zeros(n_studies)
        for i_study in range(n_studies):
            temp_ma_values = ma_values[i_study, :]
            min_ma_values[i_study] = np.min(temp_ma_values[temp_ma_values != 0])

        max_ma_values = min_ma_values * n_foci_per_study
        max_poss_value = self._compute_summarystat(max_ma_values)
        # Set up histogram with bins from 0 to max value + one bin
        N_BINS = 10000
        bins_max = max_poss_value + (max_poss_value / (N_BINS - 1))  # one extra bin
        self.null_distributions_["histogram_bins"] = np.linspace(0, bins_max, num=N_BINS)

        ma_hists = np.zeros(
            (ma_values.shape[0], self.null_distributions_["histogram_bins"].shape[0])
        )
        for i_exp in range(ma_values.shape[0]):
            # Remember that histogram uses bin edges (not centers), so it
            # returns a 1xhist_bins-1 array
            n_zeros = len(np.where(ma_values[i_exp, :] == 0)[0])
            reduced_ma_values = ma_values[i_exp, ma_values[i_exp, :] > 0]
            ma_hists[i_exp, 0] = n_zeros
            ma_hists[i_exp, 1:] = np.histogram(
                a=reduced_ma_values, bins=self.null_distributions_["histogram_bins"], density=False
            )[0]

        # Inverse of step size in histBins (0.0001) = 10000
        step = 1 / np.mean(np.diff(self.null_distributions_["histogram_bins"]))

        # Null distribution to convert ALE to p-values.
        stat_hist = ma_hists[0, :]
        for i_exp in range(1, ma_hists.shape[0]):
            temp_hist = np.copy(stat_hist)
            ma_hist = np.copy(ma_hists[i_exp, :])

            # Find histogram bins with nonzero values for each histogram.
            stat_idx = np.where(temp_hist > 0)[0]
            exp_idx = np.where(ma_hist > 0)[0]

            # Normalize histograms.
            temp_hist /= np.sum(temp_hist)
            ma_hist /= np.sum(ma_hist)

            # Perform weighted convolution of histograms.
            stat_hist = np.zeros(self.null_distributions_["histogram_bins"].shape[0])
            for j_idx in exp_idx:
                # Compute probabilities of observing each summary value in
                # histogram bins by randomly combining maps represented by
                # the MA histogram and summary value histogram.
                # Add observed probabilities to corresponding bins in summary
                # value histogram.
                probabilities = ma_hist[j_idx] * temp_hist[stat_idx]
                stat_scores = (
                    self.null_distributions_["histogram_bins"][j_idx]
                    + self.null_distributions_["histogram_bins"][stat_idx]
                )
                score_idx = np.floor(stat_scores * step).astype(int)
                np.add.at(stat_hist, score_idx, probabilities)

        # Convert aleHist into null distribution. The value in each bin
        # represents the probability of finding an ALE value (stored in
        # histBins) of that value or lower.
        null_distribution = stat_hist / np.sum(stat_hist)
        null_distribution = np.cumsum(null_distribution[::-1])[::-1]
        null_distribution /= np.max(null_distribution)
        self.null_distributions_["histogram_weights"] = null_distribution

    def _summarystat_to_p(self, stat_values, null_method="analytic"):
        """
        Compute p- and z-values from summary statistics (e.g., ALE scores) and
        either histograms from analytic null or null distribution from
        empirical null.

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
        p_values = np.ones(stat_values.shape)

        if null_method == "analytic":
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histogram_weights" in self.null_distributions_.keys()

            step = 1 / np.mean(np.diff(self.null_distributions_["histogram_bins"]))

            # Determine p- and z-values from ALE values and null distribution.
            idx = np.where(stat_values > 0)[0]
            stat_bins = round2(stat_values[idx] * step)
            p_values[idx] = self.null_distributions_["histogram_weights"][stat_bins]
        elif null_method == "empirical":
            assert "empirical_null" in self.null_distributions_.keys()

            for i_voxel in range(stat_values.shape[0]):
                p_values[i_voxel] = null_to_p(
                    stat_values[i_voxel],
                    self.null_distributions_["empirical_null"],
                    tail="upper",
                )
        else:
            raise ValueError("Argument 'null_method' must be one of: 'analytic', 'empirical'.")

        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values

    def _run_fwe_permutation(self, params):
        """
        Run a single Monte Carlo permutation of a dataset. Does vFWE, but not cFWE.
        """
        iter_ijk, iter_df = params
        iter_ijk = np.squeeze(iter_ijk)
        iter_df[["i", "j", "k"]] = iter_ijk
        iter_ma_maps = self.kernel_transformer.transform(
            iter_df, masker=self.masker, return_type="array"
        )
        iter_of_map = np.sum(iter_ma_maps, axis=0)
        iter_max_value = np.max(iter_of_map)
        return iter_max_value

    def correct_fwe_montecarlo(self, result, n_iters=10000, n_cores=-1):
        """
        Perform FWE correction using the max-value permutation method.
        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from a KDA meta-analysis.
        n_iters : :obj:`int`, optional
            Number of iterations to build the vFWE null distribution.
            Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is -1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'logp_level-voxel'.

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = KDA()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        stat_values = result.get_map("stat", return_type="array")
        null_ijk = np.vstack(np.where(self.masker.mask_img.get_fdata())).T

        n_cores = self._check_ncores(n_cores)

        rand_idx = np.random.choice(
            null_ijk.shape[0], size=(self.inputs_["coordinates"].shape[0], n_iters)
        )
        rand_ijk = null_ijk[rand_idx, :]
        iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
        iter_df = self.inputs_["coordinates"].copy()

        # Define parameters
        iter_dfs = [iter_df] * n_iters
        params = zip(iter_ijks, iter_dfs)

        if n_cores == 1:
            perm_results = []
            for pp in tqdm(params, total=n_iters):
                perm_results.append(self._run_fwe_permutation(pp))
        else:
            with mp.Pool(n_cores) as p:
                perm_results = list(tqdm(p.imap(self._run_fwe_permutation, params), total=n_iters))

        perm_max_values = perm_results

        # Voxel-level FWE
        vfwe_map = np.empty(stat_values.shape, dtype=float)
        for i_vox, val in enumerate(stat_values):
            vfwe_map[i_vox] = -np.log10(null_to_p(val, perm_max_values, "upper"))
        vfwe_map[np.isinf(vfwe_map)] = -np.log10(np.finfo(float).eps)

        images = {"logp_level-voxel": vfwe_map}
        return images
