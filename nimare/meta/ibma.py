"""Image-based meta-analysis estimators."""
from __future__ import division

import logging

import numpy as np
import pymare
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

from ..base import MetaEstimator
from ..transforms import p_to_z, t_to_z

LGR = logging.getLogger(__name__)


class Fishers(MetaEstimator):
    """An image-based meta-analytic test using t- or z-statistic images.

    Requires z-statistic images, but will be extended to work with t-statistic
    images as well.

    Notes
    -----
    Requires ``z`` images.

    Warning
    -------
    This method does not currently calculate p-values correctly. Do not use.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Fisher, R. A. (1934). Statistical methods for research workers.
      Statistical methods for research workers., (5th Ed).
      https://www.cabdirect.org/cabdirect/abstract/19351601205

    See Also
    --------
    :class:`pymare.estimators.FisherCombinationTest`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        masker = self.masker or dataset.masker
        if not isinstance(masker, NiftiMasker):
            raise ValueError(
                f"A {type(masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator. "
                "This is because aggregation, such as averaging values across ROIs, "
                "will produce invalid results."
            )

        pymare_dset = pymare.Dataset(y=self.inputs_["z_maps"])
        est = pymare.estimators.FisherCombinationTest()
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "z": est_summary.z,
            "p": est_summary.p,
        }
        return results


class Stouffers(MetaEstimator):
    """A t-test on z-statistic images.

    Requires z-statistic images.

    Parameters
    ----------
    use_sample_size : :obj:`bool`, optional
        Whether to use sample sizes for weights (i.e., "weighted Stouffer's")
        or not. Default is False.

    Notes
    -----
    Requires ``z`` images and optionally the sample size metadata field.

    Warning
    -------
    This method does not currently calculate p-values correctly. Do not use.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Stouffer, S. A., Suchman, E. A., DeVinney, L. C., Star, S. A., &
      Williams Jr, R. M. (1949). The American Soldier: Adjustment during
      army life. Studies in social psychology in World War II, vol. 1.
      https://psycnet.apa.org/record/1950-00790-000
    * Zaykin, D. V. (2011). Optimally weighted Z‐test is a powerful method for
      combining probabilities in meta‐analysis. Journal of evolutionary
      biology, 24(8), 1836-1841.
      https://doi.org/10.1111/j.1420-9101.2011.02297.x

    See Also
    --------
    :class:`pymare.estimators.StoufferCombinationTest`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(self, use_sample_size=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sample_size = use_sample_size
        if self.use_sample_size:
            self._required_inputs["sample_sizes"] = ("metadata", "sample_sizes")

    def _fit(self, dataset):
        masker = self.masker or dataset.masker
        if not isinstance(masker, NiftiMasker):
            raise ValueError(
                f"A {type(masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator. "
                "This is because aggregation, such as averaging values across ROIs, "
                "will produce invalid results."
            )

        est = pymare.estimators.StoufferCombinationTest()

        if self.use_sample_size:
            sample_sizes = np.array([np.mean(n) for n in self.inputs_["sample_sizes"]])
            weights = np.sqrt(sample_sizes)
            weight_maps = np.tile(weights, (self.inputs_["z_maps"].shape[1], 1)).T
            pymare_dset = pymare.Dataset(y=self.inputs_["z_maps"], v=weight_maps)
        else:
            pymare_dset = pymare.Dataset(y=self.inputs_["z_maps"])

        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        results = {
            "z": est_summary.z,
            "p": est_summary.p,
        }
        return results


class WeightedLeastSquares(MetaEstimator):
    """Weighted least-squares meta-regression.

    Provides the weighted least-squares estimate of the fixed effects given
    known/assumed between-study variance tau^2.
    When tau^2 = 0 (default), the model is the standard inverse-weighted
    fixed-effects meta-regression.

    Parameters
    ----------
    tau2 : :obj:`float` or 1D :class:`numpy.ndarray`, optional
        Assumed/known value of tau^2. Must be >= 0. Default is 0.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warning
    -------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Brockwell, S. E., & Gordon, I. R. (2001). A comparison of statistical
      methods for meta-analysis. Statistics in Medicine, 20(6), 825–840.
      https://doi.org/10.1002/sim.650

    See Also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, tau2=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau2 = tau2

    def _fit(self, dataset):
        masker = self.masker or dataset.masker
        if not isinstance(masker, NiftiMasker):
            LGR.warning(
                f"A {type(masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est = pymare.estimators.WeightedLeastSquares(tau2=self.tau2)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": est_summary.tau2,
            "z": est_summary.get_fe_stats()["z"].squeeze(),
            "p": est_summary.get_fe_stats()["p"].squeeze(),
            "est": est_summary.get_fe_stats()["est"].squeeze(),
        }
        return results


class DerSimonianLaird(MetaEstimator):
    """DerSimonian-Laird meta-regression estimator.

    Estimates the between-subject variance tau^2 using the DerSimonian-Laird
    (1986) method-of-moments approach.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warning
    -------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
      Controlled clinical trials, 7(3), 177-188.
    * Kosmidis, I., Guolo, A., & Varin, C. (2017). Improving the accuracy of
      likelihood-based inference in meta-analysis and meta-regression.
      Biometrika, 104(2), 489–496. https://doi.org/10.1093/biomet/asx001

    See Also
    --------
    :class:`pymare.estimators.DerSimonianLaird`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        masker = self.masker or dataset.masker
        if not isinstance(masker, NiftiMasker):
            LGR.warning(
                f"A {type(masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        est = pymare.estimators.DerSimonianLaird()
        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": est_summary.tau2,
            "z": est_summary.get_fe_stats()["z"].squeeze(),
            "p": est_summary.get_fe_stats()["p"].squeeze(),
            "est": est_summary.get_fe_stats()["est"].squeeze(),
        }
        return results


class Hedges(MetaEstimator):
    """Hedges meta-regression estimator.

    Estimates the between-subject variance tau^2 using the Hedges & Olkin (1985)
    approach.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warning
    -------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Hedges LV, Olkin I. 1985. Statistical Methods for Meta‐Analysis.

    See Also
    --------
    :class:`pymare.estimators.Hedges`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        masker = self.masker or dataset.masker
        if not isinstance(masker, NiftiMasker):
            LGR.warning(
                f"A {type(masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        est = pymare.estimators.Hedges()
        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": est_summary.tau2,
            "z": est_summary.get_fe_stats()["z"].squeeze(),
            "p": est_summary.get_fe_stats()["p"].squeeze(),
            "est": est_summary.get_fe_stats()["est"].squeeze(),
        }
        return results


class SampleSizeBasedLikelihood(MetaEstimator):
    """Method estimates with known sample sizes but unknown sampling variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Parameters
    ----------
    method : {'ml', 'reml'}, optional
        The estimation method to use.
        Either 'ml' (for maximum-likelihood) or 'reml'
        (restricted maximum-likelihood). Default is 'ml'.

    Notes
    -----
    Requires ``beta`` images and sample size from metadata.

    Homogeneity of sigma^2 across studies is assumed.
    The ML and REML solutions are obtained via SciPy’s scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warning
    -------
    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    See Also
    --------
    :class:`pymare.estimators.SampleSizeBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {
        "beta_maps": ("image", "beta"),
        "sample_sizes": ("metadata", "sample_sizes"),
    }

    def __init__(self, method="ml", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def _fit(self, dataset):
        sample_sizes = np.array([np.mean(n) for n in self.inputs_["sample_sizes"]])
        n_maps = np.tile(sample_sizes, (self.inputs_["beta_maps"].shape[1], 1)).T
        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], n=n_maps)
        est = pymare.estimators.SampleSizeBasedLikelihoodEstimator(method=self.method)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": est_summary.tau2,
            "z": est_summary.get_fe_stats()["z"].squeeze(),
            "p": est_summary.get_fe_stats()["p"].squeeze(),
            "est": est_summary.get_fe_stats()["est"].squeeze(),
        }
        return results


class VarianceBasedLikelihood(MetaEstimator):
    """A likelihood-based meta-analysis method for estimates with known variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    coefficients using the specified likelihood-based estimator (ML or REML).

    Parameters
    ----------
    method : {'ml', 'reml'}, optional
        The estimation method to use.
        Either 'ml' (for maximum-likelihood) or 'reml'
        (restricted maximum-likelihood). Default is 'ml'.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    The ML and REML solutions are obtained via SciPy’s scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warning
    -------
    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
      Controlled clinical trials, 7(3), 177-188.
    * Kosmidis, I., Guolo, A., & Varin, C. (2017). Improving the accuracy of
      likelihood-based inference in meta-analysis and meta-regression.
      Biometrika, 104(2), 489–496. https://doi.org/10.1093/biomet/asx001

    See Also
    --------
    :class:`pymare.estimators.VarianceBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, method="ml", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def _fit(self, dataset):
        masker = self.masker or dataset.masker
        if not isinstance(masker, NiftiMasker):
            LGR.warning(
                f"A {type(masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        est = pymare.estimators.VarianceBasedLikelihoodEstimator(method=self.method)

        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": est_summary.tau2,
            "z": est_summary.get_fe_stats()["z"].squeeze(),
            "p": est_summary.get_fe_stats()["p"].squeeze(),
            "est": est_summary.get_fe_stats()["est"].squeeze(),
        }
        return results


class PermutedOLS(MetaEstimator):
    r"""An analysis with permuted ordinary least squares (OLS), using nilearn.

    Parameters
    ----------
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``z`` images.

    Available correction methods: :func:`PermutedOLS.correct_fwe_montecarlo`

    Warning
    -------
    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported significance
      levels. Journal of Business & Economic Statistics, 1(4), 292-298.

    See Also
    --------
    nilearn.mass_univariate.permuted_ols : The function used for this IBMA.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(self, two_sided=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.two_sided = two_sided
        self.parameters_ = {}

    def _fit(self, dataset):
        # Use intercept as explanatory variable
        self.parameters_["tested_vars"] = np.ones((self.inputs_["z_maps"].shape[0], 1))
        self.parameters_["confounding_vars"] = None

        _, t_map, _ = permuted_ols(
            self.parameters_["tested_vars"],
            self.inputs_["z_maps"],
            confounding_vars=self.parameters_["confounding_vars"],
            model_intercept=False,  # modeled by tested_vars
            n_perm=0,
            two_sided_test=self.two_sided,
            random_state=42,
            n_jobs=1,
            verbose=0,
        )

        # Convert t to z, preserving signs
        dof = self.parameters_["tested_vars"].shape[0] - self.parameters_["tested_vars"].shape[1]
        z_map = t_to_z(t_map, dof)
        images = {"t": t_map.squeeze(), "z": z_map.squeeze()}
        return images

    def correct_fwe_montecarlo(self, result, n_iters=10000, n_cores=-1):
        """Perform FWE correction using the max-value permutation method.

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            Result object from an ALE meta-analysis.
        n_iters : :obj:`int`, optional
            The number of iterations to run in estimating the null distribution.
            Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is -1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'z_vthresh', 'p_level-voxel', 'z_level-voxel', and
            'logp_level-cluster'.

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.
        nilearn.mass_univariate.permuted_ols : The function used for this IBMA.

        Examples
        --------
        >>> meta = PermutedOLS()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo',
                                     n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        n_cores = self._check_ncores(n_cores)

        log_p_map, t_map, _ = permuted_ols(
            self.parameters_["tested_vars"],
            self.inputs_["z_maps"],
            confounding_vars=self.parameters_["confounding_vars"],
            model_intercept=False,  # modeled by tested_vars
            n_perm=n_iters,
            two_sided_test=self.two_sided,
            random_state=42,
            n_jobs=n_cores,
            verbose=0,
        )

        # Fill complete maps
        p_map = np.power(10.0, -log_p_map)

        # Convert p to z, preserving signs
        sign = np.sign(t_map)
        sign[sign == 0] = 1
        z_map = p_to_z(p_map, tail="two") * sign
        images = {"logp_level-voxel": log_p_map.squeeze(), "z_level-voxel": z_map.squeeze()}
        return images
