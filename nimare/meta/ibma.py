"""Image-based meta-analysis estimators."""
from __future__ import division

import logging

import nibabel as nib
import numpy as np
import pymare
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn.image import concat_imgs, resample_to_img
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

from nimare.base import Estimator
from nimare.transforms import p_to_z, t_to_z
from nimare.utils import _boolean_unmask, _check_ncores, get_masker

LGR = logging.getLogger(__name__)


class IBMAEstimator(Estimator):
    """Base class for meta-analysis methods in :mod:`~nimare.meta`.

    .. versionadded:: 0.0.12

        * IBMA-specific elements of ``MetaEstimator`` excised and used to create ``IBMAEstimator``.
        * Generic kwargs and args converted to named kwargs.
          All remaining kwargs are for resampling.

    """

    def __init__(self, *, mask=None, resample=False, memory_limit=None, **kwargs):
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask
        self.resample = resample
        self.memory_limit = memory_limit

        # defaults for resampling images (nilearn's defaults do not work well)
        self._resample_kwargs = {"clip": True, "interpolation": "linear"}

        # Identify any kwargs
        resample_kwargs = {k: v for k, v in kwargs.items() if k.startswith("resample__")}

        # Flag any extraneous kwargs
        other_kwargs = dict(set(kwargs.items()) - set(resample_kwargs.items()))
        if other_kwargs:
            LGR.warn(f"Unused keyword arguments found: {tuple(other_kwargs.items())}")

        # Update the default resampling parameters
        resample_kwargs = {k.split("resample__")[1]: v for k, v in resample_kwargs.items()}
        self._resample_kwargs.update(resample_kwargs)

    def _preprocess_input(self, dataset):
        """Preprocess inputs to the Estimator from the Dataset as needed."""
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)

        # Ensure that protected values are not included among _required_inputs
        assert "aggressive_mask" not in self._required_inputs.keys(), "This is a protected name."

        if "aggressive_mask" in self.inputs_.keys():
            LGR.warning("Removing existing 'aggressive_mask' from Estimator.")
            self.inputs_.pop("aggressive_mask")

        # A dictionary to collect masked image data, to be further reduced by the aggressive mask.
        temp_image_inputs = {}

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "image":
                # If no resampling is requested, check if resampling is required
                if not self.resample:
                    check_imgs = {img: nib.load(img) for img in self.inputs_[name]}
                    _check_same_fov(**check_imgs, reference_masker=mask_img, raise_error=True)
                    imgs = list(check_imgs.values())
                else:
                    # resampling will only occur if shape/affines are different
                    # making this harmless if all img shapes/affines are the same as the reference
                    imgs = [
                        resample_to_img(nib.load(img), mask_img, **self._resample_kwargs)
                        for img in self.inputs_[name]
                    ]

                # input to NiFtiLabelsMasker must be 4d
                img4d = concat_imgs(imgs, ensure_ndim=4)

                # Mask required input images using either the dataset's mask or the estimator's.
                temp_arr = masker.transform(img4d)

                # An intermediate step to mask out bad voxels.
                # Can be dropped once PyMARE is able to handle masked arrays or missing data.
                nonzero_voxels_bool = np.all(temp_arr != 0, axis=0)
                nonnan_voxels_bool = np.all(~np.isnan(temp_arr), axis=0)
                good_voxels_bool = np.logical_and(nonzero_voxels_bool, nonnan_voxels_bool)

                data = masker.transform(img4d)

                temp_image_inputs[name] = data
                if "aggressive_mask" not in self.inputs_.keys():
                    self.inputs_["aggressive_mask"] = good_voxels_bool
                else:
                    # Remove any voxels that are bad in any image-based inputs
                    self.inputs_["aggressive_mask"] = np.logical_or(
                        self.inputs_["aggressive_mask"],
                        good_voxels_bool,
                    )

        # Further reduce image-based inputs to remove "bad" voxels
        # (voxels with zeros or NaNs in any studies)
        if "aggressive_mask" in self.inputs_.keys():
            n_bad_voxels = (
                self.inputs_["aggressive_mask"].size - self.inputs_["aggressive_mask"].sum()
            )
            if n_bad_voxels:
                LGR.warning(
                    f"Masking out {n_bad_voxels} additional voxels. "
                    "The updated masker is available in the Estimator.masker attribute."
                )

            for name, raw_masked_data in temp_image_inputs.items():
                self.inputs_[name] = raw_masked_data[:, self.inputs_["aggressive_mask"]]


class Fishers(IBMAEstimator):
    """An image-based meta-analytic test using t- or z-statistic images.

    Requires z-statistic images, but will be extended to work with t-statistic images as well.

    This method is described in :footcite:t:`fisher1946statistical`.

    Notes
    -----
    Requires ``z`` images.

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.FisherCombinationTest`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator. "
                "This is because aggregation, such as averaging values across ROIs, "
                "will produce invalid results."
            )

        pymare_dset = pymare.Dataset(y=self.inputs_["z_maps"])
        est = pymare.estimators.FisherCombinationTest()
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "z": _boolean_unmask(est_summary.z.squeeze(), self.inputs_["aggressive_mask"]),
            "p": _boolean_unmask(est_summary.p.squeeze(), self.inputs_["aggressive_mask"]),
        }
        return results


class Stouffers(IBMAEstimator):
    """A t-test on z-statistic images.

    Requires z-statistic images.

    This method is described in :footcite:t:`stouffer1949american`.

    Parameters
    ----------
    use_sample_size : :obj:`bool`, optional
        Whether to use sample sizes for weights (i.e., "weighted Stouffer's") or not,
        as described in :footcite:t:`zaykin2011optimally`.
        Default is False.

    Notes
    -----
    Requires ``z`` images and optionally the sample size metadata field.

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.StoufferCombinationTest`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(self, use_sample_size=False, **kwargs):
        super().__init__(**kwargs)
        self.use_sample_size = use_sample_size
        if self.use_sample_size:
            self._required_inputs["sample_sizes"] = ("metadata", "sample_sizes")

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
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
            "z": _boolean_unmask(est_summary.z.squeeze(), self.inputs_["aggressive_mask"]),
            "p": _boolean_unmask(est_summary.p.squeeze(), self.inputs_["aggressive_mask"]),
        }
        return results


class WeightedLeastSquares(IBMAEstimator):
    """Weighted least-squares meta-regression.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Provides the weighted least-squares estimate of the fixed effects given
    known/assumed between-study variance tau^2.
    When tau^2 = 0 (default), the model is the standard inverse-weighted
    fixed-effects meta-regression.

    This method was described in :footcite:t:`brockwell2001comparison`.

    Parameters
    ----------
    tau2 : :obj:`float` or 1D :class:`numpy.ndarray`, optional
        Assumed/known value of tau^2. Must be >= 0. Default is 0.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, tau2=0, **kwargs):
        super().__init__(**kwargs)
        self.tau2 = tau2

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est = pymare.estimators.WeightedLeastSquares(tau2=self.tau2)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        # tau2 is an float, not a map, so it can't go in the results dictionary
        results = {
            "z": _boolean_unmask(
                est_summary.get_fe_stats()["z"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "p": _boolean_unmask(
                est_summary.get_fe_stats()["p"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "est": _boolean_unmask(
                est_summary.get_fe_stats()["est"].squeeze(), self.inputs_["aggressive_mask"]
            ),
        }
        return results


class DerSimonianLaird(IBMAEstimator):
    """DerSimonian-Laird meta-regression estimator.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Estimates the between-subject variance tau^2 using the :footcite:t:`dersimonian1986meta`
    method-of-moments approach :footcite:p:`dersimonian1986meta,kosmidis2017improving`.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.DerSimonianLaird`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        est = pymare.estimators.DerSimonianLaird()
        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": _boolean_unmask(est_summary.tau2.squeeze(), self.inputs_["aggressive_mask"]),
            "z": _boolean_unmask(
                est_summary.get_fe_stats()["z"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "p": _boolean_unmask(
                est_summary.get_fe_stats()["p"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "est": _boolean_unmask(
                est_summary.get_fe_stats()["est"].squeeze(), self.inputs_["aggressive_mask"]
            ),
        }
        return results


class Hedges(IBMAEstimator):
    """Hedges meta-regression estimator.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Estimates the between-subject variance tau^2 using the :footcite:t:`hedges2014statistical`
    approach.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.Hedges`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        est = pymare.estimators.Hedges()
        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": _boolean_unmask(est_summary.tau2.squeeze(), self.inputs_["aggressive_mask"]),
            "z": _boolean_unmask(
                est_summary.get_fe_stats()["z"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "p": _boolean_unmask(
                est_summary.get_fe_stats()["p"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "est": _boolean_unmask(
                est_summary.get_fe_stats()["est"].squeeze(), self.inputs_["aggressive_mask"]
            ),
        }
        return results


class SampleSizeBasedLikelihood(IBMAEstimator):
    """Method estimates with known sample sizes but unknown sampling variances.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Parameters
    ----------
    method : {'ml', 'reml'}, optional
        The estimation method to use. The available options are

        ============== =============================
        "ml" (default) Maximum likelihood
        "reml"         Restricted maximum likelihood
        ============== =============================

    Notes
    -----
    Requires ``beta`` images and sample size from metadata.

    Homogeneity of sigma^2 across studies is assumed.
    The ML and REML solutions are obtained via SciPy’s scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warnings
    --------
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

    def __init__(self, method="ml", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        sample_sizes = np.array([np.mean(n) for n in self.inputs_["sample_sizes"]])
        n_maps = np.tile(sample_sizes, (self.inputs_["beta_maps"].shape[1], 1)).T
        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], n=n_maps)
        est = pymare.estimators.SampleSizeBasedLikelihoodEstimator(method=self.method)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": _boolean_unmask(est_summary.tau2.squeeze(), self.inputs_["aggressive_mask"]),
            "z": _boolean_unmask(
                est_summary.get_fe_stats()["z"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "p": _boolean_unmask(
                est_summary.get_fe_stats()["p"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "est": _boolean_unmask(
                est_summary.get_fe_stats()["est"].squeeze(), self.inputs_["aggressive_mask"]
            ),
        }
        return results


class VarianceBasedLikelihood(IBMAEstimator):
    """A likelihood-based meta-analysis method for estimates with known variances.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    coefficients using the specified likelihood-based estimator (ML or REML)
    :footcite:p:`dersimonian1986meta,kosmidis2017improving`.

    Parameters
    ----------
    method : {'ml', 'reml'}, optional
        The estimation method to use. The available options are

        ============== =============================
        "ml" (default) Maximum likelihood
        "reml"         Restricted maximum likelihood
        ============== =============================

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    The ML and REML solutions are obtained via SciPy's scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warnings
    --------
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
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.VarianceBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, method="ml", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        est = pymare.estimators.VarianceBasedLikelihoodEstimator(method=self.method)

        pymare_dset = pymare.Dataset(y=self.inputs_["beta_maps"], v=self.inputs_["varcope_maps"])
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        results = {
            "tau2": _boolean_unmask(est_summary.tau2.squeeze(), self.inputs_["aggressive_mask"]),
            "z": _boolean_unmask(
                est_summary.get_fe_stats()["z"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "p": _boolean_unmask(
                est_summary.get_fe_stats()["p"].squeeze(), self.inputs_["aggressive_mask"]
            ),
            "est": _boolean_unmask(
                est_summary.get_fe_stats()["est"].squeeze(), self.inputs_["aggressive_mask"]
            ),
        }
        return results


class PermutedOLS(IBMAEstimator):
    r"""An analysis with permuted ordinary least squares (OLS), using nilearn.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    This approach is described in :footcite:t:`freedman1983nonstochastic`.

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

    Warnings
    --------
    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.mass_univariate.permuted_ols : The function used for this IBMA.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(self, two_sided=True, **kwargs):
        super().__init__(**kwargs)
        self.two_sided = two_sided
        self.parameters_ = {}

    def _fit(self, dataset):
        self.dataset = dataset
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
        images = {
            "t": _boolean_unmask(t_map.squeeze(), self.inputs_["aggressive_mask"]),
            "z": _boolean_unmask(z_map.squeeze(), self.inputs_["aggressive_mask"]),
        }
        return images

    def correct_fwe_montecarlo(self, result, n_iters=10000, n_cores=1):
        """Perform FWE correction using the max-value permutation method.

        .. versionchanged:: 0.0.8

            * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

        .. versionadded:: 0.0.4

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from an ALE meta-analysis.
        n_iters : :obj:`int`, optional
            The number of iterations to run in estimating the null distribution.
            Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.

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
        n_cores = _check_ncores(n_cores)

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
        images = {
            "logp_level-voxel": _boolean_unmask(
                log_p_map.squeeze(), self.inputs_["aggressive_mask"]
            ),
            "z_level-voxel": _boolean_unmask(z_map.squeeze(), self.inputs_["aggressive_mask"]),
        }
        return images
