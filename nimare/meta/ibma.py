"""Image-based meta-analysis estimators."""

from __future__ import division

import logging
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
import pymare
from joblib import Memory

try:
    # nilearn>0.10.3
    from nilearn._utils.niimg_conversions import check_same_fov
except ImportError:
    # nilearn < 0.10.3
    from nilearn._utils.niimg_conversions import _check_same_fov as check_same_fov

from nilearn.image import concat_imgs, resample_to_img
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.utils import _apply_liberal_mask
from nimare.transforms import d_to_g, p_to_z, t_to_d, t_to_z
from nimare.utils import _boolean_unmask, _check_ncores, get_masker

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


class IBMAEstimator(Estimator):
    """Base class for meta-analysis methods in :mod:`~nimare.meta`.

    .. versionchanged:: 0.2.1

        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.2.0

        * Remove `resample` and `memory_limit` arguments. Resampling is now
          performed only if shape/affines are different.

    .. versionadded:: 0.0.12

        * IBMA-specific elements of ``Estimator`` excised and used to create ``IBMAEstimator``.
        * Generic kwargs and args converted to named kwargs.
          All remaining kwargs are for resampling.

    """

    def __init__(
        self,
        aggressive_mask=True,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        *,
        mask=None,
        **kwargs,
    ):
        self.aggressive_mask = aggressive_mask

        if mask is not None:
            mask = get_masker(mask, memory=memory, memory_level=memory_level)
        self.masker = mask

        super().__init__(memory=memory, memory_level=memory_level)

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

        # Reserve the key for the correlation matrix
        self.inputs_["corr_matrix"] = None

        if self.aggressive_mask:
            # Ensure that protected values are not included among _required_inputs
            assert (
                "aggressive_mask" not in self._required_inputs.keys()
            ), "This is a protected name."

            if "aggressive_mask" in self.inputs_.keys():
                LGR.warning("Removing existing 'aggressive_mask' from Estimator.")
                self.inputs_.pop("aggressive_mask")
        else:
            # A dictionary to collect data, to be further reduced by the liberal mask.
            self.inputs_["data_bags"] = {}

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "image":
                # Resampling will only occur if shape/affines are different
                imgs = [
                    (
                        nib.load(img)
                        if check_same_fov(nib.load(img), reference_masker=mask_img)
                        else resample_to_img(nib.load(img), mask_img, **self._resample_kwargs)
                    )
                    for img in self.inputs_[name]
                ]

                # input to NiFtiLabelsMasker must be 4d
                img4d = concat_imgs(imgs, ensure_ndim=4)

                # Mask required input images using either the dataset's mask or the estimator's.
                temp_arr = masker.transform(img4d)

                # To save memory, we only save the original image array and perform masking later
                # in the estimator if self.aggressive_mask is True.
                self.inputs_[name] = temp_arr

                if self.aggressive_mask:
                    # Determine the good voxels here
                    nonzero_voxels_bool = np.all(temp_arr != 0, axis=0)
                    nonnan_voxels_bool = np.all(~np.isnan(temp_arr), axis=0)
                    good_voxels_bool = np.logical_and(nonzero_voxels_bool, nonnan_voxels_bool)

                    if "aggressive_mask" not in self.inputs_.keys():
                        self.inputs_["aggressive_mask"] = good_voxels_bool
                    else:
                        # Remove any voxels that are bad in any image-based inputs
                        self.inputs_["aggressive_mask"] = np.logical_or(
                            self.inputs_["aggressive_mask"],
                            good_voxels_bool,
                        )
                else:
                    data_bags = zip(*_apply_liberal_mask(temp_arr))

                    keys = ["values", "voxel_mask", "study_mask"]
                    self.inputs_["data_bags"][name] = [dict(zip(keys, bag)) for bag in data_bags]

        # Further reduce image-based inputs to remove "bad" voxels
        # (voxels with zeros or NaNs in any studies)
        if self.aggressive_mask:
            if n_bad_voxels := (
                self.inputs_["aggressive_mask"].size - self.inputs_["aggressive_mask"].sum()
            ):
                LGR.warning(f"Masking out {n_bad_voxels} additional voxels.")


class Fishers(IBMAEstimator):
    """An image-based meta-analytic test using t- or z-statistic images.

    Requires z-statistic images, but will be extended to work with t-statistic images as well.

    This method is described in :footcite:t:`fisher1946statistical`.

    .. versionchanged:: 0.3.0

        * New parameter: ``two_sided``, controls the type of test to be performed. In addition,
            the default is now set to True (two-sided), which differs from previous versions
            where only one-sided tests were performed.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``z`` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.FisherCombinationTest`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(self, two_sided=True, **kwargs):
        super().__init__(**kwargs)
        self.two_sided = two_sided
        self._mode = "concordant" if self.two_sided else "directed"

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}) on "
            f"{len(self.inputs_['id'])} z-statistic images using the Fisher "
            "combined probability method \\citep{fisher1946statistical}."
        )
        return description

    def _fit_model(self, stat_maps):
        """Fit the model to the data."""
        n_studies, n_voxels = stat_maps.shape

        pymare_dset = pymare.Dataset(y=stat_maps)
        est = pymare.estimators.FisherCombinationTest(mode=self._mode)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        z_map = est_summary.z.squeeze()
        p_map = est_summary.p.squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, dof_map

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

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(self.inputs_["z_maps"][:, voxel_mask])

            z_map, p_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["z_maps"].shape[1]
            z_map = np.zeros(n_voxels, dtype=float)
            p_map = np.zeros(n_voxels, dtype=float)
            dof_map = np.zeros(n_voxels, dtype=np.int32)
            for bag in self.inputs_["data_bags"]["z_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"])

        maps = {"z": z_map, "p": p_map, "dof": dof_map}
        description = self._generate_description()

        return maps, {}, description


class Stouffers(IBMAEstimator):
    """A t-test on z-statistic images.

    Requires z-statistic images.

    This method is described in :footcite:t:`stouffer1949american`.

    .. versionchanged:: 0.3.0

        * New parameter: ``two_sided``, controls the type of test to be performed. In addition,
            the default is now set to True (two-sided), which differs from previous versions
            where only one-sided tests were performed.
        * Add correction for multiple contrasts within a study.
        * New parameter: ``normalize_contrast_weights`` to normalized the weights by the
            number of contrasts in each study.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    use_sample_size : :obj:`bool`, optional
        Whether to use sample sizes for weights (i.e., "weighted Stouffer's") or not,
        as described in :footcite:t:`zaykin2011optimally`.
        Default is False.
    normalize_contrast_weights : :obj:`bool`, optional
        Whether to use number of contrast per study to normalized the weights or not.
        Default is False.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``z`` images and optionally the sample size metadata field.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.StoufferCombinationTest`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(
        self,
        use_sample_size=False,
        normalize_contrast_weights=False,
        two_sided=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_sample_size = use_sample_size
        if self.use_sample_size:
            self._required_inputs["sample_sizes"] = ("metadata", "sample_sizes")

        self.normalize_contrast_weights = normalize_contrast_weights

        self.two_sided = two_sided
        self._mode = "concordant" if self.two_sided else "directed"

    def _preprocess_input(self, dataset):
        """Preprocess additional inputs to the Estimator from the Dataset as needed."""
        super()._preprocess_input(dataset)

        study_mask = dataset.images["id"].isin(self.inputs_["id"])

        # Convert each contrast name to a unique integer value.
        labels = dataset.images["study_id"][study_mask].to_list()
        label_to_int = {label: i for i, label in enumerate(set(labels))}
        label_counts = Counter(labels)

        self.inputs_["contrast_names"] = np.array([label_to_int[label] for label in labels])
        self.inputs_["num_contrasts"] = np.array([label_counts[label] for label in labels])

        n_studies = len(self.inputs_["id"])
        if n_studies != np.unique(self.inputs_["contrast_names"]).size:
            # If all studies are not unique, we will need to correct for multiple contrasts
            # Calculate correlation matrix on valid voxels
            if self.aggressive_mask:
                voxel_mask = self.inputs_["aggressive_mask"]
                self.inputs_["corr_matrix"] = np.corrcoef(
                    self.inputs_["z_maps"][:, voxel_mask],
                    rowvar=True,
                )
            else:
                self.inputs_["corr_matrix"] = np.corrcoef(
                    self.inputs_["z_maps"],
                    rowvar=True,
                )

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}) on "
            f"{len(self.inputs_['id'])} z-statistic images using the Stouffer "
            "method \\citep{stouffer1949american}"
        )

        if self.use_sample_size:
            description += (
                ", with studies weighted by the square root of the study sample sizes, per "
                "\\cite{zaykin2011optimally}."
            )
        else:
            description += "."

        return description

    def _fit_model(self, stat_maps, study_mask=None, corr=None):
        """Fit the model to the data."""
        n_studies, n_voxels = stat_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        est = pymare.estimators.StoufferCombinationTest(mode=self._mode)

        contrast_maps, sub_corr = None, None
        if corr is not None:
            contrast_maps = np.tile(self.inputs_["contrast_names"][study_mask], (n_voxels, 1)).T
            sub_corr = corr[np.ix_(study_mask, study_mask)]

        weights = np.ones(n_studies)

        if self.normalize_contrast_weights:
            weights *= 1 / self.inputs_["num_contrasts"][study_mask]

        if self.use_sample_size:
            sample_sizes = np.array(
                [np.mean(self.inputs_["sample_sizes"][idx]) for idx in study_mask]
            )
            weights *= np.sqrt(sample_sizes)

        weight_maps = np.tile(weights, (n_voxels, 1)).T

        pymare_dset = pymare.Dataset(y=stat_maps, n=weight_maps, v=contrast_maps)
        est.fit_dataset(pymare_dset, corr=sub_corr)
        est_summary = est.summary()

        z_map = est_summary.z.squeeze()
        p_map = est_summary.p.squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, dof_map

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

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]

            result_maps = self._fit_model(
                self.inputs_["z_maps"][:, voxel_mask],
                corr=self.inputs_["corr_matrix"],
            )

            z_map, p_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["z_maps"].shape[1]
            z_map = np.zeros(n_voxels, dtype=float)
            p_map = np.zeros(n_voxels, dtype=float)
            dof_map = np.zeros(n_voxels, dtype=np.int32)
            for bag in self.inputs_["data_bags"]["z_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(
                    bag["values"], bag["study_mask"], corr=self.inputs_["corr_matrix"]
                )

        maps = {"z": z_map, "p": p_map, "dof": dof_map}
        description = self._generate_description()

        return maps, {}, description


class WeightedLeastSquares(IBMAEstimator):
    """Weighted least-squares meta-regression.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" to outputs.

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
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    tau2 : :obj:`float` or 1D :class:`numpy.ndarray`, optional
        Assumed/known value of tau^2. Must be >= 0. Default is 0.

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

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

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta images using the Weighted Least Squares approach "
            "\\citep{brockwell2001comparison}, "
            f"with an a priori tau-squared value of {self.tau2} defined across all voxels."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps)
        est = pymare.estimators.WeightedLeastSquares(tau2=self.tau2)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, est_map, se_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map = [np.zeros(n_voxels, dtype=float) for _ in range(4)]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(beta_bag["values"], varcope_bag["values"])

        # tau2 is a float, not a map, so it can't go into the results dictionary
        tables = {"level-estimator": pd.DataFrame(columns=["tau2"], data=[self.tau2])}
        maps = {"z": z_map, "p": p_map, "est": est_map, "se": se_map, "dof": dof_map}
        description = self._generate_description()

        return maps, tables, description


class DerSimonianLaird(IBMAEstimator):
    """DerSimonian-Laird meta-regression estimator.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" to outputs.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Estimates the between-subject variance tau^2 using the :footcite:t:`dersimonian1986meta`
    method-of-moments approach :footcite:p:`dersimonian1986meta,kosmidis2017improving`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.DerSimonianLaird`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta and variance images using the "
            "DerSimonian-Laird method \\citep{dersimonian1986meta}, in which tau-squared is "
            "estimated on a voxel-wise basis using the method-of-moments approach "
            "\\citep{dersimonian1986meta,kosmidis2017improving}."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps)
        est = pymare.estimators.DerSimonianLaird()
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, est_map, se_map, tau2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(5)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    tau2_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(beta_bag["values"], varcope_bag["values"])

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "dof": dof_map,
        }
        description = self._generate_description()

        return maps, {}, description


class Hedges(IBMAEstimator):
    """Hedges meta-regression estimator.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" to outputs.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Estimates the between-subject variance tau^2 using the :footcite:t:`hedges2014statistical`
    approach.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.Hedges`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta and variance images using the Hedges "
            "method \\citep{hedges2014statistical}, in which tau-squared is estimated on a "
            "voxel-wise basis."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps)
        est = pymare.estimators.Hedges()
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, est_map, se_map, tau2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(5)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    tau2_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(beta_bag["values"], varcope_bag["values"])

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "dof": dof_map,
        }
        description = self._generate_description()

        return maps, {}, description


class SampleSizeBasedLikelihood(IBMAEstimator):
    """Method estimates with known sample sizes but unknown sampling variances.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" and "sigma2" to outputs.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    method : {'ml', 'reml'}, optional
        The estimation method to use. The available options are

        ============== =============================
        "ml" (default) Maximum likelihood
        "reml"         Restricted maximum likelihood
        ============== =============================

    Notes
    -----
    Requires :term:`beta` images and sample size from metadata.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "sigma2"       Estimated within-study variance. Assumed to be the same for all studies.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Homogeneity of sigma^2 across studies is assumed.
    The ML and REML solutions are obtained via SciPy's scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warnings
    --------
    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

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

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta images using sample size-based "
            "maximum likelihood estimation, in which tau-squared and sigma-squared are estimated "
            "on a voxel-wise basis."
        )
        return description

    def _fit_model(self, beta_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        sample_sizes = np.array([np.mean(self.inputs_["sample_sizes"][idx]) for idx in study_mask])
        n_maps = np.tile(sample_sizes, (n_voxels, 1)).T

        pymare_dset = pymare.Dataset(y=beta_maps, n=n_maps)
        est = pymare.estimators.SampleSizeBasedLikelihoodEstimator(method=self.method)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        fe_stats = est_summary.get_fe_stats()

        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        sigma2_map = est.params_["sigma2"].squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, est_map, se_map, tau2_map, sigma2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, sigma2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map, sigma2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(6)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            for bag in self.inputs_["data_bags"]["beta_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    est_map[bag["voxel_mask"]],
                    se_map[bag["voxel_mask"]],
                    tau2_map[bag["voxel_mask"]],
                    sigma2_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"], bag["study_mask"])

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "sigma2": sigma2_map,
            "dof": dof_map,
        }
        description = self._generate_description()

        return maps, {}, description


class VarianceBasedLikelihood(IBMAEstimator):
    """A likelihood-based meta-analysis method for estimates with known variances.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        Add "se" output.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    coefficients using the specified likelihood-based estimator (ML or REML)
    :footcite:p:`dersimonian1986meta,kosmidis2017improving`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    method : {'ml', 'reml'}, optional
        The estimation method to use. The available options are

        ============== =============================
        "ml" (default) Maximum likelihood
        "reml"         Restricted maximum likelihood
        ============== =============================

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

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

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

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

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta and variance images using "
            "variance-based maximum likelihood estimation, in which tau-squared is estimated on a "
            "voxel-wise basis."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps)
        est = pymare.estimators.VarianceBasedLikelihoodEstimator(method=self.method)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        fe_stats = est_summary.get_fe_stats()

        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, est_map, se_map, tau2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(5)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    tau2_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(beta_bag["values"], varcope_bag["values"])

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "dof": dof_map,
        }
        description = self._generate_description()

        return maps, {}, description


class PermutedOLS(IBMAEstimator):
    r"""An analysis with permuted ordinary least squares (OLS), using nilearn.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Use beta maps instead of z maps.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    This approach is described in :footcite:t:`freedman1983nonstochastic`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``beta`` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "t"            T-statistic map from one-sample test.
    "z"            Z-statistic map from one-sample test.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Available correction methods: :func:`PermutedOLS.correct_fwe_montecarlo`

    Warnings
    --------
    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.mass_univariate.permuted_ols : The function used for this IBMA.
    """

    _required_inputs = {"beta_maps": ("image", "beta")}

    def __init__(self, two_sided=True, **kwargs):
        super().__init__(**kwargs)
        self.two_sided = two_sided
        self.parameters_ = {}

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta images using Nilearn's "
            "\\citep{10.3389/fninf.2014.00014} permuted ordinary least squares method."
        )
        return description

    def _fit_model(self, beta_maps, n_perm=0):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        # Use intercept as explanatory variable
        tested_vars = np.ones((n_studies, 1))
        confounding_vars = None

        log_p_map, t_map, _ = permuted_ols(
            tested_vars,
            beta_maps,
            confounding_vars=confounding_vars,
            model_intercept=False,  # modeled by tested_vars
            n_perm=n_perm,
            two_sided_test=self.two_sided,
            random_state=42,
            n_jobs=1,
            verbose=0,
        )

        # Convert t to z, preserving signs
        dof = n_studies - 1

        z_map = t_to_z(t_map, dof)
        dof_map = np.tile(dof, n_voxels).astype(np.int32)

        return log_p_map.squeeze(), t_map.squeeze(), z_map.squeeze(), dof_map

    def _fit(self, dataset):
        self.dataset = dataset

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(self.inputs_["beta_maps"][:, voxel_mask])

            # Skip log_p_map
            t_map, z_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps[1:])
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]
            t_map = np.zeros(n_voxels, dtype=float)
            z_map = np.zeros(n_voxels, dtype=float)
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            for bag in self.inputs_["data_bags"]["beta_maps"]:
                (
                    _,  # Skip log_p_map
                    t_map[bag["voxel_mask"]],
                    z_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"])

        maps = {"t": t_map, "z": z_map, "dof": dof_map}
        description = self._generate_description()

        return maps, {}, description

    def correct_fwe_montecarlo(self, result, n_iters=5000, n_cores=1):
        """Perform FWE correction using the max-value permutation method.

        .. versionchanged:: 0.0.8

            * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

        .. versionadded:: 0.0.4

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from an ALE meta-analysis.
        n_iters : :obj:`int`, default=5000
            The number of iterations to run in estimating the null distribution.
            Default is 5000.
        n_cores : :obj:`int`, default=1
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'p_level-voxel', 'z_level-voxel', 'logp_level-voxel'.

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

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            log_p_map, t_map, _, _ = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask], n_perm=n_iters
            )

            # Fill complete maps
            p_map = np.power(10.0, -log_p_map)

            # Convert p to z, preserving signs
            sign = np.sign(t_map)
            sign[sign == 0] = 1
            z_map = p_to_z(p_map, tail="two") * sign

            log_p_map = _boolean_unmask(log_p_map, voxel_mask)
            z_map = _boolean_unmask(z_map, voxel_mask)

        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]
            log_p_map = np.zeros(n_voxels, dtype=float)
            z_map = np.zeros(n_voxels, dtype=float)

            for bag in self.inputs_["data_bags"]["beta_maps"]:
                log_p_map_tmp, t_map_tmp, _, _ = self._fit_model(
                    self.inputs_["beta_maps"][:, bag["voxel_mask"]], n_perm=n_iters
                )

                # Fill complete maps
                p_map_tmp = np.power(10.0, -log_p_map_tmp)

                # Convert p to z, preserving signs
                sign = np.sign(t_map_tmp)
                sign[sign == 0] = 1
                z_map_tmp = p_to_z(p_map_tmp, tail="two") * sign

                log_p_map[bag["voxel_mask"]] = log_p_map_tmp.squeeze()
                z_map[bag["voxel_mask"]] = z_map_tmp.squeeze()

        maps = {"logp_level-voxel": log_p_map, "z_level-voxel": z_map}
        description = (
            "Family-wise error rate correction was performed using Nilearn's "
            "\\citep{10.3389/fninf.2014.00014} permuted OLS method, in which null distributions "
            "of test statistics were estimated using the "
            "max-value permutation method detailed in \\cite{freedman1983nonstochastic}. "
            f"{n_iters} iterations were performed to generate the null distribution."
        )

        return maps, {}, description


class FixedEffectsHedges(IBMAEstimator):
    """Fixed Effects Hedges meta-regression estimator.

    .. versionadded:: 0.4.0

    Provides the weighted least-squares estimate of the fixed effects using Hedge's g
    as the point estimate and the variance of bias-corrected Cohen's d as the variance
    estimate, and given known/assumed between-study variance tau^2.
    When tau^2 = 0 (default), the model is the standard inverse-weighted
    fixed-effects meta-regression.

    This method was described in :footcite:t:`bossier2019`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    tau2 : :obj:`float` or 1D :class:`numpy.ndarray`, optional
        Assumed/known value of tau^2. Must be >= 0. Default is 0.

    Notes
    -----
    Requires `t` images and sample size from metadata.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"t_maps": ("image", "t"), "sample_sizes": ("metadata", "sample_sizes")}

    def __init__(self, tau2=0, **kwargs):
        super().__init__(**kwargs)
        self.tau2 = tau2

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} t-statistic images using Heges' g as point estimates "
            "and the variance of bias-corrected Cohen's in a Weighted Least Squares approach "
            "\\citep{brockwell2001comparison,bossier2019}, "
            f"with an a priori tau-squared value of {self.tau2} defined across all voxels."
        )
        return description

    def _fit_model(self, t_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = t_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        sample_sizes = np.array([np.mean(self.inputs_["sample_sizes"][idx]) for idx in study_mask])
        n_maps = np.tile(sample_sizes, (n_voxels, 1)).T

        # Calculate Hedge's g maps: Standardized mean
        cohens_maps = t_to_d(t_maps, n_maps)
        hedges_maps, var_hedges_maps = d_to_g(cohens_maps, n_maps, return_variance=True)

        del n_maps, sample_sizes, cohens_maps

        pymare_dset = pymare.Dataset(y=hedges_maps, v=var_hedges_maps)
        est = pymare.estimators.WeightedLeastSquares(tau2=self.tau2)
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        dof_map = np.tile(n_studies - 1, n_voxels).astype(np.int32)

        return z_map, p_map, est_map, se_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(self.inputs_["t_maps"][:, voxel_mask])

            z_map, p_map, est_map, se_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["t_maps"].shape[1]

            z_map, p_map, est_map, se_map = [np.zeros(n_voxels, dtype=float) for _ in range(4)]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            for bag in self.inputs_["data_bags"]["t_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    est_map[bag["voxel_mask"]],
                    se_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"], bag["study_mask"])

        # tau2 is a float, not a map, so it can't go into the results dictionary
        tables = {"level-estimator": pd.DataFrame(columns=["tau2"], data=[self.tau2])}
        maps = {"z": z_map, "p": p_map, "est": est_map, "se": se_map, "dof": dof_map}
        description = self._generate_description()

        return maps, tables, description
