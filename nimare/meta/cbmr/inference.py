"""Statistical inference on fitted CBMR results."""

import copy
import logging
import re
from functools import wraps

import numpy as np
import pandas as pd
import scipy.sparse
from nilearn.image import resample_to_img

from nimare.meta import models
from nimare.meta.cbmr._helpers import (
    _INHERIT_INCIDENCE_THRESHOLD,
    _as_csr_matrix,
    _csr_row_indices,
    _is_named_pairwise_contrast,
    _normalize_named_pairwise_contrasts,
    _uses_cuda,
    _validate_incidence_threshold,
)
from nimare.meta.cbmr._torch import torch
from nimare.meta.cbmr.estimator import CBMREstimator
from nimare.meta.cbmr.results import CBMRResult
from nimare.transforms import chi2_to_nlogp, nlogp_to_z, z_to_nlogp
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _clip_p_values,
    _nlogp_to_logp_values,
    get_masker,
)

LGR = logging.getLogger(__name__)


class CBMRInference:
    """Statistical inference on fitted CBMR results.

    Notes
    -----
        This is the public inference entry point.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    device : :obj:`string`, optional
        Device type ('cpu' or 'cuda') represents the device on which operations will be allocated.
        Default is 'cpu'.
    moderator_effect : {"voxelwise", "global"}, optional
        Inference parameterization to use. ``"voxelwise"`` uses the integrated voxelwise CBMR
        inference backend with sandwich or inverse-Fisher covariance estimates. ``"global"`` uses
        the standard CBMR inference backend. Default is ``"global"``.
    method : {"sandwich", "FI"}, optional
        Covariance estimator for CBMR inference. The voxelwise default is ``"sandwich"`` because
        it uses empirical residual variation to provide standard errors that are more robust to
        model misspecification, study-level clustering, and departures from idealized Poisson
        assumptions common in coordinate-based meta-analysis. The global default is ``"FI"`` to
        preserve the historical inverse-Fisher behavior. ``"FI"`` can be more efficient when the
        likelihood, mean-variance relationship, and independence assumptions are correctly
        specified, but may be too optimistic otherwise.
    sandwich_meat : {"cluster", "iid"}, optional
        Meat estimator for voxelwise sandwich covariance. ``"cluster"`` aggregates scores by
        experiment, while ``"iid"`` treats experiment-voxel observations independently. For global
        CBMR the marginal spatial and moderator score rows are already aggregated, so both options
        use the same row-wise meat.
    sandwich_correction : {None, "hc0", "hc1", "hc3"}, optional
        Optional HC-style finite-sample/leverage correction for sandwich covariance.
    ridge : :obj:`float`, optional
        Nonnegative ridge added before inverting the Fisher information matrix.
    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, or Nilearn masker, optional
        Optional ROI mask used to restrict voxelwise inference outputs from a fitted result.
        If None, the fitted result's analysis mask is used.
    incidence_threshold : :obj:`float` or None, optional
        Drop voxels with empirical focus incidence less than or equal to this threshold when
        incidence information is available on the fitted result. Use None to keep all fitted
        voxels. By default, inherit the fitted estimator's incidence threshold.
    """

    _valid_methods = ("sandwich", "FI")
    _valid_sandwich_meats = ("cluster", "iid")
    _valid_sandwich_corrections = (None, "hc0", "hc1", "hc3")
    _voxelwise_default_method = "sandwich"
    _global_default_method = "FI"

    def __init__(
        self,
        device="cpu",
        moderator_effect="global",
        method=None,
        sandwich_meat="cluster",
        sandwich_correction="hc3",
        ridge=1e-6,
        mask=None,
        incidence_threshold=_INHERIT_INCIDENCE_THRESHOLD,
    ):
        self.moderator_effect = self._normalize_moderator_effect(moderator_effect)
        self.device = device
        self.mask = mask
        self.masker = get_masker(mask) if mask is not None else None
        self._inherit_incidence_threshold = incidence_threshold is _INHERIT_INCIDENCE_THRESHOLD
        self.incidence_threshold = (
            incidence_threshold
            if self._inherit_incidence_threshold
            else _validate_incidence_threshold(incidence_threshold)
        )
        # device check
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("CUDA not found; using device 'cpu'.")
            self.device = "cpu"

        self.result = None
        self.groups = None
        self.moderators = None

        if self.moderator_effect == "global" and method is None:
            method = self._global_default_method
        elif method is None:
            method = self._voxelwise_default_method

        self.method = self._validate_method(method)
        self.sandwich_meat = self._validate_sandwich_meat(sandwich_meat)
        self.sandwich_correction = self._validate_sandwich_correction(sandwich_correction)
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        self.ridge = ridge

        self._reset_fitted_coefficient_caches()
        self._reset_inference_caches()

    @classmethod
    def _normalize_moderator_effect(cls, moderator_effect):
        """Normalize the shared public moderator-effect selector."""
        return CBMREstimator._validate_moderator_effect(moderator_effect)

    @staticmethod
    def _validate_choice(value, choices, error_message, case_map=None):
        """Validate a string-like option against allowed choices."""
        if isinstance(value, str):
            normalized = value.lower()
            value = case_map.get(normalized, normalized) if case_map else normalized
        if value not in choices:
            raise ValueError(error_message)
        return value

    @classmethod
    def _validate_method(cls, method):
        """Validate and normalize an inference standard-error method."""
        return cls._validate_choice(
            method,
            cls._valid_methods,
            "method must be one of {'sandwich', 'FI'}.",
            case_map={"fi": "FI", "sandwich": "sandwich"},
        )

    @classmethod
    def _validate_sandwich_meat(cls, sandwich_meat):
        """Validate and normalize the sandwich meat estimator."""
        return cls._validate_choice(
            sandwich_meat,
            cls._valid_sandwich_meats,
            "sandwich_meat must be either 'cluster' or 'iid'.",
        )

    @classmethod
    def _validate_sandwich_correction(cls, sandwich_correction):
        """Validate and normalize the sandwich leverage correction."""
        return cls._validate_choice(
            sandwich_correction,
            cls._valid_sandwich_corrections,
            "sandwich_correction must be None, 'hc0', 'hc1', or 'hc3'.",
        )

    def _validate_global_sandwich_model(self):
        """Validate model support for global robust covariance."""
        model = getattr(self, "estimator", None)
        model = getattr(model, "model", None)
        if model is not None and not isinstance(model, models.PoissonEstimator):
            raise ValueError(
                "Global sandwich inference is currently supported only for "
                "model=models.PoissonEstimator."
            )

    def _check_fit(fn):
        """Check if CBMRInference instance has been fit."""

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if self.result is None:
                raise ValueError("CBMRInference instance has not been fit.")
            return fn(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def _copy_result_for_inference(result):
        """Create an inference result copy without deep-copying all stored arrays and tables."""
        copied_result = copy.copy(result)
        copied_result.estimator = copy.deepcopy(result.estimator)
        copied_result.maps = {
            map_name: np.array(map_, copy=True) for map_name, map_ in result.maps.items()
        }
        copied_result.tables = {
            table_name: table.copy(deep=True) for table_name, table in result.tables.items()
        }
        copied_result.metadata = copy.deepcopy(result.metadata)
        return copied_result

    def _reset_inference_caches(self):
        """Reset cached inference/covariance intermediates for the fitted result."""
        self._group_log_intensity_cache = {}
        self._group_null_log_intensity_cache = {}
        if self.moderator_effect in ("global", "mixed"):
            self._group_spatial_covariance_cache = {}
            self._moderator_covariance = None
            self._moderator_variance = None
            self._mixed_joint_covariance_cache = {}
        if self.moderator_effect in ("voxelwise", "mixed"):
            self._group_covariance_cache = {}

    def _reset_fitted_coefficient_caches(self):
        """Reset coefficient-derived state that belongs to one fitted result."""
        if self.moderator_effect in ("global", "mixed"):
            self._moderator_coef_table = None
        if self.moderator_effect in ("voxelwise", "mixed"):
            self._group_coefficient_cache = {}

    def _get_group_log_intensity(self, group):
        """Return cached group log-intensity values."""
        group_log_intensity = self._group_log_intensity_cache.get(group)
        if group_log_intensity is None:
            if self.moderator_effect == "global":
                group_log_intensity = np.log(self.result.maps[f"spatialIntensity_group-{group}"])
            else:  # voxelwise or mixed
                bases = self.estimator.inputs_["coef_spline_bases"]
                coefficient = self._get_group_coefficient_matrix(group)
                group_log_intensity = bases @ coefficient[-1]
            self._group_log_intensity_cache[group] = group_log_intensity
        return group_log_intensity

    def _get_group_null_log_intensity(self, group):
        """Return cached null baseline log-intensity for homogeneity testing."""
        group_null_log_intensity = self._group_null_log_intensity_cache.get(group)
        if group_null_log_intensity is None:
            if self.moderator_effect == "global":
                group_foci_per_voxel = self.estimator.inputs_["foci_per_voxel"][group]
                group_foci_per_experiment = self.estimator.inputs_["foci_per_experiment"][group]
                n_voxels, n_experiments = (
                    group_foci_per_voxel.shape[0],
                    group_foci_per_experiment.shape[0],
                )
                group_null_log_intensity = np.log(
                    np.sum(group_foci_per_voxel) / (n_voxels * n_experiments)
                )
            else:  # voxelwise or mixed
                foci = self.estimator.inputs_["foci_by_experiment_voxel"][group]
                total_foci = foci.sum()
                n_experiments, n_voxels = foci.shape
                group_null_log_intensity = np.log(max(float(total_foci), np.finfo(float).tiny))
                group_null_log_intensity -= np.log(n_experiments * n_voxels)
            self._group_null_log_intensity_cache[group] = group_null_log_intensity
        return group_null_log_intensity

    def _get_group_spatial_covariance(self, involved_groups):
        """Return cached spatial covariance for the involved groups."""
        group_key = (tuple(involved_groups), self.method, self.sandwich_correction, self.ridge)
        cov_spatial_coef = self._group_spatial_covariance_cache.get(group_key)
        if cov_spatial_coef is None:
            if self.method == "FI":
                moderators_by_group = (
                    self.estimator.inputs_["moderators_by_group"] if self.moderators else None
                )
                f_spatial_coef = self.estimator.model.fisher_info_multiple_group_spatial(
                    involved_groups,
                    self.estimator.inputs_["coef_spline_bases"],
                    moderators_by_group,
                    self.estimator.inputs_["foci_per_voxel"],
                    self.estimator.inputs_["foci_per_experiment"],
                )
                cov_spatial_coef = np.linalg.inv(f_spatial_coef)
            else:
                cov_spatial_coef = self._compute_global_spatial_sandwich_covariance(
                    involved_groups
                )
            self._group_spatial_covariance_cache[group_key] = cov_spatial_coef
        return cov_spatial_coef

    def _get_moderator_covariance(self):
        """Return cached moderator covariance and marginal variances."""
        if self._moderator_covariance is None:
            if self.moderator_effect == "mixed":
                self._moderator_covariance = self._get_mixed_global_moderator_covariance()
            elif self.method == "FI":
                moderators_by_group = (
                    self.estimator.inputs_["moderators_by_group"] if self.moderators else None
                )
                f_moderator_coef = self.estimator.model.fisher_info_multiple_group_moderator(
                    self.estimator.inputs_["coef_spline_bases"],
                    moderators_by_group,
                    self.estimator.inputs_["foci_per_voxel"],
                    self.estimator.inputs_["foci_per_experiment"],
                )
                self._moderator_covariance = np.linalg.inv(f_moderator_coef)
            else:
                self._moderator_covariance = self._compute_global_moderator_sandwich_covariance()
            self._moderator_variance = np.diag(self._moderator_covariance)
        return self._moderator_covariance, self._moderator_variance

    @staticmethod
    def _resample_mask_to_reference(mask_img, reference_img):
        """Return mask data resampled into the reference mask grid."""
        same_grid = mask_img.shape == reference_img.shape and np.allclose(
            mask_img.affine, reference_img.affine
        )
        if not same_grid:
            mask_img = resample_to_img(mask_img, reference_img, interpolation="nearest")
        return np.asanyarray(mask_img.dataobj).astype(bool)

    def _voxel_selection_for_result(self, result):
        """Return a boolean voxel selector for optional inference masking."""
        reference_img = result.masker.mask_img
        reference_data = np.asanyarray(reference_img.dataobj).astype(bool)
        n_voxels = int(reference_data.sum())
        keep_voxels = np.ones(n_voxels, dtype=bool)

        if self.masker is not None:
            mask_img = self.masker.mask_img
            requested_data = self._resample_mask_to_reference(mask_img, reference_img)
            keep_voxels &= requested_data.ravel()[np.flatnonzero(reference_data.ravel())]

        incidence_rate = result.estimator.inputs_.get("empirical_incidence_rate")
        if self.incidence_threshold is not None and incidence_rate is not None:
            incidence_rate = np.asarray(incidence_rate)
            if incidence_rate.shape[0] != n_voxels:
                raise ValueError("Stored empirical incidence rates do not match the result mask.")
            keep_voxels &= incidence_rate > self.incidence_threshold

        if not np.any(keep_voxels):
            raise ValueError(
                "No voxels survived CBMR inference masking. Lower incidence_threshold or "
                "provide a less restrictive mask."
            )
        return keep_voxels

    @staticmethod
    def _subset_sparse_columns(matrix, keep_voxels):
        """Subset sparse or dense experiment-by-voxel matrices by voxel columns."""
        if scipy.sparse.issparse(matrix):
            return _as_csr_matrix(matrix)[:, keep_voxels]
        return np.asarray(matrix)[:, keep_voxels]

    def _restrict_result_voxels(self, result):
        """Restrict a fitted result to the requested inference ROI/incidence set."""
        keep_voxels = self._voxel_selection_for_result(result)
        if np.all(keep_voxels):
            return result

        reference_img = result.masker.mask_img
        reference_data = np.asanyarray(reference_img.dataobj).astype(bool)
        kept_flat_indices = np.flatnonzero(reference_data.ravel())[keep_voxels]
        restricted_mask_data = np.zeros(reference_data.size, dtype=bool)
        restricted_mask_data[kept_flat_indices] = True
        restricted_mask_data = restricted_mask_data.reshape(reference_data.shape)
        restricted_mask_img = CBMREstimator._mask_image_from_data(
            restricted_mask_data, reference_img
        )
        restricted_masker = get_masker(restricted_mask_img)

        n_voxels = int(reference_data.sum())
        result.maps = {
            map_name: map_[keep_voxels] if map_.shape[0] == n_voxels else map_
            for map_name, map_ in result.maps.items()
        }
        result.masker = restricted_masker
        result.estimator.masker = restricted_masker
        result.estimator.inputs_["mask_img"] = restricted_mask_img
        result.estimator.inputs_["coef_spline_bases"] = result.estimator.inputs_[
            "coef_spline_bases"
        ][keep_voxels]

        incidence_rate = result.estimator.inputs_.get("empirical_incidence_rate")
        if incidence_rate is not None:
            result.estimator.inputs_["empirical_incidence_rate"] = np.asarray(incidence_rate)[
                keep_voxels
            ]

        for key in ("foci_by_experiment", "foci_by_experiment_voxel"):
            if key not in result.estimator.inputs_:
                continue
            for group, matrix in list(result.estimator.inputs_[key].items()):
                result.estimator.inputs_[key][group] = self._subset_sparse_columns(
                    matrix,
                    keep_voxels,
                )

        if "foci_by_experiment" in result.estimator.inputs_:
            for group, matrix in result.estimator.inputs_["foci_by_experiment"].items():
                result.estimator.inputs_["foci_per_voxel"][group] = np.asarray(
                    matrix.sum(axis=0),
                    dtype=np.int32,
                ).reshape((-1, 1))
                result.estimator.inputs_["foci_per_experiment"][group] = np.asarray(
                    matrix.sum(axis=1),
                    dtype=np.int32,
                ).reshape((-1, 1))
        return result

    def fit(self, result):
        """Fit CBMRInference instance.

        Parameters
        ----------
        result : :obj:`~nimare.meta.cbmr.CBMRResult`
            Fitted CBMR result containing regression coefficient tables and spatial intensity
            maps.
        """
        if not isinstance(result, CBMRResult) or result.moderator_effect != self.moderator_effect:
            raise TypeError(
                f"CBMRInference.fit with moderator_effect={self.moderator_effect!r} requires a "
                f"CBMRResult with moderator_effect={self.moderator_effect!r}."
            )

        if self._inherit_incidence_threshold:
            self.incidence_threshold = getattr(result.estimator, "incidence_threshold", None)

        self.result = self._copy_result_for_inference(result)
        self.result = self._restrict_result_voxels(self.result)
        self._reset_fitted_coefficient_caches()
        self._reset_inference_caches()
        self.estimator = self.result.estimator
        self.groups = list(self.result.groups)
        self.moderators = list(self.result.moderators)

        if self.moderator_effect in ("global", "mixed"):
            self.estimator.device = self.device
            if hasattr(self.estimator, "model"):
                self.estimator.model.device = self.device
                self.estimator.model.to(self.device)
                self.estimator.model._invalidate_tensor_inputs_cache()
            if self.method == "sandwich" and self.moderator_effect == "global":
                self._validate_global_sandwich_model()
            if self.moderator_effect == "mixed" and self.estimator.global_moderators:
                self._moderator_coef_table = (
                    self.result.tables["global_moderators_regression_coef"].to_numpy().T
                )
            elif self.moderators:
                self._moderator_coef_table = (
                    self.result.tables["moderators_regression_coef"].to_numpy().T
                )

        self._create_regular_expressions()

        self.group_reference_dict = {
            group_name: index for index, group_name in enumerate(self.groups)
        }
        self.moderator_reference_dict = {}
        if self.moderators:
            self.moderator_reference_dict = {
                moderator_name: index for index, moderator_name in enumerate(self.moderators)
            }
            for moderator_name, index in self.moderator_reference_dict.items():
                LGR.info("%s = index_%d", moderator_name, index)

    @_check_fit
    def display(self):
        """Display group and moderator names and order."""
        # Visualize group/moderator names and their indices in the contrast array.
        LGR.info("Group Reference in contrast array")
        for group, index in self.group_reference_dict.items():
            LGR.info("%s = index_%d", group, index)
        if self.moderators:
            LGR.info("Moderator Reference in contrast array")
            for moderator, index in self.moderator_reference_dict.items():
                LGR.info("%s = index_%d", moderator, index)

    def _create_regular_expressions(self):
        """Create regular expressions for parsing contrast names.

        Creates the following attributes:
        self.groups_regular_expression: regular expression for parsing group names
        self.moderators_regular_expression: regular expression for parsing moderator names

        Usage:
        >>> self.groups_regular_expression.match("group1 - group2").groupdict()
        """
        operator = "(\\ ?(?P<operator>[+-]?)\\ ??)"
        for attr in ["groups", "moderators"]:
            groups = getattr(self, attr)
            if groups:
                first_group, second_group = [
                    f"(?P<{order}>{'|'.join([re.escape(g) for g in groups])})"
                    for order in ["first", "second"]
                ]
                reg_expr = re.compile(first_group + "(" + operator + second_group + "?)")
            else:
                reg_expr = None

            setattr(self, f"{attr}_regular_expression", reg_expr)

    def _contrast_source_context(self, source):
        """Return the fitted names, parser, and index lookup for one contrast source."""
        if source == "groups":
            return self.groups, self.groups_regular_expression, self.group_reference_dict
        if source == "moderators":
            return (
                self.moderators,
                self.moderators_regular_expression,
                self.moderator_reference_dict,
            )
        return None, None, None

    def _create_named_contrast_vector(self, contrast, source):
        """Create one named contrast vector for groups or moderators."""
        regressors, regular_expression, reference_dict = self._contrast_source_context(source)
        contrast_vector = np.zeros(len(regressors))
        contrast_match = regular_expression.match(contrast)
        if contrast_match is None:
            raise ValueError(f"{contrast} is not a valid contrast.")

        contrast_parts = contrast_match.groupdict()
        if all(contrast_parts.values()):
            contrast_vector[reference_dict[contrast_parts["first"]]] = 1
            contrast_vector[reference_dict[contrast_parts["second"]]] = int(
                contrast_parts["operator"] + "1"
            )
        else:
            contrast_vector[reference_dict[contrast]] = 1
        return contrast_vector

    @_check_fit
    def create_contrast(self, contrast_name, source="groups"):
        """Create contrast matrix for generalized hypothesis testing (GLH).

        Named group contrasts may refer to a single group (for a homogeneity test) or a pairwise
        comparison such as ``group_a-group_b``. Named moderator contrasts follow the same pattern.

        Parameters
        ----------
        contrast_name : :obj:`~string` or sequence of :obj:`~string`
            Name or names of the contrasts to construct.
        source : {"groups", "moderators"}, optional
            Whether to build group or moderator contrasts.
        """
        contrast_name = _normalize_named_pairwise_contrasts(contrast_name)
        contrast_matrix = {}
        if source in ("groups", "moderators"):
            for contrast in contrast_name:
                contrast_matrix[contrast] = self._create_named_contrast_vector(contrast, source)

        return contrast_matrix

    @_check_fit
    def transform(self, t_con_groups=None, t_con_moderators=None, method=None):
        """Conduct generalized linear hypothesis (GLH) testing on CBMR estimates.

        Parameters
        ----------
        t_con_groups : bool, dict, list, tuple, str, or None, optional
            Group homogeneity or comparison specification. Use ``False`` to skip group inference.
        t_con_moderators : bool, dict, list, tuple, str, or None, optional
            Moderator effect or comparison specification. Use ``False`` to skip moderator
            inference.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for CBMR inference.
        """
        if method is not None:
            validated_method = self._validate_method(method)
            if self.moderator_effect == "global" and validated_method == "sandwich":
                self._validate_global_sandwich_model()
            if validated_method != self.method:
                self.method = validated_method
                self._reset_inference_caches()

        if self.moderator_effect in ("voxelwise", "mixed"):
            self.result.metadata["voxelwise_cbmr_inference_method"] = self.method
            if self.method == "sandwich":
                self.result.metadata["voxelwise_cbmr_sandwich_meat"] = self.sandwich_meat
                self.result.metadata["voxelwise_cbmr_sandwich_correction"] = (
                    self.sandwich_correction
                )
            else:
                self.result.metadata.pop("voxelwise_cbmr_sandwich_meat", None)
                self.result.metadata.pop("voxelwise_cbmr_sandwich_correction", None)
        if self.moderator_effect in ("global", "mixed"):
            self.result.metadata["global_cbmr_inference_method"] = self.method
            if self.method == "sandwich":
                self.result.metadata["global_cbmr_sandwich_meat"] = self.sandwich_meat
                self.result.metadata["global_cbmr_sandwich_correction"] = self.sandwich_correction
            else:
                self.result.metadata.pop("global_cbmr_sandwich_meat", None)
                self.result.metadata.pop("global_cbmr_sandwich_correction", None)

        self.t_con_groups = t_con_groups
        self.t_con_moderators = t_con_moderators

        prepared_groups = self._preprocess_t_con_regressor(source="groups")
        if prepared_groups is not None:
            self.t_con_groups, self.t_con_groups_name = prepared_groups
            self._run_group_inference()
        prepared_moderators = self._preprocess_t_con_regressor(source="moderators")
        if prepared_moderators is not None:
            self.t_con_moderators, self.t_con_moderators_name = prepared_moderators
            self._run_moderator_inference()

        return self.result

    def fit_transform(self, result, t_con_groups=None, t_con_moderators=None, method=None):
        """Fit the inference engine and conduct GLH testing on CBMR estimates.

        Parameters
        ----------
        result : :obj:`~nimare.meta.cbmr.CBMRResult`
            Fitted CBMR result containing regression coefficient tables and spatial intensity
            maps.
        t_con_groups : bool, dict, list, tuple, str, or None, optional
            Group homogeneity or comparison specification. Use ``False`` to skip group inference.
        t_con_moderators : bool, dict, list, tuple, str, or None, optional
            Moderator effect or comparison specification. Use ``False`` to skip moderator
            inference.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        """
        self.fit(result)
        return self.transform(t_con_groups, t_con_moderators, method=method)

    @_check_fit
    def _preprocess_t_con_regressor(self, source):
        """Preprocess contrast vector/matrix for GLH testing.

        Follow the steps below:
        (1) Remove groups not involved in contrast;
        (2) Standardize contrast matrix (row sum to 1);
        (3) Remove duplicate rows in contrast matrix.

        Parameters
        ----------
        source : :obj:`~string`
            Source of contrast matrix, either "groups" or "moderators".

        Returns
        -------
        t_con_regressor : :obj:`~list`
            Preprocessed contrast vector/matrix for inference on
            spatial intensity or moderators.
        t_con_regressor_name : :obj:`~list`
            Name of contrast vector/matrix for spatial intensity
        """
        t_con_regressor = getattr(self, f"t_con_{source}")
        regressors = getattr(self, f"{source}")
        if not regressors:
            if t_con_regressor in (None, False):
                return None
            raise ValueError(f"No {source} are available for CBMR inference.")

        if t_con_regressor is False:
            return None

        t_con_regressor, t_con_regressor_name = self._coerce_contrast_specification(
            t_con_regressor,
            regressors,
            source,
        )
        t_con_regressor = self._ensure_2d_contrasts(t_con_regressor)
        self._validate_contrast_shapes(t_con_regressor, len(regressors), source)
        t_con_regressor = self._remove_zero_rows(t_con_regressor, source)
        t_con_regressor = self._standardize_contrasts(t_con_regressor)
        t_con_regressor, t_con_regressor_name = self._deduplicate_contrasts(
            t_con_regressor,
            t_con_regressor_name,
        )

        return t_con_regressor, t_con_regressor_name

    def _coerce_contrast_specification(self, t_con_regressor, regressors, source):
        """Normalize user-provided contrast specifications into contrast matrices."""
        if t_con_regressor is None or t_con_regressor is True:
            default_contrasts = self.create_contrast(regressors, source=source)
            return list(default_contrasts.values()), list(default_contrasts.keys())

        if isinstance(t_con_regressor, dict):
            return list(t_con_regressor.values()), list(t_con_regressor.keys())

        if isinstance(t_con_regressor, str) or _is_named_pairwise_contrast(t_con_regressor):
            named_contrasts = self.create_contrast(t_con_regressor, source=source)
            return list(named_contrasts.values()), list(named_contrasts.keys())

        if isinstance(t_con_regressor, (list, np.ndarray)):
            if self._uses_named_contrast_list(t_con_regressor):
                named_contrasts = self.create_contrast(t_con_regressor, source=source)
                return list(named_contrasts.values()), list(named_contrasts.keys())

            contrast_matrices = [np.array(con_regressor) for con_regressor in t_con_regressor]
            for i, contrast in enumerate(contrast_matrices):
                self.result.metadata[f"GLH_{source}_{i}"] = contrast
            return contrast_matrices, None

        raise ValueError(
            f"Unsupported {source} contrast specification of type {type(t_con_regressor)}."
        )

    def _uses_named_contrast_list(self, t_con_regressor):
        """Return whether a contrast sequence is expressed entirely with names."""
        return (
            isinstance(t_con_regressor, list)
            and bool(t_con_regressor)
            and all(
                isinstance(con_regressor, str) or _is_named_pairwise_contrast(con_regressor)
                for con_regressor in t_con_regressor
            )
        )

    @staticmethod
    def _ensure_2d_contrasts(t_con_regressor):
        """Ensure each contrast array is two-dimensional."""
        return [
            con_regressor.reshape((1, -1)) if len(con_regressor.shape) == 1 else con_regressor
            for con_regressor in t_con_regressor
        ]

    @staticmethod
    def _validate_contrast_shapes(t_con_regressor, n_regressors, source):
        """Validate that each contrast has the expected regressor dimension."""
        wrong_shape = [con_regressor.shape[1] != n_regressors for con_regressor in t_con_regressor]
        if np.any(wrong_shape):
            wrong_con_regressor_idx = np.where(wrong_shape)[0].tolist()
            raise ValueError(
                f"""The shape of {str(wrong_con_regressor_idx)}th contrast vector(s) in contrast
                matrix doesn't match with {source}."""
            )

    @staticmethod
    def _remove_zero_rows(t_con_regressor, source):
        """Remove zero rows from each contrast matrix and reject empty contrasts."""
        con_regressor_zero_row = [
            np.where(np.sum(np.abs(con_regressor), axis=1) == 0)[0]
            for con_regressor in t_con_regressor
        ]
        if np.any([len(zero_row) > 0 for zero_row in con_regressor_zero_row]):
            t_con_regressor = [
                np.delete(t_con_regressor[i], con_regressor_zero_row[i], axis=0)
                for i in range(len(t_con_regressor))
            ]
            if np.any([con_regressor.shape[0] == 0 for con_regressor in t_con_regressor]):
                raise ValueError(
                    f"""One or more of contrast vector(s) in {source} contrast matrix are
                    all zeros."""
                )
        return t_con_regressor

    @staticmethod
    def _standardize_contrasts(t_con_regressor):
        """Scale each contrast row by its absolute row sum."""
        return [
            con_regressor / np.sum(np.abs(con_regressor), axis=1).reshape((-1, 1))
            for con_regressor in t_con_regressor
        ]

    @staticmethod
    def _deduplicate_contrasts(t_con_regressor, t_con_regressor_name):
        """Drop duplicate contrast matrices while preserving their original order."""
        deduplicated_contrasts = []
        deduplicated_names = [] if t_con_regressor_name is not None else None
        for index, con_regressor in enumerate(t_con_regressor):
            if any(np.array_equal(con_regressor, existing) for existing in deduplicated_contrasts):
                continue
            deduplicated_contrasts.append(con_regressor)
            if deduplicated_names is not None:
                deduplicated_names.append(t_con_regressor_name[index])
        return deduplicated_contrasts, deduplicated_names

    @_check_fit
    def _run_group_inference(self):
        """Evaluate all prepared group contrasts and write them into the result object."""
        for con_group_count, con_group in enumerate(self.t_con_groups):
            group_stats = self._evaluate_group_contrast(con_group)
            self._store_group_inference_result(con_group_count, group_stats)

    def _evaluate_group_contrast(self, con_group):
        """Compute statistics for one prepared group contrast."""
        X = self.estimator.inputs_["coef_spline_bases"]
        n_brain_voxel, spatial_coef_dim = X.shape
        involved_groups, simp_con_group, is_homogeneity_test = self._summarize_group_contrast(
            con_group
        )
        contrast_log_intensity = self._compute_group_contrast_log_intensity(
            simp_con_group,
            involved_groups,
            is_homogeneity_test,
        )
        if self.moderator_effect in ("voxelwise", "mixed"):
            cov_spatial_coef = self._get_intercept_covariance_for_groups(involved_groups)
            spatial_coef_dim = None
            n_brain_voxel = None
        else:
            cov_spatial_coef = self._get_group_spatial_covariance(involved_groups)

        if con_group.shape[0] == 1:
            z_stats_spatial, nlogp_vals_spatial = self._compute_group_wald_statistics(
                simp_con_group,
                involved_groups,
                cov_spatial_coef,
                contrast_log_intensity,
                X,
                spatial_coef_dim,
            )
            chi_sq_spatial = None
        else:
            (
                chi_sq_spatial,
                z_stats_spatial,
                nlogp_vals_spatial,
            ) = self._compute_group_glh_statistics(
                simp_con_group,
                involved_groups,
                cov_spatial_coef,
                contrast_log_intensity,
                X,
                spatial_coef_dim=spatial_coef_dim,
                n_brain_voxel=n_brain_voxel,
                is_homogeneity_test=is_homogeneity_test,
            )

        return {
            "contrast_count": con_group.shape[0],
            "chi_square": chi_sq_spatial,
            "p": _clip_p_values(np.exp(nlogp_vals_spatial), dtype=DEFAULT_FLOAT_DTYPE),
            "logp": _nlogp_to_logp_values(nlogp_vals_spatial),
            "z": z_stats_spatial,
        }

    def _summarize_group_contrast(self, con_group):
        """Return the involved groups and simplified contrast matrix."""
        involved_index = np.where(np.any(con_group != 0, axis=0))[0].tolist()
        involved_groups = [self.groups[i] for i in involved_index]
        simp_con_group = con_group[:, ~np.all(con_group == 0, axis=0)]
        is_homogeneity_test = np.all(np.count_nonzero(con_group, axis=1) == 1)
        return involved_groups, simp_con_group, is_homogeneity_test

    def _compute_group_contrast_log_intensity(
        self,
        simp_con_group,
        involved_groups,
        is_homogeneity_test,
    ):
        """Project fitted group log-intensities through one contrast matrix."""
        involved_log_intensity_per_voxel = np.stack(
            [self._get_group_log_intensity(group) for group in involved_groups],
            axis=0,
        )
        if is_homogeneity_test:
            null_log_spatial_intensity = [
                self._get_group_null_log_intensity(group) for group in involved_groups
            ]
            involved_log_intensity_per_voxel -= np.asarray(null_log_spatial_intensity).reshape(
                -1, 1
            )
        return np.matmul(simp_con_group, involved_log_intensity_per_voxel)

    @staticmethod
    def _compute_group_wald_statistics(
        simp_con_group,
        involved_groups,
        cov_spatial_coef,
        contrast_log_intensity,
        X,
        spatial_coef_dim=None,
    ):
        """Compute one-contrast Wald statistics for group inference."""
        if spatial_coef_dim is None:
            spatial_coef_dim = X.shape[1]
        n_con_group_involved = len(involved_groups)
        var_log_intensity = []
        for k in range(n_con_group_involved):
            cov_spatial_coef_k = cov_spatial_coef[
                k * spatial_coef_dim : (k + 1) * spatial_coef_dim,
                k * spatial_coef_dim : (k + 1) * spatial_coef_dim,
            ]
            var_log_intensity_k = np.sum(np.multiply(X @ cov_spatial_coef_k, X), axis=1)
            var_log_intensity.append(var_log_intensity_k)
        var_log_intensity = np.stack(var_log_intensity, axis=0)
        involved_var_log_intensity = simp_con_group**2 @ var_log_intensity
        involved_std_log_intensity = np.sqrt(np.maximum(involved_var_log_intensity, 0.0))
        z_stats_spatial = contrast_log_intensity / np.where(
            involved_std_log_intensity > 0,
            involved_std_log_intensity,
            np.inf,
        )
        tail = "one" if n_con_group_involved == 1 else "two"
        return z_stats_spatial, z_to_nlogp(z_stats_spatial, tail=tail)

    def _compute_group_glh_statistics(
        self,
        simp_con_group,
        involved_groups,
        cov_spatial_coef,
        contrast_log_intensity,
        X,
        spatial_coef_dim=None,
        n_brain_voxel=None,
        is_homogeneity_test=False,
    ):
        """Compute multi-row GLH statistics for group inference."""
        n_con_group_involved = len(involved_groups)
        m = simp_con_group.shape[0]
        if spatial_coef_dim is None:
            spatial_coef_dim = X.shape[1]
        if n_brain_voxel is None:
            n_brain_voxel = X.shape[0]
        cov_spatial_coef = cov_spatial_coef.reshape(
            n_con_group_involved,
            spatial_coef_dim,
            n_con_group_involved,
            spatial_coef_dim,
        )
        cov_log_intensity = np.einsum(
            "vi,kisj,vj->ksv",
            X,
            cov_spatial_coef,
            X,
            optimize=True,
        )
        chi_sq_spatial = self._chi_square_log_intensity(
            m,
            n_brain_voxel,
            n_con_group_involved,
            simp_con_group,
            cov_log_intensity,
            contrast_log_intensity,
        )
        nlogp_vals_spatial = chi2_to_nlogp(chi_sq_spatial, m)
        if is_homogeneity_test:
            z_stats_spatial = nlogp_to_z(nlogp_vals_spatial, tail="one")
        else:
            z_stats_spatial = nlogp_to_z(nlogp_vals_spatial, tail="two")
            if simp_con_group.shape[0] == 1:
                z_stats_spatial = z_stats_spatial * np.sign(contrast_log_intensity.flatten())
        return chi_sq_spatial, z_stats_spatial, nlogp_vals_spatial

    def _store_group_inference_result(self, con_group_count, group_stats):
        """Write one computed group-inference result into result maps."""
        contrast_name = self.t_con_groups_name[con_group_count] if self.t_con_groups_name else None
        if contrast_name:

            def key_builder(stat_name):
                return f"{stat_name}_group-{contrast_name}"

        else:

            def key_builder(stat_name):
                return f"{stat_name}_GLH_groups_{con_group_count}"

        self._store_stat_outputs(
            self.result.maps,
            group_stats,
            key_builder,
            chi_square_key="chiSquare",
        )

    @staticmethod
    def _store_stat_outputs(
        container,
        stats,
        key_builder,
        chi_square_key="chi_square",
        as_table=False,
    ):
        """Store chi-square, p, logp, and z statistic outputs with shared naming logic."""
        stat_keys = []
        if stats["contrast_count"] > 1:
            stat_keys.append(("chi_square", chi_square_key))
        stat_keys.extend([("p", "p"), ("logp", "logp"), ("z", "z")])
        for stat_name, key_name in stat_keys:
            values = stats[stat_name]
            if as_table:
                values = pd.DataFrame(data=np.array(values), columns=[stat_name])
            container[key_builder(key_name)] = values

    @_check_fit
    def _glh_con_group(self):
        """Conduct GLH testing for group-wise spatial intensity estimation.

        GLH testing allows flexible hypothesis testings on spatial
        intensity, including group-wise spatial homogeneity test and
        group comparison test.
        """
        self._run_group_inference()

    @staticmethod
    def _chi_square_log_intensity(
        m=None,
        n_brain_voxel=None,
        n_con_group_involved=None,
        simp_con_group=None,
        cov_log_intensity=None,
        contrast_log_intensity=None,
        n_voxels=None,
        n_involved_groups=None,
    ):
        """
        Calculate chi-square statistics for GLH on group-wise log intensity function.

        It is an intermediate steps for GLH testings.

        Parameters
        ----------
        m : :obj:`int`
            Number of independent GLH tests.
        n_brain_voxel : :obj:`int`
            Number of voxels within the brain mask.
        n_con_group_involved : :obj:`int`
            Number of groups involved in the GLH test.
        simp_con_group : :obj:`numpy.ndarray`
            Simplified contrast matrix for the GLH test.
        cov_log_intensity : :obj:`numpy.ndarray`
            Covariance matrix of log intensity estimation.
        contrast_log_intensity : :obj:`numpy.ndarray`
            The product of contrast matrix and log intensity estimation.

        Returns
        -------
        chi_sq_spatial : :obj:`numpy.ndarray`
            Voxel-wise chi-square statistics for GLH tests on group-wise spatial
            intensity estimations.
        """
        if n_voxels is not None:
            n_brain_voxel = n_voxels
        if n_involved_groups is not None:
            n_con_group_involved = n_involved_groups
        if cov_log_intensity.ndim == 3:
            if cov_log_intensity.shape[:2] == (n_con_group_involved, n_con_group_involved):
                cov_by_voxel = np.moveaxis(cov_log_intensity, -1, 0)
            else:
                cov_by_voxel = cov_log_intensity
        else:
            cov_by_voxel = cov_log_intensity.reshape(
                n_con_group_involved, n_con_group_involved, n_brain_voxel
            ).transpose(2, 0, 1)
        contrast_cov = np.einsum(
            "ai,vij,bj->vab",
            simp_con_group,
            cov_by_voxel,
            simp_con_group,
            optimize=True,
        )
        contrast_values = contrast_log_intensity.T
        solved = np.linalg.solve(contrast_cov, contrast_values[..., np.newaxis])
        return np.einsum("va,va->v", contrast_values, solved[..., 0], optimize=True)

    @_check_fit
    def _run_moderator_inference(self):
        """Evaluate all prepared moderator contrasts and write them into the result object."""
        if self.moderator_effect == "mixed":
            for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
                global_contrast, voxelwise_contrast = self._split_mixed_moderator_contrast(
                    con_moderator
                )
                if global_contrast is not None:
                    cov_moderator_coef, var_moderator_coef = self._get_moderator_covariance()
                    moderator_stats = self._evaluate_global_moderator_contrast(
                        global_contrast,
                        cov_moderator_coef,
                        var_moderator_coef,
                        self._moderator_coef_table,
                    )
                    self._store_moderator_inference_result(con_moderator_count, moderator_stats)
                else:
                    for group in self.groups:
                        moderator_stats = self._evaluate_voxelwise_moderator_contrast(
                            group,
                            voxelwise_contrast,
                        )
                        self._store_moderator_inference_result(
                            con_moderator_count,
                            group,
                            moderator_stats,
                        )
            return

        if self.moderator_effect == "voxelwise":
            for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
                for group in self.groups:
                    moderator_stats = self._evaluate_voxelwise_moderator_contrast(
                        group,
                        con_moderator,
                    )
                    self._store_moderator_inference_result(
                        con_moderator_count,
                        group,
                        moderator_stats,
                    )
            return

        cov_moderator_coef, var_moderator_coef = self._get_moderator_covariance()
        moderator_coef = self._moderator_coef_table
        for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
            moderator_stats = self._evaluate_global_moderator_contrast(
                con_moderator,
                cov_moderator_coef,
                var_moderator_coef,
                moderator_coef,
            )
            self._store_moderator_inference_result(con_moderator_count, moderator_stats)

    def _split_mixed_moderator_contrast(self, con_moderator):
        """Split an all-moderator contrast into either global or voxelwise coordinates."""
        global_mask = np.array(
            [moderator in self.estimator.global_moderators for moderator in self.moderators]
        )
        voxelwise_mask = np.array(
            [moderator in self.estimator.voxelwise_moderators for moderator in self.moderators]
        )
        uses_global = np.any(con_moderator[:, global_mask] != 0)
        uses_voxelwise = np.any(con_moderator[:, voxelwise_mask] != 0)
        if uses_global and uses_voxelwise:
            raise ValueError(
                "Mixed CBMR moderator contrasts cannot combine global and voxelwise moderators. "
                "Test each moderator type separately."
            )
        if uses_global:
            return con_moderator[:, global_mask], None
        if uses_voxelwise:
            return None, con_moderator[:, voxelwise_mask]
        raise ValueError("Moderator contrast does not select any fitted moderator.")

    def _evaluate_voxelwise_moderator_contrast(self, group, con_moderator):
        """Compute spatially varying statistics for one voxelwise moderator contrast."""
        bases = self.estimator.inputs_["coef_spline_bases"]
        coefficient = self._get_group_coefficient_matrix(group)
        covariance = self._get_group_covariance(group)
        intercept_column = np.zeros((con_moderator.shape[0], 1))
        augmented_contrast = np.column_stack([con_moderator, intercept_column])
        return self._compute_spatial_coefficient_statistics(
            coefficient,
            covariance,
            augmented_contrast,
            bases,
        )

    def _evaluate_global_moderator_contrast(
        self,
        con_moderator,
        cov_moderator_coef=None,
        var_moderator_coef=None,
        moderator_coef=None,
    ):
        """Compute statistics for one prepared moderator contrast."""
        m_con_moderator = con_moderator.shape[0]
        contrast_moderator_coef = np.matmul(con_moderator, moderator_coef)
        if m_con_moderator == 1:
            involved_var_moderator_coef = con_moderator**2 @ var_moderator_coef
            involved_std_moderator_coef = np.sqrt(involved_var_moderator_coef)
            z_stats_moderator = contrast_moderator_coef / involved_std_moderator_coef
            nlogp_vals_moderator = z_to_nlogp(z_stats_moderator, tail="two")
            chi_sq_moderator = None
        else:
            contrast_covariance = con_moderator @ cov_moderator_coef @ con_moderator.T
            solved = np.linalg.solve(contrast_covariance, contrast_moderator_coef)
            chi_sq_moderator = contrast_moderator_coef.T @ solved
            nlogp_vals_moderator = chi2_to_nlogp(chi_sq_moderator, m_con_moderator)
            z_stats_moderator = nlogp_to_z(nlogp_vals_moderator, tail="two")

        return {
            "contrast_count": m_con_moderator,
            "chi_square": chi_sq_moderator,
            "p": _clip_p_values(np.exp(nlogp_vals_moderator), dtype=np.float64),
            "logp": _nlogp_to_logp_values(nlogp_vals_moderator, dtype=np.float64),
            "z": z_stats_moderator,
        }

    def _store_moderator_inference_result(
        self,
        con_moderator_count,
        moderator_stats_or_group,
        moderator_stats=None,
    ):
        """Write one computed moderator-inference result into result tables."""
        if moderator_stats is not None:
            group = moderator_stats_or_group
            contrast_name = (
                self.t_con_moderators_name[con_moderator_count]
                if self.t_con_moderators_name
                else None
            )
            if contrast_name:
                key_builder = (
                    lambda stat_name: f"{stat_name}_voxelwiseModeratorEffect_"
                    f"{contrast_name}_group-{group}"
                )
            else:
                key_builder = (
                    lambda stat_name: f"{stat_name}_GLH_voxelwiseModeratorEffects_"
                    f"{con_moderator_count}_group-{group}"
                )
            self._store_stat_outputs(
                self.result.maps,
                moderator_stats,
                key_builder,
                chi_square_key="chiSquare",
            )
            return

        moderator_stats = moderator_stats_or_group
        contrast_name = (
            self.t_con_moderators_name[con_moderator_count] if self.t_con_moderators_name else None
        )
        if contrast_name:

            def key_builder(stat_name):
                return f"{stat_name}_{contrast_name}"

        else:

            def key_builder(stat_name):
                return f"{stat_name}_GLH_moderators_{con_moderator_count}"

        self._store_stat_outputs(
            self.result.tables,
            moderator_stats,
            key_builder,
            as_table=True,
        )

    @_check_fit
    def _glh_con_moderator(self):
        """Conduct Generalized linear hypothesis (GLH) testing for moderators.

        GLH testing allows flexible hypothesis testings on regression
        coefficients of moderators, including testing for
        the existence of moderator effects and difference in moderator
        effects across multiple moderator effects.
        """
        self._run_moderator_inference()

    def _get_group_augmented_moderators(self, group):
        """Return the group moderator matrix with an intercept as the final column."""
        if self.moderator_effect == "mixed":
            moderators = np.asarray(self.estimator.inputs_["voxelwise_moderators_by_group"][group])
        elif self.moderators:
            moderators = np.asarray(self.estimator.inputs_["moderators_by_group"][group])
        else:
            n_experiments = self.estimator.inputs_["foci_by_experiment_voxel"][group].shape[0]
            moderators = np.empty((n_experiments, 0), dtype=np.float64)
        return np.column_stack([moderators, np.ones((moderators.shape[0], 1))])

    def _get_group_coefficient_matrix(self, group):
        """Return a ``(n_moderators + 1, n_bases)`` coefficient matrix."""
        coefficient = self._group_coefficient_cache.get(group)
        if coefficient is not None:
            return coefficient

        n_bases = self.estimator.inputs_["coef_spline_bases"].shape[1]
        if getattr(self.estimator, "voxelwise_coef", None) is not None:
            coefficient = np.asarray(self.estimator.voxelwise_coef[group]).reshape((-1, n_bases))
        elif getattr(self.estimator, "voxelwise_model", None) is not None:
            spatial_coef = (
                self.estimator.voxelwise_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            if self.moderator_effect == "mixed":
                has_voxelwise_moderators = bool(self.estimator.voxelwise_moderators)
            else:
                has_voxelwise_moderators = bool(self.moderators)
            if has_voxelwise_moderators:
                moderator_coef = (
                    self.estimator.voxelwise_model.moderator_coef_linears[group]
                    .weight.detach()
                    .cpu()
                    .numpy()
                )
                coefficient = np.vstack([moderator_coef, spatial_coef])
            else:
                coefficient = spatial_coef.reshape((1, -1))
        else:
            spatial_coef = self.result.tables["spatial_regression_coef"].loc[group].to_numpy()
            has_voxelwise_moderators = (
                bool(self.estimator.voxelwise_moderators)
                if self.moderator_effect == "mixed"
                else bool(self.moderators)
            )
            if has_voxelwise_moderators:
                moderator_coef = self.result.tables[
                    f"voxelwise_moderator_effect_regression_coef_group-{group}"
                ].to_numpy()
                coefficient = np.vstack([moderator_coef, spatial_coef])
            else:
                coefficient = spatial_coef.reshape((1, -1))

        self._group_coefficient_cache[group] = coefficient
        return coefficient

    @staticmethod
    def _compute_glm_sandwich_covariance(
        design,
        foci,
        mean,
        ridge=1e-6,
        correction="hc3",
    ):
        """Compute robust covariance for one log-link marginal Poisson GLM."""
        design = np.asarray(design, dtype=float)
        foci = np.asarray(foci, dtype=float).reshape(-1)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        mean = np.nan_to_num(mean, nan=0.0, posinf=1e12, neginf=0.0)
        mean = np.clip(mean, 1e-12, 1e12)

        if design.ndim != 2:
            raise ValueError("design must be a two-dimensional array.")
        if design.shape[0] != foci.shape[0] or foci.shape != mean.shape:
            raise ValueError("design, foci, and mean must have matching rows.")

        fisher_info = design.T @ (design * mean[:, None])
        bread_inverse = CBMRInference._sandwich_bread_inverse(fisher_info, ridge)
        residuals = np.nan_to_num(foci - mean, nan=0.0, posinf=0.0, neginf=0.0)
        correction_factor = 1.0

        if correction == "hc1":
            n_observations, n_parameters = design.shape
            if n_observations <= n_parameters:
                raise ValueError(
                    "HC1 sandwich correction requires more observations than model columns. "
                    "Use sandwich_correction='hc0' or 'hc3' for this setting."
                )
            correction_factor = n_observations / float(n_observations - n_parameters)
        elif correction == "hc3":
            leverage = mean * np.einsum(
                "ij,jk,ik->i",
                design,
                bread_inverse,
                design,
                optimize=True,
            )
            leverage = np.nan_to_num(leverage, nan=0.0, posinf=1.0, neginf=0.0)
            leverage = np.clip(leverage, 0.0, 0.999)
            residuals = residuals / np.maximum(1.0 - leverage, 1e-6)

        meat_matrix = design.T @ (design * residuals[:, None] ** 2)
        covariance = correction_factor * bread_inverse @ meat_matrix @ bread_inverse
        return 0.5 * (covariance + covariance.T)

    def _global_group_moderator_sum(self, group):
        """Return the summed experiment-level moderator effect for one global-CBMR group."""
        if not self.moderators:
            return float(np.asarray(self.estimator.inputs_["foci_per_experiment"][group]).size)

        moderators = np.asarray(self.estimator.inputs_["moderators_by_group"][group], dtype=float)
        moderator_coef = np.asarray(self._moderator_coef_table, dtype=float)
        return float(np.exp(np.clip(moderators @ moderator_coef, -100, 100)).sum())

    def _compute_global_spatial_sandwich_covariance(self, involved_groups):
        """Compute block-diagonal robust covariance for global spatial coefficients."""
        bases = np.asarray(self.estimator.inputs_["coef_spline_bases"], dtype=float)
        n_bases = bases.shape[1]
        covariance = np.zeros((len(involved_groups) * n_bases, len(involved_groups) * n_bases))
        for group_index, group in enumerate(involved_groups):
            spatial_intensity = np.asarray(
                self.result.maps[f"spatialIntensity_group-{group}"],
                dtype=float,
            ).reshape(-1)
            mean = self._global_group_moderator_sum(group) * spatial_intensity
            group_covariance = self._compute_glm_sandwich_covariance(
                bases,
                self.estimator.inputs_["foci_per_voxel"][group],
                mean,
                ridge=self.ridge,
                correction=self.sandwich_correction,
            )
            group_slice = slice(group_index * n_bases, (group_index + 1) * n_bases)
            covariance[group_slice, group_slice] = group_covariance
        return covariance

    def _compute_global_moderator_sandwich_covariance(self):
        """Compute robust covariance for global experiment-level moderator coefficients."""
        if not self.moderators:
            return np.zeros((0, 0), dtype=np.float64)

        designs = []
        foci = []
        mean = []
        moderator_coef = np.asarray(self._moderator_coef_table, dtype=float)
        for group in self.groups:
            group_moderators = np.asarray(
                self.estimator.inputs_["moderators_by_group"][group],
                dtype=float,
            )
            spatial_sum = float(
                np.asarray(self.result.maps[f"spatialIntensity_group-{group}"], dtype=float).sum()
            )
            designs.append(group_moderators)
            foci.append(
                np.asarray(self.estimator.inputs_["foci_per_experiment"][group], dtype=float)
            )
            mean.append(
                spatial_sum * np.exp(np.clip(group_moderators @ moderator_coef, -100, 100))
            )

        return self._compute_glm_sandwich_covariance(
            np.vstack(designs),
            np.concatenate([group_foci.reshape(-1) for group_foci in foci]),
            np.concatenate([group_mean.reshape(-1) for group_mean in mean]),
            ridge=self.ridge,
            correction=self.sandwich_correction,
        )

    def _mixed_joint_parameter_layout(self):
        """Return parameter slices for mixed global and group-specific coefficient blocks."""
        n_global = len(self.estimator.global_moderators)
        n_bases = self.estimator.inputs_["coef_spline_bases"].shape[1]
        n_local_regressors = len(self.estimator.voxelwise_moderators) + 1
        local_size = n_local_regressors * n_bases
        group_slices = {}
        offset = n_global
        for group in self.groups:
            group_slices[group] = slice(offset, offset + local_size)
            offset += local_size
        return slice(0, n_global), group_slices, n_local_regressors, n_bases, offset

    def _mixed_joint_fisher_information(self):
        """Return Fisher information for the full mixed-model parameter vector."""
        global_slice, group_slices, _, _, total_size = self._mixed_joint_parameter_layout()
        fisher_info = np.zeros((total_size, total_size), dtype=float)
        bases = np.asarray(self.estimator.inputs_["coef_spline_bases"], dtype=float)

        for group in self.groups:
            global_moderators = np.asarray(
                self.estimator.inputs_["global_moderators_by_group"][group],
                dtype=float,
            )
            local_moderators = self._get_group_augmented_moderators(group)
            mean = self._get_group_mean(group)
            group_slice = group_slices[group]

            mean_by_experiment = mean.sum(axis=1)
            fisher_info[global_slice, global_slice] += global_moderators.T @ (
                global_moderators * mean_by_experiment[:, None]
            )
            fisher_info[group_slice, group_slice] = self._compute_fisher_information(
                local_moderators,
                bases,
                mean,
            )
            cross_info = np.einsum(
                "mg,mr,mv,vb->grb",
                global_moderators,
                local_moderators,
                mean,
                bases,
                optimize=True,
            ).reshape(global_slice.stop, -1)
            fisher_info[global_slice, group_slice] = cross_info
            fisher_info[group_slice, global_slice] = cross_info.T

        return fisher_info

    @staticmethod
    def _dense_residuals(foci, mean):
        """Return dense residuals from sparse or dense foci and fitted means."""
        residuals = -np.asarray(mean, dtype=float)
        if scipy.sparse.issparse(foci):
            residuals = residuals + _as_csr_matrix(foci).toarray()
        else:
            residuals = residuals + np.asarray(foci, dtype=float)
        return np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)

    def _mixed_joint_leverage_scale(
        self,
        bread_inverse,
        global_moderators,
        local_moderators,
        bases,
        mean,
        group_slice,
    ):
        """Return HC3 residual scaling for one mixed-model group."""
        if self.sandwich_correction != "hc3":
            return None

        global_slice, _, n_local_regressors, n_bases, _ = self._mixed_joint_parameter_layout()
        global_covariance = bread_inverse[global_slice, global_slice]
        local_covariance = bread_inverse[group_slice, group_slice]
        cross_covariance = bread_inverse[global_slice, group_slice].reshape(
            global_slice.stop,
            n_local_regressors,
            n_bases,
        )

        global_leverage = np.einsum(
            "mg,gh,mh->m",
            global_moderators,
            global_covariance,
            global_moderators,
            optimize=True,
        )[:, None]
        local_covariance_blocks = local_covariance.reshape(
            n_local_regressors,
            n_bases,
            n_local_regressors,
            n_bases,
        ).transpose(0, 2, 1, 3)
        local_leverage_basis = np.einsum(
            "vp,rspq,vq->rsv",
            bases,
            local_covariance_blocks,
            bases,
            optimize=True,
        )
        local_leverage = np.einsum(
            "mr,ms,rsv->mv",
            local_moderators,
            local_moderators,
            local_leverage_basis,
            optimize=True,
        )
        cross_leverage = 2 * np.einsum(
            "mg,grb,mr,vb->mv",
            global_moderators,
            cross_covariance,
            local_moderators,
            bases,
            optimize=True,
        )
        leverage = mean * (global_leverage + local_leverage + cross_leverage)
        leverage = np.nan_to_num(leverage, nan=0.0, posinf=1.0, neginf=0.0)
        leverage = np.clip(leverage, 0.0, 0.999)
        return np.maximum(1.0 - leverage, 1e-6)

    def _mixed_joint_sandwich_meat(self, bread_inverse):
        """Return sandwich meat for the full mixed-model parameter vector."""
        global_slice, group_slices, _, _, total_size = self._mixed_joint_parameter_layout()
        bases = np.asarray(self.estimator.inputs_["coef_spline_bases"], dtype=float)
        meat = np.zeros((total_size, total_size), dtype=float)
        n_observations = 0

        for group in self.groups:
            global_moderators = np.asarray(
                self.estimator.inputs_["global_moderators_by_group"][group],
                dtype=float,
            )
            local_moderators = self._get_group_augmented_moderators(group)
            mean = self._get_group_mean(group)
            residuals = self._dense_residuals(
                self.estimator.inputs_["foci_by_experiment_voxel"][group],
                mean,
            )
            group_slice = group_slices[group]
            residual_scale = self._mixed_joint_leverage_scale(
                bread_inverse,
                global_moderators,
                local_moderators,
                bases,
                mean,
                group_slice,
            )
            if residual_scale is not None:
                residuals = residuals / residual_scale

            if self.sandwich_meat == "cluster":
                n_observations += residuals.shape[0]
                group_scores = np.zeros((residuals.shape[0], total_size), dtype=float)
                group_scores[:, global_slice] = global_moderators * residuals.sum(axis=1)[:, None]
                basis_residuals = residuals @ bases
                local_scores = [
                    basis_residuals * local_moderators[:, index : index + 1]
                    for index in range(local_moderators.shape[1])
                ]
                group_scores[:, group_slice] = np.hstack(local_scores)
                meat += group_scores.T @ group_scores
            else:
                n_observations += residuals.size
                residual_square = residuals**2
                residual_square_by_experiment = residual_square.sum(axis=1)
                meat[global_slice, global_slice] += global_moderators.T @ (
                    global_moderators * residual_square_by_experiment[:, None]
                )
                meat[group_slice, group_slice] += self._sandwich_meat_matrix(
                    local_moderators,
                    bases,
                    residuals,
                    meat="iid",
                )
                cross_meat = np.einsum(
                    "mg,mr,mv,vb->grb",
                    global_moderators,
                    local_moderators,
                    residual_square,
                    bases,
                    optimize=True,
                ).reshape(global_slice.stop, -1)
                meat[global_slice, group_slice] += cross_meat
                meat[group_slice, global_slice] += cross_meat.T

        if self.sandwich_correction == "hc1":
            n_parameters = total_size
            if n_observations <= n_parameters:
                raise ValueError(
                    "HC1 sandwich correction requires more observations than model parameters. "
                    "Use sandwich_correction='hc0' or 'hc3' for this setting."
                )
            meat *= n_observations / float(n_observations - n_parameters)

        return meat

    def _get_mixed_joint_covariance(self):
        """Return cached joint covariance for all mixed-model coefficients."""
        cache_key = (self.method, self.sandwich_meat, self.sandwich_correction, self.ridge)
        covariance = self._mixed_joint_covariance_cache.get(cache_key)
        if covariance is not None:
            return covariance

        fisher_info = self._mixed_joint_fisher_information()
        bread_inverse = self._sandwich_bread_inverse(fisher_info, self.ridge)
        if self.method == "FI":
            covariance = bread_inverse
        else:
            meat = self._mixed_joint_sandwich_meat(bread_inverse)
            covariance = bread_inverse @ meat @ bread_inverse
            covariance = 0.5 * (covariance + covariance.T)
        self._mixed_joint_covariance_cache[cache_key] = covariance
        return covariance

    def _get_mixed_global_moderator_covariance(self):
        """Return the global-moderator block from the mixed joint covariance."""
        global_slice, _, _, _, _ = self._mixed_joint_parameter_layout()
        covariance = self._get_mixed_joint_covariance()
        return covariance[global_slice, global_slice]

    def _get_mixed_group_covariance(self, group):
        """Return one group-specific voxelwise/spatial block from mixed joint covariance."""
        _, group_slices, _, _, _ = self._mixed_joint_parameter_layout()
        group_slice = group_slices[group]
        covariance = self._get_mixed_joint_covariance()
        return covariance[group_slice, group_slice]

    def _get_mixed_intercept_covariance_for_groups(self, involved_groups):
        """Return joint spatial-intercept covariance for selected mixed-model groups."""
        _, group_slices, n_local_regressors, n_bases, _ = self._mixed_joint_parameter_layout()
        covariance = self._get_mixed_joint_covariance()
        intercept_covariance = np.zeros((len(involved_groups) * n_bases,) * 2)
        intercept_offset = (n_local_regressors - 1) * n_bases
        for row_index, row_group in enumerate(involved_groups):
            row_start = group_slices[row_group].start + intercept_offset
            row_slice = slice(row_start, row_start + n_bases)
            output_row = slice(row_index * n_bases, (row_index + 1) * n_bases)
            for col_index, col_group in enumerate(involved_groups):
                col_start = group_slices[col_group].start + intercept_offset
                col_slice = slice(col_start, col_start + n_bases)
                output_col = slice(col_index * n_bases, (col_index + 1) * n_bases)
                intercept_covariance[output_row, output_col] = covariance[row_slice, col_slice]
        return intercept_covariance

    def _get_group_mean(self, group):
        """Return fitted Poisson mean for one group as experiments by voxels."""
        bases = self.estimator.inputs_["coef_spline_bases"]
        moderators = self._get_group_augmented_moderators(group)
        coefficient = self._get_group_coefficient_matrix(group)
        linear_predictor = moderators @ coefficient @ bases.T
        if self.moderator_effect == "mixed" and self.estimator.global_moderators:
            global_moderators = np.asarray(
                self.estimator.inputs_["global_moderators_by_group"][group],
                dtype=float,
            )
            linear_predictor = linear_predictor + global_moderators @ self._moderator_coef_table
        return np.exp(np.clip(linear_predictor, -100, 100))

    @staticmethod
    def _compute_fisher_information(moderators, bases, mean):
        """Compute Fisher information for the Kronecker design without forming it."""
        n_moderators = moderators.shape[1]
        n_bases = bases.shape[1]
        fisher_info = np.zeros((n_moderators * n_bases, n_moderators * n_bases))
        moderator_weights = np.einsum("mr,ms,mv->rsv", moderators, moderators, mean)
        for row in range(n_moderators):
            row_slice = slice(row * n_bases, (row + 1) * n_bases)
            for col in range(row, n_moderators):
                col_slice = slice(col * n_bases, (col + 1) * n_bases)
                block = bases.T @ (bases * moderator_weights[row, col, :, None])
                fisher_info[row_slice, col_slice] = block
                if row != col:
                    fisher_info[col_slice, row_slice] = block.T
        return fisher_info

    @staticmethod
    def _sandwich_bread_inverse(fisher_info, ridge):
        """Invert the sandwich bread term with a ridge-stabilized pseudo-inverse."""
        regularized = fisher_info + ridge * np.eye(fisher_info.shape[0])
        bread_inverse = np.linalg.pinv(regularized)
        return 0.5 * (bread_inverse + bread_inverse.T)

    @staticmethod
    def _sandwich_meat_matrix(moderators, bases, residuals, meat):
        """Compute cluster- or iid-robust sandwich meat for the Kronecker design."""
        n_experiments, n_moderators = moderators.shape
        n_bases = bases.shape[1]
        n_parameters = n_moderators * n_bases
        if meat == "cluster":
            basis_residuals = bases.T @ residuals.T
            cluster_scores = np.zeros((n_parameters, n_experiments), dtype=float)
            for moderator_index in range(n_moderators):
                parameter_slice = slice(
                    moderator_index * n_bases,
                    (moderator_index + 1) * n_bases,
                )
                cluster_scores[parameter_slice, :] = (
                    basis_residuals * moderators[:, moderator_index][None, :]
                )
            return cluster_scores @ cluster_scores.T

        residual_weights = np.einsum("mr,ms,mv->rsv", moderators, moderators, residuals**2)
        meat_matrix = np.zeros((n_parameters, n_parameters), dtype=float)
        for row in range(n_moderators):
            row_slice = slice(row * n_bases, (row + 1) * n_bases)
            for col in range(row, n_moderators):
                col_slice = slice(col * n_bases, (col + 1) * n_bases)
                block = bases.T @ (bases * residual_weights[row, col, :, None])
                meat_matrix[row_slice, col_slice] = block
                if row != col:
                    meat_matrix[col_slice, row_slice] = block.T
        return meat_matrix

    @classmethod
    def _sandwich_meat_matrix_sparse_response(
        cls,
        moderators,
        bases,
        foci,
        mean,
        meat,
        residual_scale=None,
    ):
        """Compute sandwich meat from a sparse response without materializing dense foci."""
        foci = _as_csr_matrix(foci)
        if residual_scale is None:
            adjusted_mean = mean
            adjusted_foci = foci
        else:
            adjusted_mean = mean / residual_scale
            row_indices = _csr_row_indices(foci)
            adjusted_foci = scipy.sparse.csr_matrix(
                (
                    foci.data / residual_scale[row_indices, foci.indices],
                    foci.indices,
                    foci.indptr,
                ),
                shape=foci.shape,
            )

        if meat == "cluster":
            basis_residuals = (adjusted_foci @ bases).T - (bases.T @ adjusted_mean.T)
            n_bases = bases.shape[1]
            n_parameters = moderators.shape[1] * n_bases
            cluster_scores = np.zeros((n_parameters, moderators.shape[0]), dtype=float)
            for moderator_index in range(moderators.shape[1]):
                parameter_slice = slice(
                    moderator_index * n_bases,
                    (moderator_index + 1) * n_bases,
                )
                cluster_scores[parameter_slice, :] = (
                    basis_residuals * moderators[:, moderator_index][None, :]
                )
            return cluster_scores @ cluster_scores.T

        meat_matrix = cls._sandwich_meat_matrix(
            moderators,
            bases,
            adjusted_mean,
            meat="iid",
        )
        row_indices = _csr_row_indices(adjusted_foci)
        delta = (
            adjusted_foci.data**2
            - 2
            * adjusted_foci.data
            * adjusted_mean[
                row_indices,
                adjusted_foci.indices,
            ]
        )
        delta_matrix = scipy.sparse.csr_matrix(
            (delta, adjusted_foci.indices, adjusted_foci.indptr),
            shape=foci.shape,
        )
        n_bases = bases.shape[1]
        for row in range(moderators.shape[1]):
            row_slice = slice(row * n_bases, (row + 1) * n_bases)
            for col in range(row, moderators.shape[1]):
                col_slice = slice(col * n_bases, (col + 1) * n_bases)
                moderator_weight = moderators[:, row] * moderators[:, col]
                voxel_weight = np.asarray(
                    delta_matrix.multiply(moderator_weight[:, None]).sum(axis=0)
                ).ravel()
                block = bases.T @ (bases * voxel_weight[:, None])
                meat_matrix[row_slice, col_slice] += block
                if row != col:
                    meat_matrix[col_slice, row_slice] += block.T
        return meat_matrix

    @classmethod
    def _sandwich_correction_scale(cls, correction, bread_inverse, moderators, bases, mean):
        """Return residual scaling and finite-sample factor for sandwich corrections."""
        if correction is None or correction == "hc0":
            return None, 1.0

        n_experiments, n_moderators = moderators.shape
        if correction == "hc1":
            if n_experiments <= n_moderators:
                raise ValueError(
                    "HC1 sandwich correction requires more experiments than model columns. "
                    "Use sandwich_correction='hc0' or 'hc3' for this setting."
                )
            return None, n_experiments / float(n_experiments - n_moderators)

        n_bases = bases.shape[1]
        bread_inverse_blocks = bread_inverse.reshape(
            n_moderators,
            n_bases,
            n_moderators,
            n_bases,
        ).transpose(0, 2, 1, 3)
        leverage_basis = np.einsum(
            "vp,rspq,vq->rsv",
            bases,
            bread_inverse_blocks,
            bases,
            optimize=True,
        )
        leverage = mean * np.einsum(
            "mr,ms,rsv->mv",
            moderators,
            moderators,
            leverage_basis,
            optimize=True,
        )
        leverage = np.nan_to_num(leverage, nan=0.0, posinf=1.0, neginf=0.0)
        leverage = np.clip(leverage, 0.0, 0.999)
        return np.maximum(1.0 - leverage, 1e-6), 1.0

    @classmethod
    def _apply_sandwich_correction(
        cls,
        correction,
        bread_inverse,
        moderators,
        bases,
        mean,
        residuals,
    ):
        """Apply HC-style finite-sample corrections to sandwich residuals."""
        residual_scale, correction_factor = cls._sandwich_correction_scale(
            correction,
            bread_inverse,
            moderators,
            bases,
            mean,
        )
        if residual_scale is None:
            return residuals, correction_factor
        return residuals / residual_scale, correction_factor

    @classmethod
    def _compute_sandwich_covariance(
        cls,
        moderators,
        bases,
        foci,
        mean,
        ridge=1e-6,
        meat="cluster",
        correction="hc3",
    ):
        """Compute robust Poisson sandwich covariance for one voxelwise CBMR group.

        The voxelwise model has Kronecker rows ``moderator_experiment x basis_voxel``; this is
        distinct from the global-moderator covariance, where moderator effects are not spatially
        expanded over spline bases.
        """
        moderators = np.asarray(moderators, dtype=float)
        bases = np.asarray(bases, dtype=float)
        mean = np.asarray(mean, dtype=float)
        mean = np.nan_to_num(mean, nan=0.0, posinf=1e12, neginf=0.0)
        mean = np.clip(mean, 1e-12, 1e12)
        if scipy.sparse.issparse(foci):
            foci = _as_csr_matrix(foci)
            response_shape = foci.shape
        else:
            response_shape = np.asarray(foci).shape

        if response_shape != mean.shape:
            raise ValueError("foci and mean must have matching experiment-by-voxel shapes.")
        if response_shape != (moderators.shape[0], bases.shape[0]):
            raise ValueError("foci must have shape (n_experiments, n_voxels).")

        fisher_info = cls._compute_fisher_information(moderators, bases, mean)
        bread_inverse = cls._sandwich_bread_inverse(fisher_info, ridge)
        if scipy.sparse.issparse(foci):
            residual_scale, correction_factor = cls._sandwich_correction_scale(
                correction,
                bread_inverse,
                moderators,
                bases,
                mean,
            )
            meat_matrix = cls._sandwich_meat_matrix_sparse_response(
                moderators,
                bases,
                foci,
                mean,
                meat,
                residual_scale=residual_scale,
            )
        else:
            y = np.asarray(foci, dtype=float)
            residuals = np.nan_to_num(y - mean, nan=0.0, posinf=0.0, neginf=0.0)
            residuals, correction_factor = cls._apply_sandwich_correction(
                correction,
                bread_inverse,
                moderators,
                bases,
                mean,
                residuals,
            )
            meat_matrix = cls._sandwich_meat_matrix(moderators, bases, residuals, meat)
        covariance = correction_factor * bread_inverse @ meat_matrix @ bread_inverse
        return 0.5 * (covariance + covariance.T)

    def _get_group_covariance(self, group):
        """Return cached covariance of augmented coefficients for one group."""
        if self.moderator_effect == "mixed" and self.estimator.global_moderators:
            return self._get_mixed_group_covariance(group)

        cache_key = (group, self.method, self.sandwich_meat, self.sandwich_correction, self.ridge)
        covariance = self._group_covariance_cache.get(cache_key)
        if covariance is not None:
            return covariance

        bases = self.estimator.inputs_["coef_spline_bases"]
        moderators = self._get_group_augmented_moderators(group)
        mean = self._get_group_mean(group)
        if self.method == "FI":
            fisher_info = self._compute_fisher_information(moderators, bases, mean)
            covariance = self._sandwich_bread_inverse(fisher_info, self.ridge)
        else:
            covariance = self._compute_sandwich_covariance(
                moderators,
                bases,
                self.estimator.inputs_["foci_by_experiment_voxel"][group],
                mean,
                ridge=self.ridge,
                meat=self.sandwich_meat,
                correction=self.sandwich_correction,
            )
        self._group_covariance_cache[cache_key] = covariance
        return covariance

    def _get_intercept_covariance_for_groups(self, involved_groups):
        """Return block-diagonal covariance for group spatial-intercept coefficients."""
        if self.moderator_effect == "mixed" and self.estimator.global_moderators:
            return self._get_mixed_intercept_covariance_for_groups(involved_groups)

        n_bases = self.estimator.inputs_["coef_spline_bases"].shape[1]
        covariance = np.zeros((len(involved_groups) * n_bases, len(involved_groups) * n_bases))
        for group_index, group in enumerate(involved_groups):
            group_covariance = self._get_group_covariance(group)
            block = group_covariance[-n_bases:, -n_bases:]
            group_slice = slice(group_index * n_bases, (group_index + 1) * n_bases)
            covariance[group_slice, group_slice] = block
        return covariance

    @staticmethod
    def _as_list(value, default, label):
        """Normalize a scalar-or-sequence selection to a list."""
        if value is None:
            return list(default)
        if isinstance(value, str):
            return [value]
        try:
            return list(value)
        except TypeError as exc:
            raise TypeError(f"{label} must be a string, sequence of strings, or None.") from exc

    @staticmethod
    def _validate_selected_names(selected, available, label):
        """Reject requested group or moderator names that are not available."""
        invalid = [name for name in selected if name not in available]
        if invalid:
            available_names = ", ".join(available)
            invalid_names = ", ".join(invalid)
            raise ValueError(
                f"Unknown {label}: {invalid_names}. Available {label}: {available_names}."
            )

    @staticmethod
    def _validate_unit_change(unit_change):
        """Return a finite scalar moderator-unit change."""
        unit_change = float(unit_change)
        if not np.isfinite(unit_change):
            raise ValueError("unit_change must be a finite scalar.")
        return unit_change

    @staticmethod
    def _unit_change_label(unit_change):
        """Format a unit change for stable map keys."""
        label = f"{unit_change:g}".replace("-", "neg").replace(".", "p")
        return label

    def _validate_voxelwise_moderator_diagnostic_inputs(self, moderators, groups, unit_change):
        """Validate voxelwise moderator-effect diagnostic selections."""
        if self.moderator_effect not in ("voxelwise", "mixed"):
            raise ValueError(
                "Voxelwise moderator-effect diagnostics require "
                "CBMRInference(moderator_effect='voxelwise') or 'mixed'."
            )
        available_moderators = (
            self.estimator.voxelwise_moderators
            if self.moderator_effect == "mixed"
            else self.moderators
        )
        if not available_moderators:
            raise ValueError("This CBMR result does not include voxelwise moderators.")

        moderators = self._as_list(moderators, available_moderators, "moderators")
        groups = self._as_list(groups, self.groups, "groups")
        self._validate_selected_names(moderators, available_moderators, "moderators")
        self._validate_selected_names(groups, self.groups, "groups")
        unit_change = self._validate_unit_change(unit_change)
        return moderators, groups, unit_change

    def _compute_voxelwise_moderator_diagnostic_maps(self, moderator, group, unit_change):
        """Compute RI and ID maps for one moderator/group/unit-change combination."""
        bases = self.estimator.inputs_["coef_spline_bases"]
        coefficient = self._get_group_coefficient_matrix(group)
        if self.moderator_effect == "mixed":
            moderator_index = list(self.estimator.voxelwise_moderators).index(moderator)
        else:
            moderator_index = self.moderator_reference_dict[moderator]
        moderator_log_intensity_change = bases @ coefficient[moderator_index]
        relative_intensity = np.exp(
            np.clip(unit_change * moderator_log_intensity_change, -100, 100)
        )
        baseline_intensity = np.asarray(
            self.result.maps[f"spatialIntensity_group-{group}"],
            dtype=float,
        )
        intensity_difference = baseline_intensity * (relative_intensity - 1.0)
        return relative_intensity, intensity_difference

    @staticmethod
    def _validate_id_threshold(id_threshold):
        """Return a finite non-negative ID threshold for ROI filtering."""
        if id_threshold is None:
            return None
        id_threshold = float(id_threshold)
        if not np.isfinite(id_threshold) or id_threshold < 0:
            raise ValueError("id_threshold must be None or a finite non-negative scalar.")
        return id_threshold

    @staticmethod
    def _mask_relative_intensity_to_id_roi(
        relative_intensity,
        intensity_difference,
        id_threshold=None,
    ):
        """Mask RI values to voxels whose absolute ID values pass a threshold."""
        id_threshold = CBMRInference._validate_id_threshold(id_threshold)
        relative_intensity = np.asarray(relative_intensity, dtype=float)
        intensity_difference = np.asarray(intensity_difference, dtype=float)
        if relative_intensity.shape != intensity_difference.shape:
            raise ValueError(
                "relative_intensity and intensity_difference must have the same shape."
            )

        finite_difference = np.isfinite(intensity_difference)
        if not finite_difference.any():
            raise ValueError("intensity_difference must contain at least one finite value.")

        absolute_difference = np.abs(intensity_difference)
        if id_threshold is None:
            id_threshold = float(np.quantile(absolute_difference[finite_difference], 0.5))

        roi_mask = finite_difference & (absolute_difference >= id_threshold)
        masked_relative_intensity = np.where(
            roi_mask & np.isfinite(relative_intensity),
            relative_intensity,
            0.0,
        )
        return masked_relative_intensity, id_threshold

    @_check_fit
    def generate_voxelwise_moderator_effect_maps(
        self,
        moderators=None,
        groups=None,
        unit_change=1.0,
    ):
        """Generate diagnostic maps for voxelwise moderator effects.

        Parameters
        ----------
        moderators : :obj:`str`, sequence of :obj:`str`, or None, optional
            Spatially varying moderators to diagnose. If None, all fitted moderators are used.
        groups : :obj:`str`, sequence of :obj:`str`, or None, optional
            Groups for which to generate maps. If None, all fitted groups are used.
        unit_change : :obj:`float`, optional
            Moderator-unit increase to visualize. Default is 1.

        Returns
        -------
        :obj:`~nimare.meta.cbmr.CBMRResult`
            Result copy with added RI and ID maps. Relative Intensity (RI) is the multiplicative
            intensity ratio for ``unit_change`` moderator units, and Intensity Difference (ID) is
            the corresponding additive change from the fitted group baseline intensity.
        """
        moderators, groups, unit_change = self._validate_voxelwise_moderator_diagnostic_inputs(
            moderators,
            groups,
            unit_change,
        )
        unit_label = self._unit_change_label(unit_change)
        generated_maps = []

        for group in groups:
            for moderator in moderators:
                relative_intensity, intensity_difference = (
                    self._compute_voxelwise_moderator_diagnostic_maps(
                        moderator,
                        group,
                        unit_change,
                    )
                )
                relative_key = (
                    f"relativeIntensity_voxelwiseModeratorEffect_{moderator}_"
                    f"unit-{unit_label}_group-{group}"
                )
                difference_key = (
                    f"intensityDifference_voxelwiseModeratorEffect_{moderator}_"
                    f"unit-{unit_label}_group-{group}"
                )
                self.result.maps[relative_key] = relative_intensity
                self.result.maps[difference_key] = intensity_difference
                generated_maps.extend([relative_key, difference_key])

        self.result.metadata["voxelwise_moderator_effect_diagnostic_unit_change"] = unit_change
        self.result.metadata["voxelwise_moderator_effect_diagnostic_maps"] = tuple(generated_maps)
        return self.result

    @_check_fit
    def plot_voxelwise_moderator_effects(
        self,
        moderators=None,
        groups=None,
        unit_change=1.0,
        cut_coords=None,
        display_mode="ortho",
        id_threshold=None,
        threshold=None,
        figure=None,
        plot_kwargs=None,
    ):
        """Plot RI diagnostic maps within ID-defined regions of interest.

        Each requested moderator/group combination is plotted as one row. Intensity Difference
        (ID) values define the region of interest: voxels are retained when ``abs(ID)`` is greater
        than or equal to ``id_threshold``. If ``id_threshold`` is None, the 50% quantile of
        ``abs(ID)`` is used. Relative Intensity (RI) values are displayed only within that ROI.
        """
        id_threshold = self._validate_id_threshold(id_threshold)
        result = self.generate_voxelwise_moderator_effect_maps(
            moderators=moderators,
            groups=groups,
            unit_change=unit_change,
        )
        moderators, groups, unit_change = self._validate_voxelwise_moderator_diagnostic_inputs(
            moderators,
            groups,
            unit_change,
        )

        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_stat_map

        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        unit_label = self._unit_change_label(unit_change)
        n_rows = len(moderators) * len(groups)
        if figure is None:
            figure = plt.figure(figsize=(5, 3.5 * n_rows))
        axes = figure.subplots(n_rows, 1, squeeze=False)
        plot_threshold = 1e-12 if threshold is None else threshold

        for row, (group, moderator) in enumerate(
            (group, moderator) for group in groups for moderator in moderators
        ):
            relative_key = (
                f"relativeIntensity_voxelwiseModeratorEffect_{moderator}_"
                f"unit-{unit_label}_group-{group}"
            )
            difference_key = (
                f"intensityDifference_voxelwiseModeratorEffect_{moderator}_"
                f"unit-{unit_label}_group-{group}"
            )
            masked_relative_intensity, resolved_id_threshold = (
                self._mask_relative_intensity_to_id_roi(
                    result.maps[relative_key],
                    result.maps[difference_key],
                    id_threshold=id_threshold,
                )
            )
            title_suffix = f"{moderator}, group={group}, unit={unit_change:g}"
            plot_stat_map(
                result.masker.inverse_transform(masked_relative_intensity),
                axes=axes[row, 0],
                figure=figure,
                cut_coords=cut_coords,
                display_mode=display_mode,
                threshold=plot_threshold,
                title=f"RI in ID ROI: {title_suffix}, |ID| >= {resolved_id_threshold:g}",
                **plot_kwargs,
            )

        return figure

    @staticmethod
    def _contrast_covariance_by_voxel(contrast, covariance, bases):
        """Project coefficient covariance into voxel-wise contrast covariance."""
        n_regressors = contrast.shape[1]
        n_bases = bases.shape[1]
        covariance_blocks = covariance.reshape(n_regressors, n_bases, n_regressors, n_bases)
        contrast_covariance_blocks = np.einsum(
            "sr,rpuq,tu->sptq",
            contrast,
            covariance_blocks,
            contrast,
            optimize=True,
        )
        return np.einsum(
            "np,sptq,nq->nst",
            bases,
            contrast_covariance_blocks,
            bases,
            optimize=True,
        )

    @classmethod
    def _compute_spatial_coefficient_statistics(cls, coefficient, covariance, contrast, bases):
        """Compute voxelwise Wald statistics for voxelwise moderator-effect coefficients."""
        contrast_eta = contrast @ coefficient @ bases.T
        contrast_cov = cls._contrast_covariance_by_voxel(contrast, covariance, bases)

        if contrast.shape[0] == 1:
            contrast_var = contrast_cov[:, 0, 0]
            contrast_std = np.sqrt(np.maximum(contrast_var, 0.0))
            z_stats = contrast_eta[0] / np.where(contrast_std > 0, contrast_std, np.inf)
            nlogp_vals = z_to_nlogp(z_stats, tail="two")
            chi_square = None
        else:
            solved = np.linalg.solve(contrast_cov, contrast_eta.T[..., np.newaxis])
            chi_square = np.einsum("ns,ns->n", contrast_eta.T, solved[..., 0], optimize=True)
            nlogp_vals = chi2_to_nlogp(chi_square, contrast.shape[0])
            z_stats = nlogp_to_z(nlogp_vals, tail="two")

        return {
            "contrast_count": contrast.shape[0],
            "chi_square": chi_square,
            "p": _clip_p_values(np.exp(nlogp_vals), dtype=np.float64),
            "logp": _nlogp_to_logp_values(nlogp_vals, dtype=np.float64),
            "z": z_stats,
        }
