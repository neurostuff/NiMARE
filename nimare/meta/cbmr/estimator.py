"""Coordinate-based meta-regression estimator."""

import logging
import time

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta import models
from nimare.meta.cbmr._helpers import (
    DEFAULT_GROUP_NAME,
    DEFAULT_INCIDENCE_THRESHOLD,
    _as_csr_matrix,
    _uses_cuda,
    _validate_incidence_threshold,
)
from nimare.meta.cbmr._torch import torch
from nimare.meta.cbmr.basis import b_spline_bases
from nimare.meta.cbmr.optimizers import fit_voxelwise_cbmr_approximate
from nimare.meta.cbmr.results import CBMRResult
from nimare.utils import (
    dummy_encoding_moderators,
    get_masker,
    get_masker_mask_image,
    get_template,
    mm2vox,
    seed_torch,
    validate_coordinate_spaces,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


class CBMREstimator(Estimator):
    """Coordinate-based meta-regression with a spatial model.

    .. warning::
        Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed in
        a future release. Prefer :class:`~nimare.nimads.Studyset`.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    group_categories : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR allows a collection to be categorized into multiple groups according to one or more
        group categories. Default is one-group CBMR.
    moderators : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR can accommodate moderators (e.g. sample size, year of publication).
        Default is CBMR without moderators.
    global_moderators : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        Moderators with scalar, whole-brain effects. In global CBMR, this is an alias for
        ``moderators``. In mixed CBMR, these are modeled separately from voxelwise moderators.
    voxelwise_moderators : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        Moderators with spatially varying effects. In voxelwise CBMR, this is an alias for
        ``moderators``. In mixed CBMR, these are modeled separately from global moderators.
    moderator_effect : {"voxelwise", "global"}, optional
        How experiment-level moderator effects are parameterized. ``"global"`` fits the
        standard CBMR model with one coefficient per moderator and group. ``"voxelwise"`` fits
        the voxelwise moderator-effect CBMR backend, in which moderator effects vary smoothly over
        voxels.
        Default is ``"global"``.
    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, or Nilearn masker, optional
        Region-of-interest mask. If None, CBMR uses the whole 2 mm MNI152 brain mask.
    incidence_threshold : :obj:`float` or None, optional
        Drop voxels with empirical focus incidence less than or equal to this threshold after
        applying ``mask``. Empirical incidence is the fraction of experiments with at least one
        focus in a voxel. Use None to retain all voxels in ``mask``. Default is 0.001.
    model : subclass of :class:`~nimare.meta.models.GeneralLinearModelEstimator`, optional
        Stochastic model class used by CBMR. Available options are:

        - :class:`~nimare.meta.models.PoissonEstimator` (default): the most efficient
          and widely used option, but slightly less accurate because it approximates
          low-rate binomial data, cannot account for over-dispersion in foci counts,
          and may underestimate standard errors.
        - :class:`~nimare.meta.models.NegativeBinomialEstimator`: slower and
          sometimes less stable, but slightly more accurate. This model allows
          anticipated excess variance relative to Poisson via a group-wise
          overdispersion parameter shared by all experiments and voxels.
        - :class:`~nimare.meta.models.ClusteredNegativeBinomialEstimator`: a
          random-effects Poisson variant that models experiment-level latent
          characteristics shared across the brain for a given experiment.
    penalty : :obj:`~bool`, optional
        Currently, the only available option is Firth-type penalty, which penalizes the
        likelihood function by Jeffreys' invariant prior and encourages convergence.
    spline_spacing : :obj:`~int`, optional
        Spatial structure of foci counts is parameterized by the coefficients of cubic
        B-spline bases in CBMR. Spatial smoothness in CBMR is determined by spline spacing,
        which is shared across the x, y, and z dimensions. Default is 10.
    n_iter : :obj:`int`, optional
        Number of iterations allowed in the log-likelihood optimization.
        Default is 2000.
    lr : :obj:`float`, optional
        Learning rate in optimization of log-likelihood function.
        Default is 1.
    lr_decay : :obj:`float`, optional
        Multiplicative factor of learning rate decay.
        Default is 0.999.
    tol : :obj:`float`, optional
        Stopping criterion based on the change in log-likelihood between two consecutive
        iterations. Default is 1e-9.
    device : :obj:`string`, optional
        Device type ('cpu' or 'cuda') representing where operations will be allocated.
        Default is 'cpu'.
    random_state : :obj:`int`, optional
        Random seed used for torch-based weight initialization. Default is None.
    **kwargs
        Keyword arguments. Arguments for the Estimator can be assigned here,
        Additional masking controls are exposed as ``mask`` and ``incidence_threshold``.

    Attributes
    ----------
    masker : :class:`~nilearn.maskers.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMR estimators, this includes the following keys:
        coordinates,
        mask_img (brain mask image),
        id (experiment ids),
        ids_by_group (experiment ids categorized by groups),
        moderators_by_group (moderators categorized by groups, if present),
        coef_spline_bases (spatial matrix of cubic B-spline coefficients in x, y, and z),
        foci_by_experiment (experiment-by-voxel sparse focus-count matrices, categorized by
        groups),
        foci_per_voxel (voxelwise sum of foci counts across experiments, categorized by groups),
        foci_per_experiment (experiment-wise sum of foci counts across space, categorized by
        groups).

    Notes
    -----
    Follow-up inference is exposed through :class:`~nimare.meta.cbmr.CBMRResult` and
    :class:`~nimare.meta.cbmr.CBMRInference`.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}
    _group_column = "_cbmr_group"
    _valid_moderator_effects = ("global", "voxelwise", "mixed")
    _valid_backends = ("full", "approximate")

    @classmethod
    def _validate_moderator_effect(cls, moderator_effect):
        """Validate and normalize the public moderator-effect selector."""
        if isinstance(moderator_effect, str):
            moderator_effect = moderator_effect.lower()
        if moderator_effect not in cls._valid_moderator_effects:
            raise ValueError(
                "moderator_effect must be one of "
                f"{cls._valid_moderator_effects}. Got {moderator_effect!r}."
            )
        return moderator_effect

    def __init__(
        self,
        group_categories=None,
        moderators=None,
        global_moderators=None,
        voxelwise_moderators=None,
        global_moderator=None,
        moderator_effect="global",
        mask=None,
        incidence_threshold=DEFAULT_INCIDENCE_THRESHOLD,
        spline_spacing=10,
        model=models.PoissonEstimator,
        penalty=False,
        backend="full",
        n_iter=2000,
        lr=1,
        lr_decay=0.999,
        tol=1e-9,
        device="cpu",
        random_state=None,
        alpha=1.0,
        damping=1e-4,
        compute_nll=False,
        **kwargs,
    ):
        self._init_pipeline_state(
            group_categories=group_categories,
            moderators=moderators,
            global_moderators=global_moderators,
            voxelwise_moderators=voxelwise_moderators,
            global_moderator=global_moderator,
            moderator_effect=moderator_effect,
            mask=mask,
            incidence_threshold=incidence_threshold,
            spline_spacing=spline_spacing,
            model=model,
            penalty=penalty,
            backend=backend,
            n_iter=n_iter,
            lr=lr,
            lr_decay=lr_decay,
            tol=tol,
            device=device,
            random_state=random_state,
            alpha=alpha,
            damping=damping,
            compute_nll=compute_nll,
            **kwargs,
        )

    def _init_pipeline_state(
        self,
        group_categories=None,
        moderators=None,
        global_moderators=None,
        voxelwise_moderators=None,
        global_moderator=None,
        moderator_effect="global",
        mask=None,
        incidence_threshold=DEFAULT_INCIDENCE_THRESHOLD,
        spline_spacing=10,
        model=models.PoissonEstimator,
        penalty=False,
        backend="full",
        n_iter=2000,
        lr=1,
        lr_decay=0.999,
        tol=1e-9,
        device="cpu",
        random_state=None,
        alpha=1.0,
        damping=1e-4,
        compute_nll=False,
        **kwargs,
    ):
        """Initialize shared estimator state for the selected concrete CBMR pipeline."""
        super().__init__(**kwargs)
        self.moderator_effect = self._validate_moderator_effect(moderator_effect)
        if global_moderator is not None:
            if global_moderators is not None:
                raise ValueError("Use only one of global_moderator or global_moderators.")
            global_moderators = global_moderator
        self.incidence_threshold = _validate_incidence_threshold(incidence_threshold)
        if backend not in self._valid_backends:
            raise ValueError(f"backend must be one of {self._valid_backends}. Got {backend!r}.")
        if (
            self.moderator_effect in ("voxelwise", "mixed")
            and model is not models.PoissonEstimator
        ):
            raise ValueError("Voxelwise CBMR currently requires model=models.PoissonEstimator.")
        if self.moderator_effect == "mixed" and backend != "full":
            raise ValueError("Mixed CBMR currently requires backend='full'.")
        self.mask = mask
        self.masker = get_masker(mask) if mask is not None else None

        self.group_categories = group_categories
        self.global_moderators = self._as_moderator_list(global_moderators)
        self.voxelwise_moderators = self._as_moderator_list(voxelwise_moderators)
        self.moderators = self._resolve_moderators_for_effect(
            moderators,
            self.global_moderators,
            self.voxelwise_moderators,
        )

        self.spline_spacing = spline_spacing
        model_class = (
            models.PoissonEstimator if self.moderator_effect in ("voxelwise", "mixed") else model
        )
        self.model = model_class(
            penalty=penalty, lr=lr, lr_decay=lr_decay, n_iter=n_iter, tol=tol, device=device
        )
        self.penalty = penalty
        self.backend = backend
        self.n_iter = n_iter
        self.lr = lr
        self.lr_decay = lr_decay
        self.tol = tol
        self.device = device
        self.random_state = random_state
        self.alpha = alpha
        self.damping = damping
        self.compute_nll = compute_nll
        self.voxelwise_model = None
        self.voxelwise_coef = None
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("CUDA not found; using device 'cpu'.")
            self.device = "cpu"
        self.model.device = self.device

    @staticmethod
    def _as_moderator_list(moderators):
        """Normalize moderator selectors to a list without expanding categorical variables."""
        if moderators is None:
            return []
        if isinstance(moderators, str):
            return [moderators]
        return list(moderators)

    def _resolve_moderators_for_effect(
        self,
        moderators,
        global_moderators,
        voxelwise_moderators,
    ):
        """Resolve public moderator arguments into the estimator's active moderators."""
        if self.moderator_effect == "mixed":
            if moderators is not None:
                raise ValueError(
                    "Use global_moderators and/or voxelwise_moderators when "
                    "moderator_effect='mixed'."
                )
            duplicated = set(global_moderators) & set(voxelwise_moderators)
            if duplicated:
                raise ValueError(
                    "The same moderator cannot be both global and voxelwise in a mixed CBMR "
                    f"model: {sorted(duplicated)}."
                )
            return global_moderators + voxelwise_moderators

        if self.moderator_effect == "global":
            if voxelwise_moderators:
                raise ValueError(
                    "voxelwise_moderators can only be used when "
                    "moderator_effect='voxelwise' or 'mixed'."
                )
            if global_moderators:
                if moderators is not None:
                    raise ValueError(
                        "Use only one of moderators or global_moderators when "
                        "moderator_effect='global'."
                    )
                return global_moderators
            return moderators

        if global_moderators:
            raise ValueError(
                "global_moderators can only be used when " "moderator_effect='global' or 'mixed'."
            )
        if voxelwise_moderators:
            if moderators is not None:
                raise ValueError(
                    "Use only one of moderators or voxelwise_moderators when "
                    "moderator_effect='voxelwise'."
                )
            return voxelwise_moderators
        return moderators

    def _generate_description(self):
        """Generate a description of the Estimator instance.

        Returns
        -------
        description : :obj:`str`
            Description of the Estimator instance.
        """
        if self.moderator_effect == "mixed":
            moderator_parts = []
            if self.global_moderators:
                moderator_parts.append(f"global moderators: {', '.join(self.global_moderators)}")
            if self.voxelwise_moderators:
                moderator_parts.append(
                    f"voxelwise moderators: {', '.join(self.voxelwise_moderators)}"
                )
            moderators_str = (
                f"and accommodate the following moderators: {'; '.join(moderator_parts)}"
                if moderator_parts
                else ""
            )
        elif self.moderators:
            moderators_str = f"""and accommodate the following moderators:
                            {', '.join(self.moderators)}"""
        else:
            moderators_str = ""
        if self.moderator_effect == "global":
            moderator_effect_str = " Moderator effects were modeled as global effects."
        elif self.moderator_effect == "mixed":
            moderator_effect_str = (
                " Moderator effects were modeled with a mixture of global and voxelwise effects."
            )
        else:
            moderator_effect_str = " Moderator effects were modeled as voxelwise effects."
        if self.model.penalty:
            penalty_str = " Firth-type penalty is applied to ensure convergence."
        else:
            penalty_str = ""

        if type(self.model).__name__ == "PoissonEstimator":
            model_str = (
                " Here, Poisson model \\citep{eisenberg1966general} is the most basic CBMR model. "
                "It's based on the assumption that foci arise from a realisation of a continuous "
                "inhomogeneous Poisson process, so that the (discrete) voxel-wise foci counts will"
                " be independently distributed as Poisson random variables, with rate equal to the"
                " integral of the true, unobserved, continuous intensity function over each voxel."
            )
        elif type(self.model).__name__ == "NegativeBinomialEstimator":
            model_str = (
                " Negative Binomial (NB) model \\citep{barndorff1969negative} is a generalized "
                "Poisson model with over-dispersion. "
                "It's a more flexible model, but more difficult to estimate. In practice, foci"
                "counts often display over-dispersion (the variance of response variable"
                " substantially exceeds the mean), which is not captured by the Poisson model."
            )
        elif type(self.model).__name__ == "ClusteredNegativeBinomialEstimator":
            model_str = (
                " Clustered NB model \\citep{geoffroy2001poisson} can also accommodate "
                "over-dispersion in foci counts. "
                "In the NB model, the latent random variable introduces independent variation "
                "at each voxel. While in the Clustered NB model, we assert the random effects "
                "are not independent voxelwise effects, but rather latent characteristics of "
                "each experiment, and represent a shared effect over the entire brain for a "
                "given experiment."
            )

        model_description = (
            f"CBMR is a meta-regression framework that was performed with NiMARE {__version__}. "
            f"{type(self.model).__name__} model was used to model group-wise spatial intensity "
            f"functions {moderators_str}." + moderator_effect_str + model_str
        )

        optimization_description = (
            "CBMR is fitted via maximizing the log-likelihood function with L-BFGS algorithm, with"
            f" learning rate {self.lr}, learning rate decay {self.lr_decay} and "
            f"tolerance {self.tol}." + penalty_str + f" The optimization is run on {self.device}."
            f" The input dataset included {self.inputs_['coordinates'].shape[0]} foci from "
            f"{len(self.inputs_['id'])} experiments. The analysis mask included "
            f"{self.inputs_['coef_spline_bases'].shape[0]} voxels after ROI and empirical "
            f"incidence filtering."
        )

        description = model_description + "\n" + optimization_description
        return description

    def _make_result(self, dataset, maps=None, tables=None, description=""):
        """Construct a CBMR-specific result object."""
        masker = self.masker or dataset.masker
        return CBMRResult(self, mask=masker, maps=maps, tables=tables, description=description)

    def _resolve_roi_masker(self, dataset):
        """Return the user-requested ROI masker or the default 2 mm MNI brain masker."""
        if self.masker is not None:
            return get_masker(self.masker)

        default_mask_img = get_template(space="mni152_2mm", mask="brain")
        return get_masker(default_mask_img)

    @staticmethod
    def _mask_image_from_data(mask_data, reference_img):
        """Create a binary mask image aligned to a reference image."""
        header = reference_img.header.copy()
        header.set_data_dtype(np.uint8)
        return nib.Nifti1Image(mask_data.astype(np.uint8), reference_img.affine, header)

    @staticmethod
    def _build_mask_lookup(mask_data):
        """Return a flat-index lookup from full image space to masked voxel space."""
        n_mask_voxels = int(mask_data.sum())
        mask_lookup = np.full(mask_data.size, -1, dtype=np.int32)
        mask_lookup[np.flatnonzero(mask_data.ravel())] = np.arange(n_mask_voxels, dtype=np.int32)
        return mask_lookup, n_mask_voxels

    def _initialize_spatial_inputs(self, masker, mask_img):
        """Build and cache mask-derived spatial inputs used by CBMR."""
        self.inputs_["mask_img"] = mask_img
        mask_data = np.asanyarray(mask_img.dataobj).astype(bool, copy=False)
        mask_lookup, n_mask_voxels = self._build_mask_lookup(mask_data)
        self.inputs_["coef_spline_bases"] = b_spline_bases(
            masker_voxels=mask_data,
            spacing=self.spline_spacing,
        )
        return mask_data, mask_lookup, n_mask_voxels

    def _compute_empirical_incidence(self, coordinates, ids_by_group, n_mask_voxels):
        """Return the empirical voxel incidence rate across experiments."""
        n_experiments = sum(len(group_ids) for group_ids in ids_by_group.values())
        if n_experiments == 0:
            raise ValueError("CBMR requires at least one experiment.")

        foci_by_experiment = self._build_group_foci_matrices(
            coordinates,
            ids_by_group,
            n_mask_voxels,
        )
        incidence_count = np.zeros(n_mask_voxels, dtype=np.float64)
        for group_matrix in foci_by_experiment.values():
            incidence_count += np.asarray((group_matrix > 0).sum(axis=0)).ravel()
        return incidence_count / float(n_experiments)

    def _threshold_mask_by_incidence(
        self,
        mask_img,
        mask_data,
        filtered_coordinates,
        ids_by_group,
        n_mask_voxels,
    ):
        """Apply empirical-incidence filtering to an ROI mask."""
        incidence_rate = self._compute_empirical_incidence(
            filtered_coordinates,
            ids_by_group,
            n_mask_voxels,
        )
        self.inputs_["empirical_incidence_rate_roi"] = incidence_rate

        if self.incidence_threshold is None:
            keep_voxels = np.ones(n_mask_voxels, dtype=bool)
        else:
            keep_voxels = incidence_rate > self.incidence_threshold

        if not np.any(keep_voxels):
            raise ValueError(
                "No voxels survived CBMR incidence filtering. Lower incidence_threshold or "
                "provide a less restrictive mask."
            )

        thresholded_mask_data = np.zeros(mask_data.size, dtype=bool)
        roi_flat_indices = np.flatnonzero(mask_data.ravel())
        thresholded_mask_data[roi_flat_indices[keep_voxels]] = True
        thresholded_mask_data = thresholded_mask_data.reshape(mask_data.shape)
        self.inputs_["empirical_incidence_rate"] = incidence_rate[keep_voxels]
        self.inputs_["incidence_threshold"] = self.incidence_threshold
        return self._mask_image_from_data(thresholded_mask_data, mask_img)

    def _filter_coordinates_to_mask(self, coordinates, mask_img, mask_data, mask_lookup):
        """Filter coordinates to the mask and attach masked-space voxel indices."""
        if coordinates.empty:
            filtered_coordinates = coordinates.copy()
            filtered_coordinates["_cbmr_mask_index"] = pd.Series(dtype=np.int32)
            return filtered_coordinates

        ijk = mm2vox(coordinates[["x", "y", "z"]].to_numpy(), mask_img.affine)
        shape = np.asarray(mask_data.shape, dtype=np.int64)
        in_bounds = np.all((ijk >= 0) & (ijk < shape), axis=1)

        keep_mask = np.zeros(coordinates.shape[0], dtype=bool)
        mask_indices = np.empty(0, dtype=np.int32)
        if np.any(in_bounds):
            bounded_idx = np.where(in_bounds)[0]
            bounded_ijk = ijk[bounded_idx]
            flat_voxel_index = np.ravel_multi_index(bounded_ijk.T, mask_data.shape)
            in_mask = mask_data.ravel()[flat_voxel_index]
            kept_idx = bounded_idx[in_mask]
            keep_mask[kept_idx] = True
            mask_indices = mask_lookup[flat_voxel_index[in_mask]]

        n_dropped = int(coordinates.shape[0] - keep_mask.sum())
        LGR.info(
            "%d/%d coordinates fall outside of the mask. Removing them.",
            n_dropped,
            coordinates.shape[0],
        )

        filtered_coordinates = coordinates.loc[keep_mask].copy()
        filtered_coordinates["_cbmr_mask_index"] = mask_indices
        return filtered_coordinates

    @staticmethod
    def _format_group_name(group_value):
        """Normalize a group label into the public map/table naming format."""
        if isinstance(group_value, (list, tuple, np.ndarray, pd.Series)):
            return "".join(str(value).capitalize() for value in group_value)
        return str(group_value).capitalize()

    def _collect_experiment_annotations(self, dataset):
        """Return one annotation row per experiment in the collected input order."""
        experiment_annotations = (
            dataset.annotations_df[dataset.annotations_df["id"].isin(self.inputs_["id"])]
            .drop_duplicates(subset=["id"])
            .set_index("id", drop=False)
            .reindex(self.inputs_["id"])
            .reset_index(drop=True)
        )
        experiment_annotations = experiment_annotations.copy()
        experiment_annotations["id"] = self.inputs_["id"]
        return experiment_annotations

    def _assign_group_labels(self, experiment_annotations):
        """Attach normalized group labels to the aligned experiment table."""
        if self.group_categories is None:
            experiment_annotations[self._group_column] = DEFAULT_GROUP_NAME
        elif isinstance(self.group_categories, str):
            if self.group_categories not in experiment_annotations.columns:
                raise ValueError(
                    f"Category_names: {self.group_categories} does not exist in the dataset"
                )
            experiment_annotations[self._group_column] = experiment_annotations[
                self.group_categories
            ].map(self._format_group_name)
        elif isinstance(self.group_categories, list):
            missing_categories = set(self.group_categories) - set(experiment_annotations.columns)
            if missing_categories:
                raise ValueError(
                    f"Category_names: {missing_categories} do/does not exist in the dataset."
                )
            experiment_annotations[self._group_column] = experiment_annotations[
                self.group_categories
            ].apply(lambda row: self._format_group_name(row.tolist()), axis=1)
        else:
            raise ValueError("group_categories must be None, a string, or a list of strings.")

        return experiment_annotations

    def _index_experiments_by_group(self, experiment_annotations):
        """Return experiment IDs grouped in the order used by downstream summaries."""
        ids_by_group = (
            experiment_annotations.groupby(self._group_column, sort=False)["id"]
            .agg(list)
            .to_dict()
        )
        return ids_by_group

    def _build_group_moderators(self, experiment_annotations):
        """Collect moderator arrays in the same experiment order used elsewhere in CBMR."""
        if self.moderator_effect == "mixed":
            outputs = {}
            all_moderators = []
            for attr, input_key in (
                ("global_moderators", "global_moderators_by_group"),
                ("voxelwise_moderators", "voxelwise_moderators_by_group"),
            ):
                moderator_names = getattr(self, attr)
                if not moderator_names:
                    outputs[input_key] = None
                    setattr(self, attr, [])
                    continue
                experiment_annotations, moderator_names = dummy_encoding_moderators(
                    experiment_annotations,
                    moderator_names,
                )
                moderator_names = self._as_moderator_list(moderator_names)
                setattr(self, attr, moderator_names)
                all_moderators.extend(moderator_names)
                moderators_by_group = {}
                for group, group_annotations in experiment_annotations.groupby(
                    self._group_column, sort=False
                ):
                    moderators_by_group[group] = group_annotations[moderator_names].to_numpy()
                outputs[input_key] = moderators_by_group

            self.moderators = all_moderators
            outputs["moderators_by_group"] = outputs["voxelwise_moderators_by_group"]
            return experiment_annotations, outputs

        if not self.moderators:
            self.inputs_.pop("moderators_by_group", None)
            return experiment_annotations, None

        experiment_annotations, self.moderators = dummy_encoding_moderators(
            experiment_annotations, self.moderators
        )
        if isinstance(self.moderators, str):
            self.moderators = [self.moderators]

        moderators_by_group = {}
        for group, group_annotations in experiment_annotations.groupby(
            self._group_column, sort=False
        ):
            moderators_by_group[group] = group_annotations[self.moderators].to_numpy()

        return experiment_annotations, moderators_by_group

    def _build_experiment_group_inputs(self, dataset, filtered_coordinates, n_mask_voxels):
        """Assemble grouped experiment IDs, moderators, and focus summaries."""
        experiment_annotations = self._collect_experiment_annotations(dataset)
        experiment_annotations = self._assign_group_labels(experiment_annotations)
        ids_by_group = self._index_experiments_by_group(experiment_annotations)
        self.groups = list(ids_by_group.keys())

        experiment_annotations, moderators_by_group = self._build_group_moderators(
            experiment_annotations
        )
        foci_by_experiment, foci_per_voxel, foci_per_experiment = self._build_group_foci(
            filtered_coordinates,
            ids_by_group,
            n_mask_voxels,
        )
        inputs = {
            "ids_by_group": ids_by_group,
            "foci_by_experiment": foci_by_experiment,
            "foci_per_voxel": foci_per_voxel,
            "foci_per_experiment": foci_per_experiment,
        }
        if self.moderator_effect == "mixed":
            inputs.update(moderators_by_group)
        else:
            inputs["moderators_by_group"] = moderators_by_group
        if self.moderator_effect in ("voxelwise", "mixed"):
            inputs["foci_by_experiment_voxel"] = self._build_group_foci_matrices(
                filtered_coordinates,
                ids_by_group,
                n_mask_voxels,
            )
        return inputs

    @staticmethod
    def _build_group_foci_matrices(coordinates, ids_by_group, n_mask_voxels):
        """Return experiment-by-voxel foci count matrices for each group."""
        return CBMREstimator._build_group_sparse_foci_matrices(
            coordinates,
            ids_by_group,
            n_mask_voxels,
            dtype=np.float64,
        )

    @staticmethod
    def _build_group_sparse_foci_matrices(coordinates, ids_by_group, n_mask_voxels, dtype):
        """Return grouped experiment-by-voxel sparse focus-count matrices."""
        foci_by_experiment = {}
        if coordinates.empty:
            for group, group_ids in ids_by_group.items():
                foci_by_experiment[group] = scipy.sparse.csr_matrix(
                    (len(group_ids), n_mask_voxels),
                    dtype=dtype,
                )
            return foci_by_experiment

        coordinates = coordinates.loc[:, ["id", "_cbmr_mask_index"]].copy()
        for group, group_ids in ids_by_group.items():
            id_to_row = {exp_id: i for i, exp_id in enumerate(group_ids)}
            group_coordinates = coordinates.loc[coordinates["id"].isin(group_ids)]
            rows = group_coordinates["id"].map(id_to_row).to_numpy(dtype=np.int64, copy=False)
            cols = group_coordinates["_cbmr_mask_index"].to_numpy(dtype=np.int64, copy=False)
            data = np.ones(group_coordinates.shape[0], dtype=dtype)
            foci_by_experiment[group] = scipy.sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(group_ids), n_mask_voxels),
                dtype=dtype,
            )
        return foci_by_experiment

    def _as_torch_tensor(self, value):
        """Convert an array-like object to a float64 tensor on the estimator device."""
        if scipy.sparse.issparse(value):
            value = _as_csr_matrix(value).toarray()
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def _prepare_torch_inputs(self):
        """Return tensorized voxelwise moderator-effect CBMR inputs."""
        bases = self._as_torch_tensor(self.inputs_["coef_spline_bases"])
        moderators_by_group = None
        if self.moderator_effect == "mixed":
            moderator_names = self.voxelwise_moderators
            moderator_key = "voxelwise_moderators_by_group"
        else:
            moderator_names = self.moderators
            moderator_key = "moderators_by_group"
        if moderator_names:
            moderators_by_group = {
                group: self._as_torch_tensor(self.inputs_[moderator_key][group])
                for group in self.groups
            }
        global_moderators_by_group = None
        if self.moderator_effect == "mixed" and self.global_moderators:
            global_moderators_by_group = {
                group: self._as_torch_tensor(self.inputs_["global_moderators_by_group"][group])
                for group in self.groups
            }
        foci_by_experiment_voxel = {
            group: self._as_torch_tensor(self.inputs_["foci_by_experiment_voxel"][group])
            for group in self.groups
        }
        if self.moderator_effect == "mixed":
            return bases, moderators_by_group, foci_by_experiment_voxel, global_moderators_by_group
        return bases, moderators_by_group, foci_by_experiment_voxel

    def _voxelwise_cbmr_description(self, backend):
        """Generate a NiMARE-style description for voxelwise moderator-effect CBMR."""
        if self.moderator_effect == "mixed":
            moderator_text = []
            if self.global_moderators:
                moderator_text.append(
                    f"global moderator effects for {', '.join(self.global_moderators)}"
                )
            if self.voxelwise_moderators:
                moderator_text.append(
                    f"voxelwise moderator effects for {', '.join(self.voxelwise_moderators)}"
                )
            moderator_text = (
                "with " + " and ".join(moderator_text)
                if moderator_text
                else "without experiment-level moderators"
            )
        elif self.moderators:
            moderator_text = (
                "with voxelwise moderator effects for " f"{', '.join(self.moderators)}"
            )
        else:
            moderator_text = "without experiment-level moderators"
        return (
            f"Voxelwise moderator-effect CBMR was performed with the {backend} backend "
            f"{moderator_text}. The model used {len(self.groups)} group(s), "
            f"spline spacing {self.spline_spacing}, device {self.device}, and "
            f"{self.inputs_['coef_spline_bases'].shape[0]} analysis-mask voxels after ROI and "
            f"empirical incidence filtering."
        )

    def _build_group_foci(self, coordinates, ids_by_group, n_mask_voxels):
        """Summarize voxelwise and experiment-wise focus counts for each group."""
        foci_by_experiment = self._build_group_sparse_foci_matrices(
            coordinates,
            ids_by_group,
            n_mask_voxels,
            dtype=np.int32,
        )

        foci_per_voxel = {}
        foci_per_experiment = {}
        for group, group_foci_by_experiment in foci_by_experiment.items():
            group_foci_per_voxel = np.asarray(
                group_foci_by_experiment.sum(axis=0), dtype=np.int32
            ).reshape((-1, 1))
            group_foci_per_experiment = np.asarray(
                group_foci_by_experiment.sum(axis=1), dtype=np.int32
            ).reshape((-1, 1))

            foci_per_voxel[group] = group_foci_per_voxel
            foci_per_experiment[group] = group_foci_per_experiment

        return foci_by_experiment, foci_per_voxel, foci_per_experiment

    def _preprocess_input(self, dataset):
        """Mask required input images using either the Dataset's mask or the Estimator's.

        Also, categorize experiment id, voxelwise sum of foci counts across experiments,
        experiment-wise sum of foci counts across space into multiple groups. And summarize
        moderators into
        multiple groups (if exist).

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            In this method, the collection is used to (1) select the appropriate mask image,
            (2) categorize experiments into multiple groups according to group categories in
            annotations,
            (3) summarize group-wise experiment id, moderators (if exist), foci per voxel, foci
            per experiment,
            (4) extract sample size metadata and use it as one of the moderators.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            Specifically, (1) a "mask_img" key will be added (brain mask image),
            (2) an 'id' key will be added (id of all experiments in the dataset),
            (3) a 'coef_spline_bases' key will be added (spatial matrix of coefficient of cubic
            B-spline bases in x,y,z dimension),
            (4) an 'ids_by_group' key will be added (experiment id categorized by groups),
            (5) a 'moderators_by_group' key will be added (moderators categorized
            by groups) if moderators are considered,
            (6) a 'foci_by_experiment' key will be added (experiment-by-voxel sparse focus-count
            matrices, categorized by groups),
            (7) an 'foci_per_voxel' key will be added (voxelwise sum of foci count across
            experiments, categorized by groups),
            (8) an 'foci_per_experiment' key will be added (experiment-wise sum of
            foci count across space, categorized by groups).

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        validate_coordinate_spaces(self.inputs_["coordinates"])

        roi_masker = self._resolve_roi_masker(dataset)
        _, roi_mask_img = get_masker_mask_image(roi_masker)
        roi_mask_data = np.asanyarray(roi_mask_img.dataobj).astype(bool, copy=False)
        roi_mask_lookup, n_roi_voxels = self._build_mask_lookup(roi_mask_data)
        roi_filtered_coordinates = self._filter_coordinates_to_mask(
            self.inputs_["coordinates"],
            roi_mask_img,
            roi_mask_data,
            roi_mask_lookup,
        )

        experiment_annotations = self._collect_experiment_annotations(dataset)
        experiment_annotations = self._assign_group_labels(experiment_annotations)
        ids_by_group = self._index_experiments_by_group(experiment_annotations)

        analysis_mask_img = self._threshold_mask_by_incidence(
            roi_mask_img,
            roi_mask_data,
            roi_filtered_coordinates,
            ids_by_group,
            n_roi_voxels,
        )
        self.masker = get_masker(analysis_mask_img)
        _, mask_img = get_masker_mask_image(self.masker)
        mask_data, mask_lookup, n_mask_voxels = self._initialize_spatial_inputs(
            self.masker,
            mask_img,
        )
        filtered_coordinates = self._filter_coordinates_to_mask(
            self.inputs_["coordinates"],
            mask_img,
            mask_data,
            mask_lookup,
        )
        self.inputs_["coordinates"] = filtered_coordinates.drop(columns=["_cbmr_mask_index"])
        self.inputs_.update(
            self._build_experiment_group_inputs(
                dataset,
                filtered_coordinates,
                n_mask_voxels,
            )
        )

    def _fit(self, dataset):
        """Perform coordinate-based meta-regression (CBMR) on dataset.

        (1) Estimate group-wise spatial regression coefficients and its standard error via
        inverse of Fisher Information matrix; Similarly, estimate regression coefficient of
        moderators (if exist), as well as its standard error via inverse of
        Fisher Information matrix;
        (2) Estimate standard error of group-wise log intensity, group-wise intensity via delta
        method;
        (3) For NegativeBinomial or ClusteredNegativeBinomial model, estimate regression
        coefficient of overdispersion.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection to analyze.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        if self.moderator_effect in ("voxelwise", "mixed"):
            if self.backend == "approximate":
                return self._fit_approximate(dataset)
            return self._fit_full(dataset)

        init_weight_kwargs = {
            "groups": self.groups,
            "moderators": self.moderators,
            "spatial_coef_dim": self.inputs_["coef_spline_bases"].shape[1],
            "moderators_coef_dim": len(self.moderators) if self.moderators else None,
        }
        seed_torch(self.random_state, self.device)
        self.model.init_weights(**init_weight_kwargs)

        moderators_by_group = self.inputs_["moderators_by_group"] if self.moderators else None
        self.model.fit(
            self.inputs_["coef_spline_bases"],
            moderators_by_group,
            self.inputs_["foci_per_voxel"],
            self.inputs_["foci_per_experiment"],
        )

        maps, tables = self.model.summary()

        return maps, tables, self._description_text()

    def _fit_full(self, dataset):
        """Fit voxelwise moderator-effect CBMR with the full torch L-BFGS backend."""
        seed_torch(self.random_state, self.device)
        torch_inputs = self._prepare_torch_inputs()
        if self.moderator_effect == "mixed":
            (
                bases,
                moderators_by_group,
                foci_by_experiment_voxel,
                global_moderators_by_group,
            ) = torch_inputs
        else:
            bases, moderators_by_group, foci_by_experiment_voxel = torch_inputs
            global_moderators_by_group = None
        voxelwise_moderators = (
            self.voxelwise_moderators if self.moderator_effect == "mixed" else self.moderators
        )
        moderators_coef_dim = len(voxelwise_moderators) if voxelwise_moderators else None
        global_moderators_coef_dim = (
            len(self.global_moderators)
            if self.moderator_effect == "mixed" and self.global_moderators
            else None
        )
        self.voxelwise_model = models.SpatialCBMRModel(
            groups=self.groups,
            spatial_coef_dim=self.inputs_["coef_spline_bases"].shape[1],
            moderators_coef_dim=moderators_coef_dim,
            global_moderators_coef_dim=global_moderators_coef_dim,
            device=self.device,
        )
        optimizer = torch.optim.LBFGS(
            params=self.voxelwise_model.parameters(),
            lr=self.lr,
            max_iter=self.n_iter,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe",
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        start_time = time.time()

        def closure():
            """Evaluate the L-BFGS closure required by torch."""
            optimizer.zero_grad()
            loss = self.voxelwise_model(
                bases,
                moderators_by_group,
                foci_by_experiment_voxel,
                global_moderators_by_group=global_moderators_by_group,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        scheduler.step()
        LGR.info(
            "Voxelwise moderator-effect CBMR optimisation took %.1f s.",
            time.time() - start_time,
        )
        maps, tables = self._extract_torch_results(moderators_by_group)
        return maps, tables, self._voxelwise_cbmr_description("full L-BFGS")

    def _fit_approximate(self, dataset):
        """Fit voxelwise moderator-effect CBMR with the approximate backend."""
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        self.voxelwise_coef = {}
        for group in self.groups:
            foci = self.inputs_["foci_by_experiment_voxel"][group]
            if self.moderators:
                moderators = self.inputs_["moderators_by_group"][group]
            else:
                moderators = np.empty((foci.shape[0], 0), dtype=np.float64)

            augmented_moderators = np.column_stack(
                [moderators, np.ones((foci.shape[0], 1), dtype=np.float64)]
            )
            coefficient = self._voxelwise_cbmr_approximate_solver(
                augmented_moderators,
                bases,
                foci,
                tol=self.tol,
                max_iter=self.n_iter,
                alpha=self.alpha,
                damping=self.damping,
                compute_nll=self.compute_nll,
            )
            self.voxelwise_coef[group] = coefficient
            self._add_approximate_results(
                maps,
                tables,
                group,
                moderators,
                coefficient,
            )
        return maps, tables, self._voxelwise_cbmr_description("approximate")

    @property
    def _voxelwise_cbmr_approximate_solver(self):
        """Return the approximate solver used by the voxelwise backend."""
        return fit_voxelwise_cbmr_approximate

    def _extract_torch_results(self, moderators_by_group):
        """Extract maps and coefficient tables from the fitted torch model."""
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        for group in self.groups:
            spatial_coef = (
                self.voxelwise_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            maps[f"spatialIntensity_group-{group}"] = np.exp(bases @ spatial_coef)
            self._add_spatial_coef_table(tables, group, spatial_coef)
            voxelwise_moderators = (
                self.voxelwise_moderators if self.moderator_effect == "mixed" else self.moderators
            )
            if voxelwise_moderators:
                moderator_coef = (
                    self.voxelwise_model.moderator_coef_linears[group]
                    .weight.detach()
                    .cpu()
                    .numpy()
                )
                moderators = moderators_by_group[group].detach().cpu().numpy()
                self._add_moderator_maps_and_tables(
                    maps,
                    tables,
                    group,
                    moderators,
                    moderator_coef,
                )
        if self.moderator_effect == "mixed" and self.global_moderators:
            global_coef = (
                self.voxelwise_model.global_moderators_linear.weight.detach().cpu().numpy()
            )
            tables["global_moderators_regression_coef"] = pd.DataFrame(
                data=global_coef,
                columns=self.global_moderators,
            )
        return maps, tables

    @staticmethod
    def _add_spatial_coef_table(tables, group, spatial_coef):
        """Add one group to the CBMR-style spatial coefficient table."""
        columns = [f"basis_{i}" for i in range(spatial_coef.size)]
        spatial_coef_table = tables.get(
            "spatial_regression_coef",
            pd.DataFrame(columns=columns, dtype=np.float64),
        )
        spatial_coef_table = spatial_coef_table.reindex(columns=columns)
        spatial_coef_table.loc[group] = spatial_coef
        tables["spatial_regression_coef"] = spatial_coef_table

    @staticmethod
    def _append_group_moderator_table(tables, group, group_moderator_table):
        """Append one group to the aggregate voxelwise moderator-effect table."""
        indexed_table = group_moderator_table.copy()
        indexed_table.index = pd.MultiIndex.from_product(
            [[group], indexed_table.index],
            names=["group", "moderator"],
        )
        if "voxelwise_moderator_effects_regression_coef" in tables:
            tables["voxelwise_moderator_effects_regression_coef"] = pd.concat(
                [tables["voxelwise_moderator_effects_regression_coef"], indexed_table]
            )
        else:
            tables["voxelwise_moderator_effects_regression_coef"] = indexed_table

    @staticmethod
    def _add_moderator_table(tables, group, moderator_names, moderator_coef):
        """Add voxelwise moderator-effect coefficient tables for one group."""
        group_moderator_table = pd.DataFrame(
            moderator_coef,
            index=moderator_names,
            columns=[f"basis_{i}" for i in range(moderator_coef.shape[1])],
        )
        tables[f"voxelwise_moderator_effect_regression_coef_group-{group}"] = group_moderator_table
        CBMREstimator._append_group_moderator_table(
            tables,
            group,
            group_moderator_table,
        )

    def _add_moderator_maps_and_tables(self, maps, tables, group, moderators, moderator_coef):
        """Add voxelwise moderator-effect maps and tables for one group."""
        bases = self.inputs_["coef_spline_bases"]
        moderator_names = (
            self.voxelwise_moderators if self.moderator_effect == "mixed" else self.moderators
        )
        for index, moderator_name in enumerate(moderator_names):
            moderator_effect = moderators[:, index : index + 1] @ moderator_coef[index : index + 1]
            maps[f"voxelwiseModeratorEffect_{moderator_name}_group-{group}"] = (
                moderator_effect @ bases.T
            ).mean(axis=0)
        maps[f"voxelwiseModeratorEffectTotal_group-{group}"] = (
            moderators @ moderator_coef @ bases.T
        ).mean(axis=0)
        self._add_moderator_table(tables, group, moderator_names, moderator_coef)

    def _add_approximate_results(self, maps, tables, group, moderators, coefficient):
        """Add maps and coefficient tables for one approximate-backend group."""
        bases = self.inputs_["coef_spline_bases"]
        n_bases = bases.shape[1]
        coefficient = coefficient.reshape((-1, n_bases))
        moderator_coef = coefficient[:-1]
        spatial_coef = coefficient[-1]
        maps[f"spatialIntensity_group-{group}"] = np.exp(bases @ spatial_coef)
        self._add_spatial_coef_table(tables, group, spatial_coef)
        if self.moderators:
            self._add_moderator_maps_and_tables(
                maps,
                tables,
                group,
                moderators,
                moderator_coef,
            )
