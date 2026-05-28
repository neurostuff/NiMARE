"""Coordinate Based Meta Regression Methods."""

import copy
import logging
import re
import time
from functools import wraps

import numpy as np
import pandas as pd
import scipy
import scipy.sparse

try:
    import torch  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "Torch is required to use `CBMR` classes. Install with `pip install 'nimare[cbmr]'`."
    ) from e

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta import models
from nimare.meta.utils import fit_spatial_cbmr_approximate
from nimare.results import MetaResult
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _clip_p_values,
    _minimum_positive_float,
    b_spline_bases,
    dummy_encoding_moderators,
    get_masker,
    get_masker_mask_image,
    mm2vox,
    seed_torch,
    validate_coordinate_spaces,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]
DEFAULT_GROUP_NAME = "Default"


def _uses_cuda(device):
    """Return whether the provided device string targets CUDA."""
    return str(device).startswith("cuda")


def _is_named_pairwise_contrast(contrast):
    """Return whether a contrast uses tuple shorthand like (A, B)."""
    return (
        isinstance(contrast, tuple)
        and len(contrast) == 2
        and all(isinstance(part, str) for part in contrast)
    )


def _normalize_named_pairwise_contrasts(contrasts):
    """Convert tuple shorthand like (A, B) into the legacy string form."""
    if contrasts is None:
        return None
    if isinstance(contrasts, str) or _is_named_pairwise_contrast(contrasts):
        contrasts = [contrasts]

    normalized = []
    for contrast in contrasts:
        if _is_named_pairwise_contrast(contrast):
            normalized.append(f"{contrast[0]}-{contrast[1]}")
        else:
            normalized.append(contrast)
    return normalized


class CBMRResult(MetaResult):
    """Meta-analytic result for CBMR with result-centered inference helpers.

    The same result class is used for both standard global-moderator CBMR and voxelwise
    moderator-effect CBMR. Model-specific inference is selected from the fitted estimator's
    ``moderator_effect`` attribute.
    """

    @property
    def moderator_effect(self):
        """Return the fitted moderator-effect parameterization."""
        return getattr(self.estimator, "moderator_effect", "global")

    @property
    def groups(self):
        """Return fitted group names in display order."""
        return tuple(getattr(self.estimator, "groups", ()) or ())

    @property
    def moderators(self):
        """Return fitted moderator names in display order."""
        return tuple(getattr(self.estimator, "moderators", ()) or ())

    @property
    def sv_moderator_names(self):
        """Return spatially varying moderator map names."""
        return tuple(name for name in self.maps if name.startswith("svModerator_"))

    def copy(self):
        """Return a copy of the CBMR result object."""
        new = CBMRResult(
            estimator=self.estimator,
            corrector=self.corrector,
            diagnostics=self.diagnostics,
            mask=self.masker,
            maps=copy.deepcopy(self.maps),
            tables=copy.deepcopy(self.tables),
            description=self.description_,
        )
        new.metadata = copy.deepcopy(self.metadata)
        return new

    def describe_inference_inputs(self):
        """Summarize the fitted groups, moderators, and moderator-effect type."""
        return {
            "groups": self.groups,
            "moderators": self.moderators,
            "moderator_effect": self.moderator_effect,
        }

    def describe_sv_effects(self):
        """Return simple summaries for spatially varying moderator maps."""
        return {
            name: (float(values.min()), float(values.mean()), float(values.max()))
            for name, values in self.maps.items()
            if name.startswith("svModerator_")
        }

    def get_inference(self, device=None, method=None, **kwargs):
        """Return a fitted inference engine for advanced CBMR use cases."""
        inference_device = device or getattr(self.estimator, "device", "cpu")
        inference = CBMRInference(
            device=inference_device,
            moderator_effect=self.moderator_effect,
            method=method,
            **kwargs,
        )
        inference.fit(self)
        return inference

    def infer(
        self,
        group_contrasts=False,
        moderator_contrasts=False,
        device=None,
        method=None,
        **kwargs,
    ):
        """Run CBMR inference from a fitted result."""
        inference = self.get_inference(device=device, method=method, **kwargs)
        return inference.transform(
            t_con_groups=group_contrasts,
            t_con_moderators=moderator_contrasts,
        )

    def test_groups(self, groups=None, device=None, method=None, **kwargs):
        """Run one-group spatial homogeneity tests for the requested groups."""
        group_contrasts = list(self.groups) if groups is None else groups
        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=False,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_groups(self, contrasts, device=None, method=None, **kwargs):
        """Run pairwise group-comparison tests using names or (group_a, group_b) tuples."""
        group_contrasts = _normalize_named_pairwise_contrasts(contrasts)
        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=False,
            device=device,
            method=method,
            **kwargs,
        )

    def test_moderators(self, moderators=None, device=None, method=None, **kwargs):
        """Test whether the requested moderator effects differ from zero."""
        if not self.moderators:
            raise ValueError("This CBMR result does not include moderators.")
        moderator_contrasts = list(self.moderators) if moderators is None else moderators
        return self.infer(
            group_contrasts=False,
            moderator_contrasts=moderator_contrasts,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_moderators(self, contrasts, device=None, method=None, **kwargs):
        """Run pairwise moderator-comparison tests using names or tuples."""
        if not self.moderators:
            raise ValueError("This CBMR result does not include moderators.")
        moderator_contrasts = _normalize_named_pairwise_contrasts(contrasts)
        return self.infer(
            group_contrasts=False,
            moderator_contrasts=moderator_contrasts,
            device=device,
            method=method,
            **kwargs,
        )


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
    moderator_effect : {"voxelwise", "global"}, optional
        How experiment-level moderator effects are parameterized. ``"global"`` fits the
        standard CBMR model with one coefficient per moderator and group. ``"voxelwise"`` fits
        the spatially varying CBMR backend, in which moderator effects vary smoothly over voxels.
        Default is ``"voxelwise"``.
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
    lr: :obj:`float`, optional
        Learning rate in optimization of log-likelihood function.
        Default is 1.
    lr_decay: :obj:`float`, optional
        Multiplicative factor of learning rate decay.
        Default is 0.999.
    tol: :obj:`float`, optional
        Stopping criterion based on the change in log-likelihood between two consecutive
        iterations. Default is 1e-9.
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') representing where operations will be allocated.
        Default is 'cpu'.
    random_state : :obj:`int`, optional
        Random seed used for torch-based weight initialization. Default is None.
    **kwargs
        Keyword arguments. Arguments for the Estimator can be assigned here,
        Another optional argument is ``mask``.

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
    _valid_moderator_effects = ("global", "voxelwise")
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
        moderator_effect="voxelwise",
        mask=None,
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
            moderator_effect=moderator_effect,
            mask=mask,
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
        moderator_effect="voxelwise",
        mask=None,
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
        if backend not in self._valid_backends:
            raise ValueError(f"backend must be one of {self._valid_backends}. Got {backend!r}.")
        if self.moderator_effect == "voxelwise" and model is not models.PoissonEstimator:
            raise ValueError("Voxelwise CBMR currently requires model=models.PoissonEstimator.")
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.group_categories = group_categories
        self.moderators = moderators

        self.spline_spacing = spline_spacing
        model_class = models.PoissonEstimator if self.moderator_effect == "voxelwise" else model
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
        self.spatial_varying_model = None
        self.spatial_varying_coef = None
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device cpu")
            self.device = "cpu"
        self.model.device = self.device

    def _generate_description(self):
        """Generate a description of the Estimator instance.

        Returns
        -------
        description : :obj:`str`
            Description of the Estimator instance.
        """
        description = """CBMR is a meta-regression framework that can explicitly model
                    group-wise spatial intensity function, and consider the effect of
                    moderators. It consists of two components: (1) a spatial
                    model that makes use of a spline parameterization to induce a smooth
                    response; (2) a generalized linear model (Poisson, Negative Binomial
                    (NB), Clustered NB) to model group-wise spatial intensity function).
                    CBMR is fitted via maximizing the log-likelihood function with L-BFGS
                    algorithm."""
        if self.moderators:
            moderators_str = f"""and accommodate the following moderators:
                            {', '.join(self.moderators)}"""
        else:
            moderators_str = ""
        moderator_effect_str = (
            " Moderator effects were modeled as global effects."
            if self.moderator_effect == "global"
            else " Moderator effects were modeled as voxelwise effects."
        )
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
            f"{len(self.inputs_['id'])} experiments."
        )

        description = model_description + "\n" + optimization_description
        return description

    def _make_result(self, dataset, maps=None, tables=None, description=""):
        """Construct a CBMR-specific result object."""
        masker = self.masker or dataset.masker
        return CBMRResult(self, mask=masker, maps=maps, tables=tables, description=description)

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
            "moderators_by_group": moderators_by_group,
            "foci_by_experiment": foci_by_experiment,
            "foci_per_voxel": foci_per_voxel,
            "foci_per_experiment": foci_per_experiment,
        }
        if self.moderator_effect == "voxelwise":
            inputs["foci_by_experiment_voxel"] = self._build_group_foci_matrices(
                filtered_coordinates,
                ids_by_group,
                n_mask_voxels,
            )
        return inputs

    @staticmethod
    def _build_group_foci_matrices(coordinates, ids_by_group, n_mask_voxels):
        """Return experiment-by-voxel foci count matrices for each group."""
        foci_by_experiment_voxel = {}
        if coordinates.empty:
            for group, group_ids in ids_by_group.items():
                foci_by_experiment_voxel[group] = scipy.sparse.csr_matrix(
                    (len(group_ids), n_mask_voxels),
                    dtype=np.float64,
                )
            return foci_by_experiment_voxel

        coordinates = coordinates.loc[:, ["id", "_cbmr_mask_index"]].copy()
        for group, group_ids in ids_by_group.items():
            id_to_row = {exp_id: i for i, exp_id in enumerate(group_ids)}
            group_coordinates = coordinates.loc[coordinates["id"].isin(group_ids)]
            rows = group_coordinates["id"].map(id_to_row).to_numpy(dtype=np.int64, copy=False)
            cols = group_coordinates["_cbmr_mask_index"].to_numpy(dtype=np.int64, copy=False)
            data = np.ones(group_coordinates.shape[0], dtype=np.float64)
            foci_by_experiment_voxel[group] = scipy.sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(len(group_ids), n_mask_voxels),
                dtype=np.float64,
            ).tocsr()
        return foci_by_experiment_voxel

    def _as_torch_tensor(self, value):
        """Convert an array-like object to a float64 tensor on the estimator device."""
        if scipy.sparse.issparse(value):
            value = value.toarray()
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def _prepare_torch_inputs(self):
        """Return tensorized spatially varying CBMR inputs."""
        bases = self._as_torch_tensor(self.inputs_["coef_spline_bases"])
        moderators_by_group = None
        if self.moderators:
            moderators_by_group = {
                group: self._as_torch_tensor(self.inputs_["moderators_by_group"][group])
                for group in self.groups
            }
        foci_by_experiment_voxel = {
            group: self._as_torch_tensor(self.inputs_["foci_by_experiment_voxel"][group])
            for group in self.groups
        }
        return bases, moderators_by_group, foci_by_experiment_voxel

    def _spatial_cbmr_description(self, backend):
        """Generate a NiMARE-style description for spatially varying CBMR."""
        if self.moderators:
            moderator_text = (
                "with spatially varying experiment-level moderator effects for "
                f"{', '.join(self.moderators)}"
            )
        else:
            moderator_text = "without experiment-level moderators"
        return (
            f"Spatially varying CBMR was performed with the {backend} backend "
            f"{moderator_text}. The model used {len(self.groups)} group(s), "
            f"spline spacing {self.spline_spacing}, and device {self.device}."
        )

    def _build_group_foci(self, coordinates, ids_by_group, n_mask_voxels):
        """Summarize voxelwise and experiment-wise focus counts for each group."""
        grouped_coordinates = coordinates.loc[:, ["id", "_cbmr_mask_index"]].copy()
        id_to_group = pd.Series(
            {id_: group for group, group_ids in ids_by_group.items() for id_ in group_ids}
        )
        grouped_coordinates[self._group_column] = grouped_coordinates["id"].map(id_to_group)
        grouped_coordinates = grouped_coordinates[
            grouped_coordinates[self._group_column].notna()
        ].reset_index(drop=True)

        foci_by_experiment = {}
        foci_per_voxel = {}
        foci_per_experiment = {}
        for group, group_ids in ids_by_group.items():
            group_coordinates = grouped_coordinates.loc[
                grouped_coordinates[self._group_column] == group
            ]
            if group_coordinates.empty:
                group_foci_by_experiment = scipy.sparse.csr_matrix(
                    (len(group_ids), n_mask_voxels), dtype=np.int32
                )
            else:
                id_to_row = pd.Series(
                    np.arange(len(group_ids), dtype=np.int32),
                    index=group_ids,
                )
                row_indices = (
                    group_coordinates["id"].map(id_to_row).to_numpy(dtype=np.int32, copy=False)
                )
                column_indices = group_coordinates["_cbmr_mask_index"].to_numpy(
                    dtype=np.int32, copy=False
                )
                data = np.ones(group_coordinates.shape[0], dtype=np.int32)
                group_foci_by_experiment = scipy.sparse.coo_matrix(
                    (data, (row_indices, column_indices)),
                    shape=(len(group_ids), n_mask_voxels),
                    dtype=np.int32,
                ).tocsr()

            group_foci_per_voxel = np.asarray(
                group_foci_by_experiment.sum(axis=0), dtype=np.int32
            ).reshape((-1, 1))
            group_foci_per_experiment = np.asarray(
                group_foci_by_experiment.sum(axis=1), dtype=np.int32
            ).reshape((-1, 1))

            foci_by_experiment[group] = group_foci_by_experiment
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
        masker, mask_img = get_masker_mask_image(
            self.masker,
            dataset=dataset,
            message=(
                "A masker is required for coordinate-based meta-analysis. "
                "Provide a `mask` to the Estimator or initialize the Dataset with a `target` "
                "and/or `mask` so `dataset.masker` is defined."
            ),
        )
        validate_coordinate_spaces(self.inputs_["coordinates"])
        mask_data, mask_lookup, n_mask_voxels = self._initialize_spatial_inputs(masker, mask_img)
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
        coefficient of overdispersion.s

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection to analyze.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        if self.moderator_effect == "voxelwise":
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
        """Fit spatially varying CBMR with the full torch L-BFGS backend."""
        seed_torch(self.random_state, self.device)
        bases, moderators_by_group, foci_by_experiment_voxel = self._prepare_torch_inputs()
        moderators_coef_dim = len(self.moderators) if self.moderators else None
        self.spatial_varying_model = models.SpatialCBMRModel(
            groups=self.groups,
            spatial_coef_dim=self.inputs_["coef_spline_bases"].shape[1],
            moderators_coef_dim=moderators_coef_dim,
            device=self.device,
        )
        optimizer = torch.optim.LBFGS(
            params=self.spatial_varying_model.parameters(),
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
            loss = self.spatial_varying_model(
                bases,
                moderators_by_group,
                foci_by_experiment_voxel,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        scheduler.step()
        LGR.info("Spatially varying CBMR optimisation took %.1f s.", time.time() - start_time)
        maps, tables = self._extract_torch_results(moderators_by_group)
        return maps, tables, self._spatial_cbmr_description("full L-BFGS")

    def _fit_approximate(self, dataset):
        """Fit spatially varying CBMR with the approximate backend."""
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        self.spatial_varying_coef = {}
        for group in self.groups:
            foci = self.inputs_["foci_by_experiment_voxel"][group]
            if self.moderators:
                moderators = self.inputs_["moderators_by_group"][group]
            else:
                moderators = np.empty((foci.shape[0], 0), dtype=np.float64)

            augmented_moderators = np.column_stack(
                [moderators, np.ones((foci.shape[0], 1), dtype=np.float64)]
            )
            coefficient = self._get_spatial_cbmr_approximate_solver()(
                augmented_moderators,
                bases,
                foci,
                tol=self.tol,
                max_iter=self.n_iter,
                alpha=self.alpha,
                damping=self.damping,
                compute_nll=self.compute_nll,
            )
            self.spatial_varying_coef[group] = coefficient
            self._add_approximate_results(
                maps,
                tables,
                group,
                moderators,
                coefficient,
            )
        return maps, tables, self._spatial_cbmr_description("approximate")

    @staticmethod
    def _get_spatial_cbmr_approximate_solver():
        """Return the approximate solver used by the voxelwise backend."""
        return fit_spatial_cbmr_approximate

    def _extract_torch_results(self, moderators_by_group):
        """Extract maps and coefficient tables from the fitted torch model."""
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        for group in self.groups:
            spatial_coef = (
                self.spatial_varying_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            maps[f"spatialIntensity_group-{group}"] = np.exp(bases @ spatial_coef)
            self._add_spatial_coef_table(tables, group, spatial_coef)
            if self.moderators:
                moderator_coef = (
                    self.spatial_varying_model.moderator_coef_linears[group]
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
        """Append one group to the aggregate spatially varying moderator table."""
        indexed_table = group_moderator_table.copy()
        indexed_table.index = pd.MultiIndex.from_product(
            [[group], indexed_table.index],
            names=["group", "moderator"],
        )
        if "sv_moderators_regression_coef" in tables:
            tables["sv_moderators_regression_coef"] = pd.concat(
                [tables["sv_moderators_regression_coef"], indexed_table]
            )
        else:
            tables["sv_moderators_regression_coef"] = indexed_table

    @staticmethod
    def _add_moderator_table(tables, group, moderator_names, moderator_coef):
        """Add spatially varying moderator coefficient tables for one group."""
        group_moderator_table = pd.DataFrame(
            moderator_coef,
            index=moderator_names,
            columns=[f"basis_{i}" for i in range(moderator_coef.shape[1])],
        )
        tables[f"sv_moderator_regression_coef_group-{group}"] = group_moderator_table
        CBMREstimator._append_group_moderator_table(
            tables,
            group,
            group_moderator_table,
        )

    def _add_moderator_maps_and_tables(self, maps, tables, group, moderators, moderator_coef):
        """Add spatially varying moderator maps and tables for one group."""
        bases = self.inputs_["coef_spline_bases"]
        for index, moderator_name in enumerate(self.moderators):
            moderator_effect = moderators[:, index : index + 1] @ moderator_coef[index : index + 1]
            maps[f"svModerator_{moderator_name}_group-{group}"] = (
                moderator_effect @ bases.T
            ).mean(axis=0)
        maps[f"svModeratorTotal_group-{group}"] = (moderators @ moderator_coef @ bases.T).mean(
            axis=0
        )
        self._add_moderator_table(tables, group, self.moderators, moderator_coef)

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


class CBMRInference(object):
    """Statistical inference on fitted CBMR results.

    Notes
    -----
        This is the public inference entry point.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') represents the device on which operations will be allocated.
        Default is 'cpu'.
    moderator_effect : {"voxelwise", "global"}, optional
        Inference parameterization to use. ``"voxelwise"`` uses the integrated spatial CBMR
        inference backend with sandwich or inverse-Fisher covariance estimates. ``"global"`` uses
        the standard CBMR inference backend. Default is ``"voxelwise"``.
    """

    _valid_methods = ("sandwich", "FI")
    _valid_sandwich_meats = ("cluster", "iid")
    _valid_sandwich_corrections = (None, "hc0", "hc1", "hc3")
    _voxelwise_default_method = "sandwich"

    def __init__(
        self,
        device="cpu",
        moderator_effect="voxelwise",
        method=None,
        sandwich_meat="cluster",
        sandwich_correction="hc3",
        ridge=1e-6,
    ):
        self.moderator_effect = self._normalize_moderator_effect(moderator_effect)
        self.device = device
        # device check
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device 'cpu'")
            self.device = "cpu"

        self.result = None
        self.groups = None
        self.moderators = None

        if self.moderator_effect == "global":
            self._validate_global_pipeline_options(
                method,
                sandwich_meat,
                sandwich_correction,
                ridge,
            )
        else:  # voxelwise
            if method is None:
                method = "sandwich"
            self.method = self._validate_method(method)
            self.sandwich_meat = self._validate_sandwich_meat(sandwich_meat)
            self.sandwich_correction = self._validate_sandwich_correction(sandwich_correction)
            if ridge < 0:
                raise ValueError("ridge must be nonnegative.")
            self.ridge = ridge

        self._reset_inference_caches()

    @classmethod
    def _normalize_moderator_effect(cls, moderator_effect):
        """Normalize the shared public moderator-effect selector."""
        return CBMREstimator._validate_moderator_effect(moderator_effect)

    @classmethod
    def _validate_method(cls, method):
        """Validate and normalize an inference standard-error method."""
        if isinstance(method, str):
            if method.lower() == "fi":
                return "FI"
            if method.lower() == "sandwich":
                return "sandwich"
        raise ValueError("method must be one of {'sandwich', 'FI'}.")

    @classmethod
    def _validate_sandwich_meat(cls, sandwich_meat):
        """Validate and normalize the sandwich meat estimator."""
        if isinstance(sandwich_meat, str):
            sandwich_meat = sandwich_meat.lower()
        if sandwich_meat not in cls._valid_sandwich_meats:
            raise ValueError("sandwich_meat must be either 'cluster' or 'iid'.")
        return sandwich_meat

    @classmethod
    def _validate_sandwich_correction(cls, sandwich_correction):
        """Validate and normalize the sandwich leverage correction."""
        if isinstance(sandwich_correction, str):
            sandwich_correction = sandwich_correction.lower()
        if sandwich_correction not in cls._valid_sandwich_corrections:
            raise ValueError("sandwich_correction must be None, 'hc0', 'hc1', or 'hc3'.")
        return sandwich_correction

    @staticmethod
    def _validate_global_pipeline_options(method, sandwich_meat, sandwich_correction, ridge):
        """Reject voxelwise-only options when the global pipeline is selected."""
        if method is not None:
            raise ValueError("method is only supported for voxelwise moderator effects.")
        if sandwich_meat != "cluster" or sandwich_correction != "hc3" or ridge != 1e-6:
            raise ValueError(
                "sandwich_meat, sandwich_correction, and ridge are only supported for "
                "voxelwise moderator effects."
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
        if self.moderator_effect == "global":
            self._group_spatial_covariance_cache = {}
            self._moderator_covariance = None
            self._moderator_variance = None
            self._moderator_coef_table = None
        else:  # voxelwise
            self._group_covariance_cache = {}
            self._group_coefficient_cache = {}

    def _get_group_log_intensity(self, group):
        """Return cached group log-intensity values."""
        group_log_intensity = self._group_log_intensity_cache.get(group)
        if group_log_intensity is None:
            if self.moderator_effect == "global":
                group_log_intensity = np.log(self.result.maps[f"spatialIntensity_group-{group}"])
            else:  # voxelwise
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
            else:  # voxelwise
                foci = self.estimator.inputs_["foci_by_experiment_voxel"][group]
                total_foci = foci.sum()
                n_experiments, n_voxels = foci.shape
                group_null_log_intensity = np.log(max(float(total_foci), np.finfo(float).tiny))
                group_null_log_intensity -= np.log(n_experiments * n_voxels)
            self._group_null_log_intensity_cache[group] = group_null_log_intensity
        return group_null_log_intensity

    def _get_group_spatial_covariance(self, involved_groups):
        """Return cached spatial covariance for the involved groups."""
        group_key = tuple(involved_groups)
        cov_spatial_coef = self._group_spatial_covariance_cache.get(group_key)
        if cov_spatial_coef is None:
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
            self._group_spatial_covariance_cache[group_key] = cov_spatial_coef
        return cov_spatial_coef

    def _get_moderator_covariance(self):
        """Return cached moderator covariance and marginal variances."""
        if self._moderator_covariance is None:
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
            self._moderator_variance = np.diag(self._moderator_covariance)
        return self._moderator_covariance, self._moderator_variance

    def fit(self, result):
        """Fit CBMRInference instance.

        Parameters
        ----------
        result : :obj:`~nimare.meta.cbmr.CBMRResult`
            Fitted CBMR result containing regression coefficient tables and spatial intensity
            maps.
        """
        if self.moderator_effect == "voxelwise":
            if not isinstance(result, CBMRResult) or result.moderator_effect != "voxelwise":
                raise TypeError(
                    "CBMRInference.fit with moderator_effect='voxelwise' requires a "
                    "CBMRResult with moderator_effect='voxelwise'."
                )
        else:
            if not isinstance(result, CBMRResult) or result.moderator_effect != "global":
                raise TypeError(
                    "CBMRInference.fit with moderator_effect='global' requires a "
                    "CBMRResult with moderator_effect='global'."
                )

        self.result = self._copy_result_for_inference(result)
        self._reset_inference_caches()
        self.estimator = self.result.estimator
        self.groups = list(self.result.groups)
        self.moderators = list(self.result.moderators)

        if self.moderator_effect == "global":
            self.estimator.device = self.device
            self.estimator.model.device = self.device
            self.estimator.model.to(self.device)
            self.estimator.model._invalidate_tensor_inputs_cache()
            if self.moderators:
                self._moderator_coef_table = (
                    self.result.tables["moderators_regression_coef"].to_numpy().T
                )

        self.create_regular_expressions()

        self.group_reference_dict = {
            group_name: index for index, group_name in enumerate(self.groups)
        }
        self.moderator_reference_dict = {}
        if self.moderators:
            self.moderator_reference_dict = {
                moderator_name: index for index, moderator_name in enumerate(self.moderators)
            }
            for moderator_name, index in self.moderator_reference_dict.items():
                LGR.info(f"{moderator_name} = index_{index}")

    @_check_fit
    def display(self):
        """Display Groups and Moderator names and order."""
        # visialize group/moderator names and their indices in contrast array
        LGR.info("Group Reference in contrast array")
        for group, index in self.group_reference_dict.items():
            LGR.info(f"{group} = index_{index}")
        if self.moderators:
            LGR.info("Moderator Reference in contrast array")
            for moderator, index in self.moderator_reference_dict.items():
                LGR.info(f"{moderator} = index_{index}")

    def create_regular_expressions(self):
        """
        Create regular expressions for parsing contrast names.

        creates the following attributes:
        self.groups_regular_expression: regular expression for parsing group names
        self.moderators_regular_expression: regular expression for parsing moderator names

        usage:
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

            setattr(self, "{}_regular_expression".format(attr), reg_expr)

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
        if source == "groups":  # contrast matrix for spatial intensity
            for contrast in contrast_name:
                contrast_vector = np.zeros(len(self.groups))
                contrast_match = self.groups_regular_expression.match(contrast)
                # check validity of contrast name
                if contrast_match is None:
                    raise ValueError(f"{contrast} is not a valid contrast.")
                groups_contrast = contrast_match.groupdict()
                # create contrast matrix
                if all(groups_contrast.values()):  # group comparison
                    contrast_vector[self.group_reference_dict[groups_contrast["first"]]] = 1
                    contrast_vector[self.group_reference_dict[groups_contrast["second"]]] = int(
                        contrast_match["operator"] + "1"
                    )
                else:  # homogeneity test
                    contrast_vector[self.group_reference_dict[contrast]] = 1
                contrast_matrix[contrast] = contrast_vector

        elif source == "moderators":  # contrast matrix for moderator effect
            for contrast in contrast_name:
                contrast_vector = np.zeros(len(self.moderators))
                contrast_match = self.moderators_regular_expression.match(contrast)
                if contrast_match is None:
                    raise ValueError(f"{contrast} is not a valid contrast.")
                moderators_contrast = contrast_match.groupdict()
                if all(moderators_contrast.values()):  # moderator comparison
                    contrast_vector[
                        self.moderator_reference_dict[moderators_contrast["first"]]
                    ] = 1
                    contrast_vector[
                        self.moderator_reference_dict[moderators_contrast["second"]]
                    ] = int(moderators_contrast["operator"] + "1")
                else:  # moderator effect
                    contrast_vector[self.moderator_reference_dict[contrast]] = 1
                contrast_matrix[contrast] = contrast_vector

        return contrast_matrix

    @_check_fit
    def transform(self, t_con_groups=None, t_con_moderators=None, method=None):
        """Conduct generalized linear hypothesis (GLH) testing on CBMR estimates."""
        if self.moderator_effect == "voxelwise":
            if method is not None:
                validated_method = self._validate_method(method)
                if validated_method != self.method:
                    self.method = validated_method
                    self._reset_inference_caches()
            self.result.metadata["spatial_cbmr_inference_method"] = self.method
            if self.method == "sandwich":
                self.result.metadata["spatial_cbmr_sandwich_meat"] = self.sandwich_meat
                self.result.metadata["spatial_cbmr_sandwich_correction"] = self.sandwich_correction
            else:
                self.result.metadata.pop("spatial_cbmr_sandwich_meat", None)
                self.result.metadata.pop("spatial_cbmr_sandwich_correction", None)
        else:
            if method is not None:
                raise ValueError("method is only supported for voxelwise moderator effects.")

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
        """Fit and transform."""
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
        if self.moderator_effect == "voxelwise":
            cov_spatial_coef = self._get_intercept_covariance_for_groups(involved_groups)
            spatial_coef_dim = None
            n_brain_voxel = None
        else:
            cov_spatial_coef = self._get_group_spatial_covariance(involved_groups)

        if con_group.shape[0] == 1:
            z_stats_spatial, p_vals_spatial = self._compute_group_wald_statistics(
                simp_con_group,
                involved_groups,
                cov_spatial_coef,
                contrast_log_intensity,
                X,
                spatial_coef_dim,
            )
            chi_sq_spatial = None
        else:
            chi_sq_spatial, z_stats_spatial, p_vals_spatial = self._compute_group_glh_statistics(
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
            "p": p_vals_spatial,
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
        if n_con_group_involved == 1:
            p_vals_spatial = scipy.stats.norm.sf(z_stats_spatial)
        else:
            p_vals_spatial = scipy.stats.norm.sf(abs(z_stats_spatial)) * 2
        p_vals_spatial = _clip_p_values(
            p_vals_spatial,
            dtype=DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        return z_stats_spatial, p_vals_spatial

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
        p_vals_spatial = scipy.stats.chi2.sf(chi_sq_spatial, df=m)
        p_vals_spatial = _clip_p_values(
            p_vals_spatial,
            dtype=DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        if is_homogeneity_test:
            z_stats_spatial = scipy.stats.norm.isf(p_vals_spatial)
            z_stats_spatial[z_stats_spatial < 0] = 0
        else:
            z_p_values = np.maximum(
                p_vals_spatial,
                2 * _minimum_positive_float(p_vals_spatial.dtype),
            )
            z_stats_spatial = scipy.stats.norm.isf(z_p_values / 2)
            if simp_con_group.shape[0] == 1:
                z_stats_spatial *= np.sign(contrast_log_intensity.flatten())
        z_stats_spatial = np.clip(z_stats_spatial, a_min=-10, a_max=10)
        return chi_sq_spatial, z_stats_spatial, p_vals_spatial

    def _store_group_inference_result(self, con_group_count, group_stats):
        """Write one computed group-inference result into result maps."""
        contrast_name = self.t_con_groups_name[con_group_count] if self.t_con_groups_name else None
        if contrast_name:
            if group_stats["contrast_count"] > 1:
                self.result.maps[f"chiSquare_group-{contrast_name}"] = group_stats["chi_square"]
            self.result.maps[f"p_group-{contrast_name}"] = group_stats["p"]
            self.result.maps[f"z_group-{contrast_name}"] = group_stats["z"]
        else:
            if group_stats["contrast_count"] > 1:
                self.result.maps[f"chiSquare_GLH_groups_{con_group_count}"] = group_stats[
                    "chi_square"
                ]
            self.result.maps[f"p_GLH_groups_{con_group_count}"] = group_stats["p"]
            self.result.maps[f"z_GLH_groups_{con_group_count}"] = group_stats["z"]

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
        if self.moderator_effect == "voxelwise":
            for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
                for group in self.groups:
                    moderator_stats = self._evaluate_moderator_contrast(group, con_moderator)
                    self._store_moderator_inference_result(
                        con_moderator_count,
                        group,
                        moderator_stats,
                    )
            return

        cov_moderator_coef, var_moderator_coef = self._get_moderator_covariance()
        moderator_coef = self._moderator_coef_table
        for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
            moderator_stats = self._evaluate_moderator_contrast(
                con_moderator,
                cov_moderator_coef,
                var_moderator_coef,
                moderator_coef,
            )
            self._store_moderator_inference_result(con_moderator_count, moderator_stats)

    def _evaluate_moderator_contrast(
        self,
        con_moderator,
        cov_moderator_coef=None,
        var_moderator_coef=None,
        moderator_coef=None,
    ):
        """Compute statistics for one prepared moderator contrast."""
        if self.moderator_effect == "voxelwise":
            group = con_moderator
            con_moderator = cov_moderator_coef
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

        m_con_moderator = con_moderator.shape[0]
        contrast_moderator_coef = np.matmul(con_moderator, moderator_coef)
        if m_con_moderator == 1:
            involved_var_moderator_coef = con_moderator**2 @ var_moderator_coef
            involved_std_moderator_coef = np.sqrt(involved_var_moderator_coef)
            z_stats_moderator = contrast_moderator_coef / involved_std_moderator_coef
            p_vals_moderator = scipy.stats.norm.sf(abs(z_stats_moderator)) * 2
            p_vals_moderator = _clip_p_values(
                p_vals_moderator,
                dtype=np.asarray(p_vals_moderator).dtype,
                copy=False,
            )
            chi_sq_moderator = None
        else:
            contrast_covariance = con_moderator @ cov_moderator_coef @ con_moderator.T
            solved = np.linalg.solve(contrast_covariance, contrast_moderator_coef)
            chi_sq_moderator = contrast_moderator_coef.T @ solved
            p_vals_moderator = scipy.stats.chi2.sf(chi_sq_moderator, df=m_con_moderator)
            p_vals_moderator = _clip_p_values(
                p_vals_moderator,
                dtype=np.asarray(p_vals_moderator).dtype,
                copy=False,
            )
            z_p_values = np.maximum(
                p_vals_moderator,
                2 * _minimum_positive_float(np.asarray(p_vals_moderator).dtype),
            )
            z_stats_moderator = scipy.stats.norm.isf(z_p_values / 2)

        return {
            "contrast_count": m_con_moderator,
            "chi_square": chi_sq_moderator,
            "p": p_vals_moderator,
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
                if moderator_stats["contrast_count"] > 1:
                    self.result.maps[f"chiSquare_svModerator_{contrast_name}_group-{group}"] = (
                        moderator_stats["chi_square"]
                    )
                self.result.maps[f"p_svModerator_{contrast_name}_group-{group}"] = moderator_stats[
                    "p"
                ]
                self.result.maps[f"z_svModerator_{contrast_name}_group-{group}"] = moderator_stats[
                    "z"
                ]
            else:
                if moderator_stats["contrast_count"] > 1:
                    self.result.maps[
                        f"chiSquare_GLH_svModerators_{con_moderator_count}_group-{group}"
                    ] = moderator_stats["chi_square"]
                self.result.maps[f"p_GLH_svModerators_{con_moderator_count}_group-{group}"] = (
                    moderator_stats["p"]
                )
                self.result.maps[f"z_GLH_svModerators_{con_moderator_count}_group-{group}"] = (
                    moderator_stats["z"]
                )
            return

        moderator_stats = moderator_stats_or_group
        contrast_name = (
            self.t_con_moderators_name[con_moderator_count] if self.t_con_moderators_name else None
        )
        if contrast_name:
            if moderator_stats["contrast_count"] > 1:
                self.result.tables[f"chi_square_{contrast_name}"] = pd.DataFrame(
                    data=np.array(moderator_stats["chi_square"]),
                    columns=["chi_square"],
                )
            self.result.tables[f"p_{contrast_name}"] = pd.DataFrame(
                data=np.array(moderator_stats["p"]),
                columns=["p"],
            )
            self.result.tables[f"z_{contrast_name}"] = pd.DataFrame(
                data=np.array(moderator_stats["z"]),
                columns=["z"],
            )
        else:
            if moderator_stats["contrast_count"] > 1:
                self.result.tables[f"chi_square_GLH_moderators_{con_moderator_count}"] = (
                    pd.DataFrame(
                        data=np.array(moderator_stats["chi_square"]),
                        columns=["chi_square"],
                    )
                )
            self.result.tables[f"p_GLH_moderators_{con_moderator_count}"] = pd.DataFrame(
                data=np.array(moderator_stats["p"]),
                columns=["p"],
            )
            self.result.tables[f"z_GLH_moderators_{con_moderator_count}"] = pd.DataFrame(
                data=np.array(moderator_stats["z"]),
                columns=["z"],
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
        if self.moderators:
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
        if getattr(self.estimator, "spatial_varying_coef", None) is not None:
            coefficient = np.asarray(self.estimator.spatial_varying_coef[group]).reshape(
                (-1, n_bases)
            )
        elif getattr(self.estimator, "spatial_varying_model", None) is not None:
            spatial_coef = (
                self.estimator.spatial_varying_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            if self.moderators:
                moderator_coef = (
                    self.estimator.spatial_varying_model.moderator_coef_linears[group]
                    .weight.detach()
                    .cpu()
                    .numpy()
                )
                coefficient = np.vstack([moderator_coef, spatial_coef])
            else:
                coefficient = spatial_coef.reshape((1, -1))
        else:
            spatial_coef = self.result.tables["spatial_regression_coef"].loc[group].to_numpy()
            if self.moderators:
                moderator_coef = self.result.tables[
                    f"sv_moderator_regression_coef_group-{group}"
                ].to_numpy()
                coefficient = np.vstack([moderator_coef, spatial_coef])
            else:
                coefficient = spatial_coef.reshape((1, -1))

        self._group_coefficient_cache[group] = coefficient
        return coefficient

    def _get_group_mean(self, group):
        """Return fitted Poisson mean for one group as experiments by voxels."""
        bases = self.estimator.inputs_["coef_spline_bases"]
        moderators = self._get_group_augmented_moderators(group)
        coefficient = self._get_group_coefficient_matrix(group)
        return np.exp(np.clip(moderators @ coefficient @ bases.T, -100, 100))

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
    def _as_dense_response(foci):
        """Return a dense experiment-by-voxel response matrix."""
        if scipy.sparse.issparse(foci):
            return foci.toarray()
        return np.asarray(foci, dtype=float)

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
        foci = foci.tocsr()
        if residual_scale is None:
            adjusted_mean = mean
            adjusted_foci = foci
        else:
            adjusted_mean = mean / residual_scale
            foci_coo = foci.tocoo()
            adjusted_foci = scipy.sparse.csr_matrix(
                (
                    foci_coo.data / residual_scale[foci_coo.row, foci_coo.col],
                    (foci_coo.row, foci_coo.col),
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
        foci_coo = adjusted_foci.tocoo()
        delta = (
            foci_coo.data**2
            - 2
            * foci_coo.data
            * adjusted_mean[
                foci_coo.row,
                foci_coo.col,
            ]
        )
        delta_matrix = scipy.sparse.csr_matrix(
            (delta, (foci_coo.row, foci_coo.col)),
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
        """Compute robust Poisson sandwich covariance for one spatial CBMR group."""
        moderators = np.asarray(moderators, dtype=float)
        bases = np.asarray(bases, dtype=float)
        mean = np.asarray(mean, dtype=float)
        mean = np.nan_to_num(mean, nan=0.0, posinf=1e12, neginf=0.0)
        mean = np.clip(mean, 1e-12, 1e12)
        response_shape = foci.shape if scipy.sparse.issparse(foci) else np.asarray(foci).shape

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
            y = cls._as_dense_response(foci)
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
        n_bases = self.estimator.inputs_["coef_spline_bases"].shape[1]
        covariance = np.zeros((len(involved_groups) * n_bases, len(involved_groups) * n_bases))
        for group_index, group in enumerate(involved_groups):
            group_covariance = self._get_group_covariance(group)
            block = group_covariance[-n_bases:, -n_bases:]
            group_slice = slice(group_index * n_bases, (group_index + 1) * n_bases)
            covariance[group_slice, group_slice] = block
        return covariance

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
        """Compute voxel-wise Wald statistics for spatially varying coefficients."""
        contrast_eta = contrast @ coefficient @ bases.T
        contrast_cov = cls._contrast_covariance_by_voxel(contrast, covariance, bases)

        if contrast.shape[0] == 1:
            contrast_var = contrast_cov[:, 0, 0]
            contrast_std = np.sqrt(np.maximum(contrast_var, 0.0))
            z_stats = contrast_eta[0] / np.where(contrast_std > 0, contrast_std, np.inf)
            p_vals = scipy.stats.norm.sf(np.abs(z_stats)) * 2
            p_vals = _clip_p_values(
                p_vals,
                dtype=np.asarray(p_vals).dtype,
                copy=False,
            )
            chi_square = None
        else:
            solved = np.linalg.solve(contrast_cov, contrast_eta.T[..., np.newaxis])
            chi_square = np.einsum("ns,ns->n", contrast_eta.T, solved[..., 0], optimize=True)
            p_vals = scipy.stats.chi2.sf(chi_square, df=contrast.shape[0])
            p_vals = _clip_p_values(
                p_vals,
                dtype=np.asarray(p_vals).dtype,
                copy=False,
            )
            z_p_values = np.maximum(
                p_vals,
                2 * _minimum_positive_float(np.asarray(p_vals).dtype),
            )
            z_stats = scipy.stats.norm.isf(z_p_values / 2)

        return {
            "contrast_count": contrast.shape[0],
            "chi_square": chi_square,
            "p": p_vals,
            "z": z_stats,
        }
