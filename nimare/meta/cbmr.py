"""Coordinate Based Meta Regression Methods."""

import copy
import logging
import re
from functools import wraps

import nibabel as nib
import numpy as np
import pandas as pd
import scipy

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Torch is required to use `CBMR` classes. Install with `pip install 'nimare[cbmr]'`."
    ) from e

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta import models
from nimare.results import MetaResult
from nimare.studyset import normalize_collection
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    b_spline_bases,
    dummy_encoding_moderators,
    get_masker,
    mm2vox,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


def _uses_cuda(device):
    """Return whether the provided device string targets CUDA."""
    return str(device).startswith("cuda")


class CBMRResult(MetaResult):
    """Meta-analytic result for CBMR with result-centered inference helpers."""

    @property
    def groups(self):
        """Return fitted group names in display order."""
        return tuple(getattr(self.estimator, "groups", ()) or ())

    @property
    def moderators(self):
        """Return fitted moderator names in display order."""
        return tuple(getattr(self.estimator, "moderators", ()) or ())

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

    def available_groups(self):
        """Return the fitted groups that can be used in CBMR contrasts."""
        return list(self.groups)

    def available_moderators(self):
        """Return the fitted moderators that can be used in CBMR contrasts."""
        return list(self.moderators)

    def describe_inference_inputs(self):
        """Summarize the fitted groups and moderators for follow-up inference."""
        return {
            "groups": self.available_groups(),
            "moderators": self.available_moderators(),
        }

    @staticmethod
    def _is_named_pairwise_contrast(contrast):
        return (
            isinstance(contrast, tuple)
            and len(contrast) == 2
            and all(isinstance(part, str) for part in contrast)
        )

    @classmethod
    def _normalize_pairwise_contrasts(cls, contrasts):
        """Convert tuple shorthand like (A, B) into the legacy 'A-B' form."""
        if contrasts is None:
            return None
        if isinstance(contrasts, str) or cls._is_named_pairwise_contrast(contrasts):
            contrasts = [contrasts]

        normalized = []
        for contrast in contrasts:
            if cls._is_named_pairwise_contrast(contrast):
                normalized.append(f"{contrast[0]}-{contrast[1]}")
            else:
                normalized.append(contrast)
        return normalized

    def get_inference(self, device=None):
        """Return a fitted inference engine for advanced CBMR use cases."""
        inference_device = device or getattr(self.estimator, "device", "cpu")
        inference = CBMRInference(device=inference_device)
        inference.fit(self)
        return inference

    def infer(self, group_contrasts=False, moderator_contrasts=False, device=None):
        """Run CBMR inference from a fitted result.

        Parameters
        ----------
        group_contrasts : bool, dict, list, tuple, str, or None, optional
            Group homogeneity or comparison specification. Use ``False`` to skip group inference.
        moderator_contrasts : bool, dict, list, tuple, str, or None, optional
            Moderator effect or comparison specification. Use ``False`` to skip moderator
            inference.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        """
        inference = self.get_inference(device=device)
        return inference.transform(
            t_con_groups=group_contrasts,
            t_con_moderators=moderator_contrasts,
        )

    def test_groups(self, groups=None, device=None):
        """Run one-group spatial homogeneity tests for the requested groups."""
        group_contrasts = self.available_groups() if groups is None else groups
        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=False,
            device=device,
        )

    def compare_groups(self, contrasts, device=None):
        """Run pairwise group-comparison tests using names or (group_a, group_b) tuples."""
        group_contrasts = self._normalize_pairwise_contrasts(contrasts)
        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=False,
            device=device,
        )

    def test_moderators(self, moderators=None, device=None):
        """Test whether the requested moderator effects differ from zero."""
        if not self.moderators:
            raise ValueError("This CBMR result does not include experiment-level moderators.")
        moderator_contrasts = self.available_moderators() if moderators is None else moderators
        return self.infer(
            group_contrasts=False,
            moderator_contrasts=moderator_contrasts,
            device=device,
        )

    def compare_moderators(self, contrasts, device=None):
        """Run pairwise moderator-comparison tests using names or tuples."""
        if not self.moderators:
            raise ValueError("This CBMR result does not include experiment-level moderators.")
        moderator_contrasts = self._normalize_pairwise_contrasts(contrasts)
        return self.infer(
            group_contrasts=False,
            moderator_contrasts=moderator_contrasts,
            device=device,
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
        CBMR allows dataset to be categorized into mutiple groups, according to group categories.
        Default is one-group CBMR.
    moderators : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR can accommodate experiment-level moderators (e.g. sample size, year of publication).
        Default is CBMR without experiment-level moderators.
    model : : :obj:`~nimare.meta.models.GeneralLinearModel`, optional
        Stochastic models in CBMR. The available options are

        ======================= ==================================================================
        Poisson (default)       This is the most efficient and widely used method, but slightly
                                less accurate, because Poisson model is an approximation for
                                low-rate Binomial data, but cannot account over-dispersion in
                                foci counts and may underestimate the standard error.

        NegativeBinomial        This method might be slower and less stable, but slightly more
                                accurate. Negative Binomial (NB) model asserts foci counts follow
                                a NB distribution, and allows for anticipated excess variance
                                relative to Poisson (there's an group-wise overdispersion parameter
                                shared by all experiments and all voxels to index excess variance).

        ClusteredNegativeBinomial This method is also an efficient but less
                    accurate approach. Clustered NB model is
                    "random effect" Poisson model, which asserts
                    that the random effects are latent
                    characteristics of each experiment, and
                    represent a shared effect over the entire brain
                    for a given experiment.
        ======================= =================================================================
    penalty: :obj:`~bool`, optional
    Currently, the only available option is Firth-type penalty, which penalizes
    likelihood function by Jeffrey's invariant prior and guarantees convergent
    estimates.
    spline_spacing: :obj:`~int`, optional
    Spatial structure of foci counts is parameterized by coefficient of cubic B-spline bases
    in CBMR. Spatial smoothness in CBMR is determined by spline spacing, which is shared across
    x,y,z dimension.
    Default is 10 (20mm with 2mm brain atlas template).
    n_iters: :obj:`int`, optional
        Number of iterations limit in optimisation of log-likelihood function.
        Default is 10000.
    lr: :obj:`float`, optional
        Learning rate in optimization of log-likelihood function.
        Default is 1e-2 for Poisson and clustered NB model, and 1e-3 for NB model.
    lr_decay: :obj:`float`, optional
        Multiplicative factor of learning rate decay.
        Default is 0.999.
    tol: :obj:`float`, optional
        Stopping criteria w.r.t difference of log-likelihood function in two consecutive
        iterations.
        Default is 1e-2
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') represents the device on which operations will be allocated
        Default is 'cpu'
    **kwargs
        Keyword arguments. Arguments for the Estimator can be assigned here,
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.maskers.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMR estimators, there is only multiple keys:
        coordinates,
        mask_img (Niftiimage of brain mask),
        id (experiment id),
        ids_by_group (experiment id categorized by groups),
        all_group_moderators (experiment-level moderators categorized by groups if exist),
        coef_spline_bases (spatial matrix of coefficient of cubic B-spline
        bases in x,y,z dimension),
        foci_per_voxel (voxelwise sum of foci count across experiments, categorized by groups),
        foci_per_experiment (experiment-wise sum of foci count across space, categorized by
        groups).

    Notes
    -----
    Available correction methods: :meth:`~nimare.meta.cbmr.CBMRInference`.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}
    _group_column = "_cbmr_group"

    def __init__(
        self,
        group_categories=None,
        moderators=None,
        mask=None,
        spline_spacing=10,
        model=models.PoissonEstimator,
        penalty=False,
        n_iter=2000,
        lr=1,
        lr_decay=0.999,
        tol=1e-9,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.group_categories = group_categories
        self.moderators = moderators

        self.spline_spacing = spline_spacing
        self.model = model(
            penalty=penalty, lr=lr, lr_decay=lr_decay, n_iter=n_iter, tol=tol, device=device
        )
        self.penalty = penalty
        self.n_iter = n_iter
        self.lr = lr
        self.lr_decay = lr_decay
        self.tol = tol
        self.device = device
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device cpu")
            self.device = "cpu"
        self.model.device = self.device

        # Initialize optimisation parameters
        self.iter = 0

    def _generate_description(self):
        """Generate a description of the Estimator instance.

        Returns
        -------
        description : :obj:`str`
            Description of the Estimator instance.
        """
        description = """CBMR is a meta-regression framework that can explicitly model
                    group-wise spatial intensity function, and consider the effect of
                    experiment-level moderators. It consists of two components: (1) a spatial
                    model that makes use of a spline parameterization to induce a smooth
                    response; (2) a generalized linear model (Poisson, Negative Binomial
                    (NB), Clustered NB) to model group-wise spatial intensity function).
                    CBMR is fitted via maximizing the log-likelihood function with L-BFGS
                    algorithm."""
        if self.moderators:
            moderators_str = f"""and accommodate the following experiment-level moderators:
                            {', '.join(self.moderators)}"""
        else:
            moderators_str = ""
        if self.model.penalty:
            penalty_str = " Firth-type penalty is applied to ensure convergence."
        else:
            penalty_str = ""

        if type(self.model).__name__ == "PoissonEstimator":
            model_str = (
                " Here, Poisson model \\citep{eisenberg1966general} is the most basic CBMR model. "
                "It's based on the assumption that foci arise from a realisation of a (continues) "
                "inhomogeneous Poisson process, so that the (discrete) voxel-wise foci counts will"
                " be independently distributed as Poisson random variables, with rate equal to the"
                " integral of (true, unobserved, continous) intensity function over each voxels"
            )
        elif type(self.model).__name__ == "NegativeBinomialEstimator":
            model_str = (
                " Negative Binomial (NB) model \\citep{barndorff1969negative} is a generalized "
                "Poisson model with over-dispersion. "
                "It's a more flexible model, but more difficult to estimate. In practice, foci"
                "counts often display over-dispersion (the variance of response variable"
                "substantially exceeeds the mean), which is not captured by Poisson model."
            )
        elif type(self.model).__name__ == "ClusteredNegativeBinomialEstimator":
            model_str = (
                " Clustered NB model \\citep{geoffroy2001poisson} can also accommodate "
                "over-dispersion in foci counts. "
                "In NB model, the latent random variable introduces indepdentent variation"
                "at each voxel. While in Clustered NB model, we assert the random effects are not "
                "independent voxelwise effects, but rather latent characteristics of each "
                "experiment, and represent a shared effect over the entire brain for a given "
                "experiment."
            )

        model_description = (
            f"CBMR is a meta-regression framework that was performed with NiMARE {__version__}. "
            f"{type(self.model).__name__} model was used to model group-wise spatial intensity "
            f"functions {moderators_str}." + model_str
        )

        optimization_description = (
            "CBMR is fitted via maximizing the log-likelihood function with L-BFGS algorithm, with"
            f" learning rate {self.lr}, learning rate decay {self.lr_decay} and "
            + "tolerance {self.tol}."
            + penalty_str
            + f" The optimization is run on {self.device}."
            f" The input dataset included {self.inputs_['coordinates'].shape[0]} foci from "
            f"{len(self.inputs_['id'])} experiments."
        )

        description = model_description + "\n" + optimization_description
        return description

    def fit(self, dataset, drop_invalid=True):
        """Fit the estimator and return a CBMR-specific result object."""
        dataset = normalize_collection(dataset)
        self._collect_inputs(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        maps, tables, description = self._cache(self._fit, func_memory_level=1)(dataset)
        if not self.generate_description:
            description = ""

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        return CBMRResult(self, mask=masker, maps=maps, tables=tables, description=description)

    def _get_mask_img(self, dataset):
        """Select the masker and mask image for a fit."""
        masker = self.masker or dataset.masker
        if masker is None:
            raise ValueError(
                "A masker is required for coordinate-based meta-analysis. "
                "Provide a `mask` to the Estimator or initialize the Dataset with a `target` "
                "and/or `mask` so `dataset.masker` is defined."
            )
        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        return masker, mask_img

    def _validate_coordinates(self):
        """Validate coordinate space metadata consistently with other CBMA estimators."""
        coord_spaces = self.inputs_["coordinates"]["space"].dropna().unique()
        if coord_spaces.size == 0:
            raise ValueError(
                "Coordinate space information is missing from the Dataset. "
                "Ensure the Dataset coordinates include a valid 'space' column."
            )
        if coord_spaces.size > 1:
            shown = ", ".join(map(str, coord_spaces[:10]))
            suffix = "..." if coord_spaces.size > 10 else ""
            raise ValueError(
                "Mixed coordinate spaces detected in the Dataset (space column contains more than "
                f"one value: {shown}{suffix}). Provide a target space when creating the Dataset "
                "to harmonize coordinates, or convert coordinates to a common space before "
                "running a meta-analysis."
            )

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
            experiment_annotations[self._group_column] = "Default"
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
        foci_per_voxel, foci_per_experiment = self._build_group_foci(
            filtered_coordinates,
            ids_by_group,
            n_mask_voxels,
        )
        return {
            "ids_by_group": ids_by_group,
            "moderators_by_group": moderators_by_group,
            "foci_per_voxel": foci_per_voxel,
            "foci_per_experiment": foci_per_experiment,
        }

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

        if grouped_coordinates.empty:
            grouped_experiment_counts = pd.Series(dtype=np.int64)
        else:
            grouped_experiment_counts = grouped_coordinates.groupby(
                [self._group_column, "id"], sort=False
            ).size()

        foci_per_voxel = {}
        foci_per_experiment = {}
        for group, group_ids in ids_by_group.items():
            group_coordinates = grouped_coordinates.loc[
                grouped_coordinates[self._group_column] == group
            ]
            if group_coordinates.empty:
                group_foci_per_voxel = np.zeros((n_mask_voxels, 1), dtype=np.int32)
                group_experiment_counts = pd.Series(dtype=np.int64)
            else:
                group_foci_per_voxel = np.bincount(
                    group_coordinates["_cbmr_mask_index"].to_numpy(dtype=np.int64, copy=False),
                    minlength=n_mask_voxels,
                ).astype(np.int32, copy=False)
                group_foci_per_voxel = group_foci_per_voxel.reshape((-1, 1))
                group_experiment_counts = grouped_experiment_counts.xs(
                    group, level=self._group_column
                )

            group_foci_per_experiment = group_experiment_counts.reindex(
                group_ids,
                fill_value=0,
            ).to_numpy(dtype=np.int32)
            group_foci_per_experiment = group_foci_per_experiment.reshape((-1, 1))

            foci_per_voxel[group] = group_foci_per_voxel
            foci_per_experiment[group] = group_foci_per_experiment

        return foci_per_voxel, foci_per_experiment

    def _preprocess_input(self, dataset):
        """Mask required input images using either the Dataset's mask or the Estimator's.

        Also, categorize experiment id, voxelwise sum of foci counts across experiments,
        experiment-wise sum of foci counts across space into multiple groups. And summarize
        experiment-level moderators into
        multiple groups (if exist).

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            In this method, the Dataset is used to (1) select the appropriate mask image,
            (2) categorize experiments into multiple groups according to group categories in
            annotations,
            (3) summarize group-wise experiment id, moderators (if exist), foci per voxel, foci
            per experiment,
            (4) extract sample size metadata and use it as one of experiment-level moderators.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            Specifically, (1) a "mask_img" key will be added (Niftiimage of brain mask),
            (2) an 'id' key will be added (id of all experiments in the dataset),
            (3) a 'coef_spline_bases' key will be added (spatial matrix of coefficient of cubic
            B-spline bases in x,y,z dimension),
            (4) an 'ids_by_group' key will be added (experiment id categorized by groups),
            (5) an 'moderators_by_group' key will be added (experiment-level moderators categorized
            by groups) if experiment-level moderators are considered,
            (6) an 'foci_per_voxel' key will be added (voxelwise sum of foci count across
            experiments, categorized by groups),
            (7) an 'foci_per_experiment' key will be added (experiment-wise sum of
            foci count across space, categorized by groups).

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        masker, mask_img = self._get_mask_img(dataset)
        self._validate_coordinates()
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
        experiment-level moderators (if exist), as well as its standard error via inverse of
        Fisher Information matrix;
        (2) Estimate standard error of group-wise log intensity, group-wise intensity via delta
        method;
        (3) For NegativeBinomial or ClusteredNegativeBinomial model, estimate regression
        coefficient of overdispersion.s

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        init_weight_kwargs = {
            "groups": self.groups,
            "moderators": self.moderators,
            "spatial_coef_dim": self.inputs_["coef_spline_bases"].shape[1],
            "moderators_coef_dim": len(self.moderators) if self.moderators else None,
        }
        torch.manual_seed(100)
        if _uses_cuda(self.device):
            torch.cuda.manual_seed_all(100)
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


class CBMRInference(object):
    """Statistical inference on outcomes of CBMR.

    Notes
    -----
    The CBMR API design now centers inference on :class:`CBMRResult`. This class remains the
    lower-level implementation used by those result helpers.

    .. versionadded:: 0.1.0

    (intensity estimation and experiment-level moderator regressors)

    Parameters
    ----------
    result : :obj:`~nimare.cbmr.CBMREstimator`
        Results of optimized regression coefficients of CBMR, as well as their
        standard error in `tables`. Results of estimated spatial intensity function
        (per experiment) in `maps`.
    t_con_groups : :obj:`~bool` or obj:`~list` or obj:`~None`, optional
        Contrast matrix for homogeneity test or group comparison on estimated spatial
        intensity function.
        For boolean inputs, no statistical inference will be conducted for spatial intensity
        if `t_con_groups` is False, and spatial homogeneity test for groupwise intensity
        function will be conducted if `t_con_groups` is True.
        For list inputs, generialized linear hypothesis (GLH) testing will be conducted for
        each element independently. We also allow any element of `t_con_groups` in list type,
        which represents GLH is conducted for all contrasts in this element simultaneously.
        Default is homogeneity test on group-wise estimated intensity function.
    t_con_moderators : :obj:`~bool` or obj:`~list` or obj:`~None`, optional
        Contrast matrix for testing the existence of one or more
        experiment-level moderator effects.
        For boolean inputs, no statistical inference will be conducted for experiment-level
        moderators if `t_con_moderators` is False, and statistical inference on the effect of each
        experiment-level moderator will be conducted if `t_con_moderators` is True.
        For list inputs, generialized linear hypothesis (GLH) testing will be conducted for
        each element independently. We also allow any element of `t_con_moderators` in list type,
        which represents GLH is conducted for all contrasts in this element simultaneously.
        Default is statistical inference on the effect of each experiment-level moderator.
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') represents the device on which operations will be allocated.
        Default is 'cpu'.
    """

    def __init__(self, device="cpu"):
        self.device = device
        # device check
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device 'cpu'")
            self.device = "cpu"
        self.result = None
        self.groups = None
        self.moderators = None
        self._reset_inference_caches()

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
        copied_result.maps = {}
        for map_name, map_ in result.maps.items():
            copied_map = np.array(map_, copy=True)
            if (
                np.issubdtype(copied_map.dtype, np.floating)
                and copied_map.dtype != DEFAULT_FLOAT_DTYPE
            ):
                copied_map = copied_map.astype(DEFAULT_FLOAT_DTYPE)
            copied_result.maps[map_name] = copied_map

        copied_result.tables = {
            table_name: table.copy(deep=True) for table_name, table in result.tables.items()
        }
        copied_result.metadata = dict(result.metadata)
        return copied_result

    def _reset_inference_caches(self):
        """Reset cached inference intermediates for the current fitted result."""
        self._group_spatial_covariance_cache = {}
        self._group_log_intensity_cache = {}
        self._group_null_log_intensity_cache = {}
        self._moderator_covariance = None
        self._moderator_variance = None
        self._moderator_coef_table = None

    def _get_group_log_intensity(self, group):
        """Return cached group log-intensity values."""
        group_log_intensity = self._group_log_intensity_cache.get(group)
        if group_log_intensity is None:
            group_log_intensity = np.log(self.result.maps[f"spatialIntensity_group-{group}"])
            self._group_log_intensity_cache[group] = group_log_intensity
        return group_log_intensity

    def _get_group_null_log_intensity(self, group):
        """Return cached null log-intensity for group homogeneity testing."""
        group_null_log_intensity = self._group_null_log_intensity_cache.get(group)
        if group_null_log_intensity is None:
            group_foci_per_voxel = self.estimator.inputs_["foci_per_voxel"][group]
            group_foci_per_experiment = self.estimator.inputs_["foci_per_experiment"][group]
            n_voxels, n_experiments = (
                group_foci_per_voxel.shape[0],
                group_foci_per_experiment.shape[0],
            )
            group_null_log_intensity = np.log(
                np.sum(group_foci_per_voxel) / (n_voxels * n_experiments)
            )
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
        result : :obj:`~nimare.cbmr.CBMREstimator`
            Results of optimized regression coefficients of CBMR, as well as their
            standard error in `tables`. Results of estimated spatial intensity function
            (per experiment) in `maps`.
        """
        self.result = self._copy_result_for_inference(result)
        self._reset_inference_caches()
        self.estimator = self.result.estimator
        self.estimator.device = self.device
        self.estimator.model.device = self.device
        self.estimator.model.to(self.device)
        self.estimator.model._invalidate_tensor_inputs_cache()
        self.groups = self.result.estimator.groups
        self.moderators = self.result.estimator.moderators
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

    @staticmethod
    def _is_named_pairwise_contrast(contrast):
        return (
            isinstance(contrast, tuple)
            and len(contrast) == 2
            and all(isinstance(part, str) for part in contrast)
        )

    @classmethod
    def _normalize_named_contrasts(cls, contrast_name):
        """Normalize shorthand pairwise contrasts into the legacy string representation."""
        if contrast_name is None:
            return None
        if isinstance(contrast_name, str) or cls._is_named_pairwise_contrast(contrast_name):
            contrast_name = [contrast_name]

        normalized = []
        for contrast in contrast_name:
            if cls._is_named_pairwise_contrast(contrast):
                normalized.append(f"{contrast[0]}-{contrast[1]}")
            else:
                normalized.append(contrast)
        return normalized

    @_check_fit
    def create_contrast(self, contrast_name, source="groups"):
        """Create contrast matrix for generalized hypothesis testing (GLH).

        (1) if `source` is "group", create contrast matrix for GLH on spatial intensity;
        if `contrast_name` begins with 'homo_test_', followed by a valid group name,
        create a contrast matrix for one-group homogeneity test on spatial intensity;
        if `contrast_name` comes in the form of "group1VSgroup2", with valid group names
        "group1" and "group2", create a contrast matrix for group comparison on estimated
        group spatial intensity;
        (2) if `source` is "moderator", create contrast matrix for GLH on
        experiment-level moderators;
        if `contrast_name` begins with 'moderator_', followed by a valid moderator name,
        we create a contrast matrix for testing if the effect of this moderator exists;
        if `contrast_name` comes in the form of "moderator1VSmoderator2", with valid moderator
        names "modeator1" and "moderator2", we create a contrast matrix for testing if the
        effect of these two moderators are different.

        Parameters
        ----------
        contrast_name : :obj:`~string`
            Name of contrast in GLH.
        """
        contrast_name = self._normalize_named_contrasts(contrast_name)
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
    def transform(self, t_con_groups=None, t_con_moderators=None):
        """Conduct generalized linear hypothesis (GLH) testing on CBMR estimates.

        Estimate group-wise spatial regression coefficients and its standard
        error via inverse Fisher Information matrix, estimate standard error of
        group-wise log intensity, group-wise intensity via delta method. For NB
        or clustered model, estimate regression coefficient of overdispersion.
        Similarly, estimate regression coefficient of experiment-level
        moderators (if exist), as well as its standard error via Fisher
        Information matrix. Save these outcomes in `tables`. Also, estimate
        group-wise spatial intensity (per experiment) and save the results in
        `maps`.

        Parameters
        ----------
        t_con_groups : :obj:`~list`, optional
            Contrast matrix for GLH on group-wise spatial intensity estimation.
            Default is None (group-wise homogeneity test for all groups).
        t_con_moderators : :obj:`~list`, optional
            Contrast matrix for GLH on moderator effects.
            Default is None (tests if moderator effects exist for all moderators).
        """
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

    def fit_transform(self, result, t_con_groups=None, t_con_moderators=None):
        """Fit and transform."""
        self.fit(result)
        return self.transform(t_con_groups, t_con_moderators)

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
            spatial intensity or experiment-level moderators.
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

        if isinstance(t_con_regressor, str) or self._is_named_pairwise_contrast(t_con_regressor):
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
                isinstance(con_regressor, str) or self._is_named_pairwise_contrast(con_regressor)
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
                spatial_coef_dim,
                n_brain_voxel,
                is_homogeneity_test,
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
        spatial_coef_dim,
    ):
        """Compute one-contrast Wald statistics for group inference."""
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
        involved_std_log_intensity = np.sqrt(involved_var_log_intensity)
        z_stats_spatial = contrast_log_intensity / involved_std_log_intensity
        if n_con_group_involved == 1:
            p_vals_spatial = scipy.stats.norm.sf(z_stats_spatial)
        else:
            p_vals_spatial = scipy.stats.norm.sf(abs(z_stats_spatial)) * 2
        return z_stats_spatial, p_vals_spatial

    def _compute_group_glh_statistics(
        self,
        simp_con_group,
        involved_groups,
        cov_spatial_coef,
        contrast_log_intensity,
        X,
        spatial_coef_dim,
        n_brain_voxel,
        is_homogeneity_test,
    ):
        """Compute multi-row GLH statistics for group inference."""
        n_con_group_involved = len(involved_groups)
        m = simp_con_group.shape[0]
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
        p_vals_spatial = 1 - scipy.stats.chi2.cdf(chi_sq_spatial, df=m)
        if is_homogeneity_test:
            z_stats_spatial = scipy.stats.norm.isf(p_vals_spatial)
            z_stats_spatial[z_stats_spatial < 0] = 0
        else:
            z_stats_spatial = scipy.stats.norm.isf(p_vals_spatial / 2)
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

    def _chi_square_log_intensity(
        self,
        m,
        n_brain_voxel,
        n_con_group_involved,
        simp_con_group,
        cov_log_intensity,
        contrast_log_intensity,
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

    @staticmethod
    def _evaluate_moderator_contrast(
        con_moderator,
        cov_moderator_coef,
        var_moderator_coef,
        moderator_coef,
    ):
        """Compute statistics for one prepared moderator contrast."""
        m_con_moderator = con_moderator.shape[0]
        contrast_moderator_coef = np.matmul(con_moderator, moderator_coef)
        if m_con_moderator == 1:
            involved_var_moderator_coef = con_moderator**2 @ var_moderator_coef
            involved_std_moderator_coef = np.sqrt(involved_var_moderator_coef)
            z_stats_moderator = contrast_moderator_coef / involved_std_moderator_coef
            p_vals_moderator = scipy.stats.norm.sf(abs(z_stats_moderator)) * 2
            chi_sq_moderator = None
        else:
            contrast_covariance = con_moderator @ cov_moderator_coef @ con_moderator.T
            solved = np.linalg.solve(contrast_covariance, contrast_moderator_coef)
            chi_sq_moderator = contrast_moderator_coef.T @ solved
            p_vals_moderator = 1 - scipy.stats.chi2.cdf(chi_sq_moderator, df=m_con_moderator)
            z_stats_moderator = scipy.stats.norm.isf(p_vals_moderator / 2)

        return {
            "contrast_count": m_con_moderator,
            "chi_square": chi_sq_moderator,
            "p": p_vals_moderator,
            "z": z_stats_moderator,
        }

    def _store_moderator_inference_result(self, con_moderator_count, moderator_stats):
        """Write one computed moderator-inference result into result tables."""
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
        """Conduct Generalized linear hypothesis (GLH) testing for experiment-level moderators.

        GLH testing allows flexible hypothesis testings on regression
        coefficients of experiment-level moderators, including testing for
        the existence of moderator effects and difference in moderator
        effects across multiple moderator effects.
        """
        self._run_moderator_inference()
