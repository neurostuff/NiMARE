"""Coordinate-based meta-regression estimator."""

import logging
import re

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.cbmr._helpers import (
    DEFAULT_GROUP_NAME,
    DEFAULT_INCIDENCE_THRESHOLD,
    _uses_cuda,
    _validate_incidence_threshold,
)
from nimare.meta.cbmr._torch import torch
from nimare.meta.cbmr.basis import b_spline_bases
from nimare.meta.cbmr.results import CBMRResult
from nimare.utils import (
    get_masker,
    get_masker_mask_image,
    get_template,
    mm2vox,
    seed_torch,
    validate_coordinate_spaces,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


class _CBMRInputs(Estimator):
    """Shared input preparation for CBMR.

    Turns reported coordinates into the three things a model needs: an analysis mask, a spline
    basis evaluated at its voxels, and an experiment-by-voxel matrix of focus counts. Split from
    the estimator because it is substantial, independent of how the model is specified, and the
    place where a silent misalignment would be least detectable -- the formula binds terms
    against the annotation table, so its row order has to match the foci matrix exactly.

    The analysis mask is narrower than the ROI mask: voxels whose empirical focus incidence is at
    or below ``incidence_threshold`` are dropped, since they carry no information about the
    intensity there and would only widen the basis.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}
    _group_column = "_cbmr_group"

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

    def _build_experiment_group_inputs(self, dataset, filtered_coordinates, n_mask_voxels):
        """Assemble experiment IDs and the foci matrix.

        A single group, because a term-based design expresses grouping through the formula. That
        is what makes ``foci_by_experiment`` one (experiments x voxels) matrix rather than a dict
        of per-group ones, which is the shape the predictor consumes.
        """
        experiment_annotations = self._collect_experiment_annotations(dataset)
        experiment_annotations = self._assign_group_labels(experiment_annotations)
        ids_by_group = self._index_experiments_by_group(experiment_annotations)
        self.groups = list(ids_by_group.keys())

        return {
            "ids_by_group": ids_by_group,
            "foci_by_experiment": self._build_group_foci_matrices(
                filtered_coordinates,
                ids_by_group,
                n_mask_voxels,
            ),
        }

    @staticmethod
    def _build_group_foci_matrices(coordinates, ids_by_group, n_mask_voxels):
        """Return experiment-by-voxel foci count matrices for each group."""
        return _CBMRInputs._build_group_sparse_foci_matrices(
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


class CBMR(_CBMRInputs):
    """Coordinate-based meta-regression specified by a formula.

    .. versionadded:: 0.21.0

    Where :class:`CBMREstimator` takes ``group_categories``, ``moderators`` and a
    ``moderator_effect`` switch, this takes one formula in which each term states its own
    spatial resolution::

        CBMR("~ s(diagnosis:drug_status)")                 # a map per cell
        CBMR("~ s(diagnosis) + sample_size")               # plus a scalar moderator
        CBMR("~ s(diagnosis) + sample_size + s(avg_age)")  # one of each

    ``s()`` crosses a term with the spline basis, making its coefficient a map; without it the
    term gets a single coefficient. That is the whole global-versus-voxelwise distinction, and
    stating it per term removes the need for a separate "mixed" mode. It also reaches designs
    the older interface could not express at all -- ``s(sample_size)`` for a spatially varying
    moderator pooled across groups, ``s(diagnosis) + s(drug_status)`` for additive spatial main
    effects, ``diagnosis:sample_size`` for a group-specific scalar slope.

    Parameters
    ----------
    formula : :obj:`str` or :class:`~nimare.meta.cbmr.terms.Design`
        Model specification. See :mod:`nimare.meta.cbmr.terms` for the syntax, including why
        there is never a scalar intercept and how the spatial baseline is resolved.
    distribution : :obj:`str` or :class:`~nimare.meta.cbmr.distributions.Distribution`, optional
        Observation distribution: ``"poisson"``, ``"negativebinomial"`` or
        ``"clusterednegativebinomial"``. The overdispersion models need several experiments
        sharing each spatial map, so they cannot be combined with a continuously varying spatial
        term; :meth:`~nimare.meta.cbmr.distributions.Distribution.check_design` explains why.
        Default is ``"poisson"``.
    mask, incidence_threshold, spline_spacing, n_iter, lr, tol, device, random_state
        As for :class:`CBMREstimator`.

    Notes
    -----
    The per-term parameter budget is logged at fit time. Each ``s()`` term costs one basis width
    of coefficients per column -- 457 at the default spacing on the 2 mm mask, as much as
    another group's entire baseline map -- which the older single switch hid by promoting every
    moderator at once.
    """

    def __init__(
        self,
        formula,
        distribution="poisson",
        mask=None,
        incidence_threshold=DEFAULT_INCIDENCE_THRESHOLD,
        spline_spacing=10,
        n_iter=1000,
        lr=1.0,
        tol=1e-8,
        device="cpu",
        random_state=None,
        **kwargs,
    ):
        from nimare.meta.cbmr.distributions import resolve_distribution
        from nimare.meta.cbmr.terms import formula_to_design

        self.design = formula_to_design(formula)
        self.distribution = resolve_distribution(distribution)

        self.mask = mask
        self.masker = get_masker(mask) if mask is not None else None
        self.incidence_threshold = _validate_incidence_threshold(incidence_threshold)
        self.spline_spacing = spline_spacing
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.device = device
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("CUDA not found; using device 'cpu'.")
            self.device = "cpu"
        self.random_state = random_state
        # Grouping is expressed by the formula, so input preparation treats the studyset as one
        # undifferentiated group.
        self.group_categories = None
        self.groups = None
        super().__init__(**kwargs)

        self.bound_design = None
        self.predictor = None
        self.cbmr_model = None

    def _make_result(self, dataset, maps=None, tables=None, description=""):
        """Return a result that can test hypotheses over the fitted design."""
        masker = self.masker or dataset.masker
        return CBMRResult(self, mask=masker, maps=maps, tables=tables, description=description)

    def _experiment_annotations(self, dataset):
        """Return experiment annotations ordered to match the foci matrix rows."""
        annotations = self._collect_experiment_annotations(dataset)
        ids = list(self.inputs_["ids_by_group"][DEFAULT_GROUP_NAME])
        ordered = annotations.set_index("id").reindex(ids)
        missing = ordered.index[ordered.isna().all(axis=1)]
        if len(missing):
            raise ValueError(
                f"No annotations found for experiments {list(missing)[:5]}; a formula needs an "
                "annotation row per experiment."
            )
        return ordered.reset_index()

    def _fit(self, dataset):
        """Fit the formula-specified model and summarize it into maps and tables."""
        from nimare.meta.cbmr.model import CBMRModel
        from nimare.meta.cbmr.predictor import CBMRPredictor
        from nimare.meta.cbmr.terms import bind

        seed_torch(self.random_state, self.device)

        annotations = self._experiment_annotations(dataset)
        self.bound_design = bind(self.design, annotations)
        self.predictor = CBMRPredictor(self.bound_design, self.inputs_["coef_spline_bases"])

        n_bases = self.predictor.n_bases
        LGR.info(
            f"CBMR design {self.bound_design.design} over {self.predictor.n_voxels} voxels "
            f"and {self.predictor.patterns.n_experiments} experiments, "
            f"{self.predictor.patterns.n_patterns} distinct spatial map(s):\n"
            + self.bound_design.describe(n_bases)
        )

        foci = self.inputs_["foci_by_experiment"][DEFAULT_GROUP_NAME]
        self.cbmr_model = CBMRModel(self.predictor, self.distribution, device=self.device)
        self.cbmr_model.fit(foci, n_iter=self.n_iter, lr=self.lr, tol=self.tol)

        maps, tables = self._summarize(foci)
        return maps, tables, self._description_text()

    def _summarize(self, foci):
        """Turn the fitted model into result maps and tables.

        Reported per *term*, not per spatial pattern. A design with a continuously varying
        spatial term has as many patterns as experiments, and forty fitted intensity maps are
        not a useful answer; the informative object is the term's coefficient map. So a baseline
        term yields one intensity map per level, and any other spatial term yields its
        coefficient map -- a derivative of log intensity with respect to that column, which is
        what ``voxelwiseModeratorEffect_`` has always meant.
        """
        maps, tables = {}, {}
        bases = self.predictor.bases
        coefficients = self.cbmr_model.fitted_coefficients()
        errors = self.cbmr_model.standard_errors(foci)

        for block in self.bound_design.blocks:
            name = str(block.term)
            values = np.atleast_2d(coefficients[name])
            error_values = np.atleast_2d(errors[name])

            if not block.term.spatial:
                tables[f"moderatorEffect_{_table_safe(name)}"] = pd.DataFrame(
                    {
                        "column": list(block.column_names),
                        "coefficient": values.reshape(-1),
                        "standard_error": error_values.reshape(-1),
                    }
                )
                continue

            log_intensity = values @ bases.T
            for index, column in enumerate(block.column_names):
                label = _label_from_column(column)
                if block.is_baseline:
                    maps[f"spatialIntensity_group-{label}"] = np.exp(log_intensity[index])
                    maps[f"logSpatialIntensity_group-{label}"] = log_intensity[index]
                elif block.term.is_sum_to_zero:
                    # A constrained factor is not a moderator: its coefficients are contrasts
                    # among levels, measuring how a level shifts the baseline map.
                    factor = block.term.expr.replace(":", "-")
                    maps[f"spatialFactorEffect_{factor}-{label}"] = log_intensity[index]
                else:
                    maps[f"voxelwiseModeratorEffect_{label}"] = log_intensity[index]

            tables[f"spatialCoefficient_{_table_safe(name)}"] = pd.DataFrame(
                values, index=list(block.column_names)
            )
            tables[f"spatialCoefficientSE_{_table_safe(name)}"] = pd.DataFrame(
                error_values, index=list(block.column_names)
            )

        overdispersion = self.cbmr_model.overdispersion()
        if overdispersion is not None:
            tables["overdispersion"] = pd.DataFrame({"overdispersion": overdispersion})
        return maps, tables

    def _generate_description(self):
        """Describe the fitted model."""
        return (
            f"A coordinate-based meta-regression with design {self.bound_design.design} and a "
            f"{self.distribution.name} observation model was fitted with NiMARE "
            f"{__version__}, using cubic B-spline bases at spacing {self.spline_spacing}."
        )


def _table_safe(name):
    """Make a rendered term usable as a result-table key."""
    return name.replace(" ", "").replace("(", "-").replace(")", "").replace(":", "-")


def _label_from_column(column):
    """Turn a patsy column name into a map label.

    ``diagnosis[schiz]:drug[yes]`` becomes ``schiz-yes``, so a formula design produces the same
    readable group labels the older ``group_categories`` interface did. Names without a level --
    a continuous covariate, or the intercept -- pass through unchanged.
    """
    parts = []
    for piece in column.split(":"):
        match = re.search(r"\[(?:T\.)?([^\]]+)\]", piece)
        parts.append(match.group(1) if match else piece)
    label = "-".join(parts)
    return DEFAULT_GROUP_NAME if label == "1" else label
