"""Machine learning items for masked activation (MA) analysis."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import hash as joblib_hash
from nilearn.masking import unmask
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from nimare.base import NiMAREBase

LGR = logging.getLogger(__name__)

__all__ = ["MAFeatureExtractor", "make_map_reducer"]


class MAFeatureDataset(NiMAREBase):
    """Container for aligned map features, descriptors, targets, and provenance.

    Attributes
    ----------
    ids : sequence of str
        Full Studyset analysis identifiers ("<study_id>-<analysis_id>") per row.
    study_ids : sequence of str
        Study-level grouping label for each row; used for grouped splitting and leakage control.
    features : array-like or sparse matrix
        Analysis-by-feature matrix combining map features (sparse voxelwise)
        and optional descriptor features. Unreduced voxelwise map features must
        remain in a sparse representation. Reduced map features or
        descriptor-only matrices may be dense.
    feature_names : sequence of str or None
        Names for columns in `features`, aligned to column order.
    target : array-like or None
        Optional row-aligned prediction target (y), by default None.
    provenance : dict or None
        Map-generation settings and source Studyset details, including
        `missing_coordinates` and any `dropped_ids`, by default None.

    Notes
    -----
    The private `_map_features`, `_descriptor_features`, and `_masker`
    attributes keep map data, descriptor data, and voxel-ordering metadata
    available for later splitting and reduction steps.
    """

    def __init__(
        self,
        features: Any,
        ids: Sequence[str],
        study_ids: Sequence[str],
        feature_names: Sequence[str] | None = None,
        target: Any | None = None,
        provenance: dict[str, Any] | None = None,
        map_features: Any | None = None,
        descriptor_features: Any | None = None,
        masker: Any | None = None,
    ) -> None:
        n_rows, n_cols = features.shape

        if len(ids) != n_rows:
            raise ValueError("ids length must match number of rows in features")

        if len(study_ids) != n_rows:
            raise ValueError("study_ids length must match number of rows in features")

        if target is not None:
            if len(target) != n_rows:
                raise ValueError("target length must match number of rows in features")

        if feature_names is not None:
            if len(feature_names) != n_cols:
                raise ValueError("feature_names length must match number of columns in features")

        if map_features is None:
            map_features = features

        n_map_rows = map_features.shape[0]
        if n_map_rows != n_rows:
            raise ValueError("map_features row count must match number of rows in features")

        if descriptor_features is not None:
            n_descriptor_rows = descriptor_features.shape[0]
            if n_descriptor_rows != n_rows:
                raise ValueError(
                    "descriptor_features row count must match number of rows in features"
                )

        self.features = features
        self.ids = ids
        self.study_ids = study_ids
        self.feature_names = None if feature_names is None else list(feature_names)
        self.target = target
        self.provenance = provenance
        self._map_features = map_features
        self._descriptor_features = descriptor_features
        self._masker = masker

    def __repr__(self):
        """Show a concise dataset representation."""
        n_rows, n_features = self.features.shape
        return f"{self.__class__.__name__}(n_rows={n_rows}, n_features={n_features})"

    @property
    def map_columns(self):
        """slice: Columns containing map features."""
        return slice(0, self._map_features.shape[1])

    @property
    def descriptor_columns(self):
        """slice: Columns containing descriptor features."""
        return slice(self._map_features.shape[1], self.features.shape[1])

    @staticmethod
    def _slice_array_like(value: Any, row_indices: np.ndarray):
        """Slice rows from matrix-like or sequence-like data."""
        if value is None:
            return None

        if sparse.issparse(value):
            return value[row_indices].copy()

        if hasattr(value, "iloc"):
            return value.iloc[row_indices].copy()

        try:
            return value[row_indices].copy()
        except Exception:
            return [value[int(idx)] for idx in row_indices]

    def _slice_rows(self, row_indices: Any):
        """Return a row-aligned dataset slice."""
        n_rows = len(self.ids)
        resolved_indices = np.arange(n_rows)[row_indices]
        resolved_indices = np.atleast_1d(resolved_indices)

        return MAFeatureDataset(
            features=self._slice_array_like(self.features, resolved_indices),
            ids=self._slice_array_like(self.ids, resolved_indices),
            study_ids=self._slice_array_like(self.study_ids, resolved_indices),
            feature_names=None if self.feature_names is None else list(self.feature_names),
            target=self._slice_array_like(self.target, resolved_indices),
            provenance=copy.deepcopy(self.provenance),
            map_features=self._slice_array_like(self._map_features, resolved_indices),
            descriptor_features=self._slice_array_like(
                self._descriptor_features,
                resolved_indices,
            ),
            masker=self._masker,
        )

    def to_sklearn(self):
        """Export the dataset as a scikit-learn-compatible bundle.

        Returns
        -------
        sklearn.utils.Bunch-like
            Dataset bundle with attributes `data` (same as `features`),
            `target` (or `None`), `groups` (same as `study_ids`), and
            `ids`, `feature_names`, `provenance`, `map_columns`, and
            `descriptor_columns`.

        Notes
        -----
        Implementations must preserve sparsity for unreduced voxelwise
        features; reduced representations may be dense.
        """
        return Bunch(
            data=self.features,
            target=self.target,
            groups=self.study_ids,
            ids=self.ids,
            feature_names=self.feature_names,
            provenance=self.provenance,
            map_columns=self.map_columns,
            descriptor_columns=self.descriptor_columns,
        )

    def make_preprocessor(self, method: str = "truncated_svd", **kwargs: Any):
        """Build an unfitted preprocessor for map and descriptor columns.

        Parameters
        ----------
        method : str, default="truncated_svd"
            Map-feature reduction workflow name.
        **kwargs : dict
            Additional reducer-specific keyword arguments.

        Returns
        -------
        sklearn.compose.ColumnTransformer
            Unfitted transformer that reduces map columns and passes descriptor
            columns through unchanged.
        """
        return ColumnTransformer(
            [
                (
                    "maps",
                    make_map_reducer(method, masker=self._masker, **kwargs),
                    self.map_columns,
                ),
                ("descriptors", "passthrough", self.descriptor_columns),
            ]
        )

    def copy(self):
        """Return an independent copy of the dataset.

        Returns
        -------
        MAFeatureDataset
            Independent dataset copy.
        """
        return self._slice_rows(slice(None))


class MAFeatureExtractor(NiMAREBase):
    """Orchestrate conversion from a Studyset to scikit-learn feature data.

    This helper converts a NiMARE Studyset into an aligned scikit-learn Bunch.

    Parameters
    ----------
    kernel_transformer : object
        Existing NiMARE kernel transformer instance or class. No implicit
        scientific default is selected; public examples must pass an explicit
        kernel transformer.
    descriptor_fields : list of dict, optional
        Field selectors from metadata or annotations, by default None.
    descriptor_transformers : dict, optional
        Optional mapping from descriptor field selectors to explicit
        transformers or vectorizers for non-numeric descriptor fields, by
        default None. Non-None values are not yet implemented.
    target_field : dict, optional
        Optional field selector for y, by default None.
    target_transformer : object, optional
        Optional transformer or label extractor for free-text or multi-label
        targets, by default None. Non-None values are not yet implemented.
    missing_coordinates : {'include', 'drop'}, default='drop'
        Whether analyses without coordinates are retained as all-zero sparse
        rows or removed before row construction.
    cache_maps : bool, default=True
        Whether to cache generated MA map features across repeated calls.
    memory : object, optional
        Reserved for future joblib caching support, by default None. Non-None
        values are not yet implemented.
    memory_level : int, default=1
        Reserved for future joblib caching support. Values other than 1 are
        not yet implemented.
    """

    def __init__(
        self,
        kernel_transformer: Any,
        descriptor_fields: list[dict[str, str]] | None = None,
        descriptor_transformers: Any | None = None,
        target_field: dict[str, str] | None = None,
        target_transformer: Any | None = None,
        missing_coordinates: str = "drop",
        cache_maps: bool = True,
        memory: Any | None = None,
        memory_level: int = 1,
    ):
        if descriptor_transformers is not None:
            raise NotImplementedError("descriptor_transformers is not yet implemented.")
        if target_transformer is not None:
            raise NotImplementedError("target_transformer is not yet implemented.")
        if memory is not None:
            raise NotImplementedError("memory is not yet implemented.")
        if memory_level != 1:
            raise NotImplementedError("memory_level is not yet implemented.")

        self.kernel_transformer = kernel_transformer
        self.descriptor_fields = descriptor_fields
        self.descriptor_transformers = descriptor_transformers
        self.target_field = target_field
        self.target_transformer = target_transformer
        self.missing_coordinates = missing_coordinates
        self.cache_maps = cache_maps
        self.memory = memory
        self.memory_level = memory_level
        self._map_cache = {}

    @staticmethod
    def _get_selected_values(studyset: Any, selector: dict[str, str]):
        """Return selected Studyset values aligned to analysis IDs."""
        if not isinstance(selector, dict):
            raise TypeError("Field selectors must be dictionaries.")

        source = selector.get("source")
        field = selector.get("field")

        if not source or not field:
            raise ValueError("Field selectors must define 'source' and 'field'.")

        source_to_table = {
            "metadata": studyset.metadata,
            "annotations": studyset.annotations_df,
            "annotations_df": studyset.annotations_df,
        }
        table = source_to_table.get(source)
        if table is None:
            raise ValueError(f"Unsupported field selector source: {source}")

        if field not in table.columns:
            raise ValueError(f"Field '{field}' not found in Studyset {source}.")

        return table[field].to_numpy(dtype=object), field

    def transform(
        self,
        studyset: Any,
        map_reducer: str | None = None,
        map_reducer_params: dict[str, Any] | None = None,
    ):
        """Transform a Studyset into a scikit-learn-compatible dataset.

        Parameters
        ----------
        studyset : object
            NiMARE Studyset input.
        map_reducer : str, optional
            Map-feature reduction workflow name used to build an unfitted
            preprocessor.
        map_reducer_params : dict, optional
            Additional map-reducer parameters.

        Returns
        -------
        sklearn.utils.Bunch
            Dataset containing aligned feature data, targets, study groups,
            analysis IDs, provenance, and an optional unfitted preprocessor.
        """
        if self.missing_coordinates not in ("drop", "include"):
            raise ValueError("missing_coordinates must be 'drop' or 'include'")

        ids = studyset.ids
        coordinate_ids = np.unique(studyset.coordinates["id"].to_numpy())
        has_coordinates = np.isin(ids, coordinate_ids)

        if self.missing_coordinates == "drop":
            retained_rows = has_coordinates
            dropped_ids = ids[~has_coordinates].tolist()
        else:
            retained_rows = np.ones(len(ids), dtype=bool)
            dropped_ids = []

        ids = ids[retained_rows]
        study_ids = np.asarray([id_.rsplit("-", 1)[0] for id_ in ids], dtype=str)

        kernel_transformer = self.kernel_transformer
        if isinstance(kernel_transformer, type):
            kernel_transformer = kernel_transformer()

        cache_key = None
        if self.cache_maps:
            cache_key = joblib_hash(
                (
                    studyset.id,
                    studyset.coordinates,
                    studyset.metadata,
                    studyset.masker.mask_img,
                    type(kernel_transformer).__module__,
                    type(kernel_transformer).__qualname__,
                    kernel_transformer.get_params(),
                )
            )

        if self.cache_maps and cache_key in self._map_cache:
            map_features = self._map_cache[cache_key].copy()
        else:
            map_features = kernel_transformer.transform(studyset, return_type="sparse")
            if self.cache_maps:
                self._map_cache[cache_key] = map_features.copy()
                map_features = map_features.copy()

        if self.missing_coordinates == "include" and not np.all(has_coordinates):
            zero_row = sparse.csr_matrix(
                (1, map_features.shape[1]),
                dtype=map_features.dtype,
            )
            map_rows = iter(map_features)
            map_features = sparse.vstack(
                [next(map_rows) if present else zero_row for present in has_coordinates],
                format="csr",
            )

        descriptor_features = None
        descriptor_names = []
        if self.descriptor_fields:
            columns = []
            for selector in self.descriptor_fields:
                values, field = self._get_selected_values(studyset, selector)
                columns.append(values[retained_rows].astype(float))
                descriptor_names.append(field)

            descriptor_features = np.column_stack(columns)

        target = None
        if self.target_field is not None:
            target, _ = self._get_selected_values(studyset, self.target_field)
            target = target[retained_rows]

        if descriptor_features is None:
            features = map_features
        else:
            features = sparse.hstack(
                [map_features, sparse.csr_matrix(descriptor_features)],
                format="csr",
            )

        dataset = MAFeatureDataset(
            features=features,
            ids=ids,
            study_ids=study_ids,
            feature_names=[f"feature_{idx}" for idx in range(map_features.shape[1])]
            + descriptor_names,
            target=target,
            provenance={
                "source_studyset_id": getattr(studyset, "id", None),
                "source_studyset_name": getattr(studyset, "name", None),
                "missing_coordinates": self.missing_coordinates,
                "dropped_ids": dropped_ids,
            },
            map_features=map_features,
            descriptor_features=descriptor_features,
            masker=studyset.masker,
        )

        bunch = dataset.to_sklearn()
        bunch.preprocessor = None
        if map_reducer is not None:
            map_reducer_params = {} if map_reducer_params is None else dict(map_reducer_params)
            bunch.preprocessor = dataset.make_preprocessor(
                map_reducer,
                **map_reducer_params,
            )

        return bunch


class _NilearnMaskerReducer(TransformerMixin, BaseEstimator):
    """Reduce masked voxel features with a volumetric Nilearn masker."""

    def __init__(self, source_mask_img, atlas_masker, batch_size=10):
        self.source_mask_img = source_mask_img
        self.atlas_masker = atlas_masker
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Fit a cloned atlas masker in the source-mask space."""
        atlas_masker = clone(self.atlas_masker)
        atlas_masker.set_params(mask_img=self.source_mask_img)
        self.atlas_masker_ = atlas_masker.fit(self.source_mask_img)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Convert masked features to images and extract atlas features."""
        check_is_fitted(self, ["atlas_masker_"])
        reduced_batches = []
        for start in range(0, X.shape[0], self.batch_size):
            batch = X[start : start + self.batch_size]
            if sparse.issparse(batch):
                batch = batch.toarray()

            images = unmask(batch, self.source_mask_img)
            reduced_batches.append(self.atlas_masker_.transform(images))

        return np.vstack(reduced_batches)


def make_map_reducer(method: str, masker: Any | None = None, **kwargs: Any):
    """Construct a map-feature reducer.

    Parameters
    ----------
    method : {"atlas_aggregation", "truncated_svd"}
        Reduction workflow name.
    masker : object, optional
        Masker defining the map-feature voxel ordering, by default None.
    **kwargs : dict
        Additional reducer-specific keyword arguments.

    Returns
    -------
    object
        Scikit-learn-compatible transformer or pipeline.

    Raises
    ------
    ValueError
        Raised when atlas aggregation is requested without a source masker.
    NotImplementedError
        Raised for unsupported reducer methods.
    """
    if method == "truncated_svd":
        return TruncatedSVD(**kwargs)

    if method == "atlas_aggregation":
        if masker is None:
            raise ValueError("A source masker is required for atlas aggregation.")

        return _NilearnMaskerReducer(source_mask_img=masker.mask_img, **kwargs)

    raise NotImplementedError(f"Map reducer '{method}' is not yet implemented.")
