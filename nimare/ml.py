"""Machine learning items for masked activation (MA) analysis."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import hash as joblib_hash
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils import Bunch

from nimare.base import NiMAREBase

LGR = logging.getLogger(__name__)

__all__ = ["MAFeatureDataset", "MAFeatureExtractor", "make_map_reducer"]


class _GroupBoundCV:
    """Bind study groups to a scikit-learn cross-validator."""

    def __init__(self, inner: Any, groups: Any):
        self._inner = inner
        self._groups = groups

    def split(self, X: Any, y: Any | None = None, groups: Any | None = None):
        """Generate splits using the bound study groups."""
        return self._inner.split(X, y, groups=self._groups)

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ):
        """Return the number of splits from the wrapped cross-validator."""
        return self._inner.get_n_splits(X, y, self._groups)


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
            `feature_names`, `map_columns`, and `descriptor_columns`.

        Notes
        -----
        Implementations must preserve sparsity for unreduced voxelwise
        features; reduced representations may be dense.
        """
        return Bunch(
            data=self.features,
            target=self.target,
            groups=self.study_ids,
            feature_names=self.feature_names,
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

    def make_cv(
        self,
        n_splits: int = 5,
        test_size: float | int | None = None,
        random_state: int | None = None,
    ):
        """Build a cross-validator bound to this dataset's study groups.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds or shuffled splits.
        test_size : float or int, optional
            Test-group proportion or count. If None, use GroupKFold.
        random_state : int, optional
            Random seed used by GroupShuffleSplit.

        Returns
        -------
        _GroupBoundCV
            Cross-validator that always splits using this dataset's study IDs.
        """
        if test_size is None:
            if n_splits > len(np.unique(self.study_ids)):
                raise ValueError("n_splits cannot exceed the number of unique study IDs")
            inner = GroupKFold(n_splits=n_splits)
        else:
            inner = GroupShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                random_state=random_state,
            )

        return _GroupBoundCV(inner, self.study_ids)

    def copy(self):
        """Return an independent copy of the dataset.

        Returns
        -------
        MAFeatureDataset
            Independent dataset copy.
        """
        return self._slice_rows(slice(None))


class MAFeatureExtractor(NiMAREBase):
    """Orchestrate conversion from a Studyset to MA feature datasets.

    This helper converts a NiMARE Studyset into an aligned MA feature dataset.

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
        default None.
    target_field : dict, optional
        Optional field selector for y, by default None.
    target_transformer : object, optional
        Optional transformer or label extractor for free-text or multi-label
        targets, by default None.
    missing_coordinates : {'include', 'drop'}, default='drop'
        Whether analyses without coordinates are retained as all-zero sparse
        rows or removed before row construction.
    cache_maps : bool, default=True
        Whether to cache generated MA map features across repeated calls.
    memory : object, optional
        joblib Memory-like object for caching, by default None.
    memory_level : int, default=1
        Caching verbosity, by default 1.
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

    def transform(self, studyset: Any):
        """Transform a Studyset into an MA feature dataset.

        Parameters
        ----------
        studyset : object
            NiMARE Studyset input.

        Returns
        -------
        MAFeatureDataset
            Dataset containing all extracted analysis rows.
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

        return dataset


def make_map_reducer(method: str, masker: Any | None = None, **kwargs: Any):
    """Construct a map-feature reducer.

    Parameters
    ----------
    method : str
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
    NotImplementedError
        Raised for reducer methods outside the current truncated-SVD path.
    """
    if method == "truncated_svd":
        return TruncatedSVD(**kwargs)

    raise NotImplementedError(f"Map reducer '{method}' is not yet implemented.")
