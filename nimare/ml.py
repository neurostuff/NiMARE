"""Machine learning items for masked activation (MA) analysis."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import Bunch

from nimare.base import NiMAREBase

LGR = logging.getLogger(__name__)

__all__ = ["MAFeatureDataset", "MAFeatureExtractor", "make_map_reducer"]


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
            `feature_names`.

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
        )

    def _split_by_groups(
        self,
        test_size: float | int = 0.25,
        random_state: int | None = None,
    ):
        """Split row-aligned feature data while keeping study groups intact.

        This private helper is reserved for leakage-safe grouped splitting by
        study ID.
        """
        if test_size is None or test_size == 0:
            return self.copy(), None

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_idx, test_idx = next(
            splitter.split(self.features, self.target, groups=self.study_ids)
        )
        return self._slice_rows(train_idx), self._slice_rows(test_idx)

    def split(
        self,
        test_size: float | int = 0.25,
        random_state: int | None = None,
    ):
        """Split the dataset into leakage-safe train and test partitions.

        Parameters
        ----------
        test_size : float or int, default=0.25
            Proportion or count of grouped data to assign to the test partition.
        random_state : int or None, default=None
            Seed used when the splitter is stochastic.

        Returns
        -------
        (MAFeatureDataset, MAFeatureDataset or None)
            Tuple of (train_dataset, test_dataset). If no test partition is
            requested, the second element may be ``None``.
        """
        return self._split_by_groups(
            test_size=test_size,
            random_state=random_state,
        )

    def _apply_map_reducer(self, reducer: Any, fit: bool = False):
        """Apply a reducer to map features while preserving aligned metadata.

        This private helper keeps ids, study_ids, target, and provenance aligned
        with the transformed feature matrix. The reducer is applied only to
        private map features. Descriptor columns are appended afterward so they
        are not part of the fitted map-reduction model.
        """
        if fit:
            if hasattr(reducer, "fit_transform"):
                reduced_map_features = reducer.fit_transform(self._map_features)
            else:
                reducer.fit(self._map_features)
                reduced_map_features = reducer.transform(self._map_features)
        else:
            reduced_map_features = reducer.transform(self._map_features)

        if not sparse.issparse(reduced_map_features):
            reduced_map_features = np.asarray(reduced_map_features)
            if reduced_map_features.ndim == 1:
                reduced_map_features = reduced_map_features.reshape(-1, 1)

        n_rows, n_map_features = reduced_map_features.shape
        if n_rows != len(self.ids):
            raise ValueError("Reduced map feature row count must match input rows")

        descriptor_features = copy.deepcopy(self._descriptor_features)
        if descriptor_features is None:
            features = reduced_map_features
        elif sparse.issparse(reduced_map_features) or sparse.issparse(descriptor_features):
            features = sparse.hstack(
                [sparse.csr_matrix(reduced_map_features), sparse.csr_matrix(descriptor_features)],
                format="csr",
            )
        else:
            features = np.hstack(
                [np.asarray(reduced_map_features), np.asarray(descriptor_features)]
            )

        feature_names = None
        if self.feature_names is not None:
            original_map_features = self._map_features.shape[1]
            descriptor_names = list(self.feature_names[original_map_features:])
            feature_names = [f"feature_{idx}" for idx in range(n_map_features)] + descriptor_names

        return MAFeatureDataset(
            features=features,
            ids=self.ids,
            study_ids=self.study_ids,
            feature_names=feature_names,
            target=copy.deepcopy(self.target),
            provenance=copy.deepcopy(self.provenance),
            map_features=reduced_map_features,
            descriptor_features=descriptor_features,
            masker=self._masker,
        )

    def apply_map_reducer(self, reducer: Any, fit: bool = False):
        """Apply a map-feature reducer and return a transformed dataset copy.

        Parameters
        ----------
        reducer : object
            Scikit-learn-compatible transformer or pipeline used to reduce map
            features.
        fit : bool, default=False
            Whether the reducer should be fitted before transforming the map
            features.

        Returns
        -------
        MAFeatureDataset
            New dataset instance with reduced map features and preserved
            metadata (ids, study_ids, feature_names, target, provenance).
        """
        return self._apply_map_reducer(reducer, fit=fit)

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

    This helper converts a NiMARE Studyset into aligned MA feature datasets,
    optionally splits by study group and reduces map features.

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
    test_size : float or int or None, default=None
        Optional train/test split specification; ``None`` means no split.
    random_state : int or None, default=None
        Random seed for reproducible splits, by default None.
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
        test_size: float | int | None = None,
        random_state: int | None = None,
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
        self.test_size = test_size
        self.random_state = random_state
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
        map_reducer: Any | None = None,
        map_reducer_params: dict[str, Any] | None = None,
    ):
        """Validate the Studyset, orchestrate extraction and optional splitting.

        Parameters
        ----------
        studyset : object
            NiMARE Studyset input.
        map_reducer : object, optional
            Optional map-feature reducer, by default None.
        map_reducer_params : dict, optional
            Optional reducer parameters, by default None.

        Returns
        -------
        tuple
            Tuple of ``(train_dataset, test_dataset)``. If ``test_size`` is
            ``None`` or ``0.0``, return ``(full_dataset, None)``.
        """
        ids = studyset.ids
        study_ids = np.asarray([id_.rsplit("-", 1)[0] for id_ in ids], dtype=str)

        kernel_transformer = self.kernel_transformer
        if isinstance(kernel_transformer, type):
            kernel_transformer = kernel_transformer()

        cache_key = (id(studyset), id(self.kernel_transformer))

        if self.cache_maps and cache_key in self._map_cache:
            map_features = self._map_cache[cache_key].copy()
        else:
            map_features = kernel_transformer.transform(studyset, return_type="sparse")
            if map_features.shape[0] != len(ids):
                raise ValueError("Map feature row count must match Studyset ids")

            if self.cache_maps:
                self._map_cache[cache_key] = map_features.copy()
                map_features = map_features.copy()

        descriptor_features = None
        descriptor_names = []
        if self.descriptor_fields:
            columns = []
            for selector in self.descriptor_fields:
                values, field = self._get_selected_values(studyset, selector)
                columns.append(values.astype(float))
                descriptor_names.append(field)

            descriptor_features = np.column_stack(columns)

        target = None
        if self.target_field is not None:
            target, _ = self._get_selected_values(studyset, self.target_field)

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
                "dropped_ids": [],
            },
            map_features=map_features,
            descriptor_features=descriptor_features,
            masker=studyset.masker,
        )

        train_dataset, test_dataset = dataset.split(
            test_size=self.test_size,
            random_state=self.random_state,
        )
        reducer = None
        if map_reducer is not None:
            map_reducer_params = {} if map_reducer_params is None else dict(map_reducer_params)
            if isinstance(map_reducer, str):
                if self.random_state is not None:
                    map_reducer_params.setdefault("random_state", self.random_state)
                reducer = make_map_reducer(map_reducer, **map_reducer_params)
            else:
                reducer = map_reducer
                if map_reducer_params:
                    reducer.set_params(**map_reducer_params)

        if reducer is not None:
            train_dataset = train_dataset.apply_map_reducer(reducer, fit=True)
            if test_dataset is not None:
                test_dataset = test_dataset.apply_map_reducer(reducer, fit=False)

        return train_dataset, test_dataset


def make_map_reducer(method: str, **kwargs: Any):
    """Construct a map-feature reducer.

    Parameters
    ----------
    method : str
        Reduction workflow name.
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
