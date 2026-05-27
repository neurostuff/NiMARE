"""Machine learning items for masked activation (MA) analysis."""

from __future__ import annotations

import logging
from typing import Any

from nimare.base import NiMAREBase

LGR = logging.getLogger(__name__)

__all__ = ["MAFeatureDataset", "MAFeatureExtractor", "make_map_reducer"]


class MAFeatureDataset(NiMAREBase):
    """Container for aligned map features, descriptors, targets, and provenance.

    Attributes
    ----------
    ids : list of str
        Full Studyset analysis identifiers ("<study_id>-<analysis_id>") per row.
    study_ids : list of str
        Study-level grouping label for each row; used for grouped splitting and leakage control.
    features : array-like or sparse matrix
        Analysis-by-feature matrix combining map features (sparse voxelwise)
        and optional descriptor features. Unreduced voxelwise map features must
        remain in a sparse representation. Reduced map features or
        descriptor-only matrices may be dense.
    feature_names : list of str
        Names for columns in `features`, aligned to column order.
    target : array-like or None
        Optional row-aligned prediction target (y), by default None.
    provenance : dict or None
        Map-generation settings and source Studyset details, including
        `missing_coordinates` and any `dropped_ids`, by default None.
    """

    def __init__(
        self,
        features: Any,
        ids: list[str],
        study_ids: list[str],
        feature_names: list[str] | None = None,
        target: Any | None = None,
        provenance: Any | None = None,
    ) -> None:
        # Infer sizes from features. Prefer .shape for ndarray/sparse, fall back
        # to len() for generic sequences.
        try:
            n_rows = int(features.shape[0])
            n_cols = int(features.shape[1])
        except Exception:
            try:
                n_rows = int(len(features))
                n_cols = int(len(features[0])) if n_rows > 0 else 0
            except Exception:
                raise ValueError("Unable to determine number of rows or columns from features")

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

        self.features = features
        self.ids = list(ids)
        self.study_ids = list(study_ids)
        self.feature_names = feature_names
        self.target = target
        self.provenance = provenance

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

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.to_sklearn is not yet implemented.")

    def _split_by_groups(
        self,
        test_size: float = 0.25,
        random_state: int | None = None,
        cv: Any = None,
    ):
        """Split row-aligned feature data while keeping study groups intact.

        This private helper is reserved for leakage-safe grouped splitting by
        study ID.
        """
        raise NotImplementedError("MAFeatureDataset._split_by_groups is not yet implemented.")

    def split(
        self,
        test_size: float = 0.25,
        random_state: int | None = None,
        cv: Any = None,
    ):
        """Split the dataset into leakage-safe train and test partitions.

        Parameters
        ----------
        test_size : float, default=0.25
            Proportion of grouped data to assign to the test partition.
        random_state : int or None, default=None
            Seed used when the splitter is stochastic.
        cv : object, default=None
            Optional grouped cross-validation splitter.

        Returns
        -------
        (MAFeatureDataset, MAFeatureDataset)
            Tuple of (train_dataset, test_dataset). If no test partition is
            requested, the second element may be ``None``.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.split is not yet implemented.")

    def _apply_map_reducer(self, reducer: Any, fit: bool = False):
        """Apply a reducer to map features while preserving aligned metadata.

        This private helper is reserved for the future reducer workflow that
        keeps ids, study_ids, target, and provenance aligned with the
        transformed feature matrix.
        """
        raise NotImplementedError("MAFeatureDataset._apply_map_reducer is not yet implemented.")

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

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.apply_map_reducer is not yet implemented.")

    def copy(self):
        """Return an independent copy of the dataset.

        Returns
        -------
        MAFeatureDataset
            Independent dataset copy.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.copy is not yet implemented.")


class MAFeatureExtractor(NiMAREBase):
    """Orchestrate conversion from a Studyset to MA feature datasets and sklearn-ready exports.

    This helper converts a NiMARE Studyset into aligned MA feature datasets,
    optionally splits by study group, and can export sklearn-compatible
    bundles with optional map reduction.

    Parameters
    ----------
    kernel_transformer : object
        Existing NiMARE kernel transformer instance or class. No implicit
        scientific default is selected; public examples must pass an explicit
        kernel transformer.
    descriptor_fields : list of dict, optional
        Field selectors from metadata, annotations, or texts, by default None.
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
        descriptor_fields: list[Any] | None = None,
        descriptor_transformers: Any | None = None,
        target_field: Any | None = None,
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

    def _get_studyset_tables(self, studyset: Any):
        """Extract Studyset tables needed to build aligned MA features.

        This private helper is reserved for the future Studyset access path
        that gathers coordinates, metadata, annotations, texts, and IDs.
        """
        raise NotImplementedError(
            "MAFeatureExtractor._get_studyset_tables is not yet implemented."
        )

    def _stack_sparse_features(self, map_features: Any, descriptor_features: Any | None = None):
        """Combine sparse map features with optional descriptor features.

        This private helper is reserved for the future feature assembly path
        that builds sklearn-ready matrices from aligned feature blocks.
        """
        raise NotImplementedError(
            "MAFeatureExtractor._stack_sparse_features is not yet implemented."
        )

    def transform(self, studyset: Any):
        """Validate the Studyset, orchestrate extraction and optional splitting.

        Parameters
        ----------
        studyset : object
            NiMARE Studyset input.

        Returns
        -------
        tuple
            Tuple of ``(train_dataset, test_dataset)``. If ``test_size`` is
            ``None`` or ``0.0``, return ``(full_dataset, None)``.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureExtractor.transform is not yet implemented.")

    def to_sklearn(
        self,
        studyset: Any,
        map_reducer: Any | None = None,
        map_reducer_params: Any | None = None,
    ):
        """Run the full public pipeline convenience wrapper.

        Parameters
        ----------
        studyset : object
            NiMARE Studyset input.
        map_reducer : object, optional
            Optional map-feature reducer, by default None.
        map_reducer_params : object, optional
            Optional reducer parameters, by default None.

        Returns
        -------
        tuple
            Tuple of sklearn-ready exports as ``(train_bunch, test_bunch)``. If
            ``test_size`` is ``None`` or ``0.0``, return ``(full_bunch, None)``.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureExtractor.to_sklearn is not yet implemented.")


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
        This public API is scaffolded only.
    """
    raise NotImplementedError("make_map_reducer is not yet implemented.")
