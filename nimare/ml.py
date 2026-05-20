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
    map_features : array-like or sparse matrix
        Sample-by-map-feature matrix (n_samples, n_features).
        Sparse matrices remain sparse unless explicitly densified.
    sample_ids : list of str
        Identifiers for each sample.
    study_ids : list of str
        Study-group labels used for grouped splitting.
    sample_metadata : DataFrame-like
        Tabular provenance containing at least sample ID, study ID,
        analysis ID when available, and exclusion status where relevant.
    masker : object
        Masker (e.g., nilearn) defining voxel feature order.
    descriptor_features : array-like, optional
        Sample-aligned descriptor values, by default None.
    target : array-like, optional
        Sample-aligned prediction target (numeric or categorical), by default None.
    feature_names : list of str, optional
        Feature names aligned to map_features columns, by default None.
    exclusion_report : dict, optional
        Excluded analyses and reasons, by default None.
    provenance : dict, optional
        Map-generation settings and collection details, by default None.
    """

    def __init__(
        self,
        map_features: Any,
        sample_ids: list[str],
        study_ids: list[str],
        sample_metadata: Any,
        masker: Any,
        descriptor_features: Any | None = None,
        target: Any | None = None,
        feature_names: list[str] | None = None,
        exclusion_report: Any | None = None,
        provenance: Any | None = None,
    ) -> None:
        # Infer sizes from map_features. Prefer .shape for ndarray/sparse, fall back
        # to len() for generic sequences. Use int(...) to coerce numpy scalars to
        # native ints when present.
        try:
            n_samples = int(map_features.shape[0])
            n_features = int(map_features.shape[1])
        except Exception:
            try:
                # len(map_features) works for lists and other sized containers
                n_samples = int(len(map_features))
                # infer number of features from first row when possible
                n_features = int(len(map_features[0])) if n_samples > 0 else 0
            except Exception:
                raise ValueError(
                    "Unable to determine number of samples or features from map_features"
                )

        if len(sample_ids) != n_samples:
            raise ValueError("sample_ids length must match number of rows in map_features")

        if len(study_ids) != n_samples:
            raise ValueError("study_ids length must match number of rows in map_features")

        if len(sample_metadata) != n_samples:
            raise ValueError("sample_metadata length must match number of rows in map_features")

        if descriptor_features is not None:
            if len(descriptor_features) != n_samples:
                raise ValueError(
                    "descriptor_features length must match number of rows in map_features"
                )

        if target is not None:
            if len(target) != n_samples:
                raise ValueError("target length must match number of rows in map_features")

        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    "feature_names length must match number of columns in map_features"
                )

        self.map_features = map_features
        self.sample_ids = list(sample_ids)
        self.study_ids = list(study_ids)
        self.sample_metadata = sample_metadata
        self.masker = masker
        self.descriptor_features = descriptor_features
        self.target = target
        self.feature_names = feature_names
        self.exclusion_report = exclusion_report
        self.provenance = provenance

    def to_sklearn(
        self,
        include_descriptors: bool = True,
        include_target: bool = True,
        dense: bool = False,
    ):
        """Export the dataset as a scikit-learn-compatible bundle.

        Parameters
        ----------
        include_descriptors : bool, default=True
            Whether descriptor features should be included in the exported
            data matrix.
        include_target : bool, default=True
            Whether the target vector should be included in the export.
        dense : bool, default=False
            Whether to densify sparse map features during export.

        Returns
        -------
        sklearn.utils.Bunch-like
            Dataset bundle with aligned data, target, groups, metadata, and
            feature names.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.to_sklearn is not yet implemented.")

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
        tuple of MAFeatureDataset
            Train and test dataset slices.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.split is not yet implemented.")

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
            Dataset copy with reduced map features.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.apply_map_reducer is not yet implemented.")

    def get_feature_names(self):
        """Return exported feature names in column order.

        Returns
        -------
        list of str
            Feature names aligned to the exported data matrix.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureDataset.get_feature_names is not yet implemented.")

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
    """Extract aligned map features and optional descriptors/targets from collections.

    Converts NiMARE Studyset into scikit-learn-compatible :class:`MAFeatureDataset`.
    Handles kernel transformation, field resolution, and sample-group alignment.

    Parameters
    ----------
    kernel_transformer : object
        NiMARE kernel transformer (instance or class). Required; no implicit default.
    descriptor_fields : list of dict, optional
        Field selectors (source, field, kind) from metadata, annotations,
        or texts, by default None.
    descriptor_transformers : dict, optional
        Transformers/vectorizers for non-numeric descriptor fields, by default None.
        Without these, non-numeric fields are rejected.
    target_field : dict, optional
        Field selector for target variable (y), by default None.
    target_transformer : object, optional
        Transformer for free-text or multi-label targets, by default None.
    missing : {'raise', 'drop'}, default='raise'
        Strategy for missing values and field resolution failures,
        by default 'raise'.
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
        missing: str = "raise",
        memory: Any | None = None,
        memory_level: int = 1,
    ):
        self.kernel_transformer = kernel_transformer
        self.descriptor_fields = descriptor_fields
        self.descriptor_transformers = descriptor_transformers
        self.target_field = target_field
        self.target_transformer = target_transformer
        self.missing = missing
        self.memory = memory
        self.memory_level = memory_level

    def fit(self, collection: Any):
        """Validate a collection and record the feature schema.

        Parameters
        ----------
        collection : object
            NiMARE Studyset input.

        Returns
        -------
        MAFeatureExtractor
            Fitted extractor instance.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureExtractor.fit is not yet implemented.")

    def transform(self, collection: Any):
        """Transform a collection into an :class:`MAFeatureDataset`.

        Parameters
        ----------
        collection : object
            NiMARE Studyset input.

        Returns
        -------
        MAFeatureDataset
            Feature dataset created from the fitted schema.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureExtractor.transform is not yet implemented.")

    def fit_transform(self, collection: Any):
        """Fit the extractor and transform the collection in one call.

        Parameters
        ----------
        collection : object
            NiMARE Studyset input.

        Returns
        -------
        MAFeatureDataset
            Feature dataset created from the fitted schema.

        Raises
        ------
        NotImplementedError
            This public API is scaffolded only.
        """
        raise NotImplementedError("MAFeatureExtractor.fit_transform is not yet implemented.")


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
