"""Tests for the nimare.ml module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nimare.ml import MAFeatureDataset, MAFeatureExtractor, make_map_reducer
from nimare.nimads import Studyset


def _build_studyset_ml_source():
    """Create one canonical source dict for Studyset/NIMADS builder.

    The two studies intentionally reuse the same contrast ID (``task``) so that
    short analysis IDs are ambiguous and must be resolved via the full
    ``<study_id>-<contrast_id>`` identifier.
    """
    return {
        "id": "studyset_ml_source",
        "name": "Studyset ML source",
        "studies": [
            {
                "id": "study_alpha",
                "name": "Study alpha",
                "analyses": [
                    {
                        "id": "task",
                        "name": "Alpha task",
                        "metadata": {"sample_sizes": [20]},
                        "annotations": {"alpha_label": 1.0},
                        "texts": {"abstract": "Alpha study abstract."},
                        "points": [
                            {"space": "MNI", "coordinates": [1.0, 3.0, 5.0]},
                            {"space": "MNI", "coordinates": [2.0, 4.0, 6.0]},
                        ],
                        "images": [],
                    }
                ],
            },
            {
                "id": "study_beta",
                "name": "Study beta",
                "analyses": [
                    {
                        "id": "task",
                        "name": "Beta task",
                        "metadata": {"sample_sizes": [30]},
                        "annotations": {"beta_label": 1.0},
                        "texts": {"abstract": "Beta study abstract."},
                        "points": [{"space": "MNI", "coordinates": [-1.0, -3.0, -5.0]}],
                        "images": [],
                    }
                ],
            },
        ],
    }


def build_studyset():
    """Build a fresh Studyset from the studyset ML source dict."""
    return Studyset(_build_studyset_ml_source())


def test_studyset_builder_and_ambiguous_short_ids():
    """The Studyset builder should preserve full IDs and expose ambiguity."""
    studyset = build_studyset()

    assert isinstance(studyset, Studyset)
    assert studyset.ids.tolist() == ["study_alpha-task", "study_beta-task"]
    assert studyset.study_ids.tolist() == ["study_alpha", "study_beta"]
    assert list(studyset.coordinates["id"].unique()) == ["study_alpha-task", "study_beta-task"]
    assert studyset.filter_ids("task").ids.tolist() == ["study_alpha-task", "study_beta-task"]
    assert "abstract" in studyset.texts.columns
    assert "sample_sizes" in studyset.metadata.columns
    assert "alpha_label" in studyset.annotations_df.columns
    assert "beta_label" in studyset.annotations_df.columns


def test_ma_feature_dataset_initialization():
    """Test that MAFeatureDataset initializes and stores documented attributes."""
    map_features = np.array([[1.0, 2.0], [3.0, 4.0]])
    sample_ids = ["s1", "s2"]
    study_ids = ["study_a", "study_b"]
    sample_metadata = pd.DataFrame({"sample_id": sample_ids, "study_id": study_ids})
    masker = object()

    ds = MAFeatureDataset(
        map_features=map_features,
        sample_ids=sample_ids,
        study_ids=study_ids,
        sample_metadata=sample_metadata,
        masker=masker,
    )

    assert (ds.map_features == map_features).all()
    assert ds.sample_ids == sample_ids
    assert ds.study_ids == study_ids
    assert ds.sample_metadata is sample_metadata
    assert ds.masker is masker


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "sample_ids": ["s1"],
                "study_ids": ["study_a", "study_b"],
                "sample_metadata": [{}, {}],
            },
            "sample_ids length must match number of rows in map_features",
        ),
        (
            {
                "sample_ids": ["s1", "s2"],
                "study_ids": ["study_a"],
                "sample_metadata": [{}, {}],
            },
            "study_ids length must match number of rows in map_features",
        ),
        (
            {
                "sample_ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "sample_metadata": [{}],
            },
            "sample_metadata length must match number of rows in map_features",
        ),
        (
            {
                "sample_ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "sample_metadata": [{}, {}],
                "descriptor_features": [[0.1], [0.2], [0.3]],
            },
            "descriptor_features length must match number of rows in map_features",
        ),
        (
            {
                "sample_ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "sample_metadata": [{}, {}],
                "target": [1],
            },
            "target length must match number of rows in map_features",
        ),
        (
            {
                "sample_ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "sample_metadata": [{}, {}],
                "feature_names": ["f1"],
            },
            "feature_names length must match number of columns in map_features",
        ),
    ],
)
def test_ma_feature_dataset_initialization_length_mismatches(kwargs, message):
    """Test each validation branch in MAFeatureDataset initialization."""
    base_kwargs = {
        "map_features": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "sample_ids": ["s1", "s2"],
        "study_ids": ["study_a", "study_b"],
        "sample_metadata": [{}, {}],
        "masker": object(),
    }

    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MAFeatureDataset(**base_kwargs)


def test_ma_feature_dataset_initialization_map_features_shape_undetermined():
    """Test the fallback error when map_features has no usable size information."""

    class UnknownShape:
        def __len__(self):
            raise TypeError("no length")

    with pytest.raises(
        ValueError,
        match="Unable to determine number of samples or features from map_features",
    ):
        MAFeatureDataset(
            map_features=UnknownShape(),
            sample_ids=[],
            study_ids=[],
            sample_metadata=[],
            masker=object(),
        )


def test_ma_feature_dataset_methods_raise_not_implemented():
    """Dataset instance methods are scaffolded and should raise NotImplementedError."""
    map_features = np.array([[1.0, 2.0], [3.0, 4.0]])
    sample_ids = ["s1", "s2"]
    study_ids = ["study_a", "study_b"]
    sample_metadata = [{}, {}]
    masker = object()

    ds = MAFeatureDataset(
        map_features=map_features,
        sample_ids=sample_ids,
        study_ids=study_ids,
        sample_metadata=sample_metadata,
        masker=masker,
    )

    with pytest.raises(NotImplementedError):
        ds.to_sklearn()

    with pytest.raises(NotImplementedError):
        ds.split()

    with pytest.raises(NotImplementedError):
        ds.apply_map_reducer(object())

    with pytest.raises(NotImplementedError):
        ds.get_feature_names()

    with pytest.raises(NotImplementedError):
        ds.copy()


def test_ma_feature_extractor_initialization():
    """Test that MAFeatureExtractor initializes and stores documented parameters."""
    kernel_transformer = object()
    extractor = MAFeatureExtractor(
        kernel_transformer=kernel_transformer,
        descriptor_fields=[{"source": "metadata", "field": "sample_sizes"}],
        target_field={"source": "annotations", "field": "alpha_label"},
    )

    assert extractor.kernel_transformer is kernel_transformer
    assert extractor.descriptor_fields == [{"source": "metadata", "field": "sample_sizes"}]
    assert extractor.target_field == {"source": "annotations", "field": "alpha_label"}
    assert extractor.missing == "raise"


def test_ma_feature_extractor_methods_raise_not_implemented():
    """Test that MAFeatureExtractor methods currently raise NotImplementedError."""
    extractor = MAFeatureExtractor(kernel_transformer=object())

    with pytest.raises(NotImplementedError):
        extractor.fit(build_studyset())

    with pytest.raises(NotImplementedError):
        extractor.transform(build_studyset())

    with pytest.raises(NotImplementedError):
        extractor.fit_transform(build_studyset())


def test_make_map_reducer_placeholder():
    """Test that make_map_reducer currently raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        make_map_reducer("variance_threshold")
