"""Tests for the nimare.ml module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Bunch

from nimare.ml import MAFeatureDataset, MAFeatureExtractor, make_map_reducer
from nimare.nimads import Studyset
from nimare.utils import get_masker, get_template


def _build_studyset_ml_source():
    """Build the Studyset fixture used by the ML tests.

    This source intentionally includes a masker, valid MNI coordinates,
    numeric metadata, numeric annotations, text fields, unique study IDs, and
    unique full analysis IDs.
    """
    return {
        "id": "studyset_ml_source",
        "name": "Studyset ML source",
        "masker": get_masker(get_template(space="mni152_2mm", mask="brain")),
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
    source = _build_studyset_ml_source()
    return Studyset(source, mask=source["masker"])


def assert_sklearn_bunch_valid(
    bunch,
    expected_feature_rows: int,
    expected_feature_columns: int,
    expected_groups: list[str] | None = None,
    expected_target: list[int] | None = None,
):
    """Assert the shared sklearn-export contract for MA feature bundles."""
    assert isinstance(bunch, Bunch)
    assert bunch.data.shape == (expected_feature_rows, expected_feature_columns)
    assert sparse.issparse(bunch.data)
    assert len(bunch.groups) == expected_feature_rows
    assert len(bunch.target) == expected_feature_rows

    if expected_groups is not None:
        np.testing.assert_array_equal(bunch.groups, expected_groups)

    if expected_target is not None:
        np.testing.assert_array_equal(bunch.target, expected_target)

    estimator = LogisticRegression(max_iter=1000, solver="liblinear")
    estimator.fit(bunch.data, bunch.target)


def test_ma_feature_dataset_initialization():
    """Test that MAFeatureDataset initializes and stores documented attributes."""
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    ids = ["study_a-task", "study_b-task"]
    study_ids = ["study_a", "study_b"]
    feature_names = ["feature_1", "feature_2"]
    target = [0, 1]
    provenance = {"source": "unit-test"}

    ds = MAFeatureDataset(
        features=features,
        ids=ids,
        study_ids=study_ids,
        feature_names=feature_names,
        target=target,
        provenance=provenance,
    )

    assert (ds.features == features).all()
    assert ds.ids == ids
    assert ds.study_ids == study_ids
    assert ds.feature_names == feature_names
    assert ds.target == target
    assert ds.provenance is provenance


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "ids": ["s1"],
                "study_ids": ["study_a", "study_b"],
                "feature_names": ["f1", "f2"],
            },
            "ids length must match number of rows in features",
        ),
        (
            {
                "ids": ["s1", "s2"],
                "study_ids": ["study_a"],
                "feature_names": ["f1", "f2"],
            },
            "study_ids length must match number of rows in features",
        ),
        (
            {
                "ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "feature_names": ["f1", "f2"],
                "target": [1],
            },
            "target length must match number of rows in features",
        ),
        (
            {
                "ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "feature_names": ["f1"],
            },
            "feature_names length must match number of columns in features",
        ),
    ],
)
def test_ma_feature_dataset_initialization_length_mismatches(kwargs, message):
    """Test each validation branch in MAFeatureDataset initialization."""
    base_kwargs = {
        "features": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "ids": ["s1", "s2"],
        "study_ids": ["study_a", "study_b"],
        "feature_names": ["f1", "f2"],
    }

    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MAFeatureDataset(**base_kwargs)


def test_ma_feature_dataset_initialization_features_shape_undetermined():
    """Test the fallback error when features has no usable size information."""

    class UnknownShape:
        def __len__(self):
            raise TypeError("no length")

    with pytest.raises(
        ValueError,
        match="Unable to determine number of rows or columns from features",
    ):
        MAFeatureDataset(
            features=UnknownShape(),
            ids=[],
            study_ids=[],
        )


def test_ma_feature_dataset_methods_raise_not_implemented():
    """Dataset instance methods are scaffolded and should raise NotImplementedError."""
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    ids = ["s1", "s2"]
    study_ids = ["study_a", "study_b"]

    ds = MAFeatureDataset(
        features=features,
        ids=ids,
        study_ids=study_ids,
    )

    with pytest.raises(NotImplementedError):
        ds.to_sklearn()

    with pytest.raises(NotImplementedError):
        ds._split_by_groups()

    with pytest.raises(NotImplementedError):
        ds.split()

    with pytest.raises(NotImplementedError):
        ds._apply_map_reducer(object())

    with pytest.raises(NotImplementedError):
        ds.apply_map_reducer(object())

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
    assert extractor.missing_coordinates == "drop"
    assert extractor.test_size is None
    assert extractor.random_state is None
    assert extractor.cache_maps is True


def test_ma_feature_extractor_methods_raise_not_implemented():
    """Test that MAFeatureExtractor methods currently raise NotImplementedError."""
    extractor = MAFeatureExtractor(kernel_transformer=object())

    with pytest.raises(NotImplementedError):
        extractor._get_studyset_tables(build_studyset())

    with pytest.raises(NotImplementedError):
        extractor._stack_sparse_features(sparse.csr_matrix([[1.0, 0.0]]))

    with pytest.raises(NotImplementedError):
        extractor.transform(build_studyset())

    with pytest.raises(NotImplementedError):
        extractor.to_sklearn(build_studyset())


def test_make_map_reducer_placeholder():
    """Test that make_map_reducer currently raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        make_map_reducer("variance_threshold")
