"""Tests for the nimare.ml module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch

from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAFeatureDataset, MAFeatureExtractor, make_map_reducer
from nimare.nimads import Studyset
from nimare.utils import get_masker, get_template

RANDOM_SEED = 13


@pytest.fixture(scope="session")
def ml_studyset():
    """Build the shared Studyset used by the ML tests.

    The Studyset includes enough single-analysis studies for grouped train/test
    splitting, plus a masker, valid MNI coordinates, Studyset-style sample
    size metadata, numeric annotations, text fields, unique study IDs, and
    unique full analysis IDs.
    """
    coordinate_groups = {
        "motor": [
            [-38.0, -22.0, 56.0],
            [38.0, -20.0, 56.0],
            [0.0, -4.0, 58.0],
        ],
        "visual": [
            [-18.0, -96.0, 0.0],
            [18.0, -96.0, 0.0],
            [30.0, -88.0, 4.0],
        ],
    }
    studies = []

    idx = 0
    for group, coordinates in coordinate_groups.items():
        for coordinate in coordinates:
            study_id = f"study_{idx}"
            sample_sizes = [20 + idx]
            target_score = float(idx) + 0.5
            studies.append(
                {
                    "id": study_id,
                    "name": f"Study {idx}",
                    "analyses": [
                        {
                            "id": "task",
                            "name": f"Task {idx}",
                            "metadata": {
                                "sample_sizes": sample_sizes,
                            },
                            "annotations": {
                                "motor_label": float(group == "motor"),
                                "visual_label": float(group == "visual"),
                                "target_score": target_score,
                            },
                            "texts": {"abstract": f"Study {idx} abstract."},
                            "points": [{"space": "MNI", "coordinates": coordinate}],
                            "images": [],
                        }
                    ],
                }
            )
            idx += 1

    source = {
        "id": "studyset_ml_source",
        "name": "Studyset ML source",
        "masker": get_masker(get_template(space="mni152_2mm", mask="brain")),
        "studies": studies,
    }
    return Studyset(source, mask=source["masker"])


@pytest.fixture
def ma_feature_dataset(ml_studyset):
    """Build a fresh MAFeatureDataset from the shared Studyset."""
    studyset = ml_studyset
    ids = np.asarray(studyset.ids, dtype=str)
    study_ids = np.asarray([id_.rsplit("-", 1)[0] for id_ in ids], dtype=str)
    annotations = studyset.annotations_df.drop_duplicates("id").set_index("id").reindex(ids)
    motor_labels = annotations["motor_label"].to_numpy(dtype=float)
    target = annotations["target_score"].to_numpy(dtype=float)
    map_features = sparse.csr_matrix(
        np.column_stack(
            [
                motor_labels + 0.1 * target,
                2.0 * motor_labels + 0.1 * target,
            ]
        )
    )
    descriptor_features = motor_labels[:, None]
    features = sparse.hstack(
        [map_features, sparse.csr_matrix(descriptor_features)],
        format="csr",
    )

    return MAFeatureDataset(
        features=features,
        ids=ids,
        study_ids=study_ids,
        feature_names=["feature_0", "feature_1", "motor_label"],
        target=target,
        provenance={"source": {"ids": list(ids)}},
        map_features=map_features,
        descriptor_features=descriptor_features,
        masker=studyset.masker,
    )


def _get_data_column(data, column_idx):
    """Return one feature column as a dense 1D array."""
    column = data[:, column_idx]
    if sparse.issparse(column):
        return column.toarray().ravel()
    return np.asarray(column).ravel()


def assert_sklearn_bunch_valid(
    bunch,
    expected_feature_rows: int | None = None,
    expected_feature_columns: int | None = None,
    expected_groups: list[str] | None = None,
    expected_target: list[Any] | None = None,
    expected_feature_names: list[str] | None = None,
    expected_columns_by_group: dict[str, dict[str, Any]] | None = None,
    expected_target_by_group: dict[str, Any] | None = None,
    expected_sparse: bool | None = None,
    require_target: bool = True,
    estimator: Any | None = None,
):
    """Assert the shared sklearn-export contract for MA feature Bunches."""
    assert isinstance(bunch, Bunch)

    n_rows, n_columns = bunch.data.shape
    if expected_feature_rows is None:
        assert n_rows > 0
    else:
        assert n_rows == expected_feature_rows

    if expected_feature_columns is not None:
        assert n_columns == expected_feature_columns

    if expected_sparse is True:
        assert sparse.issparse(bunch.data)
    elif expected_sparse is False:
        assert not sparse.issparse(bunch.data)

    assert np.issubdtype(bunch.data.dtype, np.number)

    groups = np.asarray(bunch.groups)
    assert len(groups) == n_rows

    if expected_groups is not None:
        np.testing.assert_array_equal(groups, expected_groups)

    target = bunch.target
    if target is None:
        assert not require_target
    else:
        target = np.asarray(target)
        assert len(target) == n_rows

    if expected_target is not None:
        assert target is not None
        np.testing.assert_array_equal(target, expected_target)

    feature_names = list(bunch.feature_names)
    assert len(feature_names) == n_columns

    if expected_feature_names is not None:
        assert feature_names == list(expected_feature_names)

    if expected_target_by_group is not None:
        assert target is not None
        for row_idx, study_id in enumerate(groups):
            assert study_id in expected_target_by_group
            assert target[row_idx] == expected_target_by_group[study_id]

    if expected_columns_by_group is not None:
        for feature_name, expected_values in expected_columns_by_group.items():
            assert feature_name in feature_names
            column = _get_data_column(bunch.data, feature_names.index(feature_name))

            for row_idx, study_id in enumerate(groups):
                assert study_id in expected_values
                assert column[row_idx] == expected_values[study_id]

    if estimator is not None:
        assert target is not None
        estimator.fit(bunch.data, target)


def test_ma_feature_dataset_initialization(ma_feature_dataset):
    """Test that MAFeatureDataset initializes and stores documented attributes."""
    ds = ma_feature_dataset

    assert sparse.issparse(ds.features)
    assert ds.features.shape == (6, 3)
    assert repr(ds) == "MAFeatureDataset(n_rows=6, n_features=3)"
    np.testing.assert_array_equal(ds.ids, [f"study_{idx}-task" for idx in range(6)])
    np.testing.assert_array_equal(ds.study_ids, [f"study_{idx}" for idx in range(6)])
    assert ds.feature_names == ["feature_0", "feature_1", "motor_label"]
    np.testing.assert_array_equal(ds.target, np.arange(6, dtype=float) + 0.5)
    np.testing.assert_array_equal(ds.provenance["source"]["ids"], ds.ids)
    assert sparse.issparse(ds._map_features)
    assert ds._map_features.shape == (6, 2)
    assert ds.map_columns == slice(0, 2)
    assert ds.descriptor_columns == slice(2, 3)
    np.testing.assert_array_equal(
        ds._descriptor_features.ravel(),
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    assert ds._masker is not None


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
        (
            {
                "ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "feature_names": ["f1", "f2"],
                "map_features": np.array([[1.0, 2.0]]),
            },
            "map_features row count must match number of rows in features",
        ),
        (
            {
                "ids": ["s1", "s2"],
                "study_ids": ["study_a", "study_b"],
                "feature_names": ["f1", "f2"],
                "descriptor_features": np.array([[1.0]]),
            },
            "descriptor_features row count must match number of rows in features",
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


def test_ma_feature_dataset_to_sklearn(ma_feature_dataset):
    """Dataset export returns a valid sklearn Bunch."""
    ds = ma_feature_dataset

    bunch = ds.to_sklearn()
    assert_sklearn_bunch_valid(
        bunch,
        expected_feature_rows=len(ds.ids),
        expected_feature_columns=len(ds.feature_names),
        expected_groups=ds.study_ids,
        expected_target=ds.target,
        expected_feature_names=ds.feature_names,
        expected_sparse=True,
    )
    assert bunch.data is ds.features
    assert bunch.ids is ds.ids
    assert bunch.provenance is ds.provenance
    assert bunch.map_columns == ds.map_columns
    assert bunch.descriptor_columns == ds.descriptor_columns


def test_ma_feature_dataset_make_preprocessor(ma_feature_dataset):
    """Map reduction leaves descriptor columns unchanged."""
    ds = ma_feature_dataset
    preprocessor = ds.make_preprocessor(
        n_components=1,
        random_state=RANDOM_SEED,
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert not hasattr(preprocessor, "transformers_")

    transformed = preprocessor.fit_transform(ds.features)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    assert transformed.shape == (len(ds.ids), 2)
    np.testing.assert_array_equal(transformed[:, 1], ds._descriptor_features.ravel())


def test_ma_feature_dataset_sklearn_pipeline(ma_feature_dataset):
    """The preprocessor and grouped CV work in sklearn model selection."""
    ds = ma_feature_dataset.copy()
    ds.study_ids = np.repeat(["study_0", "study_1", "study_2"], 2)
    pipeline = make_pipeline(
        ds.make_preprocessor(n_components=1, random_state=RANDOM_SEED),
        Ridge(),
    )
    cv = GroupKFold(n_splits=3)

    scores = cross_val_score(
        pipeline,
        ds.features,
        ds.target,
        cv=cv,
        groups=ds.study_ids,
    )
    assert scores.shape == (3,)

    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": [0.5, 1.0]},
        cv=cv,
    )
    search.fit(ds.features, ds.target, groups=ds.study_ids)
    assert search.best_estimator_ is not None


def test_ma_feature_dataset_copy(ma_feature_dataset):
    """Dataset copy preserves aligned arrays without sharing mutable data."""
    ds = ma_feature_dataset

    copied = ds.copy()

    assert copied is not ds
    assert copied.features is not ds.features
    assert copied._map_features is not ds._map_features
    assert copied._descriptor_features is not ds._descriptor_features
    np.testing.assert_array_equal(copied.ids, ds.ids)
    np.testing.assert_array_equal(copied.study_ids, ds.study_ids)
    assert copied.feature_names == ds.feature_names
    np.testing.assert_array_equal(copied.target, ds.target)
    np.testing.assert_array_equal(copied.features.toarray(), ds.features.toarray())
    np.testing.assert_array_equal(copied._map_features.toarray(), ds._map_features.toarray())
    np.testing.assert_array_equal(copied._descriptor_features, ds._descriptor_features)
    copied.provenance["source"]["ids"].append("s3")
    np.testing.assert_array_equal(ds.provenance["source"]["ids"], ds.ids)
    assert copied._masker is ds._masker


def test_ma_feature_extractor_initialization():
    """Test that MAFeatureExtractor initializes and stores documented parameters."""
    kernel_transformer = object()
    extractor = MAFeatureExtractor(
        kernel_transformer=kernel_transformer,
        descriptor_fields=[{"source": "annotations", "field": "motor_label"}],
        target_field={"source": "annotations", "field": "motor_label"},
    )

    assert extractor.kernel_transformer is kernel_transformer
    assert extractor.descriptor_fields == [{"source": "annotations", "field": "motor_label"}]
    assert extractor.descriptor_transformers is None
    assert extractor.target_field == {"source": "annotations", "field": "motor_label"}
    assert extractor.target_transformer is None
    assert extractor.missing_coordinates == "drop"
    assert extractor.cache_maps is True
    assert extractor.memory is None
    assert extractor.memory_level == 1
    assert extractor._map_cache == {}


def test_ma_feature_extractor_selected_values_alignment(ml_studyset):
    """Selected Studyset field values remain aligned to analysis IDs."""
    studyset = ml_studyset
    extractor = MAFeatureExtractor(kernel_transformer=object())

    expected_motor = [study.analyses[0].annotations["motor_label"] for study in studyset.studies]
    expected_sample_sizes = [
        study.analyses[0].metadata["sample_sizes"] for study in studyset.studies
    ]

    values, field = extractor._get_selected_values(
        studyset,
        {"source": "annotations", "field": "motor_label"},
    )
    assert field == "motor_label"
    np.testing.assert_array_equal(values, expected_motor)

    values, field = extractor._get_selected_values(
        studyset,
        {"source": "metadata", "field": "sample_sizes"},
    )
    assert field == "sample_sizes"
    assert values.tolist() == expected_sample_sizes


def test_ma_feature_extractor_transform(ml_studyset):
    """Transform a Studyset into one aligned sklearn Bunch."""
    studyset = ml_studyset
    descriptor_name = "motor_label"
    target_name = "target_score"
    descriptor_by_study = {
        study.id: study.analyses[0].annotations[descriptor_name] for study in studyset.studies
    }
    target_by_study = {
        study.id: study.analyses[0].annotations[target_name] for study in studyset.studies
    }
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4, value=1),
        descriptor_fields=[{"source": "annotations", "field": descriptor_name}],
        target_field={"source": "annotations", "field": target_name},
    )

    bunch = extractor.transform(
        studyset,
        map_reducer="truncated_svd",
        map_reducer_params={"n_components": 2, "random_state": RANDOM_SEED},
    )

    assert len(bunch.ids) == len(studyset.ids)
    assert sparse.issparse(bunch.data[:, bunch.map_columns])
    assert descriptor_name in bunch.feature_names
    assert target_name not in bunch.feature_names
    assert isinstance(bunch.preprocessor, ColumnTransformer)
    assert not hasattr(bunch.preprocessor, "transformers_")

    assert_sklearn_bunch_valid(
        bunch,
        expected_feature_rows=len(studyset.ids),
        expected_feature_names=bunch.feature_names,
        expected_columns_by_group={descriptor_name: descriptor_by_study},
        expected_target_by_group=target_by_study,
        expected_sparse=True,
    )


@pytest.mark.parametrize(
    ("missing_coordinates", "keep_missing"),
    [("drop", False), ("include", True)],
)
def test_ma_feature_extractor_handles_missing_coordinates(
    ml_studyset,
    missing_coordinates,
    keep_missing,
):
    """Drop or retain analyses without coordinates."""
    studyset = ml_studyset.copy()
    missing_study = studyset.studies[2]
    missing_analysis = missing_study.analyses[0]
    missing_id = f"{missing_study.id}-{missing_analysis.id}"
    missing_analysis.points = []

    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4, value=1),
        descriptor_fields=[{"source": "annotations", "field": "motor_label"}],
        target_field={"source": "annotations", "field": "target_score"},
        missing_coordinates=missing_coordinates,
    )

    bunch = extractor.transform(studyset)

    expected_ids = studyset.ids if keep_missing else studyset.ids[studyset.ids != missing_id]
    np.testing.assert_array_equal(bunch.ids, expected_ids)
    expected_annotations = studyset.annotations_df.set_index("id").loc[expected_ids]
    np.testing.assert_array_equal(
        _get_data_column(bunch.data, bunch.feature_names.index("motor_label")),
        expected_annotations["motor_label"],
    )
    np.testing.assert_array_equal(bunch.target, expected_annotations["target_score"])
    assert bunch.provenance["dropped_ids"] == ([] if keep_missing else [missing_id])

    if keep_missing:
        missing_row = np.flatnonzero(bunch.ids == missing_id)[0]
        assert bunch.data[missing_row, bunch.map_columns].nnz == 0


def test_ma_feature_extractor_rejects_invalid_missing_coordinates(ml_studyset):
    """Reject an unsupported missing-coordinate mode."""
    extractor = MAFeatureExtractor(
        kernel_transformer=object(),
        missing_coordinates="invalid",
    )

    with pytest.raises(ValueError, match="missing_coordinates must be"):
        extractor.transform(ml_studyset)


def test_ma_feature_extractor_reuses_map_cache(ml_studyset):
    """Reuse unchanged map inputs and invalidate changed inputs."""

    class CountingKernel:
        def __init__(self, scale=1):
            self.n_calls = 0
            self.scale = scale

        def get_params(self):
            return {"scale": self.scale}

        def transform(self, studyset, return_type="sparse"):
            self.n_calls += 1
            assert return_type == "sparse"
            n_rows = len(studyset.ids)
            data = self.scale * np.arange(n_rows * 3, dtype=float).reshape(n_rows, 3)
            return sparse.csr_matrix(data)

    class NegatingCountingKernel(CountingKernel):
        def transform(self, studyset, return_type="sparse"):
            return -super().transform(studyset, return_type=return_type)

    studyset = ml_studyset.copy()
    kernel = CountingKernel()
    extractor = MAFeatureExtractor(kernel_transformer=kernel)

    first_bunch = extractor.transform(studyset)
    second_bunch = extractor.transform(studyset)

    assert kernel.n_calls == 1
    assert first_bunch.data is not second_bunch.data
    np.testing.assert_array_equal(
        first_bunch.data.toarray(),
        second_bunch.data.toarray(),
    )

    coordinates = studyset.coordinates.copy()
    coordinates.loc[0, "x"] += 1
    studyset.coordinates = coordinates
    extractor.transform(studyset)
    assert kernel.n_calls == 2

    kernel.scale = 2
    scaled_bunch = extractor.transform(studyset)
    assert kernel.n_calls == 3
    np.testing.assert_array_equal(
        scaled_bunch.data.toarray(),
        2 * second_bunch.data.toarray(),
    )

    negating_kernel = NegatingCountingKernel(scale=2)
    extractor.kernel_transformer = negating_kernel
    negated_bunch = extractor.transform(studyset)
    assert negating_kernel.n_calls == 1
    np.testing.assert_array_equal(
        negated_bunch.data.toarray(),
        -scaled_bunch.data.toarray(),
    )


def test_make_map_reducer_truncated_svd():
    """Build the truncated-SVD reducer and defer later reducers."""
    reducer = make_map_reducer(
        "truncated_svd",
        n_components=2,
        random_state=RANDOM_SEED,
    )

    assert isinstance(reducer, TruncatedSVD)
    assert reducer.n_components == 2
    assert reducer.random_state == RANDOM_SEED

    with pytest.raises(NotImplementedError):
        make_map_reducer("variance_threshold")
