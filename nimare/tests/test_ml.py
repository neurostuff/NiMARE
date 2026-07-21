"""Tests for the nimare.ml module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.utils import Bunch

from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAFeatureDataset, MAFeatureExtractor, make_map_reducer
from nimare.nimads import Studyset
from nimare.utils import get_masker, get_template


def _build_studyset_ml_source():
    """Build the shared perfect Studyset source used by the ML tests.

    The source includes enough single-analysis studies for grouped train/test
    splitting, plus a masker, valid MNI coordinates, Studyset-style sample
    size metadata, numeric annotations, text fields, unique study IDs, and
    unique full analysis IDs.
    """
    coordinates = [
        [1.0, 3.0, 5.0],
        [-1.0, -3.0, -5.0],
        [8.0, -12.0, 10.0],
        [-8.0, 12.0, 10.0],
        [18.0, -20.0, 22.0],
        [-18.0, 20.0, 22.0],
    ]
    studies = []

    for idx, coordinate in enumerate(coordinates):
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
                            "alpha_label": float(idx == 0),
                            "beta_label": float(idx == 1),
                            "target_score": target_score,
                        },
                        "texts": {"abstract": f"Study {idx} abstract."},
                        "points": [{"space": "MNI", "coordinates": coordinate}],
                        "images": [],
                    }
                ],
            }
        )

    return {
        "id": "studyset_ml_source",
        "name": "Studyset ML source",
        "masker": get_masker(get_template(space="mni152_2mm", mask="brain")),
        "studies": studies,
    }


def build_studyset():
    """Build a fresh shared perfect Studyset for ML tests."""
    source = _build_studyset_ml_source()
    return Studyset(source, mask=source["masker"])


def build_ma_feature_dataset():
    """Build a fresh MAFeatureDataset from the shared Studyset-like source."""
    source = _build_studyset_ml_source()
    ids = []
    study_ids = []
    map_rows = []
    descriptor_rows = []
    target = []

    for study in source["studies"]:
        analysis = study["analyses"][0]
        coordinate = analysis["points"][0]["coordinates"]
        ids.append(f"{study['id']}-{analysis['id']}")
        study_ids.append(study["id"])
        map_rows.append([coordinate[0], coordinate[1]])
        descriptor_rows.append([analysis["annotations"]["alpha_label"]])
        target.append(analysis["annotations"]["target_score"])

    map_features = sparse.csr_matrix(map_rows)
    descriptor_features = np.asarray(descriptor_rows, dtype=float)
    features = sparse.hstack(
        [map_features, sparse.csr_matrix(descriptor_features)],
        format="csr",
    )

    return MAFeatureDataset(
        features=features,
        ids=ids,
        study_ids=study_ids,
        feature_names=["feature_0", "feature_1", "alpha_label"],
        target=np.asarray(target, dtype=float),
        provenance={"source": {"ids": list(ids)}},
        map_features=map_features,
        descriptor_features=descriptor_features,
        masker=source["masker"],
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
    assert ds._map_features is features
    assert ds._descriptor_features is None
    assert ds._masker is None

    descriptor_features = np.array([[10.0], [20.0]])
    masker = object()
    ds = MAFeatureDataset(
        features=sparse.csr_matrix([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]]),
        ids=ids,
        study_ids=study_ids,
        feature_names=["feature_0", "feature_1", "descriptor"],
        target=target,
        provenance=provenance,
        map_features=sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]]),
        descriptor_features=descriptor_features,
        masker=masker,
    )

    assert sparse.issparse(ds._map_features)
    assert ds._descriptor_features is descriptor_features
    assert ds._masker is masker


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


def test_ma_feature_dataset_to_sklearn():
    """Dataset export returns a valid sklearn Bunch."""
    ds = build_ma_feature_dataset()

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


def test_ma_feature_dataset_split():
    """Dataset grouped split preserves row alignment and avoids study leakage."""
    ds = build_ma_feature_dataset()

    train, test = ds.split(test_size=0.5, random_state=13)

    assert set(train.study_ids).isdisjoint(test.study_ids)
    assert sorted(train.ids + test.ids) == sorted(ds.ids)

    expected_target_by_id = dict(zip(ds.ids, ds.target, strict=False))
    expected_features_by_id = dict(zip(ds.ids, ds.features.toarray(), strict=False))
    for split_dataset in (train, test):
        for row_id, row_target in zip(split_dataset.ids, split_dataset.target, strict=False):
            assert row_target == expected_target_by_id[row_id]
        for row_id, row_features in zip(
            split_dataset.ids,
            split_dataset.features.toarray(),
            strict=False,
        ):
            np.testing.assert_array_equal(row_features, expected_features_by_id[row_id])


def test_ma_feature_dataset_apply_map_reducer():
    """Map reducers transform map columns while preserving descriptors and metadata."""
    ds = build_ma_feature_dataset()
    descriptor_name = "alpha_label"
    descriptor_idx = ds.feature_names.index(descriptor_name)
    descriptor_values = _get_data_column(ds.features, descriptor_idx)
    reducer = make_map_reducer("truncated_svd", n_components=1, random_state=13)

    reduced = ds.apply_map_reducer(reducer, fit=True)

    assert reduced is not ds
    assert reduced.ids == ds.ids
    assert reduced.study_ids == ds.study_ids
    assert reduced.feature_names == ["feature_0", descriptor_name]
    assert reduced.features.shape == (len(ds.ids), 2)
    assert not sparse.issparse(reduced.features)
    np.testing.assert_array_equal(reduced.target, ds.target)
    np.testing.assert_array_equal(reduced.features[:, 1], descriptor_values)
    reduced.provenance["source"]["ids"].append("s4")
    assert ds.provenance["source"]["ids"] == ds.ids


def test_ma_feature_dataset_copy():
    """Dataset copy preserves aligned arrays without sharing mutable data."""
    ds = build_ma_feature_dataset()

    copied = ds.copy()

    assert copied is not ds
    assert copied.features is not ds.features
    assert copied._map_features is not ds._map_features
    assert copied._descriptor_features is not ds._descriptor_features
    assert copied.ids == ds.ids
    assert copied.study_ids == ds.study_ids
    assert copied.feature_names == ds.feature_names
    np.testing.assert_array_equal(copied.target, ds.target)
    np.testing.assert_array_equal(copied.features.toarray(), ds.features.toarray())
    np.testing.assert_array_equal(copied._map_features.toarray(), ds._map_features.toarray())
    np.testing.assert_array_equal(copied._descriptor_features, ds._descriptor_features)
    copied.provenance["source"]["ids"].append("s3")
    assert ds.provenance["source"]["ids"] == ds.ids
    assert copied._masker is ds._masker


def test_ma_feature_extractor_initialization():
    """Test that MAFeatureExtractor initializes and stores documented parameters."""
    kernel_transformer = object()
    extractor = MAFeatureExtractor(
        kernel_transformer=kernel_transformer,
        descriptor_fields=[{"source": "annotations", "field": "alpha_label"}],
        target_field={"source": "annotations", "field": "alpha_label"},
    )

    assert extractor.kernel_transformer is kernel_transformer
    assert extractor.descriptor_fields == [{"source": "annotations", "field": "alpha_label"}]
    assert extractor.descriptor_transformers is None
    assert extractor.target_field == {"source": "annotations", "field": "alpha_label"}
    assert extractor.target_transformer is None
    assert extractor.missing_coordinates == "drop"
    assert extractor.test_size is None
    assert extractor.random_state is None
    assert extractor.cache_maps is True
    assert extractor.memory is None
    assert extractor.memory_level == 1
    assert extractor._map_cache == {}


def test_ma_feature_extractor_studyset_values_alignment():
    """Studyset table extraction preserves row-aligned IDs and field values."""
    source = _build_studyset_ml_source()
    studyset = Studyset(source, mask=source["masker"])
    extractor = MAFeatureExtractor(kernel_transformer=object())

    tables = extractor._get_studyset_tables(studyset)
    assert set(tables) == {
        "ids",
        "study_ids",
        "coordinates",
        "metadata",
        "annotations_df",
        "texts",
        "masker",
        "space",
        "basepath",
    }
    expected_ids = [f"{study['id']}-{study['analyses'][0]['id']}" for study in source["studies"]]
    expected_study_ids = [study["id"] for study in source["studies"]]
    expected_alpha = [
        study["analyses"][0]["annotations"]["alpha_label"] for study in source["studies"]
    ]
    expected_sample_sizes = [
        study["analyses"][0]["metadata"]["sample_sizes"] for study in source["studies"]
    ]

    np.testing.assert_array_equal(tables["ids"], expected_ids)
    np.testing.assert_array_equal(tables["study_ids"], expected_study_ids)
    assert tables["masker"] is studyset.masker
    assert tables["space"] == studyset.space
    assert tables["basepath"] == studyset.basepath

    values, field = extractor._get_selected_values(
        tables,
        {"source": "annotations", "field": "alpha_label"},
    )
    assert field == "alpha_label"
    np.testing.assert_array_equal(values, expected_alpha)

    values, field = extractor._get_selected_values(
        tables,
        {"source": "metadata", "field": "sample_sizes"},
    )
    assert field == "sample_sizes"
    assert values == expected_sample_sizes


def test_ma_feature_extractor_stack_sparse_features_not_implemented():
    """The reserved sparse stacking helper remains explicitly scaffolded."""
    extractor = MAFeatureExtractor(kernel_transformer=object())

    with pytest.raises(NotImplementedError):
        extractor._stack_sparse_features(sparse.csr_matrix([[1.0, 0.0]]))


def test_ma_feature_extractor_transform():
    """Transform a perfect Studyset into grouped train/test MAFeatureDatasets."""
    source = _build_studyset_ml_source()
    studyset = Studyset(source, mask=source["masker"])
    descriptor_name = "alpha_label"
    target_name = "target_score"
    descriptor_by_study = {
        study["id"]: study["analyses"][0]["annotations"][descriptor_name]
        for study in source["studies"]
    }
    target_by_study = {
        study["id"]: study["analyses"][0]["annotations"][target_name]
        for study in source["studies"]
    }
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4, value=1),
        descriptor_fields=[{"source": "annotations", "field": descriptor_name}],
        target_field={"source": "annotations", "field": target_name},
        test_size=0.33,
        random_state=13,
    )

    train_dataset, test_dataset = extractor.transform(studyset)

    assert isinstance(train_dataset, MAFeatureDataset)
    assert isinstance(test_dataset, MAFeatureDataset)
    assert sparse.issparse(train_dataset._map_features)
    assert sparse.issparse(test_dataset._map_features)
    assert train_dataset._masker is studyset.masker
    assert test_dataset._masker is studyset.masker
    assert "alpha_label" in train_dataset.feature_names
    assert "target_score" not in train_dataset.feature_names
    assert set(train_dataset.study_ids).isdisjoint(set(test_dataset.study_ids))

    train_bunch = train_dataset.to_sklearn()
    test_bunch = test_dataset.to_sklearn()
    assert_sklearn_bunch_valid(
        train_bunch,
        expected_columns_by_group={descriptor_name: descriptor_by_study},
        expected_target_by_group=target_by_study,
        expected_sparse=True,
    )
    assert_sklearn_bunch_valid(
        test_bunch,
        expected_feature_names=train_dataset.feature_names,
        expected_columns_by_group={descriptor_name: descriptor_by_study},
        expected_target_by_group=target_by_study,
        expected_sparse=True,
    )


def test_ma_feature_extractor_reuses_map_cache():
    """Repeated extraction with unchanged settings reuses cached map features."""

    class CountingKernel:
        def __init__(self):
            self.n_calls = 0

        def transform(self, studyset, return_type="sparse"):
            self.n_calls += 1
            assert return_type == "sparse"
            n_rows = len(studyset.ids)
            data = np.arange(n_rows * 3, dtype=float).reshape(n_rows, 3)
            return sparse.csr_matrix(data)

    studyset = build_studyset()
    kernel = CountingKernel()
    extractor = MAFeatureExtractor(kernel_transformer=kernel)

    first_dataset, _ = extractor.transform(studyset)
    second_dataset, _ = extractor.transform(studyset)

    assert kernel.n_calls == 1
    assert first_dataset._map_features is not second_dataset._map_features
    np.testing.assert_array_equal(
        first_dataset._map_features.toarray(),
        second_dataset._map_features.toarray(),
    )


def test_ma_feature_extractor_to_sklearn():
    """Export a perfect Studyset as train/test sklearn Bunches and fit an estimator."""
    source = _build_studyset_ml_source()
    studyset = Studyset(source, mask=source["masker"])
    descriptor_name = "alpha_label"
    target_name = "target_score"
    descriptor_by_study = {
        study["id"]: study["analyses"][0]["annotations"][descriptor_name]
        for study in source["studies"]
    }
    target_by_study = {
        study["id"]: study["analyses"][0]["annotations"][target_name]
        for study in source["studies"]
    }
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4, value=1),
        descriptor_fields=[{"source": "annotations", "field": descriptor_name}],
        target_field={"source": "annotations", "field": target_name},
        test_size=0.33,
        random_state=13,
    )

    train_bunch, test_bunch = extractor.to_sklearn(
        studyset,
        map_reducer="truncated_svd",
    )

    assert_sklearn_bunch_valid(
        train_bunch,
        expected_columns_by_group={descriptor_name: descriptor_by_study},
        expected_target_by_group=target_by_study,
        estimator=Ridge(),
    )
    feature_names = list(train_bunch.feature_names)
    assert_sklearn_bunch_valid(
        test_bunch,
        expected_feature_names=feature_names,
        expected_columns_by_group={descriptor_name: descriptor_by_study},
        expected_target_by_group=target_by_study,
    )
    assert feature_names.count(descriptor_name) == 1
    assert target_name not in feature_names

    assert set(train_bunch.groups).isdisjoint(set(test_bunch.groups))

    original_map_feature_count = int(np.count_nonzero(studyset.masker.mask_img.get_fdata()))
    reduced_map_feature_count = len(
        [feature_name for feature_name in feature_names if feature_name != descriptor_name]
    )
    assert 0 < reduced_map_feature_count < original_map_feature_count


def test_make_map_reducer_truncated_svd():
    """Build the truncated-SVD reducer and defer later reducers."""
    reducer = make_map_reducer("truncated_svd", n_components=2, random_state=13)

    assert isinstance(reducer, TruncatedSVD)
    assert reducer.n_components == 2
    assert reducer.random_state == 13

    with pytest.raises(NotImplementedError):
        make_map_reducer("variance_threshold")
