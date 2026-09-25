"""Tests for the nimare.ml module."""

from __future__ import annotations

import time

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import Bunch

from nimare.generate import create_coordinate_studyset
from nimare.meta.kernel import MKDAKernel
from nimare.ml import (
    AtlasAggregator,
    MAFeatureDataset,
    MAFeatureExtractor,
    make_map_reducer,
)
from nimare.nimads import Studyset
from nimare.utils import get_masker, get_template

RANDOM_SEED = 13

# One focus per analysis, far enough apart that a small kernel gives every
# analysis a map no other analysis shares. The tests use that to prove a row
# kept its own map.
COORDINATES = [
    [-38.0, -22.0, 56.0],
    [38.0, -20.0, 56.0],
    [0.0, -4.0, 58.0],
    [-18.0, -96.0, 0.0],
    [18.0, -96.0, 0.0],
    [30.0, -88.0, 4.0],
    [-54.0, -24.0, 12.0],
    [54.0, -22.0, 10.0],
]

# Two of the six studies contribute two analyses, so grouping has something to do.
STUDY_LAYOUT = [
    ("study_0", 2),
    ("study_1", 1),
    ("study_2", 1),
    ("study_3", 2),
    ("study_4", 1),
    ("study_5", 1),
]


def _build_ml_studyset(ids_without_points=(), ids_without_score=(), coordinate_offset=0.0):
    """Build the Studyset the ML tests read.

    A Studyset is immutable, so the variants some tests need -- an analysis with
    no coordinates, an analysis missing a metadata value, coordinates moved off
    the shared ones -- are built here rather than edited into a finished one.
    """
    studies = []
    position = 0
    for study_id, n_analyses in STUDY_LAYOUT:
        analyses = []
        for analysis_index in range(n_analyses):
            analysis_id = f"task{analysis_index}"
            full_id = f"{study_id}-{analysis_id}"
            coordinate = COORDINATES[position]
            analyses.append(
                {
                    "id": analysis_id,
                    "name": f"Task {position}",
                    "metadata": {
                        "sample_sizes": [20 + position],
                        "comparison_task": "n-back" if position % 2 else "flanker",
                        **({} if full_id in ids_without_score else {"score": float(position)}),
                    },
                    "annotations": {
                        "motor_label": float(position < 4),
                        "target_score": float(position) + 0.5,
                    },
                    "texts": {"abstract": f"Analysis {position} abstract."},
                    "points": (
                        []
                        if full_id in ids_without_points
                        else [
                            {
                                "space": "MNI",
                                "coordinates": [axis + coordinate_offset for axis in coordinate],
                            }
                        ]
                    ),
                    "images": [],
                }
            )
            position += 1
        studies.append(
            {
                "id": study_id,
                "name": f"Study {study_id}",
                "metadata": {"year": 2000 + len(studies)},
                "analyses": analyses,
            }
        )

    masker = get_masker(get_template(space="mni152_2mm", mask="brain"))
    return Studyset(
        {
            "id": "studyset_ml_source",
            "name": "Studyset ML source",
            "masker": masker,
            "studies": studies,
        },
        mask=masker,
    )


@pytest.fixture(scope="session")
def ml_studyset():
    """Return the shared Studyset used by the ML tests."""
    return _build_ml_studyset()


@pytest.fixture(scope="session")
def small_masker():
    """Return a masker over a tiny volume, for reducer tests."""
    mask_data = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_data[:2, :2, :2] = 1
    return get_masker(nib.Nifti1Image(mask_data, np.eye(4)))


@pytest.fixture
def ma_feature_dataset(small_masker):
    """Build a small MAFeatureDataset directly, without running a kernel."""
    n_rows = 6
    ids = np.array([f"study_{idx // 2}-task{idx % 2}" for idx in range(n_rows)])
    study_ids = np.array([f"study_{idx // 2}" for idx in range(n_rows)])
    target = np.arange(n_rows, dtype=float) + 0.5
    map_features = sparse.csr_matrix(
        np.column_stack([np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) * 2.0])
    )
    descriptor_features = np.arange(n_rows, dtype=float)[:, None]

    return MAFeatureDataset(
        map_features,
        ids=ids,
        study_ids=study_ids,
        descriptor_features=descriptor_features,
        descriptor_names=["motor_label"],
        descriptors=pd.DataFrame({"motor_label": descriptor_features.ravel()}, index=ids),
        target=target,
        provenance={"studyset_id": "fixture", "dropped_ids": []},
        masker=small_masker,
    )


def _column(data, index):
    """Return one feature column as a dense 1D array."""
    column = data[:, index]
    if sparse.issparse(column):
        return column.toarray().ravel()
    return np.asarray(column).ravel()


def _map_signature(dataset):
    """Return the first non-zero map column of each row, which names its focus."""
    maps = dataset.map_features
    return [int(maps[row].indices.min()) for row in range(maps.shape[0])]


def assert_sklearn_bunch_valid(
    bunch,
    expected_rows=None,
    expected_columns=None,
    expected_sparse=None,
    require_target=True,
    estimator=None,
):
    """Assert the shared sklearn-export contract for MA feature Bunches."""
    assert isinstance(bunch, Bunch)

    n_rows, n_columns = bunch.data.shape
    assert n_rows == (expected_rows if expected_rows is not None else n_rows)
    if expected_columns is not None:
        assert n_columns == expected_columns
    if expected_sparse is not None:
        assert sparse.issparse(bunch.data) is expected_sparse

    assert np.issubdtype(bunch.data.dtype, np.number)
    assert len(bunch.groups) == n_rows
    assert len(bunch.ids) == n_rows
    assert len(bunch.feature_names) == n_columns

    if bunch.target is None:
        assert not require_target
    else:
        assert len(bunch.target) == n_rows

    if estimator is not None:
        estimator.fit(bunch.data, bunch.target)


# ---------------------------------------------------------------- container


def test_dataset_attributes(ma_feature_dataset):
    """The container exposes aligned blocks and derives the combined matrix."""
    dataset = ma_feature_dataset

    assert len(dataset) == 6
    assert dataset.shape == (6, 3)
    assert dataset.map_columns == slice(0, 2)
    assert dataset.descriptor_columns == slice(2, 3)
    assert dataset.feature_names == ["voxel_0", "voxel_1", "motor_label"]
    assert sparse.issparse(dataset.features)
    assert dataset.features.shape == (6, 3)
    np.testing.assert_array_equal(
        dataset.features.toarray()[:, dataset.descriptor_columns],
        dataset.descriptor_features,
    )
    assert "n_rows=6" in repr(dataset)
    assert "n_studies=3" in repr(dataset)


def test_dataset_features_are_built_once(ma_feature_dataset):
    """The combined matrix is derived from the blocks and cached."""
    assert ma_feature_dataset._features is None
    first = ma_feature_dataset.features
    assert ma_feature_dataset.features is first


def test_dataset_without_descriptors(small_masker):
    """A map-only dataset has no descriptor columns and needs no hstack."""
    map_features = sparse.csr_matrix(np.eye(3))
    dataset = MAFeatureDataset(map_features, ids=list("abc"), study_ids=list("abc"))

    assert dataset.features is map_features
    assert dataset.descriptor_columns == slice(3, 3)
    assert dataset.feature_names == ["voxel_0", "voxel_1", "voxel_2"]
    assert dataset.to_sklearn().target is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ids": ["a"]}, "ids has 1 entries"),
        ({"study_ids": ["a"]}, "study_ids has 1 entries"),
        ({"target": [1.0]}, "target has 1 entries"),
        ({"descriptor_features": np.zeros((1, 1))}, "descriptor_features has 1 rows"),
        ({"descriptor_features": np.zeros(2)}, "must be two-dimensional"),
        (
            {"descriptor_features": np.zeros((2, 2)), "descriptor_names": ["one"]},
            "must name every descriptor column",
        ),
        ({"map_feature_names": ["one"]}, "must name every map column"),
    ],
)
def test_dataset_rejects_misaligned_inputs(kwargs, message):
    """Every block has to describe the same analyses."""
    base = {
        "map_features": np.zeros((2, 2)),
        "ids": ["a", "b"],
        "study_ids": ["s", "s"],
    }
    base.update(kwargs)
    map_features = base.pop("map_features")

    with pytest.raises(ValueError, match=message):
        MAFeatureDataset(map_features, **base)


def test_dataset_to_sklearn(ma_feature_dataset):
    """Export carries the arrays scikit-learn workflows need."""
    dataset = ma_feature_dataset
    bunch = dataset.to_sklearn()

    assert_sklearn_bunch_valid(bunch, expected_rows=6, expected_columns=3, expected_sparse=True)
    np.testing.assert_array_equal(bunch.groups, dataset.study_ids)
    np.testing.assert_array_equal(bunch.ids, dataset.ids)
    assert bunch.provenance is dataset.provenance
    assert bunch.map_columns == dataset.map_columns

    data, target = dataset.to_sklearn(return_X_y=True)
    assert data is dataset.features
    assert target is dataset.target


def test_dataset_split_keeps_studies_together(ma_feature_dataset):
    """No study is split across partitions, and the split is reproducible."""
    dataset = ma_feature_dataset

    train, test = dataset.split(test_size=0.34, random_state=RANDOM_SEED)

    assert set(train.study_ids).isdisjoint(test.study_ids)
    assert len(train) + len(test) == len(dataset)
    assert set(train.ids) | set(test.ids) == set(dataset.ids)
    # Rows keep their own descriptors and targets.
    for part in (train, test):
        expected = [float(np.flatnonzero(dataset.ids == id_)[0]) for id_ in part.ids]
        np.testing.assert_array_equal(part.descriptor_features.ravel(), expected)
        np.testing.assert_array_equal(part.target, np.array(expected) + 0.5)
        np.testing.assert_array_equal(part.descriptors.index, part.ids)

    again = dataset.split(test_size=0.34, random_state=RANDOM_SEED)
    np.testing.assert_array_equal(again[1].ids, test.ids)


@pytest.mark.parametrize(
    ("test_size", "message"),
    [
        (0.9, "leaves one partition empty"),
        (0.0, "must be between 0 and 1"),
        (1.0, "must be between 0 and 1"),
        (3, "leaves one partition empty"),
        ("half", "must be a float or an int"),
    ],
)
def test_dataset_split_rejects_impossible_requests(ma_feature_dataset, test_size, message):
    """An unservable split is explained instead of half-returned."""
    with pytest.raises(ValueError, match=message):
        ma_feature_dataset.split(test_size=test_size, random_state=RANDOM_SEED)


def test_dataset_split_needs_two_studies(ma_feature_dataset):
    """One study cannot be split without splitting the study."""
    single = ma_feature_dataset.select_analyses(ma_feature_dataset.study_ids == "study_0")

    with pytest.raises(ValueError, match="at least 2 studies"):
        single.split()


def test_dataset_select_analyses(ma_feature_dataset):
    """Mask and position selections both keep every block aligned."""
    dataset = ma_feature_dataset

    by_mask = dataset.select_analyses(np.arange(len(dataset)) % 2 == 0)
    by_position = dataset.select_analyses([0, 2, 4])

    np.testing.assert_array_equal(by_mask.ids, by_position.ids)
    np.testing.assert_array_equal(by_mask.target, [0.5, 2.5, 4.5])
    np.testing.assert_array_equal(by_mask.map_features.toarray()[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(by_mask.descriptors["motor_label"], [0.0, 2.0, 4.0])
    assert by_mask.feature_names == dataset.feature_names

    reordered = dataset.select_analyses([2, 0])
    np.testing.assert_array_equal(reordered.ids, dataset.ids[[2, 0]])
    np.testing.assert_array_equal(reordered.target, dataset.target[[2, 0]])

    with pytest.raises(ValueError, match="one entry per analysis row"):
        dataset.select_analyses(np.ones(2, dtype=bool))


def test_dataset_copy_is_independent(ma_feature_dataset):
    """A copy shares no mutable state with the original."""
    dataset = ma_feature_dataset
    copied = dataset.copy()

    np.testing.assert_array_equal(copied.features.toarray(), dataset.features.toarray())
    copied.provenance["dropped_ids"].append("study_9-task0")
    copied.target[0] = 99.0
    copied.map_features.data[0] = 99.0

    assert dataset.provenance["dropped_ids"] == []
    assert dataset.target[0] == 0.5
    assert dataset.map_features.data[0] != 99.0
    assert copied.masker is dataset.masker


# ---------------------------------------------------------------- reduction


def test_make_preprocessor_reduces_only_map_columns(ma_feature_dataset):
    """Descriptor columns pass through the preprocessor untouched."""
    dataset = ma_feature_dataset
    preprocessor = dataset.make_preprocessor(
        "truncated_svd", n_components=1, random_state=RANDOM_SEED
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert not hasattr(preprocessor, "transformers_")

    transformed = preprocessor.fit_transform(dataset.features)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    assert transformed.shape == (len(dataset), 2)
    np.testing.assert_array_equal(transformed[:, 1], dataset.descriptor_features.ravel())


def test_make_preprocessor_keeps_unreduced_features_sparse(ma_feature_dataset):
    """A sparse-preserving reducer must not be densified on the way through."""
    dataset = ma_feature_dataset
    preprocessor = dataset.make_preprocessor("variance_threshold", threshold=0.0)

    transformed = preprocessor.fit_transform(dataset.features)

    assert sparse.issparse(transformed)


def test_make_preprocessor_accepts_transformers_and_passthrough(ma_feature_dataset):
    """The reducer may be an instance, a name, or nothing at all."""
    dataset = ma_feature_dataset

    built = dataset.make_preprocessor(
        TruncatedSVD(n_components=1), descriptor_transformer=SimpleImputer()
    )
    assert isinstance(built.transformers[0][1], TruncatedSVD)
    assert isinstance(built.transformers[1][1], SimpleImputer)

    passthrough = dataset.make_preprocessor(None)
    assert passthrough.transformers[0][1] == "passthrough"
    assert passthrough.fit_transform(dataset.features).shape == dataset.features.shape

    with pytest.raises(ValueError, match="only used when map_reducer names a workflow"):
        dataset.make_preprocessor(TruncatedSVD(), n_components=1)


def test_dataset_works_in_sklearn_model_selection(ma_feature_dataset):
    """The exported arrays drive grouped cross-validation and a grid search."""
    dataset = ma_feature_dataset
    pipeline = make_pipeline(
        dataset.make_preprocessor("truncated_svd", n_components=1, random_state=RANDOM_SEED),
        Ridge(),
    )
    cv = GroupKFold(n_splits=3)
    bunch = dataset.to_sklearn()

    scores = cross_val_score(pipeline, bunch.data, bunch.target, cv=cv, groups=bunch.groups)
    assert scores.shape == (3,)

    search = GridSearchCV(pipeline, {"ridge__alpha": [0.5, 1.0]}, cv=cv)
    search.fit(bunch.data, bunch.target, groups=bunch.groups)
    assert search.best_estimator_ is not None


def test_fit_transform_maps_then_transform_maps(ma_feature_dataset):
    """A reducer is fitted on train rows and reused on held-out rows."""
    dataset = ma_feature_dataset
    train, test = dataset.split(test_size=0.34, random_state=RANDOM_SEED)
    reducer = make_map_reducer("truncated_svd", n_components=1, random_state=RANDOM_SEED)

    reduced_train = train.fit_transform_maps(reducer)
    reduced_test = test.transform_maps(reducer)

    assert reduced_train.map_features.shape == (len(train), 1)
    assert reduced_test.map_features.shape == (len(test), 1)
    assert reduced_train.feature_names == ["truncatedsvd0", "motor_label"]
    np.testing.assert_array_equal(reduced_train.ids, train.ids)
    np.testing.assert_array_equal(reduced_train.target, train.target)
    np.testing.assert_array_equal(reduced_train.descriptor_features, train.descriptor_features)
    np.testing.assert_allclose(reduced_test.map_features, reducer.transform(test.map_features))
    assert reduced_train.provenance["map_reductions"][0]["reducer"] == "TruncatedSVD"
    assert "map_reductions" not in dataset.provenance


def test_transform_maps_requires_a_fitted_reducer(ma_feature_dataset):
    """Transforming held-out data with an unfitted reducer would fit on it."""
    with pytest.raises(NotFittedError, match="fit_transform_maps"):
        ma_feature_dataset.transform_maps(TruncatedSVD(n_components=1))


def test_fit_transform_maps_rejects_row_changes(ma_feature_dataset):
    """A reducer that drops rows breaks the alignment everything else relies on."""
    dropper = FunctionTransformer(lambda X: X[:-1])

    with pytest.raises(ValueError, match="must preserve the analysis rows"):
        ma_feature_dataset.fit_transform_maps(dropper)


@pytest.mark.parametrize(
    ("method", "kwargs", "expected_type"),
    [
        ("variance_threshold", {"threshold": 0.0}, VarianceThreshold),
        ("truncated_svd", {"n_components": 2}, TruncatedSVD),
    ],
)
def test_make_map_reducer_builds_sklearn_transformers(method, kwargs, expected_type):
    """Each named workflow returns an unfitted scikit-learn transformer."""
    reducer = make_map_reducer(method, **kwargs)

    assert isinstance(reducer, expected_type)
    assert clone(reducer) is not reducer


def test_make_map_reducer_rejects_unknown_workflows():
    """A typo names the workflows that do exist."""
    with pytest.raises(ValueError, match="Unknown map reducer"):
        make_map_reducer("pca")


def test_make_map_reducer_atlas_requires_source_masker():
    """Atlas aggregation needs the masker that defines the voxel order."""
    with pytest.raises(ValueError, match="source masker"):
        make_map_reducer("atlas_aggregation", atlas_masker=NiftiLabelsMasker(labels_img=None))


@pytest.mark.parametrize("atlas", ["labels", "maps"])
def test_atlas_aggregator_matches_nilearn(small_masker, atlas):
    """Aggregation gives what the nilearn masker gives, batch by batch."""
    n_voxels = int(small_masker.mask_img.get_fdata().sum())
    features = sparse.csr_matrix(np.arange(6 * n_voxels, dtype=float).reshape(6, n_voxels))
    affine = small_masker.mask_img.affine

    if atlas == "labels":
        labels = np.zeros((4, 4, 4), dtype=np.int16)
        labels[0, :2, :2] = 1
        labels[1, :2, :2] = 2
        atlas_masker = NiftiLabelsMasker(
            labels_img=nib.Nifti1Image(labels, affine), resampling_target="data", reports=False
        )
    else:
        maps = np.zeros((4, 4, 4, 2), dtype=float)
        maps[0, :2, :2, 0] = 1.0
        maps[1, :2, :2, 1] = 1.0
        atlas_masker = NiftiMapsMasker(
            maps_img=nib.Nifti1Image(maps, affine), resampling_target="data", reports=False
        )

    reducer = clone(
        make_map_reducer(
            "atlas_aggregation", masker=small_masker, atlas_masker=atlas_masker, batch_size=2
        )
    )
    transformed = reducer.fit_transform(features)

    reference = clone(atlas_masker)
    reference.set_params(mask_img=small_masker.mask_img)
    expected = reference.fit(small_masker.mask_img).transform(
        small_masker.inverse_transform(features.toarray())
    )

    assert transformed.shape == (6, 2)
    np.testing.assert_allclose(transformed, expected)
    assert len(reducer.get_feature_names_out()) == 2
    # The caller's masker is cloned before it is fitted in the source mask's space.
    assert atlas_masker.mask_img is None


def test_atlas_aggregator_requires_both_maskers(small_masker):
    """Neither masker can be guessed."""
    with pytest.raises(ValueError, match="atlas_masker"):
        AtlasAggregator(masker=small_masker).fit(np.zeros((2, 8)))

    labels_img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)
    with pytest.raises(ValueError, match="voxel order"):
        AtlasAggregator(atlas_masker=NiftiLabelsMasker(labels_img=labels_img)).fit(
            np.zeros((2, 8))
        )


def test_atlas_aggregator_names_reduced_features(ma_feature_dataset, small_masker):
    """Region names survive into the reduced dataset's feature names."""
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[:2, :2, :2] = 1
    atlas_masker = NiftiLabelsMasker(
        labels_img=nib.Nifti1Image(labels, small_masker.mask_img.affine),
        resampling_target="data",
        reports=False,
    )
    dataset = MAFeatureDataset(
        sparse.csr_matrix(np.random.default_rng(0).random((4, 8))),
        ids=[f"study_{idx}-task0" for idx in range(4)],
        study_ids=[f"study_{idx}" for idx in range(4)],
        masker=small_masker,
    )

    reduced = dataset.fit_transform_maps(
        make_map_reducer("atlas_aggregation", masker=small_masker, atlas_masker=atlas_masker)
    )

    assert reduced.map_features.shape == (4, 1)
    assert reduced.feature_names == ["1"]


# ---------------------------------------------------------------- extractor


def test_extractor_stores_its_configuration():
    """Parameters are stored, and the extractor is not an sklearn estimator."""
    kernel_transformer = MKDAKernel(r=4)
    extractor = MAFeatureExtractor(
        kernel_transformer=kernel_transformer,
        descriptor_fields=["motor_label"],
        target_field=("annotations", "target_score"),
    )

    assert extractor.kernel_transformer is kernel_transformer
    assert extractor.descriptor_fields == ["motor_label"]
    assert extractor.target_field == ("annotations", "target_score")
    assert extractor.missing_coordinates == "drop"
    assert extractor.missing_values == "raise"
    assert extractor.cache_maps is True
    assert extractor.memory is None
    assert not hasattr(extractor, "fit")
    assert not hasattr(extractor, "fit_transform")


def test_extractor_transform(ml_studyset):
    """A Studyset becomes one row per analysis, with everything aligned."""
    studyset = ml_studyset
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4),
        descriptor_fields=["sample_sizes", ("annotations", "motor_label")],
        target_field=("annotations", "target_score"),
    )

    dataset = extractor.transform(studyset)

    np.testing.assert_array_equal(dataset.ids, studyset.ids)
    np.testing.assert_array_equal(dataset.study_ids, studyset.metadata["study_id"])
    assert sparse.issparse(dataset.map_features)
    assert dataset.map_features.shape == (len(studyset.ids), studyset.masker.n_elements_)
    assert dataset.feature_names[-2:] == ["sample_sizes", "motor_label"]
    assert "target_score" not in dataset.feature_names

    annotations = studyset.annotations_df.set_index("id").loc[dataset.ids]
    np.testing.assert_array_equal(dataset.target, annotations["target_score"])
    np.testing.assert_array_equal(
        _column(dataset.features, dataset.feature_names.index("motor_label")),
        annotations["motor_label"],
    )
    np.testing.assert_array_equal(
        _column(dataset.features, dataset.feature_names.index("sample_sizes")),
        [20 + idx for idx in range(len(studyset.ids))],
    )

    assert_sklearn_bunch_valid(
        dataset.to_sklearn(), expected_rows=len(studyset.ids), expected_sparse=True
    )


def test_extractor_to_sklearn(ml_studyset):
    """The one-call export returns the same arrays the container would."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4), target_field=("annotations", "target_score")
    )

    bunch = extractor.to_sklearn(ml_studyset)
    data, target = extractor.to_sklearn(ml_studyset, return_X_y=True)

    assert_sklearn_bunch_valid(bunch, expected_rows=len(ml_studyset.ids), expected_sparse=True)
    assert data.shape == bunch.data.shape
    np.testing.assert_array_equal(target, bunch.target)


def test_extractor_records_provenance(ml_studyset):
    """Every row can be traced back to its Studyset and its settings."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4), descriptor_fields=["motor_label"]
    )

    provenance = extractor.transform(ml_studyset).provenance

    assert provenance["studyset_id"] == ml_studyset.id
    assert provenance["studyset_name"] == ml_studyset.name
    assert provenance["n_rows"] == len(ml_studyset.ids)
    assert provenance["kernel_transformer"]["class"] == "MKDAKernel"
    assert provenance["kernel_transformer"]["params"]["r"] == 4.0
    assert provenance["missing_coordinates"] == "drop"
    assert provenance["dropped_ids"] == []
    assert provenance["descriptor_fields"] == ["motor_label"]
    assert provenance["masker"] == ml_studyset.masker.__class__.__name__


def test_extractor_aligns_maps_by_id_not_position(ml_studyset):
    """Map rows follow the analysis they came from, whatever order the view is in.

    Kernel transformers return maps ordered by analysis id and drop the ids that
    name them, while ``Studyset.select_analyses`` accepts positions and so can
    hand back a view in any order.
    """
    extractor = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4))
    reference = extractor.transform(ml_studyset)
    expected = dict(zip(reference.ids, _map_signature(reference)))

    positions = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    reordered = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4)).transform(
        ml_studyset.select_analyses(positions)
    )

    np.testing.assert_array_equal(reordered.ids, ml_studyset.ids[positions])
    assert _map_signature(reordered) == [expected[id_] for id_ in reordered.ids]


def test_extractor_rejects_duplicate_analysis_ids(ml_studyset):
    """Duplicate ids would collapse into one map row without being noticed."""
    doubled = ml_studyset.merge(_build_ml_studyset(coordinate_offset=1.0))
    extractor = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4))

    # The merge keeps one analysis per id, so build the duplicate explicitly.
    duplicated = doubled.select_analyses(np.arange(len(doubled.ids) + 1) % len(doubled.ids))

    with pytest.raises(ValueError, match="must be unique"):
        extractor.transform(duplicated)


@pytest.mark.parametrize("missing_coordinates", ["drop", "include"])
def test_extractor_missing_coordinates(ml_studyset, missing_coordinates):
    """Coordinate-less analyses are dropped, or kept as all-zero map rows."""
    missing_id = "study_2-task0"
    studyset = _build_ml_studyset(ids_without_points={missing_id})
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4),
        descriptor_fields=["motor_label"],
        target_field=("annotations", "target_score"),
        missing_coordinates=missing_coordinates,
    )

    dataset = extractor.transform(studyset)

    if missing_coordinates == "drop":
        assert missing_id not in set(dataset.ids)
        assert len(dataset) == len(studyset.ids) - 1
        assert dataset.provenance["dropped_ids"] == [missing_id]
    else:
        assert len(dataset) == len(studyset.ids)
        assert dataset.provenance["dropped_ids"] == []
        row = int(np.flatnonzero(dataset.ids == missing_id)[0])
        assert dataset.map_features[row].nnz == 0
        assert dataset.map_features[row - 1].nnz > 0

    annotations = studyset.annotations_df.set_index("id").loc[dataset.ids]
    np.testing.assert_array_equal(dataset.target, annotations["target_score"])
    np.testing.assert_array_equal(dataset.descriptor_features.ravel(), annotations["motor_label"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"missing_coordinates": "maybe"}, "missing_coordinates must be"),
        ({"missing_values": "maybe"}, "missing_values must be"),
    ],
)
def test_extractor_rejects_unknown_option_values(ml_studyset, kwargs, message):
    """Option vocabularies are checked before any work is done."""
    extractor = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4), **kwargs)

    with pytest.raises(ValueError, match=message):
        extractor.transform(ml_studyset)


@pytest.mark.parametrize("missing_values", ["raise", "drop", "keep"])
def test_extractor_missing_descriptor_values(missing_values):
    """Missing values are reported, removed or left for a pipeline to impute."""
    missing_id = "study_3-task1"
    studyset = _build_ml_studyset(ids_without_score={missing_id})
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4),
        descriptor_fields=["score"],
        missing_values=missing_values,
    )

    if missing_values == "raise":
        with pytest.raises(ValueError, match=f"Missing values in score .*{missing_id}"):
            extractor.transform(studyset)
        return

    dataset = extractor.transform(studyset)
    if missing_values == "drop":
        assert missing_id not in set(dataset.ids)
        assert len(dataset) == len(studyset.ids) - 1
        assert dataset.provenance["missing_value_ids"] == {"score": [missing_id]}
        assert np.isfinite(dataset.descriptor_features).all()
    else:
        assert len(dataset) == len(studyset.ids)
        row = int(np.flatnonzero(dataset.ids == missing_id)[0])
        assert np.isnan(dataset.descriptor_features[row, 0])
        assert dataset.provenance["missing_value_ids"] == {"score": [missing_id]}


def test_extractor_missing_target_values_are_reported():
    """A missing target is named the same way a missing descriptor is."""
    studyset = _build_ml_studyset(ids_without_score={"study_1-task0"})
    extractor = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4), target_field="score")

    with pytest.raises(ValueError, match="study_1-task0"):
        extractor.transform(studyset)


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("comparison_task", "is categorical"),
        ("abstract", "is text"),
        ("not_a_field", "was not found in the Studyset metadata"),
        (("annotations", "sample_sizes"), "was not found in the Studyset annotations"),
        (("nowhere", "motor_label"), "Unsupported field selector source"),
        (42, "must be a field name"),
    ],
)
def test_extractor_rejects_unusable_descriptors(ml_studyset, selector, message):
    """Descriptor fields have to be numeric, and have to exist."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4), descriptor_fields=[selector]
    )

    with pytest.raises((ValueError, TypeError), match=message):
        extractor.transform(ml_studyset)


def test_extractor_rejects_repeated_descriptor_fields(ml_studyset):
    """The same field twice is a mistake, not two features."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4), descriptor_fields=["motor_label", "motor_label"]
    )

    with pytest.raises(ValueError, match="selected more than once"):
        extractor.transform(ml_studyset)


def test_extractor_reports_ambiguous_field_names(ml_studyset):
    """A name in two sources asks which one was meant."""
    annotated = ml_studyset.with_metadata("motor_label", np.ones(len(ml_studyset.ids)))
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4), descriptor_fields=["motor_label"]
    )

    with pytest.raises(ValueError, match="is ambiguous"):
        extractor.transform(annotated)


def test_extractor_reads_study_level_and_list_metadata(ml_studyset):
    """Study-level fields are inherited and list-valued fields are reduced."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4), descriptor_fields=["year", "sample_sizes"]
    )

    dataset = extractor.transform(ml_studyset)

    years = ml_studyset.metadata.set_index("id").loc[dataset.ids, "year"]
    np.testing.assert_array_equal(dataset.descriptor_features[:, 0], years)
    np.testing.assert_array_equal(
        dataset.descriptor_features[:, 1], [20 + idx for idx in range(len(dataset))]
    )


def test_extractor_exports_categorical_targets(ml_studyset):
    """A categorical target reaches y as labels, ready for a classifier."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4), target_field="comparison_task"
    )

    dataset = extractor.transform(ml_studyset)

    assert sorted(set(dataset.target)) == ["flanker", "n-back"]
    np.testing.assert_array_equal(
        dataset.target, ml_studyset.metadata.set_index("id").loc[dataset.ids, "comparison_task"]
    )


def test_extractor_applies_a_target_transformer(ml_studyset):
    """A label extractor turns a text field into one label per analysis."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4),
        target_field=("texts", "abstract"),
        target_transformer=lambda values: np.array(
            [text.split()[1] for text in values], dtype=str
        ),
    )

    dataset = extractor.transform(ml_studyset)

    np.testing.assert_array_equal(dataset.target, [str(idx) for idx in range(len(dataset))])


def test_extractor_target_transformer_may_be_a_transformer(ml_studyset):
    """A stateless scikit-learn transformer works as a label extractor too."""
    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=4),
        target_field=("annotations", "target_score"),
        target_transformer=FunctionTransformer(lambda values: np.round(values)),
    )

    dataset = extractor.transform(ml_studyset)

    np.testing.assert_array_equal(dataset.target, np.round(np.arange(8) + 0.5))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_field": ("texts", "abstract")}, "free text"),
        ({"target_field": "year", "target_transformer": object()}, "must be callable"),
    ],
)
def test_extractor_rejects_unusable_targets(ml_studyset, kwargs, message):
    """A target has to be one usable value per analysis."""
    extractor = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4), **kwargs)

    with pytest.raises((ValueError, TypeError), match=message):
        extractor.transform(ml_studyset)


def test_extractor_rejects_constant_targets(ml_studyset):
    """A target with one value has nothing to predict."""
    constant = ml_studyset.with_metadata("only_value", np.ones(len(ml_studyset.ids)))
    extractor = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4), target_field="only_value")

    with pytest.raises(ValueError, match="nothing to predict"):
        extractor.transform(constant)


def test_extractor_reuses_generated_maps(ml_studyset):
    """Comparing reducers over one Studyset generates the maps once."""

    class CountingKernel(MKDAKernel):
        calls = 0

        def transform(self, studyset, masker=None, return_type="image"):
            type(self).calls += 1
            return super().transform(studyset, masker=masker, return_type=return_type)

    extractor = MAFeatureExtractor(kernel_transformer=CountingKernel(r=4))
    first = extractor.transform(ml_studyset)
    second = extractor.transform(ml_studyset)

    assert CountingKernel.calls == 1
    np.testing.assert_array_equal(first.map_features.toarray(), second.map_features.toarray())

    # Moved coordinates and changed kernel parameters both invalidate the memo.
    extractor.transform(_build_ml_studyset(coordinate_offset=6.0))
    assert CountingKernel.calls == 2

    extractor.kernel_transformer = CountingKernel(r=8)
    extractor.transform(ml_studyset)
    assert CountingKernel.calls == 3

    without_memo = MAFeatureExtractor(kernel_transformer=CountingKernel(r=4), cache_maps=False)
    without_memo.transform(ml_studyset)
    without_memo.transform(ml_studyset)
    assert CountingKernel.calls == 5


def test_extractor_memory_is_passed_to_the_kernel(ml_studyset, tmp_path):
    """A cache location is wired up without touching the caller's kernel."""
    kernel_transformer = MKDAKernel(r=4)
    extractor = MAFeatureExtractor(
        kernel_transformer=kernel_transformer, memory=str(tmp_path), memory_level=1
    )

    extractor.transform(ml_studyset)

    assert any(tmp_path.iterdir())
    assert kernel_transformer.memory.location is None
    assert kernel_transformer.memory_level == 0


def test_extractor_accepts_a_kernel_class(ml_studyset):
    """A kernel transformer may be given as a class, as elsewhere in NiMARE."""
    dataset = MAFeatureExtractor(kernel_transformer=MKDAKernel).transform(ml_studyset)

    assert dataset.map_features.shape[0] == len(ml_studyset.ids)
    assert dataset.provenance["kernel_transformer"]["class"] == "MKDAKernel"


def test_extractor_rejects_an_empty_studyset(ml_studyset):
    """There is nothing to convert without analyses."""
    empty = ml_studyset.select_analyses(np.zeros(len(ml_studyset.ids), dtype=bool))

    with pytest.raises(ValueError, match="no analyses"):
        MAFeatureExtractor(kernel_transformer=MKDAKernel(r=4)).transform(empty)


def test_extractor_end_to_end_classification(ml_studyset):
    """The documented workflow runs from Studyset to grouped cross-validation."""
    from sklearn.linear_model import LogisticRegression

    extractor = MAFeatureExtractor(
        kernel_transformer=MKDAKernel(r=10), target_field="comparison_task"
    )
    dataset = extractor.transform(ml_studyset)
    bunch = dataset.to_sklearn()

    pipeline = make_pipeline(
        dataset.make_preprocessor("truncated_svd", n_components=2, random_state=RANDOM_SEED),
        LogisticRegression(max_iter=500),
    )
    scores = cross_val_score(
        pipeline, bunch.data, bunch.target, cv=GroupKFold(n_splits=4), groups=bunch.groups
    )

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


@pytest.mark.performance_smoke
def test_extractor_meets_the_conversion_budget():
    """A 1,000-study Studyset converts and splits inside the documented budget."""
    # resource is Unix-only, and importing it at module scope makes the whole module
    # uncollectable on Windows. Only this test needs it, and it runs on Linux.
    import resource

    _, studyset = create_coordinate_studyset(foci=5, n_studies=1000, sample_size=30, seed=42)

    start = time.time()
    dataset = MAFeatureExtractor(kernel_transformer=MKDAKernel(r=10)).transform(studyset)
    train, test = dataset.split(test_size=0.25, random_state=RANDOM_SEED)
    elapsed = time.time() - start
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    assert len(dataset) == len(studyset.ids)
    assert set(train.study_ids).isdisjoint(test.study_ids)
    assert sparse.issparse(dataset.features)
    assert elapsed <= 180, f"conversion and split took {elapsed:.0f}s"
    assert peak_gb <= 5, f"peak memory was {peak_gb:.1f} GB"
