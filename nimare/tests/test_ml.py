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
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import Bunch

from nimare import ml
from nimare.generate import create_coordinate_studyset
from nimare.meta.kernel import MKDAKernel
from nimare.ml import AtlasAggregator, FeatureSet
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
    """Build a small FeatureSet directly, without running a kernel."""
    n_rows = 6
    ids = np.array([f"study_{idx // 2}-task{idx % 2}" for idx in range(n_rows)])
    study_ids = np.array([f"study_{idx // 2}" for idx in range(n_rows)])
    target = np.arange(n_rows, dtype=float) + 0.5
    map_features = sparse.csr_matrix(
        np.column_stack([np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) * 2.0])
    )
    descriptor_features = np.arange(n_rows, dtype=float)[:, None]

    return FeatureSet(
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
    dataset = FeatureSet(map_features, ids=list("abc"), study_ids=list("abc"))

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
        FeatureSet(map_features, **base)


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
        TruncatedSVD(n_components=1, random_state=RANDOM_SEED)
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
    preprocessor = dataset.make_preprocessor(VarianceThreshold(threshold=0.0))

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

    with pytest.raises(ValueError, match="only used when the reducer is given as a class"):
        dataset.make_preprocessor(TruncatedSVD(), n_components=1)


def test_dataset_works_in_sklearn_model_selection(ma_feature_dataset):
    """The exported arrays drive grouped cross-validation and a grid search."""
    dataset = ma_feature_dataset
    pipeline = make_pipeline(
        dataset.make_preprocessor(TruncatedSVD(n_components=1, random_state=RANDOM_SEED)),
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
    reducer = TruncatedSVD(n_components=1, random_state=RANDOM_SEED)

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
    ("reducer", "kwargs", "expected_type"),
    [
        (VarianceThreshold(threshold=0.0), {}, VarianceThreshold),
        (TruncatedSVD(n_components=2), {}, TruncatedSVD),
        (SparseRandomProjection, {"n_components": 2}, SparseRandomProjection),
    ],
)
def test_make_preprocessor_takes_scikit_learn_transformers(
    ma_feature_dataset, reducer, kwargs, expected_type
):
    """A transformer, or a transformer class plus its parameters, both resolve."""
    preprocessor = ma_feature_dataset.make_preprocessor(reducer, **kwargs)

    built = preprocessor.transformers[0][1]
    assert isinstance(built, expected_type)
    for name, value in kwargs.items():
        assert built.get_params()[name] == value


def test_make_preprocessor_uses_a_built_transformer_as_given(ma_feature_dataset):
    """An instance is used as it is, and cannot be reconfigured in passing."""
    reducer = SparseRandomProjection(n_components=3)

    assert ma_feature_dataset.make_preprocessor(reducer).transformers[0][1] is reducer

    with pytest.raises(ValueError, match="only used when the reducer is given as a class"):
        ma_feature_dataset.make_preprocessor(reducer, n_components=4)


def test_make_preprocessor_rejects_things_that_are_not_reducers(ma_feature_dataset):
    """The message names what a reducer can be, including the workflows it replaced."""
    with pytest.raises(TypeError, match="TruncatedSVD"):
        ma_feature_dataset.make_preprocessor("truncated_svd")

    with pytest.raises(TypeError, match="is not a map reducer"):
        ma_feature_dataset.make_preprocessor(object())


def test_map_only_features_need_no_preprocessor(small_masker):
    """With nothing to keep the reducer away from, the reducer is returned as it is."""
    map_only = FeatureSet(
        sparse.csr_matrix(np.eye(4)),
        ids=[f"s{idx}-t" for idx in range(4)],
        study_ids=[f"s{idx}" for idx in range(4)],
        masker=small_masker,
    )
    reducer = TruncatedSVD(n_components=2)

    assert map_only.make_preprocessor(reducer) is reducer


def test_atlas_reducer_takes_the_maskers_voxel_order_from_the_feature_set(
    ma_feature_dataset, small_masker
):
    """An atlas, or an aggregator built without one, gets the feature set's masker."""
    atlas = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)

    from_atlas = ma_feature_dataset.make_preprocessor(atlas).transformers[0][1]
    unbound = AtlasAggregator(atlas=atlas)
    from_aggregator = ma_feature_dataset.make_preprocessor(unbound).transformers[0][1]

    assert from_atlas.masker is ma_feature_dataset.masker
    assert from_aggregator.masker is ma_feature_dataset.masker
    assert unbound.masker is None  # the caller's object is left alone


def test_atlas_reduction_needs_a_masker_somewhere(small_masker):
    """A feature set without a masker cannot place an atlas over its columns."""
    atlas = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)
    maskerless = FeatureSet(
        sparse.csr_matrix(np.eye(4)),
        ids=[f"s{idx}-t" for idx in range(4)],
        study_ids=[f"s{idx}" for idx in range(4)],
    )

    with pytest.raises(ValueError, match="voxel order"):
        maskerless.make_preprocessor(atlas)


def test_fit_transform_maps_asks_for_a_built_aggregator(ma_feature_dataset, small_masker):
    """An atlas has to become an aggregator first, so the fitted one can be reused."""
    atlas = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)

    with pytest.raises(TypeError, match="AtlasAggregator"):
        ma_feature_dataset.fit_transform_maps(atlas)


def _atlas_images(affine):
    """Return a 3D labels atlas and an equivalent 4D probabilistic atlas."""
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[0, :2, :2] = 1
    labels[1, :2, :2] = 2

    maps = np.zeros((4, 4, 4, 2), dtype=float)
    maps[0, :2, :2, 0] = 1.0
    maps[1, :2, :2, 1] = 1.0

    return nib.Nifti1Image(labels, affine), nib.Nifti1Image(maps, affine)


@pytest.fixture
def atlas_features(small_masker):
    """Return sparse map features over the small mask."""
    n_voxels = int(small_masker.mask_img.get_fdata().sum())
    return sparse.csr_matrix(np.arange(6 * n_voxels, dtype=float).reshape(6, n_voxels))


@pytest.mark.parametrize("atlas_kind", ["labels", "maps"])
def test_atlas_aggregator_matches_nilearn(small_masker, atlas_features, atlas_kind):
    """Aggregation gives what the nilearn masker gives, batch by batch."""
    labels_img, maps_img = _atlas_images(small_masker.mask_img.affine)

    if atlas_kind == "labels":
        atlas_img = labels_img
        reference = NiftiLabelsMasker(
            labels_img=labels_img, resampling_target="data", reports=False
        )
    else:
        atlas_img = maps_img
        reference = NiftiMapsMasker(maps_img=maps_img, resampling_target="data", reports=False)

    reducer = clone(AtlasAggregator(atlas=atlas_img, masker=small_masker, batch_size=2))
    transformed = reducer.fit_transform(atlas_features)

    reference.set_params(mask_img=small_masker.mask_img)
    expected = reference.fit(small_masker.mask_img).transform(
        small_masker.inverse_transform(atlas_features.toarray())
    )

    assert transformed.shape == (6, 2)
    np.testing.assert_allclose(transformed, expected)
    assert len(reducer.get_feature_names_out()) == 2


def test_atlas_aggregator_accepts_a_fetched_atlas(small_masker, atlas_features):
    """A Bunch from a nilearn fetcher is read for its maps and its labels."""
    _, maps_img = _atlas_images(small_masker.mask_img.affine)
    atlas = Bunch(maps=maps_img, labels=["Background", "left", "right"])

    reducer = AtlasAggregator(atlas=atlas, masker=small_masker).fit(atlas_features)

    assert isinstance(reducer.atlas_masker_, NiftiMapsMasker)
    np.testing.assert_array_equal(reducer.get_feature_names_out(), ["left", "right"])


def test_atlas_aggregator_accepts_a_labels_frame(small_masker, atlas_features):
    """DiFuMo-style label frames are read for their name column."""
    _, maps_img = _atlas_images(small_masker.mask_img.affine)
    labels = pd.DataFrame({"component": [1, 2], "difumo_names": ["first", "second"]})

    reducer = AtlasAggregator(atlas=Bunch(maps=maps_img, labels=labels), masker=small_masker)

    np.testing.assert_array_equal(
        reducer.fit(atlas_features).get_feature_names_out(), ["first", "second"]
    )


def test_atlas_aggregator_accepts_a_path(small_masker, atlas_features, tmp_path):
    """An atlas on disk is loaded rather than refused."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    path = tmp_path / "atlas.nii.gz"
    labels_img.to_filename(path)

    for atlas in (path, str(path)):
        reducer = AtlasAggregator(atlas=atlas, masker=small_masker)
        assert reducer.fit_transform(atlas_features).shape == (6, 2)
        assert isinstance(reducer.atlas_masker_, NiftiLabelsMasker)


def test_atlas_aggregator_accepts_a_fetcher_name(small_masker, atlas_features, monkeypatch):
    """A nilearn fetcher can be named, and its arguments passed through."""
    from nilearn import datasets

    _, maps_img = _atlas_images(small_masker.mask_img.affine)
    calls = {}

    def fake_fetcher(dimension=None):
        calls["dimension"] = dimension
        return Bunch(maps=maps_img, labels=["left", "right"])

    monkeypatch.setattr(datasets, "fetch_atlas_pretend", fake_fetcher, raising=False)

    reducer = AtlasAggregator(
        atlas="pretend", masker=small_masker, atlas_kwargs={"dimension": 2}
    ).fit(atlas_features)

    assert calls == {"dimension": 2}
    np.testing.assert_array_equal(reducer.get_feature_names_out(), ["left", "right"])


def test_atlas_aggregator_accepts_a_prebuilt_masker(small_masker, atlas_features):
    """A masker built by the caller is used as configured, and not modified."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    atlas_masker = NiftiLabelsMasker(
        labels_img=labels_img,
        # Named explicitly: left unnamed, nilearn invents a name per region and the
        # convention has changed between 0.12 ("0", the position) and 0.13 ("1", the
        # label value), which would pin the assertion below to one nilearn version.
        labels=["Background", "region_a", "region_b"],
        strategy="sum",
        resampling_target="data",
        reports=False,
    )

    reducer = AtlasAggregator(atlas=atlas_masker, masker=small_masker).fit(atlas_features)

    assert reducer.atlas_masker_.strategy == "sum"
    assert atlas_masker.mask_img is None
    np.testing.assert_array_equal(reducer.get_feature_names_out(), ["region_a", "region_b"])


@pytest.mark.parametrize(
    ("atlas", "message"),
    [
        (None, "requires an atlas"),
        ("not_an_atlas_anywhere", "neither a file nor a nilearn atlas fetcher"),
        (42, "is not an atlas"),
    ],
)
def test_atlas_aggregator_rejects_unusable_atlases(small_masker, atlas, message):
    """Whatever the atlas is not, the message says what it could be."""
    with pytest.raises((ValueError, TypeError), match=message):
        AtlasAggregator(atlas=atlas, masker=small_masker).fit(np.zeros((2, 8)))


def test_atlas_aggregator_rejects_a_voxel_masker(small_masker):
    """A NiftiMasker extracts voxels, which is not an aggregation."""
    with pytest.raises(ValueError, match="rather than regions"):
        AtlasAggregator(atlas=small_masker, masker=small_masker).fit(np.zeros((2, 8)))


def test_atlas_aggregator_requires_the_source_masker(small_masker):
    """The voxel order of the incoming features cannot be guessed."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)

    with pytest.raises(ValueError, match="voxel order"):
        AtlasAggregator(atlas=labels_img).fit(np.zeros((2, 8)))


@pytest.mark.parametrize("atlas_kind", ["labels", "maps"])
def test_atlas_aggregator_names_match_the_columns_it_returns(
    small_masker, atlas_features, atlas_kind
):
    """Names follow the regions nilearn actually returns, not the ones it was given.

    nilearn 0.12 keeps a region that falls outside the mask and 0.13 drops it,
    and for a maps atlas 0.13 drops it from the output without dropping it from
    ``maps_img_`` or ``n_elements_``. Counting the columns is the only reading
    that holds on both.
    """
    affine = small_masker.mask_img.affine
    if atlas_kind == "labels":
        labels = np.zeros((4, 4, 4), dtype=np.int16)
        labels[0, :2, :2] = 1
        labels[1, :2, :2] = 2
        labels[3, 3, 3] = 3  # outside the mask
        atlas = Bunch(maps=nib.Nifti1Image(labels, affine), labels=["in_a", "in_b", "outside"])
    else:
        maps = np.zeros((4, 4, 4, 3), dtype=float)
        maps[0, :2, :2, 0] = 1.0
        maps[1, :2, :2, 1] = 1.0
        maps[3, 3, 3, 2] = 1.0  # outside the mask
        atlas = Bunch(maps=nib.Nifti1Image(maps, affine), labels=["in_a", "in_b", "outside"])

    reducer = AtlasAggregator(atlas=atlas, masker=small_masker).fit(atlas_features)
    names = reducer.get_feature_names_out()

    assert len(names) == reducer.transform(atlas_features).shape[1]


def test_atlas_aggregator_names_reduced_features(small_masker, atlas_features):
    """Region names survive into the reduced dataset's feature names."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    dataset = FeatureSet(
        atlas_features,
        ids=[f"study_{idx}-task0" for idx in range(6)],
        study_ids=[f"study_{idx}" for idx in range(6)],
        masker=small_masker,
    )

    reduced = dataset.fit_transform_maps(
        AtlasAggregator(atlas=Bunch(maps=labels_img, labels=["one", "two"]), masker=small_masker)
    )

    assert reduced.map_features.shape == (6, 2)
    assert reduced.feature_names == ["one", "two"]
    assert all(type(name) is str for name in reduced.feature_names)


def test_dataset_reduces_with_any_sklearn_transformer(ma_feature_dataset):
    """A reducer NiMARE has never heard of works like the named ones."""
    reducer = SparseRandomProjection(n_components=1, random_state=RANDOM_SEED)
    train, test = ma_feature_dataset.split(test_size=0.34, random_state=RANDOM_SEED)

    reduced_train = train.fit_transform_maps(reducer)
    reduced_test = test.transform_maps(reducer)

    assert reduced_train.map_features.shape == (len(train), 1)
    assert reduced_test.map_features.shape == (len(test), 1)
    np.testing.assert_array_equal(reduced_train.ids, train.ids)


def test_make_preprocessor_accepts_an_atlas(small_masker, atlas_features):
    """The feature set supplies the voxel order, so an atlas needs nothing else."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    dataset = FeatureSet(
        atlas_features,
        ids=[f"study_{idx}-task0" for idx in range(6)],
        study_ids=[f"study_{idx}" for idx in range(6)],
        masker=small_masker,
    )

    # Map-only features: there is nothing to keep the aggregator away from.
    reducer = dataset.make_preprocessor(labels_img)

    assert isinstance(reducer, AtlasAggregator)
    assert reducer.masker is small_masker
    assert reducer.fit_transform(dataset.features).shape == (6, 2)


# ------------------------------------------------------------------ extraction


def test_public_surface_is_one_container_and_its_helpers():
    """Users meet one container; the class that converts a Studyset is internal."""
    assert set(ml.__all__) == {"AtlasAggregator", "FeatureSet"}
    assert not any(name.endswith("Extractor") for name in dir(ml) if not name.startswith("_"))
    # The container is data, not an estimator: nothing here is fitted on a Studyset.
    assert not hasattr(FeatureSet, "fit")
    assert not hasattr(FeatureSet, "fit_transform")


def test_from_studyset(ml_studyset):
    """A Studyset becomes one row per analysis, with everything aligned."""
    studyset = ml_studyset

    features = FeatureSet.from_studyset(
        studyset,
        MKDAKernel(r=4),
        descriptor_fields=["sample_sizes", ("annotations", "motor_label")],
        target_field=("annotations", "target_score"),
    )

    np.testing.assert_array_equal(features.ids, studyset.ids)
    np.testing.assert_array_equal(features.study_ids, studyset.metadata["study_id"])
    assert sparse.issparse(features.map_features)
    assert features.map_features.shape == (len(studyset.ids), studyset.masker.n_elements_)
    assert features.feature_names[-2:] == ["sample_sizes", "motor_label"]
    assert "target_score" not in features.feature_names

    annotations = studyset.annotations_df.set_index("id").loc[features.ids]
    np.testing.assert_array_equal(features.target, annotations["target_score"])
    np.testing.assert_array_equal(
        _column(features.features, features.feature_names.index("motor_label")),
        annotations["motor_label"],
    )
    np.testing.assert_array_equal(
        _column(features.features, features.feature_names.index("sample_sizes")),
        [20 + idx for idx in range(len(studyset.ids))],
    )

    assert_sklearn_bunch_valid(
        features.to_sklearn(), expected_rows=len(studyset.ids), expected_sparse=True
    )


def test_from_studyset_to_sklearn(ml_studyset):
    """The whole path from Studyset to scikit-learn arrays is two calls."""
    features = FeatureSet.from_studyset(
        ml_studyset, MKDAKernel(r=4), target_field=("annotations", "target_score")
    )

    bunch = features.to_sklearn()
    data, target = features.to_sklearn(return_X_y=True)

    assert_sklearn_bunch_valid(bunch, expected_rows=len(ml_studyset.ids), expected_sparse=True)
    assert data.shape == bunch.data.shape
    np.testing.assert_array_equal(target, bunch.target)


def test_from_studyset_records_provenance(ml_studyset):
    """Every row can be traced back to its Studyset and its settings."""
    provenance = FeatureSet.from_studyset(
        ml_studyset, MKDAKernel(r=4), descriptor_fields=["motor_label"]
    ).provenance

    assert provenance["studyset_id"] == ml_studyset.id
    assert provenance["studyset_name"] == ml_studyset.name
    assert provenance["n_rows"] == len(ml_studyset.ids)
    assert provenance["kernel_transformer"]["class"] == "MKDAKernel"
    assert provenance["kernel_transformer"]["params"]["r"] == 4.0
    assert provenance["missing_coordinates"] == "drop"
    assert provenance["dropped_ids"] == []
    assert provenance["descriptor_fields"] == ["motor_label"]
    assert provenance["masker"] == ml_studyset.masker.__class__.__name__


def test_from_studyset_aligns_maps_by_id_not_position(ml_studyset):
    """Map rows follow the analysis they came from, whatever order the view is in.

    Kernel transformers return maps ordered by analysis id and drop the ids that
    name them, while ``Studyset.select_analyses`` accepts positions and so can
    hand back a view in any order.
    """
    reference = FeatureSet.from_studyset(ml_studyset, MKDAKernel(r=4))
    expected = dict(zip(reference.ids, _map_signature(reference)))

    positions = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    reordered = FeatureSet.from_studyset(ml_studyset.select_analyses(positions), MKDAKernel(r=4))

    np.testing.assert_array_equal(reordered.ids, ml_studyset.ids[positions])
    assert _map_signature(reordered) == [expected[id_] for id_ in reordered.ids]


def test_from_studyset_rejects_duplicate_analysis_ids(ml_studyset):
    """Duplicate ids would collapse into one map row without being noticed."""
    doubled = ml_studyset.merge(_build_ml_studyset(coordinate_offset=1.0))

    # The merge keeps one analysis per id, so build the duplicate explicitly.
    duplicated = doubled.select_analyses(np.arange(len(doubled.ids) + 1) % len(doubled.ids))

    with pytest.raises(ValueError, match="must be unique"):
        FeatureSet.from_studyset(duplicated, MKDAKernel(r=4))


@pytest.mark.parametrize("missing_coordinates", ["drop", "include"])
def test_from_studyset_missing_coordinates(ml_studyset, missing_coordinates):
    """Coordinate-less analyses are dropped, or kept as all-zero map rows."""
    missing_id = "study_2-task0"
    studyset = _build_ml_studyset(ids_without_points={missing_id})

    features = FeatureSet.from_studyset(
        studyset,
        MKDAKernel(r=4),
        descriptor_fields=["motor_label"],
        target_field=("annotations", "target_score"),
        missing_coordinates=missing_coordinates,
    )

    if missing_coordinates == "drop":
        assert missing_id not in set(features.ids)
        assert len(features) == len(studyset.ids) - 1
        assert features.provenance["dropped_ids"] == [missing_id]
    else:
        assert len(features) == len(studyset.ids)
        assert features.provenance["dropped_ids"] == []
        row = int(np.flatnonzero(features.ids == missing_id)[0])
        assert features.map_features[row].nnz == 0
        assert features.map_features[row - 1].nnz > 0

    annotations = studyset.annotations_df.set_index("id").loc[features.ids]
    np.testing.assert_array_equal(features.target, annotations["target_score"])
    np.testing.assert_array_equal(features.descriptor_features.ravel(), annotations["motor_label"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"missing_coordinates": "maybe"}, "missing_coordinates must be"),
        ({"missing_values": "maybe"}, "missing_values must be"),
    ],
)
def test_from_studyset_rejects_unknown_option_values(ml_studyset, kwargs, message):
    """Option vocabularies are checked before any work is done."""
    with pytest.raises(ValueError, match=message):
        FeatureSet.from_studyset(ml_studyset, MKDAKernel(r=4), **kwargs)


@pytest.mark.parametrize("missing_values", ["raise", "drop", "keep"])
def test_from_studyset_missing_descriptor_values(missing_values):
    """Missing values are reported, removed or left for a pipeline to impute."""
    missing_id = "study_3-task1"
    studyset = _build_ml_studyset(ids_without_score={missing_id})
    call = dict(
        kernel_transformer=MKDAKernel(r=4),
        descriptor_fields=["score"],
        missing_values=missing_values,
    )

    if missing_values == "raise":
        with pytest.raises(ValueError, match=f"Missing values in score .*{missing_id}"):
            FeatureSet.from_studyset(studyset, **call)
        return

    features = FeatureSet.from_studyset(studyset, **call)
    if missing_values == "drop":
        assert missing_id not in set(features.ids)
        assert len(features) == len(studyset.ids) - 1
        assert features.provenance["missing_value_ids"] == {"score": [missing_id]}
        assert np.isfinite(features.descriptor_features).all()
    else:
        assert len(features) == len(studyset.ids)
        row = int(np.flatnonzero(features.ids == missing_id)[0])
        assert np.isnan(features.descriptor_features[row, 0])
        assert features.provenance["missing_value_ids"] == {"score": [missing_id]}


def test_from_studyset_missing_target_values_are_reported():
    """A missing target is named the same way a missing descriptor is."""
    studyset = _build_ml_studyset(ids_without_score={"study_1-task0"})

    with pytest.raises(ValueError, match="study_1-task0"):
        FeatureSet.from_studyset(studyset, MKDAKernel(r=4), target_field="score")


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
def test_from_studyset_rejects_unusable_descriptors(ml_studyset, selector, message):
    """Descriptor fields have to be numeric, and have to exist."""
    with pytest.raises((ValueError, TypeError), match=message):
        FeatureSet.from_studyset(ml_studyset, MKDAKernel(r=4), descriptor_fields=[selector])


def test_from_studyset_rejects_repeated_descriptor_fields(ml_studyset):
    """The same field twice is a mistake, not two features."""
    with pytest.raises(ValueError, match="selected more than once"):
        FeatureSet.from_studyset(
            ml_studyset, MKDAKernel(r=4), descriptor_fields=["motor_label", "motor_label"]
        )


def test_from_studyset_reports_ambiguous_field_names(ml_studyset):
    """A name in two sources asks which one was meant."""
    annotated = ml_studyset.with_metadata("motor_label", np.ones(len(ml_studyset.ids)))

    with pytest.raises(ValueError, match="is ambiguous"):
        FeatureSet.from_studyset(annotated, MKDAKernel(r=4), descriptor_fields=["motor_label"])


def test_from_studyset_reads_study_level_and_list_metadata(ml_studyset):
    """Study-level fields are inherited and list-valued fields are reduced."""
    features = FeatureSet.from_studyset(
        ml_studyset, MKDAKernel(r=4), descriptor_fields=["year", "sample_sizes"]
    )

    years = ml_studyset.metadata.set_index("id").loc[features.ids, "year"]
    np.testing.assert_array_equal(features.descriptor_features[:, 0], years)
    np.testing.assert_array_equal(
        features.descriptor_features[:, 1], [20 + idx for idx in range(len(features))]
    )


def test_from_studyset_exports_categorical_targets(ml_studyset):
    """A categorical target reaches y as labels, ready for a classifier."""
    features = FeatureSet.from_studyset(
        ml_studyset, MKDAKernel(r=4), target_field="comparison_task"
    )

    assert sorted(set(features.target)) == ["flanker", "n-back"]
    np.testing.assert_array_equal(
        features.target,
        ml_studyset.metadata.set_index("id").loc[features.ids, "comparison_task"],
    )


def test_from_studyset_applies_a_target_transformer(ml_studyset):
    """A label extractor turns a text field into one label per analysis."""
    features = FeatureSet.from_studyset(
        ml_studyset,
        MKDAKernel(r=4),
        target_field=("texts", "abstract"),
        target_transformer=lambda values: np.array(
            [text.split()[1] for text in values], dtype=str
        ),
    )

    np.testing.assert_array_equal(features.target, [str(idx) for idx in range(len(features))])


def test_from_studyset_target_transformer_may_be_a_transformer(ml_studyset):
    """A stateless scikit-learn transformer works as a label extractor too."""
    features = FeatureSet.from_studyset(
        ml_studyset,
        MKDAKernel(r=4),
        target_field=("annotations", "target_score"),
        target_transformer=FunctionTransformer(lambda values: np.round(values)),
    )

    np.testing.assert_array_equal(features.target, np.round(np.arange(8) + 0.5))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_field": ("texts", "abstract")}, "free text"),
        ({"target_field": "year", "target_transformer": object()}, "must be callable"),
    ],
)
def test_from_studyset_rejects_unusable_targets(ml_studyset, kwargs, message):
    """A target has to be one usable value per analysis."""
    with pytest.raises((ValueError, TypeError), match=message):
        FeatureSet.from_studyset(ml_studyset, MKDAKernel(r=4), **kwargs)


def test_from_studyset_rejects_constant_targets(ml_studyset):
    """A target with one value has nothing to predict."""
    constant = ml_studyset.with_metadata("only_value", np.ones(len(ml_studyset.ids)))

    with pytest.raises(ValueError, match="nothing to predict"):
        FeatureSet.from_studyset(constant, MKDAKernel(r=4), target_field="only_value")


class _CountingKernel(MKDAKernel):
    """An MKDA kernel that counts how often the maps are really computed."""

    calls = 0

    def _transform(self, mask, coordinates, return_type="sparse"):
        type(self).calls += 1
        return super()._transform(mask, coordinates, return_type=return_type)


def test_from_studyset_caches_maps_with_memory(ml_studyset, tmp_path):
    """A cache location makes a repeated conversion reuse the maps it made."""
    _CountingKernel.calls = 0

    first = FeatureSet.from_studyset(ml_studyset, _CountingKernel(r=4), memory=str(tmp_path))
    second = FeatureSet.from_studyset(ml_studyset, _CountingKernel(r=4), memory=str(tmp_path))

    assert _CountingKernel.calls == 1
    np.testing.assert_array_equal(first.map_features.toarray(), second.map_features.toarray())

    # A different kernel configuration is a different cache entry, not a stale hit.
    FeatureSet.from_studyset(ml_studyset, _CountingKernel(r=8), memory=str(tmp_path))
    assert _CountingKernel.calls == 2

    # Kernel transformers cache their maps at memory_level 2, so a lower level asks
    # for no caching at all.
    _CountingKernel.calls = 0
    FeatureSet.from_studyset(
        ml_studyset, _CountingKernel(r=4), memory=str(tmp_path), memory_level=1
    )
    FeatureSet.from_studyset(
        ml_studyset, _CountingKernel(r=4), memory=str(tmp_path), memory_level=1
    )
    assert _CountingKernel.calls == 2


def test_from_studyset_without_memory_does_not_cache(ml_studyset):
    """Without a cache location every conversion generates its own maps."""
    _CountingKernel.calls = 0

    FeatureSet.from_studyset(ml_studyset, _CountingKernel(r=4))
    FeatureSet.from_studyset(ml_studyset, _CountingKernel(r=4))

    assert _CountingKernel.calls == 2


def test_from_studyset_leaves_the_callers_kernel_alone(ml_studyset, tmp_path):
    """Wiring up a cache must not reconfigure the kernel that was passed in."""
    kernel_transformer = MKDAKernel(r=4)

    FeatureSet.from_studyset(ml_studyset, kernel_transformer, memory=str(tmp_path))

    assert any(tmp_path.iterdir())
    assert kernel_transformer.memory.location is None
    assert kernel_transformer.memory_level == 0


def test_from_studyset_accepts_a_kernel_class(ml_studyset):
    """A kernel transformer may be given as a class, as elsewhere in NiMARE."""
    features = FeatureSet.from_studyset(ml_studyset, MKDAKernel)

    assert features.map_features.shape[0] == len(ml_studyset.ids)
    assert features.provenance["kernel_transformer"]["class"] == "MKDAKernel"


def test_from_studyset_rejects_an_empty_studyset(ml_studyset):
    """There is nothing to convert without analyses."""
    empty = ml_studyset.select_analyses(np.zeros(len(ml_studyset.ids), dtype=bool))

    with pytest.raises(ValueError, match="no analyses"):
        FeatureSet.from_studyset(empty, MKDAKernel(r=4))


def test_end_to_end_classification(ml_studyset):
    """The documented workflow runs from Studyset to grouped cross-validation."""
    from sklearn.linear_model import LogisticRegression

    features = FeatureSet.from_studyset(
        ml_studyset, MKDAKernel(r=10), target_field="comparison_task"
    )
    bunch = features.to_sklearn()

    pipeline = make_pipeline(
        features.make_preprocessor(TruncatedSVD(n_components=2, random_state=RANDOM_SEED)),
        LogisticRegression(max_iter=500),
    )
    scores = cross_val_score(
        pipeline, bunch.data, bunch.target, cv=GroupKFold(n_splits=4), groups=bunch.groups
    )

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


@pytest.mark.performance_smoke
def test_from_studyset_meets_the_conversion_budget():
    """A 1,000-study Studyset converts and splits inside the documented budget."""
    # resource is Unix-only, and importing it at module scope makes the whole module
    # uncollectable on Windows. Only this test needs it, and it runs on Linux.
    import resource

    _, studyset = create_coordinate_studyset(foci=5, n_studies=1000, sample_size=30, seed=42)

    start = time.time()
    features = FeatureSet.from_studyset(studyset, MKDAKernel(r=10))
    train, test = features.split(test_size=0.25, random_state=RANDOM_SEED)
    elapsed = time.time() - start
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    assert len(features) == len(studyset.ids)
    assert set(train.study_ids).isdisjoint(test.study_ids)
    assert sparse.issparse(features.features)
    assert elapsed <= 180, f"conversion and split took {elapsed:.0f}s"
    assert peak_gb <= 5, f"peak memory was {peak_gb:.1f} GB"
