"""Tests for the nimare.ml module."""

from __future__ import annotations

import time
from pathlib import Path

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
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import Bunch

from nimare import ml
from nimare.generate import create_coordinate_studyset
from nimare.meta.kernel import MKDAKernel
from nimare.ml import AtlasAggregator
from nimare.nimads import Studyset
from nimare.utils import get_masker, get_resource_path, get_template

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


def _build_ml_studyset(
    ids_without_points=(),
    ids_without_score=(),
    coordinate_offset=0.0,
    text_annotation=False,
    indexed_annotation=False,
):
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
                        **({"task_name": f"task {position}"} if text_annotation else {}),
                        **(
                            {
                                "groups[0].count": float(position),
                                "groups[1].count": float(position) * 2,
                            }
                            if indexed_annotation
                            else {}
                        ),
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
def ma_bunch(small_masker):
    """Build a small bundle directly, without running a kernel."""
    n_rows = 6
    maps = sparse.csr_matrix(
        np.column_stack([np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) * 2.0])
    )
    descriptors = sparse.csr_matrix(np.arange(n_rows, dtype=float)[:, None])

    return Bunch(
        data=sparse.hstack([maps, descriptors], format="csr"),
        target=np.arange(n_rows, dtype=float) + 0.5,
        groups=np.array([f"study_{idx // 2}" for idx in range(n_rows)]),
        ids=np.array([f"study_{idx // 2}-task{idx % 2}" for idx in range(n_rows)]),
        feature_names=["voxel_0", "voxel_1", "motor_label"],
        map_columns=slice(0, 2),
        descriptor_columns=slice(2, 3),
        descriptor_names=["motor_label"],
        masker=small_masker,
        provenance={"studyset_id": "fixture", "dropped_ids": []},
    )


def _maps(bunch):
    """Return the map block of a bundle."""
    return bunch.data[:, bunch.map_columns]


def _descriptor_block(bunch):
    """Return the descriptor block as stored, or None when there is none."""
    columns = bunch.descriptor_columns
    return None if columns.stop <= columns.start else bunch.data[:, columns]


def _descriptors(bunch):
    """Return the descriptor block densely, for comparing values."""
    block = _descriptor_block(bunch)
    return None if block is None else _dense(block)


def _bunch(
    map_features,
    ids,
    groups=None,
    descriptors=None,
    descriptor_names=None,
    target=None,
    masker=None,
):
    """Build a bundle from blocks, the way to_bunch assembles one."""
    n_map = map_features.shape[1]
    n_descriptors = 0 if descriptors is None else descriptors.shape[1]
    data = (
        map_features
        if descriptors is None
        else sparse.hstack(
            [sparse.csr_matrix(map_features), sparse.csr_matrix(descriptors)], format="csr"
        )
    )
    return Bunch(
        data=data,
        target=target,
        groups=np.asarray(ids if groups is None else groups),
        ids=np.asarray(ids),
        feature_names=[f"voxel_{idx}" for idx in range(n_map)] + list(descriptor_names or []),
        map_columns=slice(0, n_map),
        descriptor_columns=slice(n_map, n_map + n_descriptors),
        descriptor_names=list(descriptor_names or []),
        masker=masker,
        provenance={},
    )


def _subset(bunch, rows):
    """Return the bundle restricted to ``rows``, every aligned field together."""
    out = Bunch(**bunch)
    out.data = bunch.data[rows]
    out.ids = np.asarray(bunch.ids)[rows]
    out.groups = np.asarray(bunch.groups)[rows]
    out.target = None if bunch.target is None else np.asarray(bunch.target)[rows]
    return out


def _split(bunch, test_size, random_state):
    """Return a grouped holdout, the way a caller splits a bundle."""
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(
            bunch.data, bunch.target, bunch.groups
        )
    )
    return _subset(bunch, train), _subset(bunch, test)


def _column(data, index):
    """Return one feature column as a dense 1D array."""
    column = data[:, index]
    if sparse.issparse(column):
        return column.toarray().ravel()
    return np.asarray(column).ravel()


def _dense(block):
    """Return a dense copy of a feature block."""
    return block.toarray() if sparse.issparse(block) else np.asarray(block)


def _map_signature(dataset):
    """Return the first non-zero map column of each row, which names its focus."""
    maps = _maps(dataset)
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


def test_dataset_without_descriptors(small_masker):
    """A map-only dataset has no descriptor columns and needs no hstack."""
    map_features = sparse.csr_matrix(np.eye(3))
    dataset = _bunch(map_features, ids=list("abc"))

    assert dataset.data is map_features
    assert dataset.descriptor_columns == slice(3, 3)
    assert dataset.descriptor_names == []
    assert dataset.feature_names == ["voxel_0", "voxel_1", "voxel_2"]
    assert dataset.target is None


# ---------------------------------------------------------------- reduction


def test_make_preprocessor_reduces_only_map_columns(ma_bunch):
    """Descriptor columns pass through the preprocessor untouched."""
    dataset = ma_bunch
    preprocessor = ml.make_preprocessor(
        dataset, TruncatedSVD(n_components=1, random_state=RANDOM_SEED)
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert not hasattr(preprocessor, "transformers_")

    transformed = preprocessor.fit_transform(dataset.data)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    assert transformed.shape == (dataset.data.shape[0], 2)
    np.testing.assert_array_equal(transformed[:, 1], _descriptors(dataset).ravel())


def test_make_preprocessor_keeps_unreduced_features_sparse(ma_bunch):
    """A sparse-preserving reducer must not be densified on the way through."""
    dataset = ma_bunch
    preprocessor = ml.make_preprocessor(dataset, VarianceThreshold(threshold=0.0))

    transformed = preprocessor.fit_transform(dataset.data)

    assert sparse.issparse(transformed)


def test_make_preprocessor_accepts_transformers_and_passthrough(ma_bunch):
    """The reducer may be an instance, a name, or nothing at all."""
    dataset = ma_bunch

    built = ml.make_preprocessor(
        dataset, TruncatedSVD(n_components=1), descriptor_transformer=SimpleImputer()
    )
    assert isinstance(built.transformers[0][1], TruncatedSVD)
    # The descriptor step densifies its columns before handing them over.
    assert isinstance(built.transformers[1][1].named_steps["transform"], SimpleImputer)

    passthrough = ml.make_preprocessor(dataset, None)
    assert passthrough.transformers[0][1] == "passthrough"
    assert passthrough.fit_transform(dataset.data).shape == dataset.data.shape

    with pytest.raises(ValueError, match="only used when the reducer is given as a class"):
        ml.make_preprocessor(dataset, TruncatedSVD(), n_components=1)


def test_dataset_works_in_sklearn_model_selection(ma_bunch):
    """The exported arrays drive grouped cross-validation and a grid search."""
    dataset = ma_bunch
    pipeline = make_pipeline(
        ml.make_preprocessor(dataset, TruncatedSVD(n_components=1, random_state=RANDOM_SEED)),
        Ridge(),
    )
    cv = GroupKFold(n_splits=3)
    bunch = dataset

    scores = cross_val_score(pipeline, bunch.data, bunch.target, cv=cv, groups=bunch.groups)
    assert scores.shape == (3,)

    search = GridSearchCV(pipeline, {"ridge__alpha": [0.5, 1.0]}, cv=cv)
    search.fit(bunch.data, bunch.target, groups=bunch.groups)
    assert search.best_estimator_ is not None


@pytest.mark.parametrize(
    ("reducer", "kwargs", "expected_type"),
    [
        (VarianceThreshold(threshold=0.0), {}, VarianceThreshold),
        (TruncatedSVD(n_components=2), {}, TruncatedSVD),
        (SparseRandomProjection, {"n_components": 2}, SparseRandomProjection),
    ],
)
def test_make_preprocessor_takes_scikit_learn_transformers(
    ma_bunch, reducer, kwargs, expected_type
):
    """A transformer, or a transformer class plus its parameters, both resolve."""
    preprocessor = ml.make_preprocessor(ma_bunch, reducer, **kwargs)

    built = preprocessor.transformers[0][1]
    assert isinstance(built, expected_type)
    for name, value in kwargs.items():
        assert built.get_params()[name] == value


def test_make_preprocessor_uses_a_built_transformer_as_given(ma_bunch):
    """An instance is used as it is, and cannot be reconfigured in passing."""
    reducer = SparseRandomProjection(n_components=3)

    assert ml.make_preprocessor(ma_bunch, reducer).transformers[0][1] is reducer

    with pytest.raises(ValueError, match="only used when the reducer is given as a class"):
        ml.make_preprocessor(ma_bunch, reducer, n_components=4)


def test_make_preprocessor_rejects_things_that_are_not_reducers(ma_bunch):
    """The message names what a reducer can be, including the workflows it replaced."""
    with pytest.raises(TypeError, match="TruncatedSVD"):
        ml.make_preprocessor(ma_bunch, "truncated_svd")

    with pytest.raises(TypeError, match="is not a map reducer"):
        ml.make_preprocessor(ma_bunch, object())


def test_make_preprocessor_keeps_a_reducer_off_the_descriptor_columns(ma_bunch):
    """A bare reducer would decompose the descriptor columns along with the voxels."""
    reducer = TruncatedSVD(n_components=1, random_state=RANDOM_SEED)
    descriptors = _descriptors(ma_bunch).ravel()

    scoped = ml.make_preprocessor(ma_bunch, reducer).fit_transform(ma_bunch.data)
    scoped = scoped.toarray() if sparse.issparse(scoped) else scoped
    bare = clone(reducer).fit_transform(ma_bunch.data)

    # Scoped: the map block is reduced and the descriptor column passes through.
    assert scoped.shape == (ma_bunch.data.shape[0], 2)
    np.testing.assert_allclose(scoped[:, -1], descriptors)
    # Bare: one component for everything, the descriptor folded into it.
    assert bare.shape == (ma_bunch.data.shape[0], 1)


@pytest.fixture
def descriptor_dataset(small_masker):
    """Return a feature set with three descriptors that want different treatment."""
    descriptors = np.array(
        [
            [10.0, 1.0, 5.0],
            [np.nan, 0.0, 6.0],
            [30.0, 1.0, 7.0],
            [40.0, np.nan, 8.0],
            [50.0, 0.0, 9.0],
            [60.0, 1.0, 10.0],
        ]
    )
    ids = [f"study_{idx}-task0" for idx in range(6)]
    return _bunch(
        sparse.csr_matrix(np.random.default_rng(0).random((6, 5))),
        ids=ids,
        descriptors=descriptors,
        # The middle name is the shape NeuroStore annotations come in, and a
        # ColumnTransformer step may not be named with a double underscore.
        descriptor_names=["sample_sizes", "Neurosynth_TFIDF__pain", "year"],
        masker=small_masker,
    )


def test_descriptors_can_take_one_transformer_each(descriptor_dataset):
    """A mapping treats each descriptor differently and keeps the column order."""
    preprocessor = ml.make_preprocessor(
        descriptor_dataset,
        TruncatedSVD(n_components=2, random_state=RANDOM_SEED),
        descriptor_transformer={
            "sample_sizes": SimpleImputer(strategy="median"),
            "year": StandardScaler(),
        },
    )

    out = preprocessor.fit_transform(descriptor_dataset.data)
    out = out.toarray() if sparse.issparse(out) else out
    imputed, untouched, scaled = out[:, -3], out[:, -2], out[:, -1]

    assert descriptor_dataset.descriptor_names == [
        "sample_sizes",
        "Neurosynth_TFIDF__pain",
        "year",
    ]
    assert imputed[1] == 40.0  # the median of the column, in place of its NaN
    assert np.isnan(untouched[3])  # not named, so passed through as it was
    np.testing.assert_allclose(scaled.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(), 1.0)


def test_descriptor_transformers_are_handed_dense_columns(descriptor_dataset):
    """A scaler refuses to centre sparse data, and descriptors are not sparse."""
    preprocessor = ml.make_preprocessor(
        descriptor_dataset,
        TruncatedSVD(n_components=2, random_state=RANDOM_SEED),
        descriptor_transformer=StandardScaler(),
    )

    out = preprocessor.fit_transform(descriptor_dataset.data)
    out = out.toarray() if sparse.issparse(out) else out

    assert out.shape == (6, 5)


def test_descriptor_mapping_rejects_names_that_are_not_descriptors(descriptor_dataset):
    """A typo names the descriptors this feature set actually has."""
    with pytest.raises(ValueError, match="No descriptor called 'nope'"):
        ml.make_preprocessor(
            descriptor_dataset,
            TruncatedSVD(n_components=2),
            descriptor_transformer={"nope": StandardScaler()},
        )


def test_descriptor_transformer_without_descriptors_is_an_error(small_masker):
    """Asking for descriptor handling on a map-only feature set is a mistake."""
    ids = [f"s{idx}-t" for idx in range(4)]
    map_only = _bunch(sparse.csr_matrix(np.eye(4)), ids=ids, masker=small_masker)

    with pytest.raises(ValueError, match="no descriptor columns"):
        ml.make_preprocessor(map_only, TruncatedSVD(n_components=2), SimpleImputer())


def test_map_only_features_need_no_preprocessor(small_masker):
    """With nothing to keep the reducer away from, the reducer is returned as it is."""
    map_only = _bunch(
        sparse.csr_matrix(np.eye(4)),
        ids=[f"s{idx}-t" for idx in range(4)],
        groups=[f"s{idx}" for idx in range(4)],
        masker=small_masker,
    )
    reducer = TruncatedSVD(n_components=2)

    assert ml.make_preprocessor(map_only, reducer) is reducer


def test_atlas_reducer_takes_the_maskers_voxel_order_from_the_feature_set(ma_bunch, small_masker):
    """An atlas, or an aggregator built without one, gets the feature set's masker."""
    atlas = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)

    from_atlas = ml.make_preprocessor(ma_bunch, atlas).transformers[0][1]
    unbound = AtlasAggregator(atlas=atlas)
    from_aggregator = ml.make_preprocessor(ma_bunch, unbound).transformers[0][1]

    assert from_atlas.masker is ma_bunch.masker
    assert from_aggregator.masker is ma_bunch.masker
    assert unbound.masker is None  # the caller's object is left alone


def test_atlas_reduction_needs_a_masker_somewhere(small_masker):
    """A feature set without a masker cannot place an atlas over its columns."""
    atlas = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)
    maskerless = _bunch(
        sparse.csr_matrix(np.eye(4)),
        ids=[f"s{idx}-t" for idx in range(4)],
        groups=[f"s{idx}" for idx in range(4)],
    )

    with pytest.raises(ValueError, match="voxel order"):
        ml.make_preprocessor(maskerless, atlas)


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
    dataset = _bunch(
        atlas_features,
        ids=[f"study_{idx}-task0" for idx in range(6)],
        groups=[f"study_{idx}" for idx in range(6)],
        masker=small_masker,
    )

    reducer = ml.make_preprocessor(dataset, Bunch(maps=labels_img, labels=["one", "two"]))
    reduced = reducer.fit_transform(dataset.data)
    names = reducer.get_feature_names_out()

    assert reduced.shape == (6, 2)
    assert list(names) == ["one", "two"]
    # scikit-learn's contract: an array of strings, which renders without a repr
    assert np.issubdtype(names.dtype, np.str_)
    assert [f"{name}" for name in names] == ["one", "two"]


def test_dataset_reduces_with_any_sklearn_transformer(ma_bunch):
    """A reducer NiMARE has never heard of works like the named ones."""
    reducer = SparseRandomProjection(n_components=1, random_state=RANDOM_SEED)
    train, test = _split(ma_bunch, 0.34, RANDOM_SEED)

    # Fitted on the training rows only, then applied to the held-out ones.
    reduced_train = reducer.fit_transform(_maps(train))
    reduced_test = reducer.transform(_maps(test))

    assert reduced_train.shape == (train.data.shape[0], 1)
    assert reduced_test.shape == (test.data.shape[0], 1)
    with pytest.raises(NotFittedError):
        clone(reducer).transform(_maps(test))


def test_make_preprocessor_accepts_an_atlas(small_masker, atlas_features):
    """The feature set supplies the voxel order, so an atlas needs nothing else."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    dataset = _bunch(
        atlas_features,
        ids=[f"study_{idx}-task0" for idx in range(6)],
        groups=[f"study_{idx}" for idx in range(6)],
        masker=small_masker,
    )

    # Map-only features: there is nothing to keep the aggregator away from.
    reducer = ml.make_preprocessor(dataset, labels_img)

    assert isinstance(reducer, AtlasAggregator)
    assert reducer.masker is small_masker
    assert reducer.fit_transform(dataset.data).shape == (6, 2)


# ------------------------------------------------------------------ extraction


def test_public_surface_is_a_studyset_method_and_three_helpers():
    """Conversion belongs to the Studyset; nimare.ml holds what it cannot answer."""
    assert set(ml.__all__) == {"AtlasAggregator", "describe_fields", "make_preprocessor"}
    assert not any(name.endswith("Extractor") for name in dir(ml) if not name.startswith("_"))
    # There is no container class left to meet.
    assert not hasattr(ml, "FeatureSet")
    assert callable(Studyset.to_bunch)


def test_from_studyset(ml_studyset):
    """A Studyset becomes one row per analysis, with everything aligned."""
    studyset = ml_studyset

    features = studyset.to_bunch(
        MKDAKernel(r=4),
        descriptor_fields=["sample_sizes", ("annotations", "motor_label")],
        target_field=("annotations", "target_score"),
    )

    np.testing.assert_array_equal(features.ids, studyset.ids)
    np.testing.assert_array_equal(features.groups, studyset.metadata["study_id"])
    assert sparse.issparse(_maps(features))
    assert _maps(features).shape == (len(studyset.ids), studyset.masker.n_elements_)
    assert features.feature_names[-2:] == ["sample_sizes", "motor_label"]
    assert "target_score" not in features.feature_names

    annotations = studyset.annotations_df.set_index("id").loc[features.ids]
    np.testing.assert_array_equal(features.target, annotations["target_score"])
    np.testing.assert_array_equal(
        _column(features.data, features.feature_names.index("motor_label")),
        annotations["motor_label"],
    )
    np.testing.assert_array_equal(
        _column(features.data, features.feature_names.index("sample_sizes")),
        [20 + idx for idx in range(len(studyset.ids))],
    )

    assert_sklearn_bunch_valid(features, expected_rows=len(studyset.ids), expected_sparse=True)


def test_from_studyset_to_sklearn(ml_studyset):
    """The whole path from Studyset to scikit-learn arrays is two calls."""
    features = ml_studyset.to_bunch(MKDAKernel(r=4), target_field=("annotations", "target_score"))

    bunch = features
    data, target = (features.data, features.target)

    assert_sklearn_bunch_valid(bunch, expected_rows=len(ml_studyset.ids), expected_sparse=True)
    assert data.shape == bunch.data.shape
    np.testing.assert_array_equal(target, bunch.target)


def test_from_studyset_records_provenance(ml_studyset):
    """Every row can be traced back to its Studyset and its settings."""
    provenance = ml_studyset.to_bunch(
        MKDAKernel(r=4), descriptor_fields=["motor_label"]
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
    reference = ml_studyset.to_bunch(MKDAKernel(r=4))
    expected = dict(zip(reference.ids, _map_signature(reference)))

    positions = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    reordered = ml_studyset.select_analyses(positions).to_bunch(MKDAKernel(r=4))

    np.testing.assert_array_equal(reordered.ids, ml_studyset.ids[positions])
    assert _map_signature(reordered) == [expected[id_] for id_ in reordered.ids]


def test_from_studyset_rejects_duplicate_analysis_ids(ml_studyset):
    """Duplicate ids would collapse into one map row without being noticed."""
    doubled = ml_studyset.merge(_build_ml_studyset(coordinate_offset=1.0))

    # The merge keeps one analysis per id, so build the duplicate explicitly.
    duplicated = doubled.select_analyses(np.arange(len(doubled.ids) + 1) % len(doubled.ids))

    with pytest.raises(ValueError, match="must be unique"):
        duplicated.to_bunch(MKDAKernel(r=4))


@pytest.mark.parametrize("missing_coordinates", ["drop", "include"])
def test_from_studyset_missing_coordinates(ml_studyset, missing_coordinates):
    """Coordinate-less analyses are dropped, or kept as all-zero map rows."""
    missing_id = "study_2-task0"
    studyset = _build_ml_studyset(ids_without_points={missing_id})

    features = studyset.to_bunch(
        MKDAKernel(r=4),
        descriptor_fields=["motor_label"],
        target_field=("annotations", "target_score"),
        missing_coordinates=missing_coordinates,
    )

    if missing_coordinates == "drop":
        assert missing_id not in set(features.ids)
        assert features.data.shape[0] == len(studyset.ids) - 1
        assert features.provenance["dropped_ids"] == [missing_id]
    else:
        assert features.data.shape[0] == len(studyset.ids)
        assert features.provenance["dropped_ids"] == []
        row = int(np.flatnonzero(features.ids == missing_id)[0])
        assert _maps(features)[row].nnz == 0
        assert _maps(features)[row - 1].nnz > 0

    annotations = studyset.annotations_df.set_index("id").loc[features.ids]
    np.testing.assert_array_equal(features.target, annotations["target_score"])
    np.testing.assert_array_equal(_descriptors(features).ravel(), annotations["motor_label"])


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
        ml_studyset.to_bunch(MKDAKernel(r=4), **kwargs)


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
            studyset.to_bunch(**call)
        return

    features = studyset.to_bunch(**call)
    if missing_values == "drop":
        assert missing_id not in set(features.ids)
        assert features.data.shape[0] == len(studyset.ids) - 1
        assert features.provenance["missing_value_ids"] == {"score": [missing_id]}
        assert np.isfinite(_descriptors(features)).all()
    else:
        assert features.data.shape[0] == len(studyset.ids)
        row = int(np.flatnonzero(features.ids == missing_id)[0])
        assert np.isnan(_descriptors(features)[row, 0])
        assert features.provenance["missing_value_ids"] == {"score": [missing_id]}


def test_from_studyset_missing_target_values_are_reported():
    """A missing target is named the same way a missing descriptor is."""
    studyset = _build_ml_studyset(ids_without_score={"study_1-task0"})

    with pytest.raises(ValueError, match="study_1-task0"):
        studyset.to_bunch(MKDAKernel(r=4), target_field="score")


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
        ml_studyset.to_bunch(MKDAKernel(r=4), descriptor_fields=[selector])


@pytest.fixture(scope="session")
def neurosynth_studyset():
    """Return the bundled Neurosynth Studyset, which annotates with 3,228 labels."""
    return Studyset(str(Path(get_resource_path()) / "neurosynth_laird_studyset.json"))


def test_annotation_labels_are_selected_by_pattern(neurosynth_studyset):
    """A pattern takes every matching label, under its own name."""
    studyset = neurosynth_studyset
    labels = [
        column
        for column in studyset.annotations_df.columns
        if column.startswith("Neurosynth_TFIDF__pa")
    ]

    features = studyset.to_bunch(
        MKDAKernel(r=10), descriptor_fields=[("annotations", "Neurosynth_TFIDF__pa*")]
    )

    assert features.descriptor_names == labels
    assert "Neurosynth_TFIDF__pain" in features.descriptor_names
    assert _descriptors(features).shape == (features.data.shape[0], len(labels))
    # The names survive whole, double underscores and all.
    assert features.feature_names[-len(labels) :] == labels

    expected = studyset.annotations_df.set_index("id").loc[features.ids, labels].to_numpy()
    np.testing.assert_allclose(_dense(_descriptors(features)), expected)


def test_annotation_labels_stay_sparse(neurosynth_studyset):
    """A whole annotation is thousands of mostly-empty columns, and stays sparse."""
    features = neurosynth_studyset.to_bunch(
        MKDAKernel(r=10),
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")],
    )

    descriptors = _descriptor_block(features)
    assert sparse.issparse(descriptors)
    assert descriptors.shape[1] > 3000
    assert descriptors.nnz < descriptors.shape[0] * descriptors.shape[1] / 10
    assert sparse.issparse(features.data)
    # Splitting and exporting keep it sparse too.
    train, _ = _split(features, 0.25, RANDOM_SEED)
    assert sparse.issparse(_descriptor_block(train))
    assert sparse.issparse(train.data)


def test_absent_annotation_labels_are_zero_not_missing(neurosynth_studyset):
    """A label no analysis carries is a zero, so missing_values has nothing to say."""
    features = neurosynth_studyset.to_bunch(
        MKDAKernel(r=10),
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")],
        missing_values="raise",
    )

    assert features.provenance["missing_value_ids"] == {}
    assert np.isfinite(_dense(_descriptors(features))).all()


def test_annotation_pattern_mixes_with_other_descriptors(neurosynth_studyset):
    """A pattern block and a scalar field sit side by side, in selection order."""
    studyset = neurosynth_studyset.with_metadata(
        "year", np.arange(len(neurosynth_studyset.ids), dtype=float)
    )

    features = studyset.to_bunch(
        MKDAKernel(r=10),
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__pai*"), "year"],
    )

    assert features.descriptor_names[-1] == "year"
    np.testing.assert_allclose(
        _dense(_descriptors(features))[:, -1], np.arange(features.data.shape[0], dtype=float)
    )


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        (("annotations", "Neurosynth_NOPE__*"), "matches no annotation label"),
        (("metadata", "sample_*"), "cannot come from metadata"),
        ("no_such_prefix_*", "matches no annotation label"),
    ],
)
def test_annotation_pattern_errors(neurosynth_studyset, selector, message):
    """A pattern that names nothing says so, and patterns are for annotations."""
    with pytest.raises(ValueError, match=message):
        neurosynth_studyset.to_bunch(MKDAKernel(r=10), descriptor_fields=[selector])


def test_annotation_pattern_records_what_was_asked_for(neurosynth_studyset):
    """Provenance keeps the selector, not the thousands of names it expanded to."""
    features = neurosynth_studyset.to_bunch(
        MKDAKernel(r=10),
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__pai*")],
    )

    assert features.provenance["descriptor_fields"] == [["annotations", "Neurosynth_TFIDF__pai*"]]
    assert features.provenance["n_descriptor_features"] == len(features.descriptor_names)


@pytest.fixture(scope="session")
def neurostore_studyset():
    """Return the bundled NeuroStore studyset, whose labels are named with brackets."""
    return Studyset(Path(get_resource_path()) / "nback_vs_flanker_studyset_2026-07")


def test_exact_field_names_win_over_pattern_matching(neurostore_studyset):
    """A label whose own name contains a glob character is still selectable.

    878 of the labels in this bundled studyset are named like
    ``ParticipantDemographicsExtractor.groups[0].BMI``.
    """
    studyset = neurostore_studyset.slice(neurostore_studyset.ids[:6])
    frame = studyset.annotations_df
    label = next(
        column
        for column in frame.columns
        if "[" in column and pd.api.types.is_numeric_dtype(frame[column])
    )

    features = studyset.to_bunch(
        MKDAKernel(r=10),
        descriptor_fields=[("annotations", label)],
        missing_values="keep",
    )

    assert features.descriptor_names == [label]
    np.testing.assert_allclose(
        _dense(_descriptors(features)).ravel(),
        frame.set_index("id").loc[features.ids, label].to_numpy(dtype=float),
    )


def test_pattern_refuses_non_numeric_labels():
    """A label a glob matches is checked the way a label named exactly is."""
    studyset = _build_ml_studyset(text_annotation=True)

    with pytest.raises(ValueError, match="not numeric"):
        studyset.to_bunch(MKDAKernel(r=4), descriptor_fields=[("annotations", "task_nam*")])

    # Named exactly, the same label is refused for the same reason.
    with pytest.raises(ValueError, match="is categorical"):
        studyset.to_bunch(MKDAKernel(r=4), descriptor_fields=[("annotations", "task_name")])


def test_patterns_span_several_annotations(ml_studyset):
    """Selection and extraction must agree about what the annotations hold."""
    studyset = ml_studyset.with_annotation(
        "second", ["extra_term"], np.arange(len(ml_studyset.ids), dtype=float)[:, None]
    )

    features = studyset.to_bunch(MKDAKernel(r=4), descriptor_fields=[("annotations", "*_term")])

    assert features.descriptor_names == ["extra_term"]


def test_a_dropped_analysis_cannot_also_be_missing():
    """missing_values speaks for the rows that are kept, not the ones already gone."""
    missing_id = "study_2-task0"
    studyset = _build_ml_studyset(ids_without_points={missing_id}, ids_without_score={missing_id})

    features = studyset.to_bunch(
        MKDAKernel(r=4), descriptor_fields=["score"], missing_values="raise"
    )

    assert features.provenance["dropped_ids"] == [missing_id]
    assert features.provenance["missing_value_ids"] == {}


def test_a_target_is_constant_only_over_the_rows_that_are_kept():
    """The minority class can disappear with the analyses that had no coordinates."""
    studyset = _build_ml_studyset(ids_without_points={"study_0-task0", "study_0-task1"})
    labels = ["rare" if id_.startswith("study_0") else "common" for id_ in studyset.ids]
    studyset = studyset.with_metadata("grp", np.array(labels, dtype=object))

    with pytest.raises(ValueError, match="single value 'common' for every analysis that was kept"):
        studyset.to_bunch(MKDAKernel(r=4), target_field="grp")


def test_study_level_metadata_is_inherited_even_when_an_analysis_declares_it(ml_studyset):
    """One analysis declaring a field must not hide the study-level value from its siblings."""
    features = ml_studyset.to_bunch(MKDAKernel(r=4), descriptor_fields=["year"])

    years = ml_studyset.metadata.set_index("id").loc[features.ids, "year"]
    np.testing.assert_array_equal(_descriptors(features)[:, 0], years)
    assert np.isfinite(_descriptors(features)[:, 0]).all()


def test_atlas_aggregator_forgets_the_previous_atlas(atlas_features, small_masker):
    """A refit with a different atlas must not report the first one's region count."""
    affine = small_masker.mask_img.affine
    two_regions = np.zeros((4, 4, 4), dtype=np.int16)
    two_regions[0, :2, :2] = 1
    two_regions[1, :2, :2] = 2
    one_region = np.ones((4, 4, 4), dtype=np.int16)

    reducer = AtlasAggregator(atlas=nib.Nifti1Image(two_regions, affine), masker=small_masker)
    reducer.fit_transform(atlas_features)
    assert len(reducer.get_feature_names_out()) == 2

    reducer.set_params(atlas=nib.Nifti1Image(one_region, affine))
    reducer.fit(atlas_features)
    assert len(reducer.get_feature_names_out()) == 1


def test_from_studyset_rejects_repeated_descriptor_fields(ml_studyset):
    """The same field twice is a mistake, not two features."""
    with pytest.raises(ValueError, match="selected more than once"):
        ml_studyset.to_bunch(MKDAKernel(r=4), descriptor_fields=["motor_label", "motor_label"])


def test_from_studyset_reports_ambiguous_field_names(ml_studyset):
    """A name in two sources asks which one was meant."""
    annotated = ml_studyset.with_metadata("motor_label", np.ones(len(ml_studyset.ids)))

    with pytest.raises(ValueError, match="is ambiguous"):
        annotated.to_bunch(MKDAKernel(r=4), descriptor_fields=["motor_label"])


def test_from_studyset_reads_study_level_and_list_metadata(ml_studyset):
    """Study-level fields are inherited and list-valued fields are reduced."""
    features = ml_studyset.to_bunch(MKDAKernel(r=4), descriptor_fields=["year", "sample_sizes"])

    years = ml_studyset.metadata.set_index("id").loc[features.ids, "year"]
    np.testing.assert_array_equal(_descriptors(features)[:, 0], years)
    np.testing.assert_array_equal(
        _descriptors(features)[:, 1], [20 + idx for idx in range(features.data.shape[0])]
    )


def test_from_studyset_exports_categorical_targets(ml_studyset):
    """A categorical target reaches y as labels, ready for a classifier."""
    features = ml_studyset.to_bunch(MKDAKernel(r=4), target_field="comparison_task")

    assert sorted(set(features.target)) == ["flanker", "n-back"]
    np.testing.assert_array_equal(
        features.target,
        ml_studyset.metadata.set_index("id").loc[features.ids, "comparison_task"],
    )


def test_from_studyset_applies_a_target_transformer(ml_studyset):
    """A label extractor turns a text field into one label per analysis."""
    features = ml_studyset.to_bunch(
        MKDAKernel(r=4),
        target_field=("texts", "abstract"),
        target_transformer=lambda values: np.array(
            [text.split()[1] for text in values], dtype=str
        ),
    )

    np.testing.assert_array_equal(
        features.target, [str(idx) for idx in range(features.data.shape[0])]
    )


def test_from_studyset_target_transformer_may_be_a_transformer(ml_studyset):
    """A stateless scikit-learn transformer works as a label extractor too."""
    features = ml_studyset.to_bunch(
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
        ml_studyset.to_bunch(MKDAKernel(r=4), **kwargs)


def test_from_studyset_rejects_constant_targets(ml_studyset):
    """A target with one value has nothing to predict."""
    constant = ml_studyset.with_metadata("only_value", np.ones(len(ml_studyset.ids)))

    with pytest.raises(ValueError, match="nothing to predict"):
        constant.to_bunch(MKDAKernel(r=4), target_field="only_value")


class _CountingKernel(MKDAKernel):
    """An MKDA kernel that counts how often the maps are really computed."""

    calls = 0

    def _transform(self, mask, coordinates, return_type="sparse"):
        type(self).calls += 1
        return super()._transform(mask, coordinates, return_type=return_type)


def test_from_studyset_caches_maps_with_memory(ml_studyset, tmp_path):
    """A cache location makes a repeated conversion reuse the maps it made."""
    _CountingKernel.calls = 0

    first = ml_studyset.to_bunch(_CountingKernel(r=4), memory=str(tmp_path))
    second = ml_studyset.to_bunch(_CountingKernel(r=4), memory=str(tmp_path))

    assert _CountingKernel.calls == 1
    np.testing.assert_array_equal(_maps(first).toarray(), _maps(second).toarray())

    # A different kernel configuration is a different cache entry, not a stale hit.
    ml_studyset.to_bunch(_CountingKernel(r=8), memory=str(tmp_path))
    assert _CountingKernel.calls == 2

    # Kernel transformers cache their maps at memory_level 2, so a lower level asks
    # for no caching at all.
    _CountingKernel.calls = 0
    ml_studyset.to_bunch(_CountingKernel(r=4), memory=str(tmp_path), memory_level=1)
    ml_studyset.to_bunch(_CountingKernel(r=4), memory=str(tmp_path), memory_level=1)
    assert _CountingKernel.calls == 2


def test_from_studyset_without_memory_does_not_cache(ml_studyset):
    """Without a cache location every conversion generates its own maps."""
    _CountingKernel.calls = 0

    ml_studyset.to_bunch(_CountingKernel(r=4))
    ml_studyset.to_bunch(_CountingKernel(r=4))

    assert _CountingKernel.calls == 2


def test_from_studyset_leaves_the_callers_kernel_alone(ml_studyset, tmp_path):
    """Wiring up a cache must not reconfigure the kernel that was passed in."""
    kernel_transformer = MKDAKernel(r=4)

    ml_studyset.to_bunch(kernel_transformer, memory=str(tmp_path))

    assert any(tmp_path.iterdir())
    assert kernel_transformer.memory.location is None
    assert kernel_transformer.memory_level == 0


def test_from_studyset_accepts_a_kernel_class(ml_studyset):
    """A kernel transformer may be given as a class, as elsewhere in NiMARE."""
    features = ml_studyset.to_bunch(MKDAKernel)

    assert _maps(features).shape[0] == len(ml_studyset.ids)
    assert features.provenance["kernel_transformer"]["class"] == "MKDAKernel"


def test_from_studyset_rejects_an_empty_studyset(ml_studyset):
    """There is nothing to convert without analyses."""
    empty = ml_studyset.select_analyses(np.zeros(len(ml_studyset.ids), dtype=bool))

    with pytest.raises(ValueError, match="no analyses"):
        empty.to_bunch(MKDAKernel(r=4))


def test_end_to_end_classification(ml_studyset):
    """The documented workflow runs from Studyset to grouped cross-validation."""
    from sklearn.linear_model import LogisticRegression

    features = ml_studyset.to_bunch(MKDAKernel(r=10), target_field="comparison_task")
    bunch = features

    pipeline = make_pipeline(
        ml.make_preprocessor(features, TruncatedSVD(n_components=2, random_state=RANDOM_SEED)),
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
    features = studyset.to_bunch(MKDAKernel(r=10))
    train, test = _split(features, 0.25, RANDOM_SEED)
    elapsed = time.time() - start
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    assert features.data.shape[0] == len(studyset.ids)
    assert set(train.groups).isdisjoint(test.groups)
    assert sparse.issparse(features.data)
    assert elapsed <= 180, f"conversion and split took {elapsed:.0f}s"
    assert peak_gb <= 5, f"peak memory was {peak_gb:.1f} GB"


def test_describe_fields_reports_every_source(ml_studyset):
    """Every field a selector may name is reported, with the kind extraction will use."""
    fields = ml.describe_fields(ml_studyset)

    assert set(fields.columns) == {
        "source",
        "field",
        "kind",
        "coverage",
        "n_unique",
        "example",
    }
    assert set(fields.source) == {"metadata", "annotations", "texts"}

    by_field = fields.set_index("field")
    assert by_field.loc["sample_sizes", "kind"] == "numeric"
    assert by_field.loc["comparison_task", "kind"] == "categorical"
    assert by_field.loc["abstract", "kind"] == "text"
    assert by_field.loc["motor_label", "source"] == "annotations"
    # study-level metadata is inherited, so every analysis reports a year
    assert by_field.loc["year", "coverage"] == pytest.approx(1.0)


def test_describe_fields_agrees_with_what_extraction_reads():
    """A field described as numeric is one from_studyset appends as a number."""
    studyset = _build_ml_studyset(ids_without_score=("study_0-task0",))
    fields = ml.describe_fields(studyset).set_index("field")

    assert fields.loc["score", "kind"] == "numeric"
    assert fields.loc["score", "coverage"] < 1.0

    features = studyset.to_bunch(
        kernel_transformer=MKDAKernel(r=1),
        descriptor_fields=["score"],
        missing_values="keep",
    )
    column = _descriptors(features)
    column = column.toarray() if sparse.issparse(column) else np.asarray(column)
    reported = np.isfinite(column[:, 0]).mean()
    assert reported == pytest.approx(fields.loc["score", "coverage"])


def test_describe_fields_can_drop_the_long_tail(ml_studyset):
    """min_coverage removes the fields too sparse to model on."""
    everything = ml.describe_fields(ml_studyset)
    covered = ml.describe_fields(ml_studyset, min_coverage=1.0)

    assert len(covered) < len(everything)
    assert (covered.coverage == 1.0).all()
    assert set(ml.describe_fields(ml_studyset, source="annotations").source) == {"annotations"}


def test_an_indexed_label_is_selectable_by_pattern():
    """``groups[0].*`` names group zero, though fnmatch reads brackets as a class."""
    studyset = _build_ml_studyset(indexed_annotation=True)
    features = studyset.to_bunch(
        kernel_transformer=MKDAKernel(r=1),
        descriptor_fields=[("annotations", "groups[0].*")],
    )

    assert features.descriptor_names == ["groups[0].count"]


def test_a_character_class_pattern_still_selects_as_one():
    """Escaping is a fallback, so a pattern that already matches is left alone."""
    studyset = _build_ml_studyset(indexed_annotation=True)
    features = studyset.to_bunch(
        kernel_transformer=MKDAKernel(r=1),
        descriptor_fields=[("annotations", "[mt]*")],
    )

    assert features.descriptor_names == ["motor_label", "target_score"]


def test_describe_fields_of_an_empty_studyset_is_an_empty_report(ml_studyset):
    """No analyses is a report with no coverage, not an error."""
    fields = ml.describe_fields(ml_studyset.slice(analyses=[]))

    assert list(fields.columns) == [
        "source",
        "field",
        "kind",
        "coverage",
        "n_unique",
        "example",
    ]
    assert (fields.coverage == 0.0).all()


def test_missing_values_can_differ_between_target_and_descriptors():
    """A descriptor gap is imputable and a target gap is not, so they can differ."""
    studyset = _build_ml_studyset(ids_without_score=("study_0-task0",))

    features = studyset.to_bunch(
        kernel_transformer=MKDAKernel(r=1),
        descriptor_fields=["score"],
        target_field=("annotations", "target_score"),
        missing_values={"target": "drop", "descriptors": "keep"},
    )

    # the analysis with no score is kept, because only its descriptor is missing
    assert "study_0-task0" in set(features.ids)
    column = _descriptors(features)
    column = column.toarray() if sparse.issparse(column) else np.asarray(column)
    assert not np.isfinite(column).all()
    assert np.isfinite(np.asarray(features.target, dtype=float)).all()


def test_a_missing_target_is_dropped_while_descriptors_are_kept():
    """The role mapping drops the rows whose target is missing, and only those."""
    studyset = _build_ml_studyset(ids_without_score=("study_0-task0",))

    features = studyset.to_bunch(
        kernel_transformer=MKDAKernel(r=1),
        target_field="score",
        descriptor_fields=[("annotations", "motor_label")],
        missing_values={"target": "drop", "descriptors": "keep"},
    )

    assert "study_0-task0" not in set(features.ids)
    assert np.isfinite(np.asarray(features.target, dtype=float)).all()


def test_missing_values_rejects_a_role_it_does_not_know():
    """A mapping key that is not a role is a mistake worth naming."""
    studyset = _build_ml_studyset()

    with pytest.raises(ValueError, match="sets a policy per role"):
        studyset.to_bunch(
            kernel_transformer=MKDAKernel(r=1),
            target_field="score",
            missing_values={"targets": "drop"},
        )


def test_missing_values_rejects_an_unknown_policy_in_a_mapping():
    """The policies are the same three whether or not a mapping is used."""
    studyset = _build_ml_studyset()

    with pytest.raises(ValueError, match="must be 'raise', 'drop' or 'keep'"):
        studyset.to_bunch(
            kernel_transformer=MKDAKernel(r=1),
            target_field="score",
            missing_values={"target": "impute"},
        )


def test_to_bunch_splits_without_splitting_a_study(ml_studyset):
    """test_size adds row positions, and a study belongs to exactly one side."""
    bunch = ml_studyset.to_bunch(MKDAKernel(r=4), test_size=0.5, random_state=RANDOM_SEED)

    assert set(bunch.groups[bunch.train]).isdisjoint(bunch.groups[bunch.test])
    assert sorted(np.concatenate([bunch.train, bunch.test]).tolist()) == list(
        range(bunch.data.shape[0])
    )
    assert len(bunch.train) and len(bunch.test)


def test_to_bunch_leaves_the_split_out_unless_it_is_asked_for(ml_studyset):
    """The bundle has the same keys as before when test_size is not given."""
    bunch = ml_studyset.to_bunch(MKDAKernel(r=4))

    assert "train" not in bunch
    assert "test" not in bunch


def test_to_bunch_split_is_the_one_a_group_splitter_would_make(ml_studyset):
    """It is GroupShuffleSplit over groups, so a caller can reproduce it."""
    bunch = ml_studyset.to_bunch(MKDAKernel(r=4), test_size=0.5, random_state=RANDOM_SEED)

    n_test = len(set(bunch.groups[bunch.test]))
    expected_train, expected_test = next(
        GroupShuffleSplit(n_splits=1, test_size=n_test, random_state=RANDOM_SEED).split(
            np.zeros(len(bunch.groups)), groups=bunch.groups
        )
    )

    np.testing.assert_array_equal(bunch.train, expected_train)
    np.testing.assert_array_equal(bunch.test, expected_test)


@pytest.mark.parametrize(
    ("test_size", "message"),
    [
        (0.9, "leaves one partition empty"),
        (0.0, "must be between 0 and 1"),
        (1.0, "must be between 0 and 1"),
        (99, "leaves one partition empty"),
        ("half", "must be a float or an int"),
    ],
)
def test_to_bunch_rejects_impossible_splits(ml_studyset, test_size, message):
    """A split that cannot be made is named rather than silently emptied."""
    with pytest.raises(ValueError, match=message):
        ml_studyset.to_bunch(MKDAKernel(r=4), test_size=test_size)


def test_to_bunch_split_needs_two_studies(ml_studyset):
    """One study cannot be split, since its analyses may not be separated."""
    one_study = ml_studyset.slice(ids=["study_0"], filter_level="study")

    with pytest.raises(ValueError, match="at least 2 studies"):
        one_study.to_bunch(MKDAKernel(r=4), test_size=0.5)


@pytest.mark.parametrize(
    "descriptor_transformer",
    [
        pytest.param(SimpleImputer(strategy="median"), id="one-transformer"),
        pytest.param({"motor_label": StandardScaler()}, id="per-descriptor-mapping"),
        pytest.param({}, id="mapping-naming-none-of-them"),
        pytest.param("passthrough", id="passthrough"),
    ],
)
def test_preprocessor_names_descriptors_after_their_fields(ma_bunch, descriptor_transformer):
    """A fitted coefficient has to be readable back to the field it belongs to."""
    preprocessor = ml.make_preprocessor(
        ma_bunch,
        TruncatedSVD(n_components=1, random_state=RANDOM_SEED),
        descriptor_transformer=descriptor_transformer,
    )
    preprocessor.fit(ma_bunch.data)

    names = [str(name) for name in preprocessor.get_feature_names_out()]

    # the ColumnTransformer selects by position, so without help this would be
    # the positional 'x2' rather than the descriptor's own name
    assert names[-1].endswith("motor_label")
    assert len(names) == 2


def test_preprocessor_feature_names_reach_the_estimator(ma_bunch):
    """The names line up with the coefficients a fitted model exposes."""
    pipeline = make_pipeline(
        ml.make_preprocessor(
            ma_bunch,
            TruncatedSVD(n_components=1, random_state=RANDOM_SEED),
            descriptor_transformer=SimpleImputer(strategy="median"),
        ),
        Ridge(),
    )
    pipeline.fit(ma_bunch.data, ma_bunch.target)

    names = pipeline[:-1].get_feature_names_out()

    assert len(names) == len(pipeline[-1].coef_)


@pytest.mark.parametrize(
    ("transformer", "stays_sparse"),
    [
        pytest.param("passthrough", True, id="passthrough"),
        pytest.param(MaxAbsScaler(), True, id="sparse-safe-scaler"),
        pytest.param(StandardScaler(with_mean=False), True, id="scaler-told-not-to-centre"),
        pytest.param(SimpleImputer(strategy="median"), True, id="imputer"),
        pytest.param(StandardScaler(), False, id="scaler-that-centres"),
        pytest.param(MinMaxScaler(), False, id="scaler-that-refuses-sparse"),
    ],
)
def test_only_transformers_that_need_dense_columns_get_them(
    neurosynth_studyset, transformer, stays_sparse
):
    """A label block thousands wide must not be densified to please a scaler."""
    bunch = neurosynth_studyset.to_bunch(
        MKDAKernel(r=10), descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")]
    )
    block = bunch.data[:, bunch.descriptor_columns]
    assert block.shape[1] > 1000

    preprocessor = ml.make_preprocessor(
        bunch,
        TruncatedSVD(n_components=2, random_state=RANDOM_SEED),
        descriptor_transformer=transformer,
    )
    preprocessor.fit(bunch.data)
    out = preprocessor.named_transformers_["descriptors"].transform(block)

    assert sparse.issparse(out) is stays_sparse
    assert out.shape == block.shape


def test_a_descriptor_transformer_that_centres_still_works(ma_bunch):
    """Densifying is the fallback, so a centring scaler is not refused."""
    preprocessor = ml.make_preprocessor(
        ma_bunch,
        TruncatedSVD(n_components=1, random_state=RANDOM_SEED),
        descriptor_transformer=StandardScaler(),
    )

    out = preprocessor.fit_transform(ma_bunch.data)
    out = out.toarray() if sparse.issparse(out) else out

    # centred, which is what StandardScaler was asked for
    assert out[:, -1].mean() == pytest.approx(0.0, abs=1e-9)
