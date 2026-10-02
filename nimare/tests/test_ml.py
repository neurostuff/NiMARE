"""Tests for the nimare.ml module."""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import Bunch

from nimare import ml
from nimare.generate import create_coordinate_studyset
from nimare.meta.kernel import ALEKernel, MKDAKernel
from nimare.ml import MAKernel, MaskerTransformer
from nimare.ml._helpers import _FeatureNames, _to_dense
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
    """Return a masker over a tiny volume, for transformer tests."""
    mask_data = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_data[:2, :2, :2] = 1
    return get_masker(nib.Nifti1Image(mask_data, np.eye(4)))


@pytest.fixture
def ma_bunch(small_masker):
    """Build a small bundle directly, without reading a Studyset.

    The voxel block spans the masker's image grid, as a real bundle's peak
    columns do, so a masker or a kernel can act on it.
    """
    n_rows = 6
    n_grid = int(np.prod(small_masker.mask_img.shape))
    voxels = np.zeros((n_rows, n_grid), dtype=float)
    voxels[:, 0] = np.arange(n_rows, dtype=float)
    voxels[:, 1] = np.arange(n_rows, dtype=float) * 2.0
    descriptors = sparse.csr_matrix(np.arange(n_rows, dtype=float)[:, None])

    return Bunch(
        data=sparse.hstack([sparse.csr_matrix(voxels), descriptors], format="csr"),
        target=np.arange(n_rows, dtype=float) + 0.5,
        groups=np.array([f"study_{idx // 2}" for idx in range(n_rows)]),
        ids=np.array([f"study_{idx // 2}-task{idx % 2}" for idx in range(n_rows)]),
        feature_names=_FeatureNames(n_grid, ["motor_label"]),
        voxel_columns=slice(0, n_grid),
        descriptor_columns=slice(n_grid, n_grid + 1),
        descriptor_names=["motor_label"],
        masker=small_masker,
        provenance={"studyset_id": "fixture", "dropped_ids": []},
    )


def _voxels(bunch):
    """Return the voxel block of a bundle."""
    return bunch.data[:, bunch.voxel_columns]


def _descriptor_block(bunch):
    """Return the descriptor block as stored, or None when there is none."""
    columns = bunch.descriptor_columns
    return None if columns.stop <= columns.start else bunch.data[:, columns]


def _descriptors(bunch):
    """Return the descriptor block densely, for comparing values."""
    block = _descriptor_block(bunch)
    return None if block is None else _dense(block)


def _bunch(
    voxel_features,
    ids,
    groups=None,
    descriptors=None,
    descriptor_names=None,
    target=None,
    masker=None,
):
    """Build a bundle from blocks, the way to_bunch assembles one."""
    n_map = voxel_features.shape[1]
    n_descriptors = 0 if descriptors is None else descriptors.shape[1]
    data = (
        voxel_features
        if descriptors is None
        else sparse.hstack(
            [sparse.csr_matrix(voxel_features), sparse.csr_matrix(descriptors)], format="csr"
        )
    )
    return Bunch(
        data=data,
        target=target,
        groups=np.asarray(ids if groups is None else groups),
        ids=np.asarray(ids),
        feature_names=_FeatureNames(n_map, descriptor_names or []),
        voxel_columns=slice(0, n_map),
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


def _step_transformer(column_transformer, index=0):
    """Return the transformer a ColumnTransformer step really applies.

    make_nimare_column_transformer wraps each one so that column names and
    sparsity survive, so the transformer itself is the wrapper's named
    ``"transform"`` step.
    """
    step = column_transformer.transformers[index][1]
    return step.named_steps["transform"] if isinstance(step, Pipeline) else step


def _column(data, index):
    """Return one feature column as a dense 1D array."""
    column = data[:, index]
    if sparse.issparse(column):
        return column.toarray().ravel()
    return np.asarray(column).ravel()


def _dense(block):
    """Return a dense copy of a feature block."""
    return block.toarray() if sparse.issparse(block) else np.asarray(block)


def _peak_signature(dataset):
    """Return the first peak column of each row, which names its focus."""
    peaks = _voxels(dataset)
    return [int(peaks[row].indices.min()) for row in range(peaks.shape[0])]


# ---------------------------------------------------------------- container


def test_dataset_without_descriptors(small_masker):
    """A map-only dataset has no descriptor columns and needs no hstack."""
    voxel_features = sparse.csr_matrix(np.eye(3))
    dataset = _bunch(voxel_features, ids=list("abc"))

    assert dataset.data is voxel_features
    assert dataset.descriptor_columns == slice(3, 3)
    assert dataset.descriptor_names == []
    assert list(dataset.feature_names) == ["voxel_0", "voxel_1", "voxel_2"]
    assert dataset.target is None


# ---------------------------------------------------------------- reduction


def test_column_transformer_reduces_only_voxel_columns(ma_bunch):
    """Descriptor columns pass through the preprocessor untouched."""
    dataset = ma_bunch
    preprocessor = ml.make_nimare_column_transformer(
        dataset,
        (TruncatedSVD(n_components=1, random_state=RANDOM_SEED), "voxels"),
        ("passthrough", "descriptors"),
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert not hasattr(preprocessor, "transformers_")

    transformed = preprocessor.fit_transform(dataset.data)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    assert transformed.shape == (dataset.data.shape[0], 2)
    np.testing.assert_array_equal(transformed[:, 1], _descriptors(dataset).ravel())


def test_column_transformer_keeps_unreduced_features_sparse(ma_bunch):
    """A sparse-preserving transformer must not be densified on the way through."""
    dataset = ma_bunch
    preprocessor = ml.make_nimare_column_transformer(
        dataset, (VarianceThreshold(threshold=0.0), "voxels"), ("passthrough", "descriptors")
    )

    transformed = preprocessor.fit_transform(dataset.data)

    assert sparse.issparse(transformed)


def test_column_transformer_accepts_transformers_and_passthrough(ma_bunch):
    """The transformer may be an instance, a name, or nothing at all."""
    dataset = ma_bunch

    built = ml.make_nimare_column_transformer(
        dataset,
        (TruncatedSVD(n_components=1), "voxels"),
        (SimpleImputer(), "descriptors"),
    )
    assert isinstance(_step_transformer(built, 0), TruncatedSVD)
    assert isinstance(_step_transformer(built, 1), SimpleImputer)

    passthrough = ml.make_nimare_column_transformer(
        dataset, ("passthrough", "voxels"), ("passthrough", "descriptors")
    )
    assert [name for name, _, _ in passthrough.transformers] == [
        "passthrough-1",
        "passthrough-2",
    ]
    assert passthrough.fit_transform(dataset.data).shape == dataset.data.shape

    with pytest.raises(ValueError, match="is not a \\(transformer, columns\\) pair"):
        ml.make_nimare_column_transformer(dataset, TruncatedSVD())


def test_dataset_works_in_sklearn_model_selection(ma_bunch):
    """The exported arrays drive grouped cross-validation and a grid search."""
    dataset = ma_bunch
    pipeline = make_pipeline(
        ml.make_nimare_column_transformer(
            dataset,
            (TruncatedSVD(n_components=1, random_state=RANDOM_SEED), "voxels"),
            ("passthrough", "descriptors"),
        ),
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
    "transformer",
    [
        pytest.param(TruncatedSVD(n_components=1), id="truncated-svd"),
        pytest.param(VarianceThreshold(threshold=0.0), id="variance-threshold"),
        pytest.param(SparseRandomProjection(n_components=1), id="sparse-random-projection"),
    ],
)
def test_column_transformer_takes_scikit_learn_transformers(ma_bunch, transformer):
    """Any scikit-learn transformer is used as the caller built it."""
    preprocessor = ml.make_nimare_column_transformer(
        ma_bunch, (transformer, "voxels"), ("passthrough", "descriptors")
    )

    assert _step_transformer(preprocessor) is transformer


def test_column_transformer_uses_a_built_transformer_as_given(ma_bunch):
    """An instance is used as it is, and cannot be reconfigured in passing."""
    transformer = SparseRandomProjection(n_components=3)

    assert (
        _step_transformer(
            ml.make_nimare_column_transformer(
                ma_bunch, (transformer, "voxels"), ("passthrough", "descriptors")
            )
        )
        is transformer
    )

    # transformers are configured by the caller, as they are for sklearn's own
    # make_column_transformer, so stray parameters are a TypeError
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ml.make_nimare_column_transformer(ma_bunch, (transformer, "voxels"), n_components=4)


def test_column_transformer_rejects_things_that_are_not_transformers(ma_bunch):
    """The message names what a transformer slot accepts."""
    with pytest.raises(ValueError, match="A string may only be 'passthrough' or 'drop'"):
        ml.make_nimare_column_transformer(ma_bunch, ("truncated_svd", "voxels"))

    with pytest.raises(TypeError, match="cannot transform the voxel columns"):
        ml.make_nimare_column_transformer(ma_bunch, (object(), "voxels"))


def test_column_transformer_keeps_a_transformer_off_the_descriptor_columns(ma_bunch):
    """A bare transformer would decompose the descriptor columns along with the voxels."""
    transformer = TruncatedSVD(n_components=1, random_state=RANDOM_SEED)
    descriptors = _descriptors(ma_bunch).ravel()

    scoped = ml.make_nimare_column_transformer(
        ma_bunch, (transformer, "voxels"), ("passthrough", "descriptors")
    ).fit_transform(ma_bunch.data)
    scoped = scoped.toarray() if sparse.issparse(scoped) else scoped
    bare = clone(transformer).fit_transform(ma_bunch.data)

    # Scoped: the voxel block is reduced and the descriptor column passes through.
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
    """A descriptor is named the way a ColumnTransformer names a frame column."""
    preprocessor = ml.make_nimare_column_transformer(
        descriptor_dataset,
        (TruncatedSVD(n_components=2, random_state=RANDOM_SEED), "voxels"),
        (SimpleImputer(strategy="median"), "sample_sizes"),
        ("passthrough", "Neurosynth_TFIDF__pain"),
        (StandardScaler(), "year"),
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
    preprocessor = ml.make_nimare_column_transformer(
        descriptor_dataset,
        (TruncatedSVD(n_components=2, random_state=RANDOM_SEED), "voxels"),
        (StandardScaler(), "descriptors"),
    )

    out = preprocessor.fit_transform(descriptor_dataset.data)
    out = out.toarray() if sparse.issparse(out) else out

    assert out.shape == (6, 5)


def test_a_column_name_that_is_not_a_block_or_a_descriptor_is_named(descriptor_dataset):
    """A typo names the blocks and the descriptors this bundle actually has."""
    with pytest.raises(ValueError, match="'nope' names neither a block nor a descriptor"):
        ml.make_nimare_column_transformer(
            descriptor_dataset,
            (TruncatedSVD(n_components=2), "voxels"),
            (StandardScaler(), "nope"),
        )


def test_an_unclaimed_block_is_not_dropped_quietly(descriptor_dataset):
    """A ColumnTransformer drops what nobody claims; here either block is worth refusing."""
    transformer = TruncatedSVD(n_components=2, random_state=RANDOM_SEED)

    with pytest.raises(ValueError, match="No transformer covers 3 of the descriptors columns"):
        ml.make_nimare_column_transformer(descriptor_dataset, (transformer, "voxels"))

    # the other way round matters more: this one would drop the whole matrix
    with pytest.raises(ValueError, match="No transformer covers 5 of the voxels columns"):
        ml.make_nimare_column_transformer(descriptor_dataset, (SimpleImputer(), "descriptors"))


@pytest.mark.parametrize(
    ("extra", "kwargs", "width"),
    [
        pytest.param([("drop", "descriptors")], {}, 2, id="dropping-descriptors-on-purpose"),
        pytest.param([], {"remainder": "passthrough"}, 5, id="keeping-them-untouched"),
    ],
)
def test_a_block_may_be_dropped_when_that_is_what_is_meant(
    descriptor_dataset, extra, kwargs, width
):
    """The refusal is about silence, not about the outcome."""
    preprocessor = ml.make_nimare_column_transformer(
        descriptor_dataset,
        (TruncatedSVD(n_components=2, random_state=RANDOM_SEED), "voxels"),
        *extra,
        **kwargs,
    )

    assert preprocessor.fit_transform(descriptor_dataset.data).shape[1] == width


def test_map_only_features_need_no_column_transformer(small_masker):
    """With no descriptors there is nothing to keep a transformer away from."""
    map_only = _bunch(
        sparse.csr_matrix(np.eye(4)),
        ids=[f"s{idx}-t" for idx in range(4)],
        groups=[f"s{idx}" for idx in range(4)],
        masker=small_masker,
    )

    assert map_only.descriptor_columns.stop == map_only.descriptor_columns.start
    # so the whole matrix is voxels and the transformer can be used on its own
    np.testing.assert_allclose(
        TruncatedSVD(n_components=2, random_state=RANDOM_SEED).fit_transform(map_only.data),
        TruncatedSVD(n_components=2, random_state=RANDOM_SEED).fit_transform(
            map_only.data[:, map_only.voxel_columns]
        ),
    )


def test_masker_transformer_takes_its_source_masker_from_the_bunch(ma_bunch, small_masker):
    """An atlas, or an aggregator built without one, gets the feature set's masker."""
    atlas = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)

    from_atlas = _step_transformer(
        ml.make_nimare_column_transformer(
            ma_bunch, (atlas, "voxels"), ("passthrough", "descriptors")
        )
    )
    unbound = MaskerTransformer(masker=atlas)
    from_aggregator = _step_transformer(
        ml.make_nimare_column_transformer(
            ma_bunch, (unbound, "voxels"), ("passthrough", "descriptors")
        )
    )

    assert from_atlas.source_masker is ma_bunch.masker
    assert from_aggregator.source_masker is ma_bunch.masker
    assert unbound.source_masker is None  # the caller's object is left alone


def test_atlas_reduction_needs_a_masker_somewhere(small_masker):
    """A feature set without a masker cannot place an atlas over its columns."""
    atlas = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), small_masker.mask_img.affine)
    maskerless = _bunch(
        sparse.csr_matrix(np.eye(4)),
        ids=[f"s{idx}-t" for idx in range(4)],
        groups=[f"s{idx}" for idx in range(4)],
    )

    with pytest.raises(ValueError, match="voxel order"):
        ml.make_nimare_column_transformer(
            maskerless, (atlas, "voxels"), ("passthrough", "descriptors")
        )


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
def labels_atlas(small_masker):
    """Return a 3D labels atlas on the small masker's grid."""
    return _atlas_images(small_masker.mask_img.affine)[0]


@pytest.fixture
def atlas_features(small_masker):
    """Return sparse features in the small masker's own voxel space."""
    n_voxels = int(small_masker.mask_img.get_fdata().sum())
    return sparse.csr_matrix(np.arange(6 * n_voxels, dtype=float).reshape(6, n_voxels))


@pytest.mark.parametrize("atlas_kind", ["labels", "maps"])
def test_masker_transformer_matches_nilearn(small_masker, atlas_features, atlas_kind):
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

    transformer = clone(
        MaskerTransformer(masker=atlas_img, source_masker=small_masker, batch_size=2)
    )
    transformed = transformer.fit_transform(atlas_features)

    reference.set_params(mask_img=small_masker.mask_img)
    expected = reference.fit(small_masker.mask_img).transform(
        small_masker.inverse_transform(atlas_features.toarray())
    )

    assert transformed.shape == (6, 2)
    np.testing.assert_allclose(transformed, expected)
    assert len(transformer.get_feature_names_out()) == 2


def test_masker_transformer_accepts_a_fetched_atlas(small_masker, atlas_features):
    """A Bunch from a nilearn fetcher is read for its maps and its labels."""
    _, maps_img = _atlas_images(small_masker.mask_img.affine)
    atlas = Bunch(maps=maps_img, labels=["Background", "left", "right"])

    transformer = MaskerTransformer(masker=atlas, source_masker=small_masker).fit(atlas_features)

    assert isinstance(transformer.masker_, NiftiMapsMasker)
    np.testing.assert_array_equal(transformer.get_feature_names_out(), ["left", "right"])


def test_masker_transformer_accepts_a_labels_frame(small_masker, atlas_features):
    """DiFuMo-style label frames are read for their name column."""
    _, maps_img = _atlas_images(small_masker.mask_img.affine)
    labels = pd.DataFrame({"component": [1, 2], "difumo_names": ["first", "second"]})

    transformer = MaskerTransformer(
        masker=Bunch(maps=maps_img, labels=labels), source_masker=small_masker
    )

    np.testing.assert_array_equal(
        transformer.fit(atlas_features).get_feature_names_out(), ["first", "second"]
    )


def test_masker_transformer_accepts_a_path(small_masker, atlas_features, tmp_path):
    """An atlas on disk is loaded from its path."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    path = tmp_path / "atlas.nii.gz"
    labels_img.to_filename(path)

    for atlas in (path, str(path)):
        transformer = MaskerTransformer(masker=atlas, source_masker=small_masker)
        assert transformer.fit_transform(atlas_features).shape == (6, 2)
        assert isinstance(transformer.masker_, NiftiLabelsMasker)


def test_masker_transformer_accepts_a_fetcher_name(small_masker, atlas_features, monkeypatch):
    """A nilearn fetcher can be named, and its arguments passed through."""
    from nilearn import datasets

    _, maps_img = _atlas_images(small_masker.mask_img.affine)
    calls = {}

    def fake_fetcher(dimension=None):
        calls["dimension"] = dimension
        return Bunch(maps=maps_img, labels=["left", "right"])

    monkeypatch.setattr(datasets, "fetch_atlas_pretend", fake_fetcher, raising=False)

    transformer = MaskerTransformer(
        masker="pretend", source_masker=small_masker, masker_kwargs={"dimension": 2}
    ).fit(atlas_features)

    assert calls == {"dimension": 2}
    np.testing.assert_array_equal(transformer.get_feature_names_out(), ["left", "right"])


def test_masker_transformer_accepts_a_prebuilt_masker(small_masker, atlas_features):
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

    transformer = MaskerTransformer(masker=atlas_masker, source_masker=small_masker).fit(
        atlas_features
    )

    assert transformer.masker_.strategy == "sum"
    assert atlas_masker.mask_img is None
    np.testing.assert_array_equal(transformer.get_feature_names_out(), ["region_a", "region_b"])


@pytest.mark.parametrize(
    ("atlas", "message"),
    [
        (None, "requires something to apply"),
        ("not_an_atlas_anywhere", "neither a file nor a nilearn atlas fetcher"),
        (42, "is not an atlas"),
    ],
)
def test_masker_transformer_rejects_unusable_atlases(small_masker, atlas, message):
    """Whatever the atlas is not, the message says what it could be."""
    with pytest.raises((ValueError, TypeError), match=message):
        MaskerTransformer(masker=atlas, source_masker=small_masker).fit(np.zeros((2, 8)))


def test_a_voxel_masker_reaches_the_voxel_columns(atlas_features, small_masker):
    """Smoothing and standardizing are voxel maskers, not aggregations."""
    plain = MaskerTransformer(NiftiMasker(), source_masker=small_masker)
    smoothed = MaskerTransformer(NiftiMasker(smoothing_fwhm=4), source_masker=small_masker)

    out = smoothed.fit(atlas_features).transform(atlas_features)

    # a voxel masker returns voxels, so the width is unchanged
    assert out.shape == atlas_features.shape
    # and it is a real transformation, not a pass through
    assert not np.allclose(out, plain.fit(atlas_features).transform(atlas_features))


def test_masker_transformer_requires_the_source_masker(small_masker):
    """The voxel order of the incoming features cannot be guessed."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)

    with pytest.raises(ValueError, match="voxel order"):
        MaskerTransformer(masker=labels_img).fit(np.zeros((2, 8)))


@pytest.mark.parametrize("atlas_kind", ["labels", "maps"])
def test_masker_transformer_names_match_the_columns_it_returns(
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

    transformer = MaskerTransformer(masker=atlas, source_masker=small_masker).fit(atlas_features)
    names = transformer.get_feature_names_out()

    assert len(names) == transformer.transform(atlas_features).shape[1]


def test_masker_transformer_names_reduced_features(small_masker, atlas_features):
    """Region names survive into the reduced dataset's feature names."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    dataset = _bunch(
        atlas_features,
        ids=[f"study_{idx}-task0" for idx in range(6)],
        groups=[f"study_{idx}" for idx in range(6)],
        masker=small_masker,
    )

    transformer = ml.make_nimare_column_transformer(
        dataset, (Bunch(maps=labels_img, labels=["one", "two"]), "voxels")
    )
    reduced = transformer.fit_transform(dataset.data)
    names = transformer.get_feature_names_out()

    assert reduced.shape == (6, 2)
    # prefixed by the step name, as scikit-learn does by default
    assert [str(name) for name in names] == [
        "maskertransformer__one",
        "maskertransformer__two",
    ]


def test_dataset_reduces_with_any_sklearn_transformer(ma_bunch):
    """A transformer NiMARE has never heard of works like the named ones."""
    transformer = SparseRandomProjection(n_components=1, random_state=RANDOM_SEED)
    train, test = _split(ma_bunch, 0.34, RANDOM_SEED)

    # Fitted on the training rows only, then applied to the held-out ones.
    reduced_train = transformer.fit_transform(_voxels(train))
    reduced_test = transformer.transform(_voxels(test))

    assert reduced_train.shape == (train.data.shape[0], 1)
    assert reduced_test.shape == (test.data.shape[0], 1)
    with pytest.raises(NotFittedError):
        clone(transformer).transform(_voxels(test))


def test_column_transformer_accepts_an_atlas(small_masker, atlas_features):
    """The feature set supplies the voxel order, so an atlas needs nothing else."""
    labels_img, _ = _atlas_images(small_masker.mask_img.affine)
    dataset = _bunch(
        atlas_features,
        ids=[f"study_{idx}-task0" for idx in range(6)],
        groups=[f"study_{idx}" for idx in range(6)],
        masker=small_masker,
    )

    # Map-only features: there is nothing to keep the aggregator away from.
    transformer = ml.make_nimare_column_transformer(
        dataset, (labels_img, "voxels"), ("passthrough", "descriptors")
    )

    assert isinstance(_step_transformer(transformer), MaskerTransformer)
    assert _step_transformer(transformer).source_masker is small_masker
    assert transformer.fit_transform(dataset.data).shape == (6, 2)


# ------------------------------------------------------------------ extraction


def test_public_surface_is_a_studyset_method_and_six_helpers():
    """Conversion belongs to the Studyset; nimare.ml holds what it cannot answer."""
    assert set(ml.__all__) == {
        "MAKernel",
        "MaskerTransformer",
        "clear_map_cache",
        "coefficient_image",
        "describe_fields",
        "make_nimare_column_transformer",
    }
    assert callable(Studyset.to_bunch)


def test_from_studyset(ml_studyset):
    """A Studyset becomes one row per analysis, with everything aligned."""
    studyset = ml_studyset

    features = studyset.to_bunch(
        descriptor_fields=["sample_sizes", ("annotations", "motor_label")],
        target_field=("annotations", "target_score"),
    )

    np.testing.assert_array_equal(features.ids, studyset.ids)
    np.testing.assert_array_equal(features.groups, studyset.metadata["study_id"])
    assert sparse.issparse(_voxels(features))
    # peak columns span the whole image grid, not just the mask
    grid = int(np.prod(studyset.masker.mask_img.shape))
    assert _voxels(features).shape == (len(studyset.ids), grid)
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

    assert isinstance(features, Bunch)
    assert np.issubdtype(features.data.dtype, np.number)
    assert len(features.groups) == len(studyset.ids)
    assert len(features.ids) == len(studyset.ids)
    assert len(features.target) == len(studyset.ids)
    assert len(features.feature_names) == features.data.shape[1]


def test_from_studyset_records_provenance(ml_studyset):
    """Every row can be traced back to its Studyset and its settings."""
    provenance = ml_studyset.to_bunch(descriptor_fields=["motor_label"]).provenance

    assert provenance["studyset_id"] == ml_studyset.id
    assert provenance["studyset_name"] == ml_studyset.name
    assert provenance["n_rows"] == len(ml_studyset.ids)
    assert provenance["missing_coordinates"] == "drop"
    assert provenance["dropped_ids"] == []
    assert provenance["descriptor_fields"] == ["motor_label"]
    assert provenance["masker"] == ml_studyset.masker.__class__.__name__


def test_from_studyset_aligns_peaks_by_id_not_position(ml_studyset):
    """Peak rows follow the analysis they came from, whatever order the view is in.

    ``Studyset.select_analyses`` accepts positions and so can hand back a view
    in any order.
    """
    reference = ml_studyset.to_bunch()
    expected = dict(zip(reference.ids, _peak_signature(reference)))

    positions = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    reordered = ml_studyset.select_analyses(positions).to_bunch()

    np.testing.assert_array_equal(reordered.ids, ml_studyset.ids[positions])
    assert _peak_signature(reordered) == [expected[id_] for id_ in reordered.ids]


def test_from_studyset_rejects_duplicate_analysis_ids(ml_studyset):
    """Duplicate ids would collapse into one row without being noticed."""
    doubled = ml_studyset.merge(_build_ml_studyset(coordinate_offset=1.0))

    # The merge keeps one analysis per id, so build the duplicate explicitly.
    duplicated = doubled.select_analyses(np.arange(len(doubled.ids) + 1) % len(doubled.ids))

    with pytest.raises(ValueError, match="must be unique"):
        duplicated.to_bunch()


@pytest.mark.parametrize("missing_coordinates", ["drop", "include"])
def test_from_studyset_missing_coordinates(ml_studyset, missing_coordinates):
    """Coordinate-less analyses are dropped, or kept as all-zero peak rows."""
    missing_id = "study_2-task0"
    studyset = _build_ml_studyset(ids_without_points={missing_id})

    features = studyset.to_bunch(
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
        assert _voxels(features)[row].nnz == 0
        assert _voxels(features)[row - 1].nnz > 0

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
        ml_studyset.to_bunch(**kwargs)


@pytest.mark.parametrize("missing_values", ["raise", "drop", "keep"])
def test_from_studyset_missing_descriptor_values(missing_values):
    """Missing values are reported, removed or left for a pipeline to impute."""
    missing_id = "study_3-task1"
    studyset = _build_ml_studyset(ids_without_score={missing_id})
    call = dict(
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
        studyset.to_bunch(target_field="score")


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("abstract", "is text"),
        ("not_a_field", "was not found in the Studyset metadata"),
        (("annotations", "sample_sizes"), "was not found in the Studyset annotations"),
        (("nowhere", "motor_label"), "Unsupported field selector source"),
        (42, "must be a field name"),
    ],
)
def test_from_studyset_rejects_unusable_descriptors(ml_studyset, selector, message):
    """Descriptor fields have to be readable as columns, and have to exist."""
    with pytest.raises((ValueError, TypeError), match=message):
        ml_studyset.to_bunch(descriptor_fields=[selector])


@pytest.fixture(scope="session")
def neurosynth_studyset(testdata_laird_studyset):
    """Return the Neurosynth Studyset from conftest, which annotates with 3,228 labels."""
    return testdata_laird_studyset


def test_annotation_labels_are_selected_by_pattern(neurosynth_studyset):
    """A pattern takes every matching label, under its own name."""
    studyset = neurosynth_studyset
    labels = [
        column
        for column in studyset.annotations_df.columns
        if column.startswith("Neurosynth_TFIDF__pa")
    ]

    features = studyset.to_bunch(descriptor_fields=[("annotations", "Neurosynth_TFIDF__pa*")])

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
        neurosynth_studyset.to_bunch(descriptor_fields=[selector])


def test_annotation_pattern_records_what_was_asked_for(neurosynth_studyset):
    """Provenance keeps the selector, not the thousands of names it expanded to."""
    features = neurosynth_studyset.to_bunch(
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
        studyset.to_bunch(descriptor_fields=[("annotations", "task_nam*")])

    # Named exactly, the same label is one field rather than a matrix of them,
    # so it is coded and its categories travel with the bundle.
    features = studyset.to_bunch(descriptor_fields=[("annotations", "task_name")])
    assert features.descriptor_categories["task_name"]


def test_patterns_span_several_annotations(ml_studyset):
    """Selection and extraction must agree about what the annotations hold."""
    studyset = ml_studyset.with_annotation(
        "second", ["extra_term"], np.arange(len(ml_studyset.ids), dtype=float)[:, None]
    )

    features = studyset.to_bunch(descriptor_fields=[("annotations", "*_term")])

    assert features.descriptor_names == ["extra_term"]


def test_a_dropped_analysis_cannot_also_be_missing():
    """missing_values speaks for the rows that are kept, not the ones already gone."""
    missing_id = "study_2-task0"
    studyset = _build_ml_studyset(ids_without_points={missing_id}, ids_without_score={missing_id})

    features = studyset.to_bunch(descriptor_fields=["score"], missing_values="raise")

    assert features.provenance["dropped_ids"] == [missing_id]
    assert features.provenance["missing_value_ids"] == {}


def test_a_target_is_constant_only_over_the_rows_that_are_kept():
    """The minority class can disappear with the analyses that had no coordinates."""
    studyset = _build_ml_studyset(ids_without_points={"study_0-task0", "study_0-task1"})
    labels = ["rare" if id_.startswith("study_0") else "common" for id_ in studyset.ids]
    studyset = studyset.with_metadata("grp", np.array(labels, dtype=object))

    with pytest.raises(ValueError, match="single value 'common' for every analysis that was kept"):
        studyset.to_bunch(target_field="grp")


def test_study_level_metadata_is_inherited_even_when_an_analysis_declares_it(ml_studyset):
    """One analysis declaring a field must not hide the study-level value from its siblings."""
    features = ml_studyset.to_bunch(descriptor_fields=["year"])

    years = ml_studyset.metadata.set_index("id").loc[features.ids, "year"]
    np.testing.assert_array_equal(_descriptors(features)[:, 0], years)
    assert np.isfinite(_descriptors(features)[:, 0]).all()


def test_masker_transformer_forgets_the_previous_atlas(atlas_features, small_masker):
    """A refit with a different atlas must not report the first one's region count."""
    affine = small_masker.mask_img.affine
    two_regions = np.zeros((4, 4, 4), dtype=np.int16)
    two_regions[0, :2, :2] = 1
    two_regions[1, :2, :2] = 2
    one_region = np.ones((4, 4, 4), dtype=np.int16)

    transformer = MaskerTransformer(
        masker=nib.Nifti1Image(two_regions, affine), source_masker=small_masker
    )
    transformer.fit_transform(atlas_features)
    assert len(transformer.get_feature_names_out()) == 2

    transformer.set_params(masker=nib.Nifti1Image(one_region, affine))
    transformer.fit(atlas_features)
    assert len(transformer.get_feature_names_out()) == 1


def test_from_studyset_rejects_repeated_descriptor_fields(ml_studyset):
    """The same field twice is a mistake, not two features."""
    with pytest.raises(ValueError, match="selected more than once"):
        ml_studyset.to_bunch(descriptor_fields=["motor_label", "motor_label"])


def test_from_studyset_reports_ambiguous_field_names(ml_studyset):
    """A name in two sources asks which one was meant."""
    annotated = ml_studyset.with_metadata("motor_label", np.ones(len(ml_studyset.ids)))

    with pytest.raises(ValueError, match="is ambiguous"):
        annotated.to_bunch(descriptor_fields=["motor_label"])


def test_from_studyset_reads_study_level_and_list_metadata(ml_studyset):
    """Study-level fields are inherited and list-valued fields are reduced."""
    features = ml_studyset.to_bunch(descriptor_fields=["year", "sample_sizes"])

    years = ml_studyset.metadata.set_index("id").loc[features.ids, "year"]
    np.testing.assert_array_equal(_descriptors(features)[:, 0], years)
    np.testing.assert_array_equal(
        _descriptors(features)[:, 1], [20 + idx for idx in range(features.data.shape[0])]
    )


def test_from_studyset_exports_categorical_targets(ml_studyset):
    """A categorical target reaches y as labels, ready for a classifier."""
    features = ml_studyset.to_bunch(target_field="comparison_task")

    assert sorted(set(features.target)) == ["flanker", "n-back"]
    np.testing.assert_array_equal(
        features.target,
        ml_studyset.metadata.set_index("id").loc[features.ids, "comparison_task"],
    )


def test_from_studyset_applies_a_target_transformer(ml_studyset):
    """A label extractor turns a text field into one label per analysis."""
    features = ml_studyset.to_bunch(
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
        ml_studyset.to_bunch(**kwargs)


def test_from_studyset_rejects_constant_targets(ml_studyset):
    """A target with one value has nothing to predict."""
    constant = ml_studyset.with_metadata("only_value", np.ones(len(ml_studyset.ids)))

    with pytest.raises(ValueError, match="nothing to predict"):
        constant.to_bunch(target_field="only_value")


def test_from_studyset_rejects_an_empty_studyset(ml_studyset):
    """There is nothing to convert without analyses."""
    empty = ml_studyset.select_analyses(np.zeros(len(ml_studyset.ids), dtype=bool))

    with pytest.raises(ValueError, match="no analyses"):
        empty.to_bunch()


def test_end_to_end_classification(ml_studyset):
    """The documented workflow runs from Studyset to grouped cross-validation."""
    from sklearn.linear_model import LogisticRegression

    features = ml_studyset.to_bunch(target_field="comparison_task")
    bunch = features

    pipeline = make_pipeline(
        ml.make_nimare_column_transformer(
            features,
            (TruncatedSVD(n_components=2, random_state=RANDOM_SEED), "voxels"),
            ("passthrough", "descriptors"),
        ),
        LogisticRegression(max_iter=500),
    )
    scores = cross_val_score(
        pipeline, bunch.data, bunch.target, cv=GroupKFold(n_splits=4), groups=bunch.groups
    )

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


@pytest.mark.performance_smoke
def test_from_studyset_meets_the_conversion_budget():
    """A 1,000-study Studyset converts and splits as a read, not a computation.

    The budget was three minutes and five gigabytes while conversion ran a
    kernel over every analysis. It now reads peaks, so anything near the old
    budget would be a regression rather than a pass.
    """
    # resource is Unix-only, and importing it at module scope makes the whole module
    # uncollectable on Windows. Only this test needs it, and it runs on Linux.
    import resource

    _, studyset = create_coordinate_studyset(foci=5, n_studies=1000, sample_size=30, seed=42)

    start = time.time()
    features = studyset.to_bunch()
    train, test = _split(features, 0.25, RANDOM_SEED)
    elapsed = time.time() - start
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    assert features.data.shape[0] == len(studyset.ids)
    assert set(train.groups).isdisjoint(test.groups)
    assert sparse.issparse(features.data)
    assert elapsed <= 10, f"conversion and split took {elapsed:.1f}s"
    assert peak_gb <= 2, f"peak memory was {peak_gb:.1f} GB"


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
        descriptor_fields=[("annotations", "groups[0].*")],
    )

    assert features.descriptor_names == ["groups[0].count"]


def test_a_character_class_pattern_still_selects_as_one():
    """Escaping is a fallback, so a pattern that already matches is left alone."""
    studyset = _build_ml_studyset(indexed_annotation=True)
    features = studyset.to_bunch(
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
            target_field="score",
            missing_values={"targets": "drop"},
        )


def test_missing_values_rejects_an_unknown_policy_in_a_mapping():
    """The policies are the same three whether or not a mapping is used."""
    studyset = _build_ml_studyset()

    with pytest.raises(ValueError, match="must be 'raise', 'drop' or 'keep'"):
        studyset.to_bunch(
            target_field="score",
            missing_values={"target": "impute"},
        )


def test_to_bunch_splits_without_splitting_a_study(ml_studyset):
    """test_size adds row positions, and a study belongs to exactly one side."""
    bunch = ml_studyset.to_bunch(test_size=0.5, random_state=RANDOM_SEED)

    assert set(bunch.groups[bunch.train]).isdisjoint(bunch.groups[bunch.test])
    assert sorted(np.concatenate([bunch.train, bunch.test]).tolist()) == list(
        range(bunch.data.shape[0])
    )
    assert len(bunch.train) and len(bunch.test)


def test_to_bunch_leaves_the_split_out_unless_it_is_asked_for(ml_studyset):
    """The bundle has the same keys as before when test_size is not given."""
    bunch = ml_studyset.to_bunch()

    assert "train" not in bunch
    assert "test" not in bunch


def test_to_bunch_split_is_the_one_a_group_splitter_would_make(ml_studyset):
    """It is GroupShuffleSplit over groups, so a caller can reproduce it."""
    bunch = ml_studyset.to_bunch(test_size=0.5, random_state=RANDOM_SEED)

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
        ml_studyset.to_bunch(test_size=test_size)


def test_to_bunch_split_needs_two_studies(ml_studyset):
    """One study cannot be split, since its analyses may not be separated."""
    one_study = ml_studyset.slice(ids=["study_0"], filter_level="study")

    with pytest.raises(ValueError, match="at least 2 studies"):
        one_study.to_bunch(test_size=0.5)


@pytest.mark.parametrize(
    "descriptor_transformer",
    [
        pytest.param(SimpleImputer(strategy="median"), id="one-transformer"),
        pytest.param(StandardScaler(), id="a-scaler-that-centres"),
        pytest.param(MaxAbsScaler(), id="a-sparse-safe-scaler"),
        pytest.param("passthrough", id="passthrough"),
    ],
)
def test_column_transformer_names_descriptors_after_their_fields(ma_bunch, descriptor_transformer):
    """A fitted coefficient has to be readable back to the field it belongs to."""
    preprocessor = ml.make_nimare_column_transformer(
        ma_bunch,
        (TruncatedSVD(n_components=1, random_state=RANDOM_SEED), "voxels"),
        (descriptor_transformer, "descriptors"),
    )
    preprocessor.fit(ma_bunch.data)

    names = [str(name) for name in preprocessor.get_feature_names_out()]

    # the ColumnTransformer selects by position, so without help this would be
    # the positional 'x2' rather than the descriptor's own name
    assert names[-1].endswith("motor_label")
    assert len(names) == 2


def test_column_transformer_feature_names_reach_the_estimator(ma_bunch):
    """The names line up with the coefficients a fitted model exposes."""
    pipeline = make_pipeline(
        ml.make_nimare_column_transformer(
            ma_bunch,
            (TruncatedSVD(n_components=1, random_state=RANDOM_SEED), "voxels"),
            (SimpleImputer(strategy="median"), "descriptors"),
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
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")]
    )
    block = bunch.data[:, bunch.descriptor_columns]
    assert block.shape[1] > 1000

    preprocessor = ml.make_nimare_column_transformer(
        bunch,
        (TruncatedSVD(n_components=2, random_state=RANDOM_SEED), "voxels"),
        (transformer, "descriptors"),
    )
    preprocessor.fit(bunch.data)
    out = preprocessor.transformers_[-1][1].transform(block)

    assert sparse.issparse(out) is stays_sparse
    assert out.shape == block.shape


def test_a_descriptor_transformer_that_centres_still_works(ma_bunch):
    """Densifying is the fallback, so a centring scaler is not refused."""
    preprocessor = ml.make_nimare_column_transformer(
        ma_bunch,
        (TruncatedSVD(n_components=1, random_state=RANDOM_SEED), "voxels"),
        (StandardScaler(), "descriptors"),
    )

    out = preprocessor.fit_transform(ma_bunch.data)
    out = out.toarray() if sparse.issparse(out) else out

    # centred, which is what StandardScaler was asked for
    assert out[:, -1].mean() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- MAKernel


def _kernel_step(bunch, kernel=None):
    """Return a fitted MAKernel over a bundle's peak columns."""
    step = MAKernel(kernel or MKDAKernel(r=4), source_masker=bunch.masker)
    return step, step.fit_transform(_voxels(bunch))


def test_ma_kernel_matches_a_direct_kernel_call(ml_studyset):
    """Convolving the peak columns gives the maps the kernel gives the Studyset."""
    bunch = ml_studyset.to_bunch()
    _, maps = _kernel_step(bunch)

    direct = MKDAKernel(r=4).transform(ml_studyset, return_type="sparse")
    direct = sparse.csr_matrix(direct) if not sparse.issparse(direct) else direct.tocsr()
    order = np.argsort(np.asarray(ml_studyset.ids, dtype=str))

    np.testing.assert_array_equal(maps.toarray()[order], direct.toarray())


def test_ma_kernel_output_is_in_the_maskers_voxel_space(ml_studyset):
    """Peaks come in over the grid and maps go out over the mask."""
    bunch = ml_studyset.to_bunch()
    step, maps = _kernel_step(bunch)

    assert _voxels(bunch).shape[1] == int(np.prod(bunch.masker.mask_img.shape))
    assert maps.shape == (len(ml_studyset.ids), bunch.masker.n_elements_)
    assert len(step.get_feature_names_out()) == maps.shape[1]


def test_ma_kernel_keeps_a_focus_outside_the_mask(small_masker):
    """A peak outside the mask still reaches into it through the kernel.

    This is why the peak columns span the whole image grid: in the mask's own
    column space such a focus has nowhere to be recorded, and its kernel is
    lost.
    """
    shape = small_masker.mask_img.shape
    outside = np.ravel_multi_index((2, 0, 0), shape)
    assert not small_masker.mask_img.get_fdata()[2, 0, 0]

    peaks = sparse.csr_matrix(
        ([1.0], ([0], [outside])), shape=(1, int(np.prod(shape))), dtype=float
    )
    maps = MAKernel(MKDAKernel(r=2), source_masker=small_masker).fit_transform(peaks)

    assert maps.nnz > 0


def test_ma_kernel_gives_an_analysis_without_peaks_an_empty_map(small_masker):
    """A row with no peaks keeps its place, as an all-zero map."""
    shape = small_masker.mask_img.shape
    peaks = sparse.lil_matrix((3, int(np.prod(shape))), dtype=float)
    peaks[1, np.ravel_multi_index((0, 0, 0), shape)] = 1.0

    maps = MAKernel(MKDAKernel(r=1), source_masker=small_masker).fit_transform(peaks.tocsr())

    assert maps.shape[0] == 3
    assert maps[0].nnz == 0 and maps[2].nnz == 0
    assert maps[1].nnz > 0


def test_ma_kernel_refuses_a_width_it_cannot_line_up_with_rows(small_masker):
    """Study-wise ALE needs a sample size per row, which a matrix cannot carry."""
    from nimare.meta.kernel import ALEKernel

    grid = int(np.prod(small_masker.mask_img.shape))
    peaks = sparse.csr_matrix((2, grid), dtype=float)

    with pytest.raises(ValueError, match="derives a separate kernel width"):
        MAKernel(ALEKernel(), source_masker=small_masker).fit(peaks)

    # A width that holds across analyses is fine either way round.
    MAKernel(ALEKernel(fwhm=4), source_masker=small_masker).fit(peaks)
    MAKernel(ALEKernel(sample_size=20), source_masker=small_masker).fit(peaks)


def test_ma_kernel_refuses_columns_that_are_not_the_grid(small_masker):
    """Handing it masked columns instead of peaks is caught, not convolved."""
    n_voxels = small_masker.n_elements_
    with pytest.raises(ValueError, match="span the whole image grid"):
        MAKernel(MKDAKernel(r=1), source_masker=small_masker).fit(
            sparse.csr_matrix((2, n_voxels), dtype=float)
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_masker": None}, "requires the source_masker"),
        ({"kernel": None}, "requires a kernel transformer"),
    ],
)
def test_ma_kernel_requires_both_of_its_parts(small_masker, kwargs, message):
    """Neither the kernel nor the space it works in has a sensible default."""
    settings = {"kernel": MKDAKernel(r=1), "source_masker": small_masker, **kwargs}
    grid = int(np.prod(small_masker.mask_img.shape))

    with pytest.raises(ValueError, match=message):
        MAKernel(**settings).fit(sparse.csr_matrix((2, grid), dtype=float))


def test_ma_kernel_accepts_a_kernel_class(small_masker):
    """A kernel transformer may be given as a class, as elsewhere in NiMARE."""
    grid = int(np.prod(small_masker.mask_img.shape))
    step = MAKernel(MKDAKernel, source_masker=small_masker)
    step.fit(sparse.csr_matrix((2, grid), dtype=float))

    assert isinstance(step.kernel_, MKDAKernel)


def test_ma_kernel_bandwidth_is_an_ordinary_hyperparameter(ml_studyset):
    """The point of the move: the kernel tunes like any other step."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    pipeline = make_pipeline(
        MAKernel(MKDAKernel(r=4), source_masker=bunch.masker),
        TruncatedSVD(n_components=2, random_state=RANDOM_SEED),
        Ridge(),
    )

    assert pipeline.get_params()["makernel__kernel__r"] == 4.0

    search = GridSearchCV(
        pipeline,
        {"makernel__kernel__r": [2.0, 6.0]},
        cv=GroupKFold(n_splits=2),
    )
    search.fit(_voxels(bunch), bunch.target, groups=bunch.groups)

    assert search.best_params_["makernel__kernel__r"] in (2.0, 6.0)


def test_masker_transformer_reads_either_column_space(small_masker, labels_atlas):
    """An atlas may act on the peaks directly, or on the maps a kernel made."""
    shape = small_masker.mask_img.shape
    peaks = sparse.lil_matrix((3, int(np.prod(shape))), dtype=float)
    peaks[0, np.ravel_multi_index((0, 0, 0), shape)] = 1.0
    peaks[1, np.ravel_multi_index((1, 1, 1), shape)] = 1.0
    peaks = peaks.tocsr()

    on_peaks = MaskerTransformer(labels_atlas, source_masker=small_masker, batch_size=2)
    counts = on_peaks.fit_transform(peaks)
    assert on_peaks.on_grid_ is True

    maps = MAKernel(MKDAKernel(r=2), source_masker=small_masker).fit_transform(peaks)
    on_maps = MaskerTransformer(labels_atlas, source_masker=small_masker, batch_size=2)
    regions = on_maps.fit_transform(maps)
    assert on_maps.on_grid_ is False

    assert counts.shape == regions.shape == (3, on_peaks.get_feature_names_out().size)
    # a labels masker averages within each region, so one peak among four
    # voxels reads as 0.25 and the kernel that spreads it reads as more
    assert counts[0].sum() == pytest.approx(0.25)
    assert regions[0].sum() > counts[0].sum()
    assert not counts[2].any()


def test_masker_transformer_refuses_a_width_from_neither_space(small_masker, labels_atlas):
    """A width matching neither the mask nor its grid is a mistake, not a guess."""
    with pytest.raises(ValueError, match="either over the masker's voxels"):
        MaskerTransformer(labels_atlas, source_masker=small_masker).fit(
            np.zeros((2, small_masker.n_elements_ + 1))
        )


def test_sparse_support_is_read_per_instance():
    """Whether a transformer reads sparse input can depend on its arguments."""
    from nimare.ml.compose import _handles_sparse

    assert _handles_sparse(TruncatedSVD(n_components=50)) is True
    assert _handles_sparse(TruncatedSVD(n_components=1)) is True
    # StandardScaler centres by default, which sparse input rules out
    assert _handles_sparse(StandardScaler()) is False
    assert _handles_sparse(StandardScaler(with_mean=False)) is True
    assert _handles_sparse(MaxAbsScaler()) is True
    # no scikit-learn encoder reads sparse input
    assert _handles_sparse(OneHotEncoder()) is False


def test_nimare_transformers_declare_that_they_read_sparse_input(ma_bunch):
    """Keep a wide voxel block sparse through NiMARE's own transformers.

    A voxel block spans the whole image grid, so densifying one to please a
    transformer that would have taken it sparse costs gigabytes.
    """
    from nimare.ml.compose import _handles_sparse

    kernel = MAKernel(MKDAKernel(r=10), source_masker=ma_bunch.masker)

    assert _handles_sparse(kernel) is True
    assert _handles_sparse(MaskerTransformer(ma_bunch.masker)) is True
    assert _handles_sparse(make_pipeline(kernel, TruncatedSVD(n_components=2))) is True


def test_a_sparse_voxel_block_is_not_densified_before_a_transformer(ma_bunch):
    """The block stays sparse on its way into a transformer that accepts it."""
    seen = []

    class _Recording(TruncatedSVD):
        # a ColumnTransformer calls fit_transform, so recording only fit would
        # observe the sparse probe rather than the block
        def fit_transform(self, X, y=None):
            seen.append(sparse.issparse(X))
            return super().fit_transform(X, y)

    preprocessor = ml.make_nimare_column_transformer(
        ma_bunch,
        (_Recording(n_components=2, random_state=RANDOM_SEED), "voxels"),
        ("passthrough", "descriptors"),
    )
    assert preprocessor.transformers[0][1].named_steps["name"].func is None

    preprocessor.fit_transform(ma_bunch.data)

    # the probe fits a clone first, so the block itself is the last call
    assert seen[-1] is True


def test_a_pipeline_is_probed_whole_not_by_its_first_step(ma_bunch):
    """A step that takes sparse input may also pass it on.

    ``SimpleImputer`` accepts sparse columns and hands sparse columns to
    whatever follows, so a ``StandardScaler`` behind it still refuses to centre
    them. Probing only the first step called that chain sparse-safe and left
    the real failure to surface inside cross-validation.
    """
    from nimare.ml.compose import _handles_sparse

    preserves_sparsity = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    assert _handles_sparse(preserves_sparsity) is False
    assert _handles_sparse(SimpleImputer(strategy="median")) is True

    # the same chain is fine once the scaler is told not to centre
    sparse_safe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False))
    assert _handles_sparse(sparse_safe) is True


def test_a_descriptor_chain_that_needs_dense_input_gets_it(ma_bunch):
    """The whole point of the probe: the chain runs rather than raising."""
    preprocessor = ml.make_nimare_column_transformer(
        ma_bunch,
        ("drop", "voxels"),
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), "descriptors"),
    )
    out = preprocessor.fit_transform(ma_bunch.data)

    assert out.shape == (ma_bunch.data.shape[0], 1)
    assert float(np.mean(out)) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------- categorical descriptors


def test_a_categorical_descriptor_becomes_a_code_and_its_categories(ml_studyset):
    """A string field enters the numeric matrix as a position, not a value."""
    features = ml_studyset.to_bunch(descriptor_fields=[("metadata", "comparison_task")])

    categories = features.descriptor_categories["comparison_task"]
    assert categories == ["flanker", "n-back"]

    codes = _dense(_descriptor_block(features))[:, 0]
    assert set(np.unique(codes)) <= set(range(len(categories)))

    # every row's code names the label that row actually carries
    raw = ml_studyset.metadata.set_index("id").loc[features.ids, "comparison_task"]
    np.testing.assert_array_equal(
        [categories[int(code)] for code in codes], raw.to_numpy(dtype=str)
    )


def test_a_numeric_descriptor_has_no_categories(ml_studyset):
    """Only a coded field appears in descriptor_categories."""
    features = ml_studyset.to_bunch(descriptor_fields=["year", ("metadata", "comparison_task")])

    assert set(features.descriptor_categories) == {"comparison_task"}


def test_a_text_descriptor_is_still_refused(ml_studyset):
    """Free text has no reading as a column, coded or otherwise."""
    with pytest.raises(ValueError, match="is text"):
        ml_studyset.to_bunch(descriptor_fields=[("texts", "abstract")])


@pytest.mark.parametrize("missing_values", ["raise", "drop", "keep"])
def test_a_missing_category_follows_the_missing_policy(missing_values):
    """A gap in a categorical field is a gap, not a category of its own."""
    missing_id = "study_3-task1"
    studyset = _build_ml_studyset()
    labels = np.array(
        ["red" if index % 2 else "blue" for index in range(len(studyset.ids))], dtype=object
    )
    labels[list(studyset.ids).index(missing_id)] = None
    studyset = studyset.with_metadata("colour", labels)

    call = {"descriptor_fields": ["colour"], "missing_values": missing_values}
    if missing_values == "raise":
        with pytest.raises(ValueError, match=f"Missing values in colour .*{missing_id}"):
            studyset.to_bunch(**call)
        return

    features = studyset.to_bunch(**call)
    assert features.descriptor_categories["colour"] == ["blue", "red"]
    codes = _dense(_descriptor_block(features))[:, 0]
    if missing_values == "drop":
        assert missing_id not in set(features.ids)
        assert np.isfinite(codes).all()
    else:
        row = int(np.flatnonzero(features.ids == missing_id)[0])
        assert np.isnan(codes[row])


def test_an_encoder_gets_the_categories_and_gives_back_the_labels(ml_studyset):
    """The caller picks the encoder; the bundle supplies what it cannot know."""
    features = ml_studyset.to_bunch(descriptor_fields=[("metadata", "comparison_task")])
    preprocessor = ml.make_nimare_column_transformer(
        features,
        ("drop", "voxels"),
        (OneHotEncoder(handle_unknown="ignore"), "comparison_task"),
    )
    preprocessor.fit(features.data)

    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    assert names == ["comparison_task_flanker", "comparison_task_n-back"]


def test_an_encoder_is_told_the_categories_so_folds_agree(ml_studyset):
    """A fold missing a category must not change how many columns a model gets."""
    features = ml_studyset.to_bunch(descriptor_fields=[("metadata", "comparison_task")])
    preprocessor = ml.make_nimare_column_transformer(
        features,
        ("drop", "voxels"),
        (OneHotEncoder(handle_unknown="ignore"), "comparison_task"),
    )
    encoder = _step_transformer(preprocessor, index=1)
    np.testing.assert_array_equal(encoder.categories, [np.array([0.0, 1.0])])

    # a subset carrying only one of the two categories still yields two columns
    codes = _dense(_descriptor_block(features))[:, 0]
    one_category = np.flatnonzero(codes == codes[0])
    reduced = preprocessor.fit_transform(features.data[one_category])
    assert reduced.shape == (len(one_category), 2)


def test_a_caller_may_choose_a_different_encoding(ml_studyset):
    """Nothing about the code prefers one-hot; it is the caller's decision."""
    features = ml_studyset.to_bunch(
        descriptor_fields=[("metadata", "comparison_task")],
        target_field=("annotations", "target_score"),
    )
    preprocessor = ml.make_nimare_column_transformer(
        features,
        ("drop", "voxels"),
        (OrdinalEncoder(), "comparison_task"),
    )
    out = preprocessor.fit_transform(features.data, features.target)

    assert out.shape == (len(features.ids), 1)


def test_a_coded_category_cannot_reach_a_model_raw(ml_studyset):
    """A code is a label's position, so passing it through is a wrong answer."""
    features = ml_studyset.to_bunch(descriptor_fields=[("metadata", "comparison_task"), "year"])

    with pytest.raises(ValueError, match="cannot be passed through"):
        ml.make_nimare_column_transformer(
            features,
            ("drop", "voxels"),
            ("passthrough", "comparison_task"),
            (SimpleImputer(), "year"),
        )

    with pytest.raises(ValueError, match="also covers"):
        ml.make_nimare_column_transformer(
            features, ("drop", "voxels"), (StandardScaler(), "descriptors")
        )


def test_a_coded_category_may_be_dropped_on_purpose(ml_studyset):
    """The guard is about silence, not about the outcome."""
    features = ml_studyset.to_bunch(descriptor_fields=[("metadata", "comparison_task"), "year"])
    preprocessor = ml.make_nimare_column_transformer(
        features,
        ("drop", "voxels"),
        ("drop", "comparison_task"),
        (SimpleImputer(), "year"),
    )
    out = preprocessor.fit_transform(features.data)

    assert out.shape == (len(features.ids), 1)


# --------------------------------------------------------------- performance


def test_a_named_block_stays_a_slice(ma_bunch):
    """A block spec must not become a list of every column it covers.

    A sparse matrix sliced by a ``slice`` skips scipy's fancy-index path, and
    the list itself was large enough to drag the garbage collector in.
    """
    from nimare.ml.compose import _by_name, _columns_of

    assert _by_name(ma_bunch, ["voxels"]) == ma_bunch.voxel_columns
    columns, _ = _columns_of(ma_bunch, "voxels")
    assert isinstance(columns, slice)

    # several names concatenate, in the order they were given
    both = _by_name(ma_bunch, ["descriptors", "voxels"])
    assert both[0] == ma_bunch.descriptor_columns.start
    assert both[1] == ma_bunch.voxel_columns.start


def test_the_source_masker_is_not_refitted_per_fold(small_masker, labels_atlas):
    """Resolving the source masker must not fit it again.

    ``clone`` strips a nilearn masker's fitted state, so a transformer that
    asked for a fitted one paid a fresh nilearn fit on every fold.
    """
    fits = []

    class _CountingMasker(type(small_masker)):
        def fit(self, imgs=None, y=None):
            fits.append(1)
            return super().fit(imgs, y)

    source = clone(small_masker)
    source.__class__ = _CountingMasker
    grid = int(np.prod(small_masker.mask_img.shape))
    peaks = sparse.csr_matrix((2, grid), dtype=float)

    MAKernel(MKDAKernel(r=1), source_masker=source).fit(peaks)
    MaskerTransformer(labels_atlas, source_masker=source).fit(peaks)

    assert fits == []


# --------------------------------------------------------------- map cache


def _row_signatures(peaks):
    """Return what identifies each row of a peak block, as the cache keys it."""
    peaks = peaks.tocsr()
    return [
        (
            peaks.indices[peaks.indptr[r] : peaks.indptr[r + 1]].tobytes(),
            peaks.data[peaks.indptr[r] : peaks.indptr[r + 1]].tobytes(),
        )
        for r in range(peaks.shape[0])
    ]


def _grid_peaks(masker, rows=3):
    """Return a small grid-space peak block with one peak per row."""
    shape = masker.mask_img.shape
    peaks = sparse.lil_matrix((rows, int(np.prod(shape))), dtype=float)
    for row in range(rows):
        peaks[row, np.ravel_multi_index((row % 2, row % 2, 0), shape)] = 1.0
    return peaks.tocsr()


def test_a_cache_changes_the_cost_and_nothing_else(ml_studyset):
    """Cached maps must equal the maps convolved afresh."""
    ml.clear_map_cache()
    bunch = ml_studyset.to_bunch()
    peaks = _voxels(bunch)

    plain = MAKernel(MKDAKernel(r=4), source_masker=bunch.masker).fit_transform(peaks)
    cached = MAKernel(MKDAKernel(r=4), source_masker=bunch.masker, cache=True).fit_transform(peaks)

    assert abs(plain - cached).nnz == 0
    served = ml.clear_map_cache()
    assert served["misses"] == len(set(_row_signatures(peaks)))
    assert served["hits"] == peaks.shape[0] - served["misses"]


def test_a_cache_serves_the_second_pass(ml_studyset):
    """A row convolved once is not convolved again."""
    ml.clear_map_cache()
    bunch = ml_studyset.to_bunch()
    peaks = _voxels(bunch)

    first = MAKernel(MKDAKernel(r=4), source_masker=bunch.masker, cache=True).fit_transform(peaks)
    second = MAKernel(MKDAKernel(r=4), source_masker=bunch.masker, cache=True).fit_transform(peaks)

    assert abs(first - second).nnz == 0
    served = ml.clear_map_cache()
    assert served["misses"] == len(set(_row_signatures(peaks)))
    assert served["hits"] == 2 * peaks.shape[0] - served["misses"]


def test_a_cache_survives_cloning(small_masker):
    """Folds share maps because the cache belongs to the process, not the step."""
    ml.clear_map_cache()
    peaks = _grid_peaks(small_masker)
    step = MAKernel(MKDAKernel(r=1), source_masker=small_masker, cache=True)

    step.fit_transform(peaks)
    clone(step).fit(peaks).transform(peaks)

    # the clone convolved nothing: every row it was given was already there
    served = ml.clear_map_cache()
    assert served["misses"] == len(set(_row_signatures(peaks)))
    assert served["hits"] == 2 * peaks.shape[0] - served["misses"]


def test_a_cached_row_names_the_kernel_that_made_it(small_masker):
    """MKDAKernel(r=1) and KDAKernel(r=1) report the same parameters.

    Keying on the parameters alone would serve one kernel's maps for the
    other's, which is the whole reason the class is in the key.
    """
    from nimare.meta.kernel import KDAKernel

    assert MKDAKernel(r=1).get_params() == KDAKernel(r=1).get_params()

    ml.clear_map_cache()
    peaks = _grid_peaks(small_masker)
    mkda = MAKernel(MKDAKernel(r=1), source_masker=small_masker, cache=True).fit_transform(peaks)
    kda = MAKernel(KDAKernel(r=1), source_masker=small_masker, cache=True).fit_transform(peaks)

    direct = MAKernel(KDAKernel(r=1), source_masker=small_masker).fit_transform(peaks)
    assert abs(kda - direct).nnz == 0
    assert mkda.shape == kda.shape
    # the two kernels share no rows, so each row was convolved by each of them
    served = ml.clear_map_cache()
    assert served["misses"] == 2 * len(set(_row_signatures(peaks)))


def test_a_cached_row_names_the_mask_it_was_made_in(small_masker):
    """The same peaks in a different mask are different maps."""
    ml.clear_map_cache()
    other = get_masker(
        nib.Nifti1Image(np.ones(small_masker.mask_img.shape, dtype=np.uint8), np.eye(4))
    )
    peaks = _grid_peaks(small_masker)

    first = MAKernel(MKDAKernel(r=1), source_masker=small_masker, cache=True).fit_transform(peaks)
    second = MAKernel(MKDAKernel(r=1), source_masker=other, cache=True).fit_transform(peaks)

    assert first.shape[1] != second.shape[1]
    # the same peaks in a different mask were convolved again, not served
    served = ml.clear_map_cache()
    assert served["misses"] == 2 * len(set(_row_signatures(peaks)))


def test_cached_rows_are_copies_not_views(small_masker):
    """A slice of a CSR's indices is a view that pins the array it came from."""
    from nimare.ml.kernel import _MAPS

    ml.clear_map_cache()
    MAKernel(MKDAKernel(r=1), source_masker=small_masker, cache=True).fit_transform(
        _grid_peaks(small_masker)
    )

    assert _MAPS.rows
    for indices, data in _MAPS.rows.values():
        assert indices.base is None
        assert data.base is None
    ml.clear_map_cache()


def test_a_cache_can_be_emptied(small_masker):
    """Holding the whole feature matrix is opt-in, and so is letting it go."""
    ml.clear_map_cache()
    peaks = _grid_peaks(small_masker)
    MAKernel(MKDAKernel(r=1), source_masker=small_masker, cache=True).fit_transform(peaks)

    served = ml.clear_map_cache()
    assert served["rows"] and served["misses"]
    assert ml.clear_map_cache() == {"rows": 0, "hits": 0, "misses": 0}


def test_a_cache_holds_across_the_folds_of_a_cross_validation(ml_studyset):
    """The point of the cache: every row convolved once, whatever the split."""
    ml.clear_map_cache()
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    peaks = _voxels(bunch)
    pipeline = make_pipeline(
        MAKernel(MKDAKernel(r=4), source_masker=bunch.masker, cache=True),
        TruncatedSVD(n_components=2, random_state=RANDOM_SEED),
        Ridge(),
    )

    cross_val_score(pipeline, peaks, bunch.target, groups=bunch.groups, cv=GroupKFold(2))

    # every row is convolved once however many folds touch it
    served = ml.clear_map_cache()
    assert served["misses"] == served["rows"]
    assert served["hits"] > 0
    assert served["hits"] + served["misses"] > peaks.shape[0]


# ---------------------------------------------------------- coefficient_image


@pytest.mark.parametrize("transformer", [None, TruncatedSVD(2, random_state=RANDOM_SEED)])
def test_coefficient_image_puts_weights_back_in_the_brain(ml_studyset, transformer):
    """A weight per feature becomes a weight per voxel, whatever reduced them."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    steps = [MAKernel(MKDAKernel(r=4), source_masker=bunch.masker)]
    steps += [transformer] if transformer is not None else []
    pipeline = make_pipeline(*steps, Ridge()).fit(_voxels(bunch), bunch.target)

    image = ml.coefficient_image(pipeline, bunch)

    assert image.shape == bunch.masker.mask_img.shape
    np.testing.assert_allclose(image.affine, bunch.masker.mask_img.affine)
    assert np.count_nonzero(np.asarray(image.dataobj))


@pytest.mark.parametrize(
    "step",
    [
        StandardScaler(),
        StandardScaler(with_mean=False),
        MaxAbsScaler(),
        MinMaxScaler(),
        RobustScaler(),
        TruncatedSVD(3, random_state=RANDOM_SEED),
        PCA(3, random_state=RANDOM_SEED),
        PCA(3, whiten=True, random_state=RANDOM_SEED),
        VarianceThreshold(),
        make_pipeline(StandardScaler(), PCA(3, random_state=RANDOM_SEED)),
    ],
    ids=lambda step: type(step).__name__ + ("_whiten" if getattr(step, "whiten", False) else ""),
)
def test_coefficient_image_weights_reproduce_the_model(ml_studyset, step):
    """The voxel weights score every map as the fitted model does, up to its intercept.

    A model scoring ``w @ (A x + c)`` scores ``(A.T w) @ x`` plus a constant, so that is what a
    weight over voxels has to be; reading ``w`` back as a point with ``inverse_transform``
    gives ``A^-1 (w - c)`` instead.
    """
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    # ALE maps rather than binary MKDA ones: on a 0/1 map most scalers' scales are 1, and a
    # weight read back through them comes out right by accident
    kernel = MAKernel(ALEKernel(fwhm=8), source_masker=bunch.masker)
    # a centred target keeps the predictions near zero, so float32 rounding of a large
    # intercept does not hide an offset read back as a weight (PCA's mean_)
    pipeline = make_pipeline(
        kernel, FunctionTransformer(_to_dense, accept_sparse=True), step, Ridge()
    ).fit(_voxels(bunch), bunch.target - bunch.target.mean())

    image = ml.coefficient_image(pipeline, bunch)

    _assert_weights_reproduce(pipeline, _voxels(bunch), image, bunch.masker)


@pytest.mark.parametrize(
    "step, message",
    [
        (QuantileTransformer(n_quantiles=5), "QuantileTransformer"),
        (MinMaxScaler(clip=True), "clip=True clips values"),
    ],
    ids=["quantile", "clipped_minmax"],
)
def test_coefficient_image_refuses_a_step_it_cannot_read_weights_through(
    ml_studyset, step, message
):
    """Having an inverse_transform is not enough: weights move back by the transpose."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    pipeline = make_pipeline(
        MAKernel(MKDAKernel(r=4), source_masker=bunch.masker),
        FunctionTransformer(_to_dense, accept_sparse=True),
        step,
        Ridge(),
    ).fit(_voxels(bunch), bunch.target)

    with pytest.raises(ValueError, match=message):
        ml.coefficient_image(pipeline, bunch)


def _zero_variance_pca():
    """Return a whitened PCA with a zero-variance last component, as a rank-deficient fit has."""
    rng = np.random.default_rng(RANDOM_SEED)
    pca = PCA(3, whiten=True, random_state=RANDOM_SEED).fit(rng.normal(size=(20, 5)))
    pca.explained_variance_[-1] = 0.0
    return pca


def _pca_floors_whitening():
    """Report whether this scikit-learn's PCA.transform floors a zero whitening scale.

    Later releases floor it at machine epsilon; earlier ones, such as 1.4, divide by zero
    and return inf, which no model downstream can be fitted on.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return bool(np.isfinite(_zero_variance_pca().transform(np.ones((1, 5)))).all())


def test_coefficient_image_keeps_a_zero_variance_component_finite():
    """A whitened component with no variance does not turn the weights into inf."""
    from nimare.ml.interpret import _pull_back

    weights = np.random.default_rng(RANDOM_SEED).normal(size=(1, 3))

    assert np.isfinite(_pull_back(_zero_variance_pca(), weights)).all()


@pytest.mark.skipif(
    not _pca_floors_whitening(),
    reason="this scikit-learn's PCA.transform divides by a zero whitening scale",
)
def test_coefficient_image_reads_a_zero_variance_component_as_pca_transforms_it():
    """A whitened component with no variance is scaled as PCA.transform scales it.

    PCA.transform floors a whitening scale at machine epsilon; dividing a weight by the
    unfloored zero gives inf where the transform gives a finite feature.
    """
    from nimare.ml.interpret import _pull_back

    rng = np.random.default_rng(RANDOM_SEED + 1)
    pca = _zero_variance_pca()
    weights = rng.normal(size=(1, 3))

    pulled = _pull_back(pca, weights)

    probe = rng.normal(size=(4, 5))
    linear_part = pca.transform(probe) - pca.transform(np.zeros((1, 5)))
    np.testing.assert_allclose(probe @ pulled.T, linear_part @ weights.T, rtol=1e-6)


def _brain_atlases(mask_img):
    """Return a labels atlas with unequal regions and three overlapping maps over a mask."""
    inside = np.asarray(mask_img.dataobj) > 0
    i = np.indices(inside.shape)[0]
    # three slabs along x of very different sizes, so a region's size matters
    labels = np.where(inside, 1 + (i > 25) + (i > 70), 0).astype(np.int16)
    centres = np.array([[30, 50, 40], [45, 60, 45], [60, 50, 40]])
    grid = np.stack(np.indices(inside.shape), axis=-1)
    maps = np.stack([np.exp(-((grid - c) ** 2).sum(-1) / (2 * 12.0**2)) for c in centres], axis=-1)
    return (
        nib.Nifti1Image(labels, mask_img.affine),
        nib.Nifti1Image(maps * inside[..., None], mask_img.affine),
    )


def _atlas_reducer(name, mask_img):
    labels_img, maps_img = _brain_atlases(mask_img)
    if name == "labels_mean":
        return NiftiLabelsMasker(labels_img=labels_img, resampling_target="data", reports=False)
    if name == "labels_sum":
        return NiftiLabelsMasker(
            labels_img=labels_img, strategy="sum", resampling_target="data", reports=False
        )
    if name == "maps":
        return NiftiMapsMasker(maps_img=maps_img, resampling_target="data", reports=False)
    if name == "smoothing":
        return NiftiMasker(smoothing_fwhm=6)
    return NiftiMasker()


def _assert_weights_reproduce(pipeline, voxels, image, masker):
    """Assert the voxel weights score every MA map as the pipeline does, up to a constant.

    The predictions and the weight image are float32, so the offset can only be constant
    to float32 rounding of the values involved, which can exceed a small fraction of the
    predictions' spread when that spread is narrow. A weight read back wrongly is off by
    the order of the spread itself.
    """
    maps = _dense(pipeline[0].transform(voxels)).astype(float)
    weights = get_masker(masker.mask_img).transform(image).ravel().astype(float)
    predicted = np.asarray(pipeline.predict(voxels), dtype=float)
    assert np.ptp(predicted) > 0
    scored = maps @ weights
    offset = predicted - scored
    rounding = 32 * np.finfo(np.float32).eps * (np.abs(predicted).max() + np.abs(scored).max())
    assert np.ptp(offset) <= 1e-5 * np.ptp(predicted) + rounding


@pytest.mark.parametrize("reducer", ["labels_mean", "labels_sum", "maps", "smoothing", "voxels"])
def test_coefficient_image_reads_an_atlas_back_per_voxel(ml_studyset, reducer):
    """A region's weight is shared out over its voxels as the reduction weighted them."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    pipeline = make_pipeline(
        MAKernel(ALEKernel(fwhm=8), source_masker=bunch.masker),
        MaskerTransformer(
            _atlas_reducer(reducer, bunch.masker.mask_img), source_masker=bunch.masker
        ),
        # ALE region values are small, and the default alpha would shrink them to nothing
        Ridge(alpha=1e-6),
    ).fit(_voxels(bunch), bunch.target - bunch.target.mean())

    image = ml.coefficient_image(pipeline, bunch)

    _assert_weights_reproduce(pipeline, _voxels(bunch), image, bunch.masker)


def test_coefficient_image_paints_regions_when_asked(ml_studyset):
    """atlas='region' shows each region's weight on every one of its voxels."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    labels_img, _ = _brain_atlases(bunch.masker.mask_img)
    pipeline = make_pipeline(
        MAKernel(ALEKernel(fwhm=8), source_masker=bunch.masker),
        MaskerTransformer(
            _atlas_reducer("labels_mean", bunch.masker.mask_img), source_masker=bunch.masker
        ),
        Ridge(alpha=1e-6),
    ).fit(_voxels(bunch), bunch.target)

    painted = np.asarray(ml.coefficient_image(pipeline, bunch, atlas="region").dataobj)
    per_voxel = np.asarray(ml.coefficient_image(pipeline, bunch).dataobj)

    labels = np.asarray(labels_img.dataobj)
    for region, weight in enumerate(pipeline[-1].coef_, start=1):
        inside = labels == region
        np.testing.assert_allclose(painted[inside], weight, rtol=1e-6)
        np.testing.assert_allclose(per_voxel[inside], weight / inside.sum(), rtol=1e-6)


@pytest.mark.parametrize(
    "reducer, message",
    [
        (
            lambda img: NiftiLabelsMasker(
                labels_img=_brain_atlases(img)[0],
                strategy="maximum",
                resampling_target="data",
                reports=False,
            ),
            "not linear in the voxels",
        ),
        (lambda img: NiftiMasker(standardize="zscore_sample"), "depends on the other rows"),
    ],
)
def test_coefficient_image_refuses_an_atlas_with_no_linear_part(ml_studyset, reducer, message):
    """A maximum, or a standardisation across rows, has no transpose to read weights through."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    pipeline = make_pipeline(
        MAKernel(ALEKernel(fwhm=8), source_masker=bunch.masker),
        MaskerTransformer(reducer(bunch.masker.mask_img), source_masker=bunch.masker),
        Ridge(),
    ).fit(_voxels(bunch), bunch.target)

    with pytest.raises(ValueError, match=message):
        ml.coefficient_image(pipeline, bunch)


def _haufe(maps, scores):
    """Return cov(maps, scores) @ pinv(cov(scores)), one row per score."""
    maps = maps - maps.mean(axis=0)
    scores = scores - scores.mean(axis=0)
    n = len(scores)
    return ((maps.T @ scores / (n - 1)) @ np.linalg.pinv(scores.T @ scores / (n - 1))).T


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [StandardScaler(), PCA(3, random_state=RANDOM_SEED)],
        # refused for weights, since it has no linear part; a pattern needs none
        [QuantileTransformer(n_quantiles=5)],
    ],
    ids=["model_on_voxels", "scaler_and_pca", "quantile"],
)
@pytest.mark.parametrize("n_targets", [1, 2])
def test_coefficient_image_pattern_is_the_haufe_pattern(ml_studyset, steps, n_targets):
    """A pattern is how each voxel covaries with the model's scores, whatever came between."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    voxels = _voxels(bunch)
    rng = np.random.default_rng(RANDOM_SEED)
    target = (
        bunch.target
        if n_targets == 1
        else np.column_stack([bunch.target, rng.normal(size=len(bunch.target))])
    )
    pipeline = make_pipeline(
        MAKernel(ALEKernel(fwhm=8), source_masker=bunch.masker),
        FunctionTransformer(_to_dense, accept_sparse=True),
        *[clone(step) for step in steps],
        Ridge(),
    ).fit(voxels, target)

    image = ml.coefficient_image(pipeline, bunch, kind="pattern", X=voxels)

    maps = _dense(pipeline[0].transform(voxels)).astype(float)
    expected = _haufe(maps, pipeline.predict(voxels).reshape(len(maps), -1))
    pattern = get_masker(bunch.masker.mask_img).transform(image).reshape(n_targets, -1)
    # MA maps are float32, so agreement is to float32 precision
    np.testing.assert_allclose(pattern, expected, atol=1e-4 * np.abs(expected).max())


def test_coefficient_image_pattern_reads_through_a_column_transformer(ml_studyset):
    """The scores include the descriptors; the pattern is over the voxels the kernel made."""
    bunch = ml_studyset.to_bunch(
        target_field=("annotations", "target_score"), descriptor_fields=["year"]
    )
    pipeline = make_pipeline(
        ml.make_nimare_column_transformer(
            bunch,
            (
                make_pipeline(
                    MAKernel(ALEKernel(fwhm=8), source_masker=bunch.masker),
                    TruncatedSVD(2, random_state=RANDOM_SEED),
                ),
                "voxels",
            ),
            (StandardScaler(), "descriptors"),
        ),
        Ridge(),
    ).fit(bunch.data, bunch.target)

    image = ml.coefficient_image(pipeline, bunch, kind="pattern")

    kernel = MAKernel(ALEKernel(fwhm=8), source_masker=bunch.masker).fit(_voxels(bunch))
    maps = _dense(kernel.transform(_voxels(bunch))).astype(float)
    expected = _haufe(maps, pipeline.predict(bunch.data)[:, None]).ravel()
    pattern = get_masker(bunch.masker.mask_img).transform(image).ravel()
    # MA maps are float32, so agreement is to float32 precision
    np.testing.assert_allclose(pattern, expected, atol=1e-4 * np.abs(expected).max())


def test_coefficient_image_checks_what_it_is_asked_for(ml_studyset):
    """An unknown kind or atlas reading, or a pattern of a model with no readout, is refused."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    pipeline = make_pipeline(MAKernel(MKDAKernel(r=4), source_masker=bunch.masker), Ridge()).fit(
        _voxels(bunch), bunch.target
    )
    with pytest.raises(ValueError, match="kind must be one of"):
        ml.coefficient_image(pipeline, bunch, kind="weight")
    with pytest.raises(ValueError, match="atlas must be one of"):
        ml.coefficient_image(pipeline, bunch, atlas="regions")

    from sklearn.tree import DecisionTreeRegressor

    tree = make_pipeline(
        MAKernel(MKDAKernel(r=4), source_masker=bunch.masker),
        DecisionTreeRegressor(random_state=RANDOM_SEED),
    ).fit(_voxels(bunch), bunch.target)
    with pytest.raises(ValueError, match="needs the linear readout"):
        ml.coefficient_image(tree, bunch, kind="pattern")


def test_coefficient_image_reads_through_a_column_transformer(ml_studyset):
    """Only the weights over the voxel block name places in the brain."""
    bunch = ml_studyset.to_bunch(
        target_field=("annotations", "target_score"), descriptor_fields=["year"]
    )
    pipeline = make_pipeline(
        ml.make_nimare_column_transformer(
            bunch,
            (
                make_pipeline(
                    MAKernel(MKDAKernel(r=4), source_masker=bunch.masker),
                    TruncatedSVD(2, random_state=RANDOM_SEED),
                ),
                "voxels",
            ),
            (StandardScaler(), "descriptors"),
        ),
        Ridge(),
    ).fit(bunch.data, bunch.target)

    image = ml.coefficient_image(pipeline, bunch)

    assert image.shape == bunch.masker.mask_img.shape


def test_coefficient_image_takes_weights_it_is_given(ml_studyset):
    """Permutation importance gives weights the model does not carry itself."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    pipeline = make_pipeline(
        MAKernel(MKDAKernel(r=4), source_masker=bunch.masker),
        TruncatedSVD(2, random_state=RANDOM_SEED),
        Ridge(),
    ).fit(_voxels(bunch), bunch.target)

    image = ml.coefficient_image(pipeline, bunch, coef=np.array([1.0, -1.0]))

    assert image.shape == bunch.masker.mask_img.shape


def test_coefficient_image_says_what_it_cannot_undo(ml_studyset):
    """A step with no inverse stops the walk rather than guessing past it."""
    bunch = ml_studyset.to_bunch(target_field=("annotations", "target_score"))
    voxels = _voxels(bunch)

    unreadable = make_pipeline(
        MAKernel(MKDAKernel(r=4), source_masker=bunch.masker),
        FunctionTransformer(lambda x: _dense(x)[:, :3]),
        Ridge(),
    ).fit(voxels, bunch.target)
    with pytest.raises(ValueError, match="no 'inverse_func' cannot be undone"):
        ml.coefficient_image(unreadable, bunch)

    from sklearn.cluster import KMeans

    unweighted = make_pipeline(
        MAKernel(MKDAKernel(r=4), source_masker=bunch.masker),
        TruncatedSVD(2, random_state=RANDOM_SEED),
        KMeans(n_clusters=2, n_init=2, random_state=RANDOM_SEED),
    ).fit(voxels)
    with pytest.raises(ValueError, match="neither 'coef_' nor 'feature_importances_'"):
        ml.coefficient_image(unweighted, bunch)
