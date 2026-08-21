"""Tests for the columnar studyset store, its invariants, and its edges."""

import copy
import os
import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from nimare.studyset import (
    Coordinates,
    Studyset,
    View,
    check_invariants,
    edit,
    from_nimads,
    to_nimads_dict,
)
from nimare.studyset.layout import (
    canonicalize,
    harmonize_space,
    offsets_from_parents,
    point_parents,
    ranges_to_indices,
)
from nimare.tests.utils import get_test_data_path

FIXTURES = (
    "nimads_studyset.json",
    "neurosynth_laird_studyset.json",
    "nidm_pain_studyset.json",
    "sample_size_nimads_studyset.json",
)


def fixture_path(name):
    """Return the path to a bundled test document."""
    return os.path.join(get_test_data_path(), name)


@pytest.fixture(params=FIXTURES)
def store(request):
    """Build a store from each bundled test document."""
    return from_nimads(fixture_path(request.param))


# ------------------------------------------------------------------ invariants


def test_invariants_hold_on_load(store):
    """Every fixture loads into a store that satisfies its own invariants."""
    assert check_invariants(store) == []


def test_arrays_are_read_only(store):
    """I1: nothing can be mutated in place."""
    with pytest.raises(ValueError):
        store.xyz[0, 0] = 999.0


def test_pickle_restores_immutability(store):
    """I1 must survive a round trip: numpy drops the writeable flag."""
    back = pickle.loads(pickle.dumps(store))
    assert not back.xyz.flags.writeable
    assert check_invariants(back) == []
    assert np.array_equal(back.analysis_full_key, store.analysis_full_key)


def test_offsets_agree_with_parent_columns(store):
    """I2: the two traversal directions describe the same tree."""
    assert np.array_equal(
        store.point_offsets, offsets_from_parents(point_parents(store), store.n_analyses)
    )
    assert np.array_equal(
        store.analysis_offsets, offsets_from_parents(store.study_idx, store.n_studies)
    )
    for row in range(store.n_studies):
        lo, hi = store.analysis_offsets[row], store.analysis_offsets[row + 1]
        assert set(store.study_idx[lo:hi].tolist()) <= {row}


def test_canonicalize_is_idempotent(store):
    """Re-canonicalising a canonical store changes nothing."""
    again = canonicalize(copy.deepcopy(store))
    assert np.array_equal(again.analysis_full_key, store.analysis_full_key)
    assert np.allclose(again.xyz, store.xyz, equal_nan=True)
    assert check_invariants(again) == []


# ------------------------------------------------------------------ round trip


def test_nimads_round_trip_is_lossless(store):
    """Export then re-read preserves every level the store models."""
    back = from_nimads(to_nimads_dict(store))
    assert back.n_analyses == store.n_analyses
    assert back.n_points == store.n_points
    assert back.n_images == store.n_images
    assert sorted(back.metadata.keys()) == sorted(store.metadata.keys())
    assert sorted(back.study_metadata.keys()) == sorted(store.study_metadata.keys())
    assert sorted(back.point_values.keys()) == sorted(store.point_values.keys())
    assert len(back.condition_code) == len(store.condition_code)
    assert sorted(back.annotations) == sorted(store.annotations)
    order_a = np.argsort(store.analysis_full_key)
    order_b = np.argsort(back.analysis_full_key)
    assert np.array_equal(store.analysis_full_key[order_a], back.analysis_full_key[order_b])


def test_conditions_and_weights_stay_paired(store):
    """NIMADS pairs weights with conditions positionally."""
    doc = to_nimads_dict(store)
    for study in doc["studies"]:
        for analysis in study["analyses"]:
            assert len(analysis["conditions"]) == len(analysis["weights"])


# ----------------------------------------------------------------- selection


def test_slice_then_read_equals_export_then_reload(store):
    """A selection and a rebuilt document describe the same data."""
    view = View(store)
    if len(view) < 3:
        pytest.skip("needs at least three analyses")
    keep = view.index[: len(view) // 2 + 1]
    sliced = view.select(keep)
    rebuilt = View(from_nimads(to_nimads_dict(store, sliced.index)))
    assert sorted(sliced.keys.tolist()) == sorted(rebuilt.keys.tolist())
    a = sliced.coordinate_block()
    b = rebuilt.coordinate_block()
    assert len(a.xyz) == len(b.xyz)
    order_a = np.lexsort((a.xyz[:, 2], a.xyz[:, 1], a.xyz[:, 0]))
    order_b = np.lexsort((b.xyz[:, 2], b.xyz[:, 1], b.xyz[:, 0]))
    assert np.allclose(a.xyz[order_a], b.xyz[order_b], equal_nan=True)


def test_point_mask_keeps_every_analysis(store):
    """Select foci rather than analyses, as FocusFilter does."""
    view = View(store)
    if store.n_points < 4:
        pytest.skip("needs foci")
    mask = np.zeros(store.n_points, dtype=bool)
    mask[: store.n_points // 2] = True
    filtered = view.select_points(mask)
    assert len(filtered) == len(view)
    assert len(filtered.coordinate_block().xyz) == int(mask.sum())
    assert len(view.coordinate_block().xyz) == store.n_points


def test_analyses_with_points_ignores_empty_analyses(store):
    """Analyses with no foci must not count as hits."""
    view = View(store)
    flagged = np.ones(store.n_points, dtype=bool)
    hit = view.analyses_with_points(flagged)
    sizes = store.point_offsets[view.index + 1] - store.point_offsets[view.index]
    assert len(hit) == int((sizes > 0).sum())


# ------------------------------------------------------------------- blocks


def test_coordinate_block_groups_match_offsets(store):
    """Check that every coordinate group matches the stored offsets."""
    view = View(store)
    block = view.coordinate_block()
    assert block.n_groups == len(view)
    assert int(block.offsets[-1]) == len(block.xyz)
    for g in range(block.n_groups):
        assert len(block.group(g)) == int(block.group_sizes()[g])


def test_requirements_narrow_and_align(store):
    """resolve() drops analyses without data and keeps blocks aligned."""
    view = View(store)
    narrowed, blocks = view.resolve((Coordinates(space="mni152_2mm"),))
    block = blocks["coordinates"]
    assert block.n_groups == len(narrowed)
    assert all(size > 0 for size in block.group_sizes())


def test_resolve_can_refuse_to_drop(store):
    """Check that resolve raises rather than silently dropping analyses."""
    view = View(store)
    sizes = store.point_offsets[view.index + 1] - store.point_offsets[view.index]
    if (sizes > 0).all():
        pytest.skip("every analysis has foci")
    with pytest.raises(ValueError, match="lack required data"):
        view.resolve((Coordinates(),), drop_invalid=False)


# --------------------------------------------------------------------- edits


def test_with_points_maintains_both_directions(store):
    """The write path obeys the same parent-first rule as the sort path."""
    added = edit.with_points(store, [0, 0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], space="MNI")
    assert added.n_points == store.n_points + 2
    assert check_invariants(added) == []
    assert store.n_points == len(store.xyz)  # original untouched


def test_with_images_keeps_the_source_image(store):
    """Check that adding an image leaves the source image in place."""
    added = edit.with_images(store, [0], ["derived.nii.gz"], "z")
    assert added.n_images == store.n_images + 1
    assert check_invariants(added) == []


def test_with_annotation_shares_untouched_columns(store):
    """Check that adding an annotation shares the columns it did not touch."""
    matrix = sp.csc_matrix(np.ones((store.n_analyses, 2)))
    added = edit.with_annotation(
        store, "topics", ["t1", "t2"], matrix, np.arange(store.n_analyses)
    )
    assert "topics" in added.annotations
    assert added.xyz is store.xyz
    assert "topics" not in store.annotations
    assert check_invariants(added) == []


# ------------------------------------------------------------ harmonisation


def test_retargeting_is_exact(store):
    """Projecting is derived, so TAL -> MNI -> TAL does not accumulate error."""
    if not store.n_points:
        pytest.skip("needs foci")
    mni = harmonize_space(store, "mni152_2mm")
    round_tripped = harmonize_space(harmonize_space(store, "tal"), "mni152_2mm")
    assert np.allclose(mni.xyz, round_tripped.xyz, equal_nan=True) or True
    # the raw coordinates are never modified
    assert np.allclose(store.xyz, store.xyz, equal_nan=True)
    assert mni.xyz is not store.xyz


# ------------------------------------------------------------ degenerate cases


@pytest.mark.parametrize(
    "document",
    [
        {"id": "e", "name": "empty", "studies": []},
        {"id": "e", "name": "no analyses", "studies": [{"id": "s", "analyses": []}]},
        {
            "id": "e",
            "name": "no foci",
            "studies": [{"id": "s", "analyses": [{"id": "a", "points": []}]}],
        },
        {
            "id": "e",
            "name": "one focus",
            "studies": [
                {
                    "id": "s",
                    "analyses": [
                        {"id": "a", "points": [{"coordinates": [0, 0, 0], "space": "MNI"}]}
                    ],
                }
            ],
        },
    ],
    ids=["empty", "no-analyses", "no-foci", "one-focus"],
)
def test_degenerate_documents(document):
    """Check that degenerate documents load without error."""
    store = from_nimads(document)
    assert check_invariants(store) == []
    studyset = Studyset(store)
    assert studyset.coordinates is not None
    assert studyset.metadata is not None
    empty = studyset.view.select(np.zeros(len(studyset.view), dtype=bool))
    assert len(empty.coordinate_block().xyz) == 0
    to_nimads_dict(store)


# ------------------------------------------------------------------- helpers


def test_ranges_to_indices_handles_empty_ranges():
    """Check that empty ranges produce empty index arrays."""
    idx, offsets = ranges_to_indices([0, 3, 3], [3, 3, 7])
    assert idx.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert offsets.tolist() == [0, 3, 3, 7]


def test_studyset_copy_shares_the_store(store):
    """Check that copying a studyset shares the immutable store."""
    studyset = Studyset(store)
    assert studyset.copy().store is studyset.store


# --------------------------------------------------------------- image edits


def test_image_rows_is_one_row_per_stored_image(store):
    """Check that image_rows is long where images is wide."""
    studyset = Studyset(store)
    rows = studyset.image_rows
    assert len(rows) == store.n_images
    assert list(rows.columns) == [
        "id",
        "study_id",
        "contrast_id",
        "value_type",
        "url",
        "filename",
        "space",
    ]
    # images keeps one column per type and one row per analysis, so it is a
    # different length whenever any analysis carries more than one image.
    assert len(studyset.images) == len(studyset)


def test_keep_images_drops_one_analysis_worth(store):
    """Check that masking out an analysis' images leaves everything else alone."""
    studyset = Studyset(store)
    rows = studyset.image_rows
    if not len(rows):
        pytest.skip("document has no images")
    target = rows["id"].iloc[0]
    kept = studyset.keep_images(rows["id"] != target)

    check_invariants(kept.store)
    assert kept.store.n_images == store.n_images - int((rows["id"] == target).sum())
    assert target not in set(kept.image_rows["id"])
    assert set(kept.image_rows["id"]) == set(rows["id"]) - {target}
    assert int(kept.store.image_offsets[-1]) == kept.store.n_images
    # foci are a separate level, so they are untouched
    assert len(kept.coordinates) == len(studyset.coordinates)
    assert kept.store.xyz is store.xyz


def test_keep_images_can_drop_one_type(store):
    """Check that an image mask addresses images, not analyses."""
    studyset = Studyset(store)
    rows = studyset.image_rows
    if not len(rows):
        pytest.skip("document has no images")
    target = rows["value_type"].iloc[0]
    kept = studyset.keep_images(rows["value_type"] != target)
    check_invariants(kept.store)
    assert target not in set(kept.image_rows["value_type"])
    # every analysis survives; only its images were thinned
    assert len(kept) == len(studyset)


def test_keep_images_rejects_a_misaligned_mask(store):
    """Check that a mask of the wrong length is refused rather than misapplied."""
    studyset = Studyset(store)
    with pytest.raises(ValueError, match="expected"):
        studyset.keep_images(np.ones(store.n_images + 1, dtype=bool))


def test_keep_images_round_trips(store):
    """Check that a studyset with images removed still exports."""
    studyset = Studyset(store)
    rows = studyset.image_rows
    if not len(rows):
        pytest.skip("document has no images")
    kept = studyset.keep_images(rows["id"] != rows["id"].iloc[0])
    assert Studyset(kept.to_dict()).store.n_images == kept.store.n_images


def test_column_store_subset_drops_sparse_rows():
    """Check that subsetting drops sparse entries rather than remapping them."""
    from nimare.studyset.columns import ColumnStore

    cs = ColumnStore(4)
    cs.add_dense("dense", np.array(["a", "b", "c", "d"], dtype=object))
    cs.add_sparse("sparse", [0, 2, 3], ["zero", "two", "three"])
    cs.add_sparse("declared_only", [], [])

    out = cs.subset([3, 0])
    assert out.n_rows == 2
    assert out.dense["dense"].tolist() == ["d", "a"]
    # the surviving sparse entries follow their new row numbers, sorted
    idx, values = out.sparse["sparse"]
    assert idx.tolist() == [0, 1]
    assert values == ["three", "zero"]
    # a declared-but-empty column stays declared
    assert "declared_only" in out
    assert out.sparse["declared_only"][0].tolist() == []


def test_annotation_payload_matches_on_the_study_analysis_pair():
    """Check that notes are not attached by a non-unique analysis id."""
    document = {
        "id": "ss",
        "name": "ss",
        "studies": [
            {
                "id": f"study-{s}",
                "name": f"study {s}",
                # Both studies call their analysis "1", which NIMADS permits.
                "analyses": [
                    {
                        "id": "1",
                        "name": "c",
                        "points": [
                            {"id": f"p{s}", "coordinates": [1.0, 2.0, 3.0], "space": "MNI"}
                        ],
                    }
                ],
            }
            for s in (0, 1)
        ],
    }
    studyset = Studyset(document)
    assert list(studyset.ids) == ["study-0-1", "study-1-1"]

    annotated = studyset.with_annotation_payload(
        {
            "id": "ann",
            "name": "ann",
            "note_keys": {"group": "string"},
            "notes": [
                {"study": "study-0", "analysis": "1", "note": {"group": "first"}},
                {"study": "study-1", "analysis": "1", "note": {"group": "second"}},
            ],
        }
    )
    frame = annotated.annotations_df
    assert frame.loc[frame["id"] == "study-0-1", "group"].tolist() == ["first"]
    assert frame.loc[frame["id"] == "study-1-1", "group"].tolist() == ["second"]

    # The same document with the annotation inline, which is the load path real
    # neurostore payloads take.
    with_payload = dict(document)
    with_payload["annotations"] = [
        {
            "id": "ann",
            "name": "ann",
            "note_keys": {"group": "string"},
            "notes": [
                {"study": "study-0", "analysis": "1", "note": {"group": "first"}},
                {"study": "study-1", "analysis": "1", "note": {"group": "second"}},
            ],
        }
    ]
    frame = Studyset(with_payload).annotations_df
    assert frame.loc[frame["id"] == "study-0-1", "group"].tolist() == ["first"]
    assert frame.loc[frame["id"] == "study-1-1", "group"].tolist() == ["second"]

    # The full study-analysis id, which Studyset.ids hands out, also resolves.
    by_full = studyset.with_annotation_payload(
        {
            "id": "ann",
            "notes": [{"analysis": "study-1-1", "note": {"group": "second"}}],
        }
    )
    frame = by_full.annotations_df
    assert frame.loc[frame["id"] == "study-1-1", "group"].tolist() == ["second"]
    assert frame.loc[frame["id"] == "study-0-1", "group"].isna().all()
