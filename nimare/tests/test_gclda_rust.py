"""Regression tests for the Rust GCLDA trainer against the Python implementation.

These tests are skipped unless the Rust binary has been built:

    cd rust/gclda && cargo build --release
"""

import json
import os
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nimare import annotate, decode

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BINARY = os.path.join(REPO_ROOT, "rust", "gclda", "target", "release", "gclda-train")

requires_rust = pytest.mark.skipif(
    not os.path.exists(BINARY),
    reason="gclda-train not built; run `cd rust/gclda && cargo build --release`",
)


@pytest.fixture(scope="module")
def small_corpus():
    """Build a small, deterministic corpus shared by the regression tests."""
    rng = np.random.default_rng(0)
    n_docs, n_terms, n_peaks = 12, 8, 60
    ids = [f"study-{i:03d}" for i in range(n_docs)]
    counts = pd.DataFrame(
        rng.integers(0, 4, size=(n_docs, n_terms)),
        index=ids,
        columns=[f"term_{j}" for j in range(n_terms)],
    )
    counts.iloc[:, 0] = 0  # one all-zero term, to exercise column dropping
    doc_for_peak = rng.integers(0, n_docs, size=n_peaks)
    coords = pd.DataFrame(
        {
            "id": [ids[d] for d in doc_for_peak],
            "x": rng.uniform(-60, 60, n_peaks).round(1),
            "y": rng.uniform(-90, 60, n_peaks).round(1),
            "z": rng.uniform(-50, 70, n_peaks).round(1),
        }
    )
    return counts, coords


@requires_rust
def test_rust_loader_exposes_decoder_interface(small_corpus, mni_mask, tmp_path):
    """The loaded result must expose exactly what nimare.decode consumes."""
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    result = annotate.gclda_rs.train_gclda_rust(
        counts,
        coords,
        mask=mask_path,
        out_dir=str(tmp_path / "out"),
        binary=BINARY,
        n_topics=4,
        n_regions=2,
        symmetric=True,
        seed_init=1,
        n_iters=3,
        loglikely_freq=1,
    )

    n_vox = int(np.asanyarray(mni_mask.dataobj).astype(bool).sum())
    assert result.p_topic_g_voxel_.shape == (n_vox, 4)
    assert result.p_voxel_g_topic_.shape == (n_vox, 4)
    assert result.p_topic_g_word_.shape == (len(result.vocabulary), 4)
    assert result.p_word_g_topic_.shape == (len(result.vocabulary), 4)
    assert result.mask is not None
    # The all-zero term must have been dropped.
    assert "term_0" not in result.vocabulary
    assert len(result.vocabulary) == 7


@requires_rust
def test_rust_loader_decoders_agree_between_mmap_and_resident(small_corpus, mni_mask, tmp_path):
    """gclda.decode/encode must agree between mmap'd and resident arrays.

    A memory-mapped array can behave differently from an in-memory one
    inside downstream numpy operations (e.g. ``np.dot``) if the loader gets
    dtype, shape, or byte order wrong. The required test above never calls
    the decoders at all, so it cannot catch that -- this test drives the
    actual ``nimare.decode``/``nimare.decode.encode`` entry points against
    both an ``mmap=True`` and an ``mmap=False`` load of the *same* trained
    model directory and checks they agree exactly.
    """
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)
    out_dir = str(tmp_path / "out")

    annotate.gclda_rs.train_gclda_rust(
        counts,
        coords,
        mask=mask_path,
        out_dir=out_dir,
        binary=BINARY,
        n_topics=4,
        n_regions=2,
        symmetric=True,
        seed_init=1,
        n_iters=3,
        loglikely_freq=1,
    )

    mmap_model = annotate.gclda_rs.load_gclda_model(out_dir, mask=mask_path, mmap=True)
    resident_model = annotate.gclda_rs.load_gclda_model(out_dir, mask=mask_path, mmap=False)

    # mmap=True must actually produce a memmap, not silently fall back.
    assert isinstance(mmap_model.p_topic_g_voxel_, np.memmap)
    assert not isinstance(resident_model.p_topic_g_voxel_, np.memmap)

    arr = np.zeros(mni_mask.shape, np.int32)
    arr[40:44, 45:49, 40:44] = 1
    roi_img = nib.Nifti1Image(arr, mni_mask.affine)

    for model in (mmap_model, resident_model):
        decoded_roi, _ = decode.discrete.gclda_decode_roi(model, roi_img)
        assert isinstance(decoded_roi, pd.DataFrame)

        decoded_map, _ = decode.continuous.gclda_decode_map(model, roi_img)
        assert isinstance(decoded_map, pd.DataFrame)

        encoded_img, _ = decode.encode.gclda_encode(model, "term 1 term 2")
        assert isinstance(encoded_img, nib.Nifti1Image)

    roi_mmap, _ = decode.discrete.gclda_decode_roi(mmap_model, roi_img)
    roi_resident, _ = decode.discrete.gclda_decode_roi(resident_model, roi_img)
    np.testing.assert_array_equal(roi_mmap["Weight"].to_numpy(), roi_resident["Weight"].to_numpy())

    map_mmap, _ = decode.continuous.gclda_decode_map(mmap_model, roi_img)
    map_resident, _ = decode.continuous.gclda_decode_map(resident_model, roi_img)
    np.testing.assert_array_equal(map_mmap["Weight"].to_numpy(), map_resident["Weight"].to_numpy())

    encoded_mmap, _ = decode.encode.gclda_encode(mmap_model, "term 1 term 2")
    encoded_resident, _ = decode.encode.gclda_encode(resident_model, "term 1 term 2")
    np.testing.assert_array_equal(encoded_mmap.get_fdata(), encoded_resident.get_fdata())


@requires_rust
def test_rust_loader_params_round_trip_types(small_corpus, mni_mask, tmp_path):
    """``.params`` values must round-trip with correct Python types.

    A plausible bug: ``symmetric`` coming back as the string ``"true"``
    (truthy for *any* non-empty string, so a bug like this would pass a
    careless ``if model.params["symmetric"]`` check) instead of the bool
    ``True``. json.load should get this right automatically, but nothing
    else in this file checks it.
    """
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    result = annotate.gclda_rs.train_gclda_rust(
        counts,
        coords,
        mask=mask_path,
        out_dir=str(tmp_path / "out"),
        binary=BINARY,
        n_topics=4,
        n_regions=2,
        symmetric=True,
        seed_init=1,
        n_iters=3,
        loglikely_freq=1,
    )

    assert result.params["symmetric"] is True
    assert isinstance(result.params["n_topics"], int) and result.params["n_topics"] == 4
    assert isinstance(result.params["n_regions"], int) and result.params["n_regions"] == 2
    assert isinstance(result.params["seed_init"], int) and result.params["seed_init"] == 1
    assert isinstance(result.params["alpha"], float)
    assert isinstance(result.ids, list) and all(isinstance(i, str) for i in result.ids)


@requires_rust
def test_export_gclda_tsvs_round_trips_through_rust_reader(small_corpus, mni_mask, tmp_path):
    """``export_gclda_tsvs`` output must round-trip through the real reader.

    An ``id``-labeled first counts column and a named ``id`` column in
    coordinates are what the Rust TSV reader expects -- verified here by
    running the real binary against it, not by inspecting the TSV text.
    """
    counts, coords = small_corpus
    stage_dir = tmp_path / "staged"
    counts_path, coords_path = annotate.gclda_rs.export_gclda_tsvs(counts, coords, stage_dir)

    with open(counts_path, encoding="utf-8") as fo:
        header = fo.readline().rstrip("\n").split("\t")
    assert header[0] == "id"

    coords_header = pd.read_csv(coords_path, sep="\t", nrows=0).columns.tolist()
    assert {"id", "x", "y", "z"} <= set(coords_header)

    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)
    out_dir = str(tmp_path / "out")

    completed = subprocess.run(
        [
            BINARY,
            "--counts",
            counts_path,
            "--coordinates",
            coords_path,
            "--mask",
            mask_path,
            "--out-dir",
            out_dir,
            "--n-topics",
            "4",
            "--n-regions",
            "2",
            "--symmetric",
            "true",
            "--seed-init",
            "1",
            "--n-iters",
            "1",
            "--loglikely-freq",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    with open(os.path.join(out_dir, "model.json"), encoding="utf-8") as fo:
        meta = json.load(fo)
    # ids are the sorted-as-strings intersection of count/coordinate IDs.
    assert meta["ids"] == sorted(meta["ids"])


INTEGER_ARRAYS = (
    "wtoken_topic_idx", "peak_topic_idx", "peak_region_idx",
    "n_peak_tokens_doc_by_topic", "n_peak_tokens_region_by_topic",
    "n_word_tokens_word_by_topic", "n_word_tokens_doc_by_topic",
    "total_n_word_tokens_by_topic",
)
FLOAT_ARRAYS = ("regions_mu", "regions_sigma", "regions_precision", "regions_log_norm")


@requires_rust
@pytest.mark.parametrize(
    "n_regions,symmetric", [(2, True), (4, True), (1, False), (3, False)]
)
def test_rust_matches_python_every_iteration(
    small_corpus, mni_mask, tmp_path, n_regions, symmetric
):
    """Rust and Python state must be identical after EVERY iteration.

    Comparing only endpoints would leave a divergence introduced at
    iteration 3 undiagnosable. This reports the first differing iteration
    and the first differing element.
    """
    counts, coords = small_corpus
    n_iters = 12
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    py_dir = tmp_path / "py_state"
    py_dir.mkdir()
    model = annotate.gclda.GCLDAModel(
        counts, coords, mask=mask_path, n_topics=4, n_regions=n_regions,
        symmetric=symmetric, seed_init=1,
    )
    model.fit(n_iters=n_iters, loglikely_freq=n_iters, dump_state_dir=str(py_dir))

    rs_dir = tmp_path / "rs_state"
    counts_path, coords_path = annotate.gclda_rs.export_gclda_tsvs(
        counts, coords, str(tmp_path / "inputs")
    )
    subprocess.run(
        [
            BINARY, "--counts", counts_path, "--coordinates", coords_path,
            "--mask", mask_path, "--out-dir", str(tmp_path / "rs_out"),
            "--n-topics", "4", "--n-regions", str(n_regions),
            "--symmetric", "true" if symmetric else "false",
            "--seed-init", "1", "--n-iters", str(n_iters),
            "--loglikely-freq", str(n_iters),
            "--dump-state-dir", str(rs_dir),
        ],
        check=True,
    )

    for it in range(1, n_iters + 1):
        py = np.load(py_dir / f"iter_{it:05d}.npz")
        for name in INTEGER_ARRAYS:
            rs = np.load(rs_dir / f"iter_{it:05d}" / f"{name}.npy")
            expected = py[name]
            if not np.array_equal(rs.ravel(), expected.ravel()):
                bad = np.flatnonzero(rs.ravel() != expected.ravel())[0]
                pytest.fail(
                    f"{name} diverged at iteration {it}, first at flat index {bad}: "
                    f"rust={rs.ravel()[bad]} python={expected.ravel()[bad]}"
                )
        for name in FLOAT_ARRAYS:
            rs = np.load(rs_dir / f"iter_{it:05d}" / f"{name}.npy")
            expected = py[name]
            # Shapes may differ harmlessly: Python stores regions_mu as
            # (T, R, 1, 3) while Rust writes (T, R, 3). Compare raveled values.
            # ascontiguousarray is required before .view() -- viewing a
            # non-contiguous array raises.
            rb = np.ascontiguousarray(rs.ravel(), dtype=np.float64).view(np.uint64)
            eb = np.ascontiguousarray(expected.ravel(), dtype=np.float64).view(np.uint64)
            assert rb.size == eb.size, f"{name} size mismatch at iteration {it}"
            if not np.array_equal(rb, eb):
                bad = np.flatnonzero(rb != eb)[0]
                pytest.fail(
                    f"{name} diverged (bitwise) at iteration {it}, flat index {bad}: "
                    f"rust={rs.ravel()[bad]!r} python={expected.ravel()[bad]!r}"
                )
