"""Tests for the nimare.diagnostics module."""

import logging
import os.path as op
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.maskers import NiftiLabelsMasker
from scipy.spatial.distance import cdist

from nimare import diagnostics
from nimare.meta import cbma, ibma
from nimare.tests.utils import get_test_data_path


def test_summarize_cluster_values_masked_array():
    """Cluster summaries should preserve cluster order in masked-array mode."""
    values = np.array([0.0, 1.0, 3.0, 2.0, 4.0], dtype=float)
    cluster_summary_context = {
        "mode": "masked_array",
        "cluster_indices": [
            np.array([1, 2], dtype=int),
            np.array([3, 4], dtype=int),
        ],
    }

    reduced = diagnostics._summarize_cluster_values(
        values, masker=None, cluster_summary_context=cluster_summary_context
    )

    assert np.allclose(reduced, np.array([2.0, 3.0]))


def test_summarize_cluster_values_image_mode():
    """Cluster summaries should preserve cluster order in image mode."""
    mask_data = np.ones((2, 2, 1), dtype=np.int8)
    mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

    class DummyMasker:
        """Minimal masker with the inverse_transform API used by diagnostics."""

        def __init__(self, mask_img):
            self.mask_img_ = mask_img

        def inverse_transform(self, data):
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]

            vol = np.zeros(mask_data.shape, dtype=float)
            vol[mask_data.astype(bool)] = arr[0]
            return nib.Nifti1Image(vol, affine=mask_img.affine)

    label_data = np.zeros(mask_data.shape, dtype=np.int16)
    label_data[0, 0, 0] = 1
    label_data[0, 1, 0] = 1
    label_data[1, 0, 0] = 2
    label_data[1, 1, 0] = 2
    label_img = nib.Nifti1Image(label_data, affine=np.eye(4))

    cluster_masker = NiftiLabelsMasker(label_img, **diagnostics._cluster_masker_kwargs())
    cluster_masker.fit(label_img)

    cluster_summary_context = {
        "mode": "image",
        "cluster_masker": cluster_masker,
    }
    values = np.array([1.0, 3.0, 2.0, 4.0], dtype=float)

    reduced = diagnostics._summarize_cluster_values(
        values,
        masker=DummyMasker(mask_img),
        cluster_summary_context=cluster_summary_context,
    )

    assert np.allclose(reduced, np.array([2.0, 3.0]))


def test_cluster_ids_survive_a_map_with_no_background():
    """A fully suprathreshold map has one real cluster, not zero.

    Nilearn derives cluster ids with ``np.unique(label_map)[1:]``, which assumes a background
    label is present. NiMARE pads a border so that call succeeds, but it then crops the border
    off again -- so deriving ids the same way here would report no clusters for a map whose
    table lists several.
    """
    rng = np.random.RandomState(0)
    data = rng.uniform(3.0, 5.0, size=(5, 5, 5)).astype(np.float32)
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))

    assert diagnostics._needs_background_border(img, 1.65)
    clusters_table, label_maps = diagnostics._get_clusters_table(
        img, 1.65, 0, True, return_label_maps=True
    )

    assert not clusters_table.empty
    ids = diagnostics._cluster_ids(label_maps[0].dataobj)
    assert ids == [1]
    # The border must be gone again, so the label map still lines up with the input.
    assert label_maps[0].shape == img.shape
    assert np.array_equal(label_maps[0].affine, img.affine)


def test_count_foci_per_cluster_matches_all_pairs_distances():
    """Focus counts must match measuring every cluster voxel against every focus.

    :func:`nimare.diagnostics._count_foci_per_cluster` only looks at the voxels neighbouring
    each focus, rather than scanning the volume once per cluster. The two agree because no
    voxel further away than a focus's own neighbourhood can be within one voxel of it, and
    this pins that down over shapes, label dtypes, out-of-bounds foci and foci that sit
    exactly on a voxel centre.
    """

    def all_pairs(label_arr, clust_ids, ijk):
        counts = []
        for c_val in clust_ids:
            cluster_idx = np.vstack(np.where(label_arr == c_val))
            distances = cdist(cluster_idx.T, ijk)
            counts.append(np.sum(np.any(distances < 1, axis=0)))
        return np.array(counts)

    rng = np.random.default_rng(0)
    compared = 0
    for i_trial in range(60):
        shape = tuple(rng.integers(3, 9, size=3))
        n_labels = int(rng.integers(1, 6))
        dtype = rng.choice([np.int16, np.int32, np.float64])
        label_arr = rng.integers(0, n_labels + 1, size=shape).astype(dtype)
        clust_ids = diagnostics._cluster_ids(label_arr)
        if not clust_ids:
            continue

        # Foci are drawn past the edges of the volume as well as inside it, and every third
        # trial puts them exactly on voxel centres, which is the boundary case for "within
        # one voxel".
        ijk = rng.uniform(-1.5, np.array(shape) + 1.5, size=(int(rng.integers(1, 12)), 3))
        if i_trial % 3 == 0:
            ijk = np.round(ijk)

        np.testing.assert_array_equal(
            diagnostics._count_foci_per_cluster(label_arr, clust_ids, ijk),
            all_pairs(label_arr, clust_ids, ijk),
            err_msg=f"trial {i_trial}, shape {shape}",
        )
        compared += 1

    assert compared > 40, "too few trials produced clusters to compare"


def test_count_foci_per_cluster_with_no_foci():
    """A study contributing no foci scores zero everywhere, not an empty array."""
    label_arr = np.array([[[1, 1], [0, 2]]], dtype=np.int16)

    counts = diagnostics._count_foci_per_cluster(label_arr, [1, 2], np.empty((0, 3)))

    assert np.array_equal(counts, [0, 0])


def test_cluster_ids_excludes_background():
    """The ordinary case is unchanged: label 0 is background, not a cluster."""
    label_arr = np.array([[0, 1, 1], [0, 0, 2]])

    assert diagnostics._cluster_ids(label_arr) == [1, 2]


def test_get_target_value_map_prefers_deterministic_priority():
    """Target value map selection should use explicit key priority."""
    result = SimpleNamespace(maps={"z": None, "est": None, "stat": None})

    assert diagnostics._get_target_value_map(result) == "stat"


def test_get_target_value_map_raises_for_unsupported_maps():
    """Unsupported map sets should fail loudly."""
    result = SimpleNamespace(maps={"doggy": None})

    with pytest.raises(ValueError, match="No supported map found"):
        diagnostics._get_target_value_map(result)


def test_diagnostics_voxel_thresh_deprecated_alias():
    """voxel_thresh should remain a deprecated alias for target_threshold."""
    with pytest.warns(FutureWarning, match="voxel_thresh"):
        counter = diagnostics.FocusCounter(target_image="z", voxel_thresh=1.0)

    assert counter.target_threshold == 1.0


def test_diagnostics_target_and_voxel_threshold_error():
    """Supplying both threshold names should fail explicitly."""
    with pytest.raises(ValueError, match="target_threshold"):
        diagnostics.FocusCounter(target_image="z", target_threshold=1.0, voxel_thresh=1.0)


@pytest.mark.parametrize(
    "target_image,source_map",
    [
        ("z_desc-size_level-cluster_corr-FWE_method-montecarlo", "z"),
        ("z_desc-mass_level-cluster_corr-FWE_method-montecarlo", "z"),
        (
            "z_desc-uniformitySize_level-cluster_corr-FWE_method-montecarlo",
            "z_desc-uniformity",
        ),
        (
            "z_desc-group1MinusGroup2Mass_level-cluster_corr-FWE_method-montecarlo",
            "z_desc-group1MinusGroup2",
        ),
        (
            "stat_desc-balancedGroup1MinusGroup2Size_level-cluster_corr-FWE_method-montecarlo",
            "stat_desc-balancedGroup1MinusGroup2",
        ),
    ],
)
def test_peak_value_map_is_derived_from_target_image(target_image, source_map):
    """Original peak maps should be derived from corrected-cluster map names."""
    assert diagnostics._peak_value_map_from_cluster_target(target_image) == source_map


def test_corrected_cluster_table_uses_thresholded_support_and_original_z():
    """Corrected cluster tables should report original-z peaks inside thresholded support."""
    target_image = "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
    mask_data = np.ones((7, 7, 7), dtype=bool)
    mask_img = nib.Nifti1Image(mask_data.astype(np.int8), affine=np.eye(4))

    class DummyMasker:
        def __init__(self, mask_img):
            self.mask_img_ = mask_img

        def transform(self, img):
            return np.asanyarray(img.dataobj)[mask_data].reshape(1, -1)

        def inverse_transform(self, data):
            arr = np.asarray(data)
            if arr.ndim > 1:
                arr = np.squeeze(arr, axis=0)
            out = np.zeros(mask_data.shape, dtype=arr.dtype)
            out[mask_data] = arr
            return nib.Nifti1Image(out, affine=mask_img.affine)

    class DummyResult:
        def __init__(self, maps, masker):
            self.maps = maps
            self.tables = {}
            self.diagnostics = []
            self.masker = masker
            self.estimator = SimpleNamespace(
                masker=masker,
                inputs_={
                    "id": ["study1"],
                    "coordinates": pd.DataFrame(
                        {"id": ["study1"], "x": [1.0], "y": [1.0], "z": [1.0]}
                    ),
                },
            )

        def get_map(self, name, return_type="image"):
            values = self.maps[name]
            if return_type == "array":
                return values
            return self.masker.inverse_transform(values)

    corrected = np.zeros(mask_data.shape, dtype=float)
    corrected[(1, 1, 1)] = 6.0
    corrected[(1, 1, 2)] = 2.0
    corrected[(1, 2, 1)] = 2.0
    corrected[(5, 5, 5)] = 1.2
    corrected[(5, 5, 4)] = 1.2

    original_z = np.zeros(mask_data.shape, dtype=float)
    original_z[(1, 1, 1)] = 3.0
    original_z[(1, 1, 2)] = 9.0
    original_z[(1, 2, 1)] = 2.0
    original_z[(5, 5, 5)] = 8.0
    original_z[(5, 5, 4)] = 8.0
    original_z[(3, 3, 3)] = 99.0

    masker = DummyMasker(mask_img)
    result = DummyResult(
        {
            target_image: masker.transform(nib.Nifti1Image(corrected, affine=mask_img.affine))[0],
            "z": masker.transform(nib.Nifti1Image(original_z, affine=mask_img.affine))[0],
        },
        masker,
    )

    counter = diagnostics.FocusCounter(target_image=target_image, target_threshold=1.64)
    result = counter.transform(result)

    clusters_table = result.tables[f"{target_image}_tab-clust"]
    peak_row = clusters_table.loc[clusters_table["Peak Stat"].idxmax()]

    assert clusters_table.shape[0] == 1
    assert peak_row["Peak Stat"] == pytest.approx(9.0)
    assert peak_row[["X", "Y", "Z"]].to_numpy().tolist() == [1.0, 1.0, 2.0]
    assert not np.any(np.isclose(clusters_table["Peak Stat"], 8.0))
    assert not np.any(np.isclose(clusters_table["Peak Stat"], 99.0))


def test_is_voxelwise_masker_uses_round_trip_when_mask_count_mismatches():
    """Voxelwise detection should fall back to a round-trip feature-shape check."""
    mask_data = np.array([[[1], [0]], [[1], [1]]], dtype=np.int8)
    mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

    class DummyMasker:
        def __init__(self, mask_img):
            self.mask_img_ = mask_img

        def inverse_transform(self, data):
            arr = np.asarray(data)
            if arr.ndim > 1:
                arr = np.squeeze(arr, axis=0)
            return nib.Nifti1Image(arr.reshape((2, 2, 1)), affine=mask_img.affine)

        def transform(self, img):
            return np.asanyarray(img.dataobj).reshape(1, 4)

    masker = DummyMasker(mask_img)

    assert diagnostics._is_voxelwise_masker(masker, 4)
    assert masker._nimare_mask_voxel_count == 3


@pytest.mark.parametrize(
    "estimator,meta_type,n_samples,target_image,voxel_thresh",
    [
        (cbma.ALE, "cbma", "onesample", "z", 1.65),
        (cbma.MKDADensity, "cbma", "onesample", "z", 1.65),
        (cbma.KDA, "cbma", "onesample", "z", 1.65),
        (cbma.MKDAChi2, "cbma", "twosample", "z_desc-uniformity", 1.65),
        (ibma.Fishers, "ibma", "onesample", "z", 0.1),
        (ibma.Stouffers, "ibma", "onesample", "z", 0.1),
        (ibma.WeightedLeastSquares, "ibma", "onesample", "z", 0.1),
        (ibma.DerSimonianLaird, "ibma", "onesample", "z", 0.1),
        (ibma.Hedges, "ibma", "onesample", "z", 0.1),
        # (ibma.SampleSizeBasedLikelihood, "ibma", "onesample", "z"),
        # (ibma.VarianceBasedLikelihood, "ibma", "onesample", "z"),
        # (ibma.PermutedOLS, "ibma", "onesample", "z"),
    ],
)
def test_jackknife_smoke(
    testdata_ibma,
    testdata_cbma_full,
    estimator,
    meta_type,
    n_samples,
    target_image,
    voxel_thresh,
):
    """Smoke test the Jackknife method."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])

    meta = estimator()
    testdata = testdata_ibma if meta_type == "ibma" else testdata_cbma_full
    res = meta.fit(dset1, dset2) if n_samples == "twosample" else meta.fit(testdata)

    jackknife = diagnostics.Jackknife(target_image=target_image, target_threshold=voxel_thresh)
    results = jackknife.transform(res)

    image_name = "_".join(target_image.split("_")[1:])
    image_name = f"_{image_name}" if image_name else image_name

    # Whether an IBMA result has one tail or two depends on whether its z map has negative
    # values below the threshold, so resolve the table by what is actually present: a
    # two-tailed IBMA merges both tails into one table, everything else splits by tail.
    table_name = f"{target_image}_diag-Jackknife_tab-counts"
    contribution_table = results.tables.get(
        table_name,
        results.tables.get(f"{table_name}_tail-positive"),
    )
    assert contribution_table is not None

    clusters_table = results.tables[f"{target_image}_tab-clust"]
    label_maps = results.maps[f"label{image_name}_tail-positive"]
    ids_ = meta.inputs_["id"] if n_samples == "onesample" else meta.inputs_["id1"]

    assert contribution_table.shape[0] == len(ids_)
    assert clusters_table.shape[0] >= contribution_table.shape[1] - 1
    assert len(label_maps) > 0


def test_jackknife_with_zero_clusters(testdata_cbma_full):
    """Ensure that Jackknife will work with zero clusters."""
    meta = cbma.ALE()
    res = meta.fit(testdata_cbma_full)

    jackknife = diagnostics.Jackknife(target_image="z", target_threshold=10)
    results = jackknife.transform(res)

    contribution_table = results.tables["z_diag-Jackknife_tab-counts"]
    clusters_table = results.tables["z_tab-clust"]
    label_maps = results.maps["label_tail-positive"]
    assert contribution_table is None
    assert clusters_table.empty
    assert not label_maps


def test_jackknife_with_custom_masker_smoke(testdata_ibma):
    """Ensure that Jackknife will work with NiftiLabelsMaskers.

    CBMAs don't work with NiftiLabelsMaskers and VarianceBasedLikelihood takes ~1 minute,
    which is too long for a single test, so I'm just using SampleSizeBasedLikelihood.
    """
    atlas = op.join(get_test_data_path(), "test_pain_dataset", "atlas.nii.gz")
    masker = NiftiLabelsMasker(atlas)

    meta = ibma.SampleSizeBasedLikelihood(mask=masker)
    res = meta.fit(testdata_ibma)

    jackknife = diagnostics.Jackknife(target_image="z", target_threshold=0.5)
    results = jackknife.transform(res)
    contribution_table = results.tables["z_diag-Jackknife_tab-counts_tail-positive"]
    assert contribution_table.shape[0] == len(meta.inputs_["id"])

    # A Jackknife with a target_image that isn't present in the MetaResult raises a ValueError.
    with pytest.raises(ValueError):
        jackknife = diagnostics.Jackknife(target_image="doggy", target_threshold=0.5)
        jackknife.transform(res)


def test_jackknife_needs_one_more_analysis_than_the_estimator(testdata_ibma):
    """The refit is what shrinks the studyset, so the floor is the estimator's plus one."""
    jackknife = diagnostics.Jackknife(target_image="z", target_threshold=0.5)

    with pytest.raises(ValueError) as exc_info:
        jackknife.transform(ibma.Fishers().fit(testdata_ibma.slice(testdata_ibma.ids[:2])))

    message = str(exc_info.value)
    assert "Jackknife needs at least 3 analyses" in message
    assert "Fishers needs at least 2" in message

    results = jackknife.transform(ibma.Fishers().fit(testdata_ibma.slice(testdata_ibma.ids[:3])))
    assert results.tables["z_diag-Jackknife_tab-counts"].shape[0] == 3


def test_jackknife_leaves_cbma_alone(testdata_cbma_full):
    """CBMA estimators declare no floor, so a two-experiment jackknife must still run."""
    ids_ = sorted(set(testdata_cbma_full.coordinates["id"]))[:2]
    res = cbma.ALE().fit(testdata_cbma_full.slice(ids_))
    jackknife = diagnostics.Jackknife(target_image="z", target_threshold=0.1)

    results = jackknife.transform(res)

    assert results.tables["z_diag-Jackknife_tab-counts_tail-positive"].shape[0] == 2


def test_resampled_stability_refuses_replicates_below_the_estimator_floor(testdata_ibma):
    """Leaving one of two out fits each replicate on a single analysis."""
    res = ibma.Fishers().fit(testdata_ibma.slice(testdata_ibma.ids[:2]))
    stability = diagnostics.ResampledStability(target_image="z", resampling_policy="leave_1_out")

    with pytest.raises(ValueError) as exc_info:
        stability.transform(res)

    message = str(exc_info.value)
    assert "keeps 1 of 2 analyses per replicate" in message
    assert "Fishers needs at least 2" in message


def test_focuscounter_negative_tail_label_map_naming(testdata_cbma_full):
    """Ensure single-tail negative clusters are labeled as negative."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    meta = cbma.ALE()
    res = meta.fit(dset)

    masker = res.estimator.masker
    mask_img = masker.mask_img_
    mask_data = mask_img.get_fdata().astype(bool)

    neg_data = np.zeros(mask_img.shape, dtype=float)
    ijk = np.column_stack(np.where(mask_data))[0]
    neg_data[tuple(ijk)] = -5.0
    neg_img = nib.Nifti1Image(neg_data, mask_img.affine)
    res.maps["z"] = np.squeeze(masker.transform(neg_img))

    counter = diagnostics.FocusCounter(target_image="z", target_threshold=1.0)
    results = counter.transform(res)

    assert "label_tail-negative" in results.maps
    assert results.maps["label_tail-negative"] is not None
    assert "label_tail-positive" not in results.maps
    assert "z_diag-FocusCounter_tab-counts_tail-negative" in results.tables


def test_focuscounter_positive_tail_label_map_naming(testdata_cbma_full):
    """Ensure single-tail positive clusters are labeled as positive."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    meta = cbma.ALE()
    res = meta.fit(dset)

    masker = res.estimator.masker
    mask_img = masker.mask_img_
    mask_data = mask_img.get_fdata().astype(bool)

    pos_data = np.zeros(mask_img.shape, dtype=float)
    ijk = np.column_stack(np.where(mask_data))[0]
    pos_data[tuple(ijk)] = 5.0
    pos_img = nib.Nifti1Image(pos_data, mask_img.affine)
    res.maps["z"] = np.squeeze(masker.transform(pos_img))

    counter = diagnostics.FocusCounter(target_image="z", target_threshold=1.0)
    results = counter.transform(res)

    assert "label_tail-positive" in results.maps
    assert results.maps["label_tail-positive"] is not None
    assert "label_tail-negative" not in results.maps
    assert "z_diag-FocusCounter_tab-counts_tail-positive" in results.tables


def test_focuscounter_single_tail_mixed_sign_warning(testdata_cbma_full, monkeypatch, caplog):
    """Ensure mixed-sign single-tail path warns and defaults to positive."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    meta = cbma.ALE()
    res = meta.fit(dset)

    masker = res.estimator.masker
    mask_img = masker.mask_img_
    mask_data = mask_img.get_fdata().astype(bool)

    pos_data = np.zeros(mask_img.shape, dtype=float)
    ijk = np.column_stack(np.where(mask_data))[0]
    pos_data[tuple(ijk)] = 5.0
    pos_img = nib.Nifti1Image(pos_data, mask_img.affine)
    res.maps["z"] = np.squeeze(masker.transform(pos_img))

    def _fake_infer(_label_maps, _clusters_table, _n_clusters):
        return ["positive"], "positive", True

    monkeypatch.setattr(diagnostics, "_infer_label_map_tails", _fake_infer)
    caplog.set_level(logging.WARNING, logger="nimare.diagnostics")

    counter = diagnostics.FocusCounter(target_image="z", target_threshold=1.0)
    results = counter.transform(res)

    assert any("Mixed-sign clusters detected" in r.message for r in caplog.records)
    assert "label_tail-positive" in results.maps
    assert "label_tail-negative" not in results.maps


def test_focuscounter_pairwise_negative_tail_uses_group2(testdata_cbma_full):
    """Ensure pairwise negative-tail diagnostics use group2 study IDs."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:4])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[4:8])
    meta = cbma.MKDAChi2()
    res = meta.fit(dset1, dset2)

    masker = res.estimator.masker
    mask_img = masker.mask_img_
    mask_data = mask_img.get_fdata().astype(bool)

    neg_data = np.zeros(mask_img.shape, dtype=float)
    ijk = np.column_stack(np.where(mask_data))[0]
    neg_data[tuple(ijk)] = -5.0
    neg_img = nib.Nifti1Image(neg_data, mask_img.affine)
    res.maps["z_desc-uniformity"] = np.squeeze(masker.transform(neg_img))

    counter = diagnostics.FocusCounter(target_image="z_desc-uniformity", target_threshold=1.0)
    results = counter.transform(res)

    table_key = "z_desc-uniformity_diag-FocusCounter_tab-counts_tail-negative"
    assert table_key in results.tables
    assert results.tables[table_key]["id"].tolist() == list(res.estimator.inputs_["id2"])


@pytest.mark.parametrize(
    "estimator,meta_type,n_samples,target_image",
    [
        (cbma.ALE, "cbma", "onesample", "z"),
        (cbma.MKDADensity, "cbma", "onesample", "z"),
        (cbma.KDA, "cbma", "onesample", "z"),
        (cbma.MKDAChi2, "cbma", "twosample", "z_desc-uniformity"),
        (ibma.Stouffers, "ibma", "onesample", "z"),
    ],
)
def test_focuscounter_smoke(
    testdata_ibma,
    testdata_cbma_full,
    estimator,
    meta_type,
    n_samples,
    target_image,
):
    """Smoke test the FocusCounter method."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])

    meta = estimator()
    testdata = testdata_ibma if meta_type == "ibma" else testdata_cbma_full
    res = meta.fit(dset1, dset2) if n_samples == "twosample" else meta.fit(testdata)

    counter = diagnostics.FocusCounter(target_image=target_image, target_threshold=1.65)
    if meta_type == "ibma":
        with pytest.raises(ValueError):
            counter.transform(res)
    else:
        results = counter.transform(res)

        image_name = "_".join(target_image.split("_")[1:])
        image_name = f"_{image_name}" if image_name else image_name

        contribution_table = results.tables[
            f"{target_image}_diag-FocusCounter_tab-counts_tail-positive"
        ]
        clusters_table = results.tables[f"{target_image}_tab-clust"]
        label_maps = results.maps[f"label{image_name}_tail-positive"]
        ids_ = meta.inputs_["id"] if n_samples == "onesample" else meta.inputs_["id1"]

        assert contribution_table.shape[0] == len(ids_)
        assert clusters_table.shape[0] >= contribution_table.shape[1] - 1
        assert len(label_maps) > 0


def test_focusfilter(testdata_laird):
    """Ensure that the FocusFilter removes out-of-mask coordinates.

    The Laird dataset contains 16 foci outside of the MNI brain mask, which the filter should
    remove.
    """
    n_coordinates_all = testdata_laird.coordinates.shape[0]
    ffilter = diagnostics.FocusFilter()
    filtered_dset = ffilter.transform(testdata_laird)
    n_coordinates_filtered = filtered_dset.coordinates.shape[0]
    assert n_coordinates_all == 1117
    assert n_coordinates_filtered == 1101
    assert n_coordinates_filtered <= n_coordinates_all
