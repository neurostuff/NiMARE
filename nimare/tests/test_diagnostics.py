"""Tests for the nimare.diagnostics module."""

import logging
import os.path as op

import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiLabelsMasker

from nimare import diagnostics
from nimare.meta import cbma, ibma
from nimare.tests.utils import get_test_data_path


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

    jackknife = diagnostics.Jackknife(target_image=target_image, voxel_thresh=voxel_thresh)
    results = jackknife.transform(res)

    image_name = "_".join(target_image.split("_")[1:])
    image_name = f"_{image_name}" if image_name else image_name

    # For ibma.WeightedLeastSquares we have both positive and negative tail combined.
    contribution_table = (
        results.tables[f"{target_image}_diag-Jackknife_tab-counts"]
        if estimator == ibma.WeightedLeastSquares
        else results.tables[f"{target_image}_diag-Jackknife_tab-counts_tail-positive"]
    )

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

    jackknife = diagnostics.Jackknife(target_image="z", voxel_thresh=10)
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

    jackknife = diagnostics.Jackknife(target_image="z", voxel_thresh=0.5)
    results = jackknife.transform(res)
    contribution_table = results.tables["z_diag-Jackknife_tab-counts_tail-positive"]
    assert contribution_table.shape[0] == len(meta.inputs_["id"])

    # A Jackknife with a target_image that isn't present in the MetaResult raises a ValueError.
    with pytest.raises(ValueError):
        jackknife = diagnostics.Jackknife(target_image="doggy", voxel_thresh=0.5)
        jackknife.transform(res)


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

    counter = diagnostics.FocusCounter(target_image="z", voxel_thresh=1.0)
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

    counter = diagnostics.FocusCounter(target_image="z", voxel_thresh=1.0)
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

    counter = diagnostics.FocusCounter(target_image="z", voxel_thresh=1.0)
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

    counter = diagnostics.FocusCounter(target_image="z_desc-uniformity", voxel_thresh=1.0)
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

    counter = diagnostics.FocusCounter(target_image=target_image, voxel_thresh=1.65)
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
