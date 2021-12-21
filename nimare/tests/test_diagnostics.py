"""Tests for the nimare.diagnostics module."""
import os.path as op

import pytest
from nilearn.input_data import NiftiLabelsMasker

from nimare.diagnostics import Jackknife
from nimare.meta import cbma, ibma
from nimare.tests.utils import get_test_data_path


@pytest.mark.parametrize(
    "estimator,meta_type,n_samples,target_image",
    [
        (cbma.ALE, "cbma", "onesample", "z"),
        (cbma.MKDADensity, "cbma", "onesample", "z"),
        (cbma.KDA, "cbma", "onesample", "z"),
        (cbma.MKDAChi2, "cbma", "twosample", "z_desc-consistency"),
        (ibma.Fishers, "ibma", "onesample", "z"),
        (ibma.Stouffers, "ibma", "onesample", "z"),
        (ibma.WeightedLeastSquares, "ibma", "onesample", "z"),
        (ibma.DerSimonianLaird, "ibma", "onesample", "z"),
        (ibma.Hedges, "ibma", "onesample", "z"),
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
):
    """Smoke test the Jackknife method."""
    meta = estimator()
    testdata = testdata_ibma if meta_type == "ibma" else testdata_cbma_full
    if n_samples == "twosample":
        res = meta.fit(testdata, testdata)
    else:
        res = meta.fit(testdata)

    jackknife = Jackknife(target_image=target_image, voxel_thresh=1.65)

    if n_samples == "twosample":
        with pytest.raises(AttributeError):
            jackknife.transform(res)
    else:
        cluster_table, labeled_img = jackknife.transform(res)
        assert cluster_table.shape[0] == len(meta.inputs_["id"]) + 1


def test_jackknife_with_custom_masker_smoke(testdata_ibma):
    """Ensure that Jackknife will work with NiftiLabelsMaskers.

    CBMAs don't work with NiftiLabelsMaskers and VarianceBasedLikelihood takes ~1 minute,
    which is too long for a single test, so I'm just using SampleSizeBasedLikelihood.
    """
    atlas = op.join(get_test_data_path(), "test_pain_dataset", "atlas.nii.gz")
    masker = NiftiLabelsMasker(atlas)

    meta = ibma.SampleSizeBasedLikelihood(mask=masker)
    res = meta.fit(testdata_ibma)

    jackknife = Jackknife(target_image="z", voxel_thresh=0.5)
    cluster_table, labeled_img = jackknife.transform(res)
    assert cluster_table.shape[0] == len(meta.inputs_["id"]) + 1

    # A Jackknife with a target_image that isn't present in the MetaResult raises a ValueError.
    with pytest.raises(ValueError):
        jackknife = Jackknife(target_image="doggy", voxel_thresh=0.5)
        jackknife.transform(res)
