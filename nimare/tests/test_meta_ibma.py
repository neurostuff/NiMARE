"""Test nimare.meta.ibma (image-based meta-analytic estimators)."""

import logging
import os.path as op

import numpy as np
import pytest
from nilearn.image import concat_imgs
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import clone

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import ibma
from nimare.tests.utils import get_test_data_path


@pytest.mark.parametrize(
    "meta,meta_kwargs,corrector,corrector_kwargs,maps",
    [
        pytest.param(
            ibma.Fishers,
            {},
            FDRCorrector,
            {"method": "indep", "alpha": 0.001},
            ("z", "p", "dof"),
            id="Fishers",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": False, "groupby": False},
            None,
            {},
            ("z", "p", "dof"),
            id="Stouffers",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": True, "groupby": False},
            None,
            {},
            ("z", "p", "dof"),
            id="Stouffers_sample_weighted",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": False},
            None,
            {},
            ("z", "p", "dof"),
            id="Stouffers_grouped",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": True},
            None,
            {},
            ("z", "p", "dof"),
            id="Stouffers_sample_grouped",
        ),
        pytest.param(
            ibma.WeightedLeastSquares,
            {"tau2": 0},
            None,
            {},
            ("z", "p", "est", "se", "dof"),
            id="WeightedLeastSquares",
        ),
        pytest.param(
            ibma.DerSimonianLaird,
            {},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "dof"),
            id="DerSimonianLaird",
        ),
        pytest.param(
            ibma.Hedges,
            {},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "dof"),
            id="Hedges",
        ),
        pytest.param(
            ibma.SampleSizeBasedLikelihood,
            {"method": "ml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "sigma2", "dof"),
            id="SampleSizeBasedLikelihood_ml",
        ),
        pytest.param(
            ibma.SampleSizeBasedLikelihood,
            {"method": "reml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "sigma2", "dof"),
            id="SampleSizeBasedLikelihood_reml",
        ),
        pytest.param(
            ibma.VarianceBasedLikelihood,
            {"method": "ml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "dof"),
            id="VarianceBasedLikelihood_ml",
        ),
        pytest.param(
            ibma.VarianceBasedLikelihood,
            {"method": "reml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "dof"),
            id="VarianceBasedLikelihood_reml",
        ),
        pytest.param(
            ibma.PermutedOLS,
            {"two_sided": True},
            FWECorrector,
            {"method": "montecarlo", "n_iters": 100, "n_cores": 1},
            ("t", "z", "dof"),
            id="PermutedOLS",
        ),
        pytest.param(
            ibma.FixedEffectsHedges,
            {"tau2": 0},
            None,
            {},
            ("z", "p", "est", "se", "dof"),
            id="FixedEffectsHedges",
        ),
    ],
)
@pytest.mark.parametrize("aggressive_mask", [True, False], ids=["aggressive", "liberal"])
def test_ibma_smoke(
    testdata_ibma,
    meta,
    aggressive_mask,
    meta_kwargs,
    corrector,
    corrector_kwargs,
    maps,
):
    """Smoke test for IBMA estimators."""
    meta = meta(aggressive_mask=aggressive_mask, **meta_kwargs)
    results = meta.fit(testdata_ibma)
    for expected_map in maps:
        assert expected_map in results.maps.keys()

    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert results.get_map("z", return_type="array").ndim == 1
    z_img = results.get_map("z")
    assert z_img.ndim == 3
    assert z_img.shape == (10, 10, 10)
    if corrector:
        corr = corrector(**corrector_kwargs)
        corr_results = corr.transform(results)
        assert isinstance(corr_results, nimare.results.MetaResult)
        assert isinstance(corr_results.description_, str)
        assert corr_results.get_map("z", return_type="array").ndim == 1
        assert corr_results.get_map("z").ndim == 3


@pytest.mark.parametrize(
    "estimator,expectation,masker_source",
    [
        (ibma.Fishers, "error", "estimator"),
        (ibma.Stouffers, "error", "estimator"),
        (ibma.WeightedLeastSquares, "warning", "estimator"),
        (ibma.DerSimonianLaird, "warning", "estimator"),
        (ibma.Hedges, "warning", "estimator"),
        (ibma.SampleSizeBasedLikelihood, "no warning", "estimator"),
        (ibma.VarianceBasedLikelihood, "warning", "estimator"),
        (ibma.PermutedOLS, "no warning", "estimator"),
    ],
)
def test_ibma_with_custom_masker(testdata_ibma, caplog, estimator, expectation, masker_source):
    """Ensure voxel-to-ROI reduction works, but only for Estimators that allow it.

    Notes
    -----
    Currently masker_source is not used, but ultimately we will want to test cases where the
    Dataset uses a NiftiLabelsMasker.
    """
    atlas = op.join(get_test_data_path(), "test_pain_dataset", "atlas.nii.gz")
    masker = NiftiLabelsMasker(atlas)

    dset = testdata_ibma
    # Pin aggressive masking on: the assertions below are about a label that has no good
    # data in every study being dropped, which is what aggressive masking does. The default
    # (aggressive_mask=False) instead recovers that label from the studies that do have it.
    meta = estimator(mask=masker, aggressive_mask=True)

    if expectation == "error":
        with pytest.raises(ValueError):
            meta.fit(dset)
    elif expectation == "warning":
        with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
            results = meta.fit(dset)
            assert "will likely produce biased results" in caplog.text
        caplog.clear()
    else:
        with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
            results = meta.fit(dset)
            assert "will likely produce biased results" not in caplog.text
        caplog.clear()

    # Only fit the estimator if it doesn't raise a ValueError
    if expectation != "error":
        assert isinstance(results, nimare.results.MetaResult)
        # There are five "labels", but one of them has no good data,
        # so the outputs should be 4 long.
        assert results.maps["z"].shape == (5,)
        assert np.isnan(results.maps["z"][0])
        assert results.get_map("z").shape == (10, 10, 10)


@pytest.mark.parametrize(
    "resample_kwargs",
    [
        {},
        {"resample__clip": False, "resample__interpolation": "continuous"},
    ],
)
def test_ibma_resampling(testdata_ibma_resample, resample_kwargs):
    """Test image-based resampling performance."""
    meta = ibma.Fishers(**resample_kwargs)
    results = meta.fit(testdata_ibma_resample)

    assert isinstance(results, nimare.results.MetaResult)


@pytest.mark.parametrize("aggressive_mask", [True, False], ids=["aggressive", "liberal"])
def test_stouffers_multiple_contrasts(testdata_ibma_multiple_contrasts, aggressive_mask):
    """Test Stouffer's correction with multiple contrasts."""
    meta = ibma.Stouffers(aggressive_mask=aggressive_mask)
    results = meta.fit(testdata_ibma_multiple_contrasts)

    assert isinstance(results, nimare.results.MetaResult)
    assert results.get_map("z", return_type="array").ndim == 1
    z_img = results.get_map("z")
    assert z_img.ndim == 3
    assert z_img.shape == (10, 10, 10)


def _z_image_paths(dataset):
    """Return the z-image paths that are actually on disk, as the estimators see them."""
    return [
        path for path in dataset.images["z"].values if isinstance(path, str) and op.isfile(path)
    ]


def test_mask_images_matches_a_single_4d_transform(testdata_ibma):
    """Masking images one at a time must give exactly what masking them together gave.

    Images are masked individually so that a full-resolution 4D copy of the whole studyset is
    never held in memory. NiMARE's NiftiMasker only selects voxels, which does not depend on
    how many images the masker is handed at once.
    """
    masker = testdata_ibma.masker
    mask_img = masker.mask_img
    filenames = _z_image_paths(testdata_ibma)

    estimator = ibma.Fishers(aggressive_mask=False)
    per_image = estimator._mask_images(masker, mask_img, filenames)

    imgs = [estimator._load_image(f, mask_img) for f in filenames]
    together = masker.transform(concat_imgs(imgs, ensure_ndim=4))

    assert per_image.shape == together.shape == (len(filenames), together.shape[1])
    assert np.array_equal(per_image, together, equal_nan=True)


@pytest.mark.parametrize(
    "masker_params",
    [{"standardize": "zscore_sample"}, {"detrend": True}, {"smoothing_fwhm": 4.0}],
)
def test_mask_images_keeps_one_4d_transform_when_the_masker_does_more_than_select(
    testdata_ibma, masker_params
):
    """Anything beyond voxel selection must still see every image at once.

    Standardizing and detrending change the answer outright; smoothing only shifts the last
    bits of a float32 result, but reproducing the previous output exactly is the point.
    """
    masker = clone(testdata_ibma.masker).set_params(**masker_params)
    masker.fit()
    mask_img = masker.mask_img
    filenames = _z_image_paths(testdata_ibma)

    estimator = ibma.Fishers(aggressive_mask=False)
    masked = estimator._mask_images(masker, mask_img, filenames)

    imgs = [estimator._load_image(f, mask_img) for f in filenames]
    together = masker.transform(concat_imgs(imgs, ensure_ndim=4))
    assert np.array_equal(masked, together, equal_nan=True)
    # Confirm the parameter really does change the answer, so that this test would notice the
    # per-image shortcut being taken here.
    plain = clone(testdata_ibma.masker).fit()
    assert not np.array_equal(masked, plain.transform(concat_imgs(imgs, ensure_ndim=4)))


def test_mask_images_keeps_one_4d_transform_for_a_labels_masker(testdata_ibma):
    """A NiftiLabelsMasker averages within labels, so it also keeps the 4D path."""
    atlas = op.join(get_test_data_path(), "test_pain_dataset", "atlas.nii.gz")
    masker, mask_img = nimare.utils.get_masker_mask_image(
        nimare.utils.get_masker(NiftiLabelsMasker(atlas))
    )
    masker.fit()
    filenames = _z_image_paths(testdata_ibma)

    estimator = ibma.Fishers(aggressive_mask=False)
    masked = estimator._mask_images(masker, mask_img, filenames)

    imgs = [estimator._load_image(f, mask_img) for f in filenames]
    together = masker.transform(concat_imgs(imgs, ensure_ndim=4))
    assert np.array_equal(masked, together, equal_nan=True)


def test_mask_images_rejects_an_empty_input(testdata_ibma):
    """An image input with no images is a broken dataset, not an empty result."""
    estimator = ibma.Fishers(aggressive_mask=False)

    with pytest.raises(ValueError, match="No images were found"):
        estimator._mask_images(testdata_ibma.masker, testdata_ibma.masker.mask_img, [])
