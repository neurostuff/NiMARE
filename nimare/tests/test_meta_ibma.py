"""Test nimare.meta.ibma (image-based meta-analytic estimators)."""
import logging
import os.path as op
from contextlib import ExitStack as does_not_raise

import pytest
from nilearn.input_data import NiftiLabelsMasker

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import ibma

from .utils import get_test_data_path


@pytest.mark.parametrize(
    "meta,meta_kwargs,corrector,corrector_kwargs",
    [
        pytest.param(
            ibma.Fishers,
            {},
            FDRCorrector,
            {"method": "indep", "alpha": 0.001},
            id="Fishers",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": False},
            None,
            {},
            id="Stouffers",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": True},
            None,
            {},
            id="Stouffers_weighted",
        ),
        pytest.param(
            ibma.WeightedLeastSquares,
            {"tau2": 0},
            None,
            {},
            id="WeightedLeastSquares",
        ),
        pytest.param(
            ibma.DerSimonianLaird,
            {},
            None,
            {},
            id="DerSimonianLaird",
        ),
        pytest.param(
            ibma.Hedges,
            {},
            None,
            {},
            id="Hedges",
        ),
        pytest.param(
            ibma.SampleSizeBasedLikelihood,
            {"method": "ml"},
            None,
            {},
            id="SampleSizeBasedLikelihood_ml",
        ),
        pytest.param(
            ibma.SampleSizeBasedLikelihood,
            {"method": "reml"},
            None,
            {},
            id="SampleSizeBasedLikelihood_reml",
        ),
        pytest.param(
            ibma.VarianceBasedLikelihood,
            {"method": "ml"},
            None,
            {},
            id="VarianceBasedLikelihood_ml",
        ),
        pytest.param(
            ibma.VarianceBasedLikelihood,
            {"method": "reml"},
            None,
            {},
            id="VarianceBasedLikelihood_reml",
        ),
        pytest.param(
            ibma.PermutedOLS,
            {"two_sided": True},
            FWECorrector,
            {"method": "montecarlo", "n_iters": 100, "n_cores": 1},
            id="PermutedOLS",
        ),
    ],
)
def test_ibma_smoke(testdata_ibma, meta, meta_kwargs, corrector, corrector_kwargs):
    """Smoke test for IBMA estimators."""
    meta = meta(**meta_kwargs)
    res = meta.fit(testdata_ibma)
    z_img = res.get_map("z")
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)
    assert res.get_map("z", return_type="array").ndim == 1
    assert z_img.ndim == 3
    assert z_img.shape == (10, 10, 10)
    if corrector:
        corr = corrector(**corrector_kwargs)
        cres = corr.transform(res)
        assert cres.get_map("z", return_type="array").ndim == 1
        assert cres.get_map("z").ndim == 3


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
    meta = estimator(mask=masker)

    if expectation == "error":
        with pytest.raises(ValueError):
            meta.fit(dset)
    elif expectation == "warning":
        with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
            meta.fit(dset)
            assert "will likely produce biased results" in caplog.text
        caplog.clear()
    else:
        with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
            meta.fit(dset)
            assert "will likely produce biased results" not in caplog.text
        caplog.clear()

    # Only fit the estimator if it doesn't raise a ValueError
    if expectation != "error":
        assert isinstance(meta.results, nimare.results.MetaResult)
        assert meta.results.maps["z"].shape == (5,)
        assert meta.results.get_map("z").shape == (10, 10, 10)


@pytest.mark.parametrize(
    "resample,resample_kwargs,expectation",
    [
        (False, {}, pytest.raises(ValueError)),
        (None, {}, pytest.raises(ValueError)),
        (True, {}, does_not_raise()),
        (
            True,
            {"resample__clip": False, "resample__interpolation": "continuous"},
            does_not_raise(),
        ),
    ],
)
def test_ibma_resampling(testdata_ibma_resample, resample, resample_kwargs, expectation):
    """Test image-based resampling performance."""
    meta = ibma.Fishers(resample=resample, **resample_kwargs)
    with expectation:
        meta.fit(testdata_ibma_resample)
    if isinstance(expectation, does_not_raise):
        assert isinstance(meta.results, nimare.results.MetaResult)
