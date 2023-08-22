"""Test nimare.meta.ibma (image-based meta-analytic estimators)."""
import logging
import os.path as op

import numpy as np
import pytest
from nilearn.input_data import NiftiLabelsMasker

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
            ("z", "p"),
            id="Fishers",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": False},
            None,
            {},
            ("z", "p"),
            id="Stouffers",
        ),
        pytest.param(
            ibma.Stouffers,
            {"use_sample_size": True},
            None,
            {},
            ("z", "p"),
            id="Stouffers_weighted",
        ),
        pytest.param(
            ibma.WeightedLeastSquares,
            {"tau2": 0},
            None,
            {},
            ("z", "p", "est", "se"),
            id="WeightedLeastSquares",
        ),
        pytest.param(
            ibma.DerSimonianLaird,
            {},
            None,
            {},
            ("z", "p", "est", "se", "tau2"),
            id="DerSimonianLaird",
        ),
        pytest.param(
            ibma.Hedges,
            {},
            None,
            {},
            ("z", "p", "est", "se", "tau2"),
            id="Hedges",
        ),
        pytest.param(
            ibma.SampleSizeBasedLikelihood,
            {"method": "ml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "sigma2"),
            id="SampleSizeBasedLikelihood_ml",
        ),
        pytest.param(
            ibma.SampleSizeBasedLikelihood,
            {"method": "reml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2", "sigma2"),
            id="SampleSizeBasedLikelihood_reml",
        ),
        pytest.param(
            ibma.VarianceBasedLikelihood,
            {"method": "ml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2"),
            id="VarianceBasedLikelihood_ml",
        ),
        pytest.param(
            ibma.VarianceBasedLikelihood,
            {"method": "reml"},
            None,
            {},
            ("z", "p", "est", "se", "tau2"),
            id="VarianceBasedLikelihood_reml",
        ),
        pytest.param(
            ibma.PermutedOLS,
            {"two_sided": True},
            FWECorrector,
            {"method": "montecarlo", "n_iters": 100, "n_cores": 1},
            ("t", "z"),
            id="PermutedOLS",
        ),
    ],
)
def test_ibma_smoke(testdata_ibma, meta, meta_kwargs, corrector, corrector_kwargs, maps):
    """Smoke test for IBMA estimators."""
    meta = meta(**meta_kwargs)
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
    meta = estimator(mask=masker)

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
