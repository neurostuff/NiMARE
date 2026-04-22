"""Tests for the JALE-derived NiMARE functionality."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from nimare.correct import FWECorrector
from nimare.diagnostics import ResampledStability
from nimare.meta.cbma import (
    ALE,
    ALESubtraction,
    BalancedALESubstraction,
    MKDAChi2,
    MKDADensity,
)
from nimare.meta.cbma.predictive import feature_extraction
from nimare.results import MetaResult
from nimare.transforms import MaskedContrastTransformer

JALE_ROOT = Path(__file__).resolve().parents[2] / "JALE"


@pytest.mark.skipif(not JALE_ROOT.exists(), reason="Local JALE checkout is unavailable.")
def test_predictive_feature_extraction_matches_jale():
    """The packaged ALE predictive features should match JALE exactly."""
    sys.path.insert(0, str(JALE_ROOT))
    try:
        from jale.core.utils.cutoff_prediction import (
            feature_extraction as jale_feature_extraction,
        )
    finally:
        sys.path.pop(0)

    nsub = np.array([12, 20, 33, 18], dtype=float)
    nfoci = np.array([5, 7, 9, 6], dtype=float)
    ours = feature_extraction(4, nsub, nfoci)
    theirs = jale_feature_extraction(4, nsub, nfoci)
    np.testing.assert_allclose(ours, theirs)


def test_resampled_stability_smoke(testdata_cbma_full):
    """A ResampledStability run should produce a bounded voxelwise stability map."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    result = ALE().fit(dset)
    result = FWECorrector(method="montecarlo", n_iters=2, n_cores=1).transform(result)
    diagnostic = ResampledStability(
        target_image="z_level-voxel_corr-FWE_method-montecarlo",
        resampling_policy="leave_1_out",
        n_cores=1,
    )
    result = diagnostic.transform(result)
    map_name = "z_level-voxel_corr-FWE_method-montecarlo_diag-ResampledStability"
    values = result.get_map(map_name, return_type="array")
    assert np.all(values >= 0)
    assert np.all(values <= 1)


def test_balanced_ale_substraction_smoke(testdata_cbma_full):
    """A BalancedALESubstraction run should emit balanced subtraction outputs."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[5:10])
    result = BalancedALESubstraction(
        target_n=3,
        n_subsamples=4,
        difference_iterations=2,
        n_iters=4,
        n_cores=1,
    ).fit(dset1, dset2)
    assert "stat_desc-balancedGroup1MinusGroup2" in result.maps
    assert "z_desc-balancedGroup1MinusGroup2" in result.maps
    assert "stat_desc-conjunction" in result.maps


def test_predictive_corrector_smoke_if_xgboost_available(testdata_cbma_full):
    """Predictive ALE correction should expose thresholded vFWE and cFWE maps."""
    pytest.importorskip("xgboost")
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:8])
    result = ALE().fit(dset)
    result = FWECorrector(method="predictive").transform(result)
    assert "z_level-voxel_corr-FWE_method-predictive" in result.maps
    assert "z_desc-size_level-cluster_corr-FWE_method-predictive" in result.maps


def test_masked_contrast_transformer_smoke_alesubtraction(testdata_cbma_full):
    """A MaskedContrastTransformer run should add a winner-masked ALE subtraction map."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[5:10])
    group1_result = ALE().fit(dset1)
    group2_result = ALE().fit(dset2)
    pairwise_values = group1_result.get_map("z", return_type="array") - group2_result.get_map(
        "z", return_type="array"
    )
    pairwise_result = MetaResult(
        ALESubtraction(),
        mask=dset1.masker,
        maps={"z_desc-group1MinusGroup2": pairwise_values},
    )

    transformed = MaskedContrastTransformer().transform(
        pairwise_result,
        group1_result,
        group2_result,
    )

    masked_name = "z_desc-group1MinusGroup2_desc-winnerMasked"
    conjunction_name = "z_desc-conjunction"
    assert masked_name in transformed.maps
    assert conjunction_name in transformed.maps


def test_masked_contrast_transformer_smoke_mkdachi2(testdata_cbma_full):
    """A MaskedContrastTransformer run should work on MKDAChi2 association maps too."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[5:10])
    pairwise_result = MKDAChi2().fit(dset1, dset2)
    group1_result = MKDADensity().fit(dset1)
    group2_result = MKDADensity().fit(dset2)

    transformed = MaskedContrastTransformer().transform(
        pairwise_result,
        group1_result,
        group2_result,
    )

    masked_name = "z_desc-association_desc-winnerMasked"
    conjunction_name = "z_desc-conjunction"
    assert masked_name in transformed.maps
    assert conjunction_name in transformed.maps
