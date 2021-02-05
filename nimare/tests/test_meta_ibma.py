"""
Test nimare.meta.ibma (image-based meta-analytic estimators).
"""
import os.path as op
from contextlib import ExitStack as does_not_raise

from nilearn.input_data import NiftiLabelsMasker
import pytest

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import ibma

from .utils import get_test_data_path


def test_Fishers(testdata_ibma):
    """
    Smoke test for Fisher's.
    """
    meta = ibma.Fishers()
    res = meta.fit(testdata_ibma)
    corr = FDRCorrector(method="indep", alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_Stouffers(testdata_ibma):
    """
    Smoke test for Stouffer's, not weighted by sample size.
    """
    meta = ibma.Stouffers(use_sample_size=False)
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_Stouffers_weighted(testdata_ibma):
    """
    Smoke test for Stouffer's, weighted by sample size.
    """
    meta = ibma.Stouffers(use_sample_size=True)
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_WeightedLeastSquares(testdata_ibma):
    """
    Smoke test for WeightedLeastSquares.
    """
    meta = ibma.WeightedLeastSquares(tau2=0)
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_DerSimonianLaird(testdata_ibma):
    """
    Smoke test for DerSimonianLaird.
    """
    meta = ibma.DerSimonianLaird()
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_Hedges(testdata_ibma):
    """
    Smoke test for Hedges.
    """
    meta = ibma.Hedges()
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_SampleSizeBasedLikelihood_ml(testdata_ibma):
    """
    Smoke test for SampleSizeBasedLikelihood with ML.
    """
    meta = ibma.SampleSizeBasedLikelihood(method="ml")
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_SampleSizeBasedLikelihood_reml(testdata_ibma):
    """
    Smoke test for SampleSizeBasedLikelihood with REML.
    """
    meta = ibma.SampleSizeBasedLikelihood(method="reml")
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_VarianceBasedLikelihood_ml(testdata_ibma):
    """
    Smoke test for VarianceBasedLikelihood with ML.
    """
    meta = ibma.VarianceBasedLikelihood(method="ml")
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_VarianceBasedLikelihood_reml(testdata_ibma):
    """
    Smoke test for VarianceBasedLikelihood with REML.
    """
    meta = ibma.VarianceBasedLikelihood(method="reml")
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert isinstance(res, nimare.results.MetaResult)


def test_PermutedOLS(testdata_ibma):
    """
    Smoke test for PermutedOLS with FWE correction.
    """
    meta = ibma.PermutedOLS(two_sided=True)
    meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    corr = FWECorrector(method="montecarlo", n_iters=100, n_cores=1)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.base.MetaResult)


def test_ibma_with_custom_masker(testdata_ibma):
    """ Ensure voxel-to-ROI reduction works. """
    atlas = op.join(get_test_data_path(), "test_pain_dataset", "atlas.nii.gz")
    masker = NiftiLabelsMasker(atlas)
    meta = ibma.Fishers(mask=masker)
    meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.results.MetaResult)
    assert meta.results.maps["z"].shape == (5,)


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
    meta = ibma.Fishers(resample=resample, **resample_kwargs)
    with expectation:
        meta.fit(testdata_ibma_resample)
    if isinstance(expectation, does_not_raise):
        assert isinstance(meta.results, nimare.results.MetaResult)
