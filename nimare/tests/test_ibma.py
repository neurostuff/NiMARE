"""
Test nimare.meta.ibma (image-based meta-analytic estimators).
"""
import os.path as op

from nilearn.input_data import NiftiLabelsMasker

import nimare
from nimare.meta import ibma
from nimare.correct import FDRCorrector
from ..utils import get_resource_path
from .utils import get_test_data_path


def test_Fishers(testdata_ibma):
    """
    Smoke test for Fisher's.
    """
    meta = ibma.Fishers()
    res = meta.fit(testdata_ibma)
    corr = FDRCorrector(method='indep', alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_Stouffers(testdata_ibma):
    """
    Smoke test for Stouffer's, not weighted by sample size.
    """
    meta = ibma.Stouffers(use_sample_size=False)
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_Stouffers_weighted(testdata_ibma):
    """
    Smoke test for Stouffer's, weighted by sample size.
    """
    meta = ibma.Stouffers(use_sample_size=True)
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_SampleSizeBased_ml(testdata_ibma):
    """
    Smoke test for SampleSizeBased with ML.
    """
    meta = ibma.SampleSizeBased(method='ml')
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_SampleSizeBased_reml(testdata_ibma):
    """
    Smoke test for SampleSizeBased with REML.
    """
    meta = ibma.SampleSizeBased(method='reml')
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_WeightedLeastSquares(testdata_ibma):
    """
    Smoke test for WeightedLeastSquares.
    """
    meta = ibma.WeightedLeastSquares(tau2=0)
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_DerSimonianLaird(testdata_ibma):
    """
    Smoke test for DerSimonianLaird.
    """
    meta = ibma.DerSimonianLaird()
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_Hedges(testdata_ibma):
    """
    Smoke test for Hedges.
    """
    meta = ibma.Hedges()
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_VarianceBasedLikelihood_ml(testdata_ibma):
    """
    Smoke test for VarianceBasedLikelihood with ML.
    """
    meta = ibma.VarianceBasedLikelihood(method='ml')
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_VarianceBasedLikelihood_reml(testdata_ibma):
    """
    Smoke test for VarianceBasedLikelihood with REML.
    """
    meta = ibma.VarianceBasedLikelihood(method='reml')
    res = meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)


def test_RandomEffectsGLM_theoretical(testdata_ibma):
    """
    Smoke test for RandomEffectsGLM with theoretical null (i.e., t-test).
    """
    meta = ibma.RandomEffectsGLM(null='theoretical')
    meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_RandomEffectsGLM_empirical(testdata_ibma):
    """
    Smoke test for RandomEffectsGLM with empirical null (i.e., con permutation).
    """
    meta = ibma.RandomEffectsGLM(null='empirical', n_iters=10)
    meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_ibma_with_custom_masker(testdata_ibma):
    """ Ensure voxel-to-ROI reduction works. """
    atlas = op.join(get_test_data_path(), 'test_pain_dataset', 'atlas.nii.gz')
    masker = NiftiLabelsMasker(atlas)
    meta = ibma.Fishers(mask=masker)
    meta.fit(testdata_ibma)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert meta.results.maps['z'].shape == (5, )
