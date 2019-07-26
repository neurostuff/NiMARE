"""
Test nimare.meta.cbma.mkda (KDA-based meta-analytic algorithms).
"""
import pytest

import nimare
from nimare.meta.cbma import mkda
from nimare.correct import FWECorrector


def test_mkda_density(cbma_testdata1):
    """
    Smoke test for MKDADensity
    """
    meta = mkda.MKDADensity(n_iters=5, n_cores=1)
    meta.fit(cbma_testdata1)
    corr = FWECorrector(method='permutation')
    result_corr = corr.transform(meta.results, voxel_thresh=0.001,
                                 n_iters=5, n_cores=1)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(result_corr, nimare.base.MetaResult)


def test_mkda_chi2_fdr(cbma_testdata1):
    """
    Smoke test for MKDAChi2
    """
    mkda_meta = mkda.MKDAChi2(corr='fdr', n_cores=1)
    mkda_meta.fit(cbma_testdata1, cbma_testdata1)
    assert isinstance(mkda_meta.results, nimare.base.MetaResult)


def test_mkda_chi2_fwe(cbma_testdata1, cbma_testdata2):
    """
    Smoke test for MKDAChi2
    """
    mkda_meta = mkda.MKDAChi2(n_iters=5, corr='fwe', n_cores=1)
    mkda_meta.fit(cbma_testdata1, cbma_testdata2)
    assert isinstance(mkda_meta.results, nimare.base.MetaResult)


def test_kda_density(cbma_testdata1):
    """
    Smoke test for KDA
    """
    kda_meta = mkda.KDA(n_iters=5, n_cores=1)
    kda_meta.fit(cbma_testdata1)
    assert isinstance(kda_meta.results, nimare.base.MetaResult)
