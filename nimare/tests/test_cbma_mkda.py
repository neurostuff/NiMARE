"""
Test nimare.meta.cbma.mkda (KDA-based meta-analytic algorithms).
"""
import pytest

import nimare
from nimare.meta.cbma import mkda
from nimare.correct import FWECorrector, FDRCorrector


def test_mkda_density():
    """
    Smoke test for MKDADensity
    """
    meta = mkda.MKDADensity()
    res = meta.fit(pytest.cbma_testdata1)
    corr = FWECorrector(method='permutation', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_mkda_chi2_fdr():
    """
    Smoke test for MKDAChi2
    """
    meta = mkda.MKDAChi2()
    res = meta.fit(pytest.cbma_testdata1, pytest.cbma_testdata1)
    corr = FDRCorrector(method='fdr_bh', alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_mkda_chi2_fwe():
    """
    Smoke test for MKDAChi2
    """
    meta = mkda.MKDAChi2()
    res = meta.fit(pytest.cbma_testdata1, pytest.cbma_testdata2)
    corr = FWECorrector(method='permutation', n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_kda_density():
    """
    Smoke test for KDA
    """
    meta = mkda.KDA()
    res = meta.fit(pytest.cbma_testdata1)
    corr = FWECorrector(method='permutation', n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)
