"""
Test nimare.meta.cbma.mkda (KDA-based meta-analytic algorithms).
"""
import nimare
from nimare.meta.cbma import mkda
from nimare.correct import FWECorrector, FDRCorrector


def test_mkda_density(testdata):
    """
    Smoke test for MKDADensity
    """
    meta = mkda.MKDADensity()
    res = meta.fit(testdata['dset'])
    corr = FWECorrector(method='montecarlo', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_mkda_chi2_fdr(testdata):
    """
    Smoke test for MKDAChi2
    """
    meta = mkda.MKDAChi2()
    res = meta.fit(testdata['dset'], testdata['dset'])
    corr = FDRCorrector(method='bh', alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_mkda_chi2_fwe(testdata):
    """
    Smoke test for MKDAChi2
    """
    meta = mkda.MKDAChi2()
    res = meta.fit(testdata['dset'], testdata['dset'])
    corr = FWECorrector(method='montecarlo', n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_kda_density(testdata):
    """
    Smoke test for KDA
    """
    meta = mkda.KDA()
    res = meta.fit(testdata['dset'])
    corr = FWECorrector(method='montecarlo', n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)
