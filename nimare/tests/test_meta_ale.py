"""
Test nimare.meta.ale (ALE/SCALE meta-analytic algorithms).
"""
import nimare
from nimare.meta import ale
from nimare.correct import FWECorrector, FDRCorrector


def test_ale(testdata_cbma):
    """
    Smoke test for ALE
    """
    meta = ale.ALE()
    res = meta.fit(testdata_cbma)
    assert 'ale' in res.maps.keys()
    assert 'p' in res.maps.keys()
    assert 'z' in res.maps.keys()
    assert isinstance(res, nimare.base.MetaResult)
    # Test MCC methods
    corr = FWECorrector(method='montecarlo', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.base.MetaResult)
    assert 'z_level-cluster_corr-FWE_method-montecarlo' in cres.maps.keys()
    assert 'z_level-voxel_corr-FWE_method-montecarlo' in cres.maps.keys()
    assert 'logp_level-cluster_corr-FWE_method-montecarlo' in cres.maps.keys()
    assert 'logp_level-voxel_corr-FWE_method-montecarlo' in cres.maps.keys()
    corr = FWECorrector(method='bonferroni')
    cres = corr.transform(res)
    assert isinstance(cres, nimare.base.MetaResult)
    corr = FDRCorrector(method='indep', alpha=0.05)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.base.MetaResult)


def test_ale_subtraction(testdata_cbma):
    """
    Smoke test for ALESubtraction
    """
    meta1 = ale.ALE()
    res1 = meta1.fit(testdata_cbma)

    meta2 = ale.ALE()
    res2 = meta2.fit(testdata_cbma)

    sub_meta = ale.ALESubtraction(n_iters=10)
    sub_meta.fit(meta1, meta2)
    assert isinstance(sub_meta.results, nimare.base.MetaResult)
    assert 'z_desc-group1MinusGroup2' in sub_meta.results.maps.keys()
