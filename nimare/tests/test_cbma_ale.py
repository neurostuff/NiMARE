"""
Test nimare.meta.cbma.ale (ALE/SCALE meta-analytic algorithms).
"""
import nimare
from nimare.meta.cbma import ale
from nimare.correct import FWECorrector, FDRCorrector


def test_ale(testdata):
    """
    Smoke test for ALE
    """
    meta = ale.ALE()
    res = meta.fit(testdata['dset'])
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


def test_ale_subtraction(testdata):
    """
    Smoke test for ALE
    """
    meta1 = ale.ALE()
    res1 = meta1.fit(testdata['dset'])

    meta2 = ale.ALE()
    res2 = meta2.fit(testdata['dset'])

    corr = FWECorrector(method='montecarlo', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    cres1 = corr.transform(res1)
    cres2 = corr.transform(res2)

    sub_meta = ale.ALESubtraction(n_iters=10)
    sub_meta.fit(
        meta1, meta2,
        image1=cres1.get_map('logp_level-cluster_corr-FWE_method-montecarlo', return_type='image'),
        image2=cres2.get_map('logp_level-cluster_corr-FWE_method-montecarlo', return_type='image'))
    assert isinstance(sub_meta.results, nimare.base.MetaResult)
