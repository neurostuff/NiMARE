"""
Test nimare.meta.cbma.ale (ALE/SCALE meta-analytic algorithms).
"""
import nimare
from nimare.meta.cbma import ale
from nimare.correct import FWECorrector


def test_ale(cbma_testdata1):
    """
    Smoke test for ALE
    """
    meta = ale.ALE()
    res = meta.fit(cbma_testdata1)
    assert isinstance(res, nimare.base.MetaResult)
    corr = FWECorrector(method='bonferroni')
    result_corr = corr.transform(res)
    assert isinstance(result_corr, nimare.base.MetaResult)
    corr = FWECorrector(method='permutation', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    corr_res = corr.transform(meta.results)
    assert isinstance(corr_res, nimare.base.MetaResult)


def test_ale_subtraction(cbma_testdata1, cbma_testdata2):
    """
    Smoke test for ALE
    """
    meta1 = ale.ALE()
    res1 = meta1.fit(cbma_testdata1)

    meta2 = ale.ALE()
    res2 = meta2.fit(cbma_testdata1)

    corr = FWECorrector(method='permutation', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    corr_res1 = corr.transform(res1)
    corr_res2 = corr.transform(res2)

    sub_meta = ale.ALESubtraction(n_iters=5)
    sub_meta.fit(
        meta1, meta2,
        image1=corr_res1.get_map('logp_level-cluster_corr-FWE_method-permutation', return_type='image'),
        image2=corr_res2.get_map('logp_level-cluster_corr-FWE_method-permutation', return_type='image'))
    assert isinstance(sub_meta.results, nimare.base.MetaResult)
