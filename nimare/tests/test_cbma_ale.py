"""
Test nimare.meta.cbma.ale (ALE/SCALE meta-analytic algorithms).
"""
import pytest

import nimare
from nimare.meta.cbma import ale
from nimare.correct import FWECorrector


def test_ale():
    """
    Smoke test for ALE
    """
    meta = ale.ALE()
    res = meta.fit(pytest.cbma_testdata1)
    assert isinstance(res, nimare.base.MetaResult)
    corr = FWECorrector(method='bonferroni')
    cres = corr.transform(res)
    assert isinstance(cres, nimare.base.MetaResult)
    corr = FWECorrector(method='permutation', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.base.MetaResult)


def test_ale_subtraction():
    """
    Smoke test for ALE
    """
    meta1 = ale.ALE()
    res1 = meta1.fit(pytest.cbma_testdata1)

    meta2 = ale.ALE()
    res2 = meta2.fit(pytest.cbma_testdata1)

    corr = FWECorrector(method='permutation', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    cres1 = corr.transform(res1)
    cres2 = corr.transform(res2)

    sub_meta = ale.ALESubtraction()
    sub_meta.fit(
        meta1, meta2,
        image1=cres1.get_map('logp_level-cluster_corr-FWE_method-permutation', return_type='image'),
        image2=cres2.get_map('logp_level-cluster_corr-FWE_method-permutation', return_type='image'))
    assert isinstance(sub_meta.results, nimare.base.MetaResult)
