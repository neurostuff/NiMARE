"""
Test nimare.meta.cbma.ale (ALE/SCALE meta-analytic algorithms).
"""
import pytest

import nimare
from nimare.meta.cbma import ale
from nimare.correct import FWECorrector


def test_ale(cbma_testdata1):
    """
    Smoke test for ALE
    """
    ale_meta = ale.ALE()
    ale_meta.fit(cbma_testdata1)
    corr = FWECorrector(method='permutation')
    result_corr = corr.transform(ale_meta.results, voxel_thresh=0.001,
                                 n_iters=5, n_cores=1)
    assert isinstance(ale_meta.results, nimare.base.MetaResult)
    assert isinstance(result_corr, nimare.base.MetaResult)


def test_ale_subtraction(cbma_testdata1, cbma_testdata2):
    """
    Smoke test for ALE
    """
    ale_meta1 = ale.ALE()
    ale_meta1.fit(cbma_testdata1)

    ale_meta2 = ale.ALE()
    ale_meta2.fit(cbma_testdata1)

    corr = FWECorrector(method='permutation')
    result_corr1 = corr.transform(ale_meta1.results, voxel_thresh=0.001,
                                  n_iters=5, n_cores=1)
    result_corr2 = corr.transform(ale_meta2.results, voxel_thresh=0.001,
                                  n_iters=5, n_cores=1)

    sub_meta = ale.ALESubtraction(n_iters=5)
    sub_meta.fit(
        ale_meta1, ale_meta2,
        image1=result_corr1.get_map('logp_level-cluster_corr-FWE_method-permutation', return_type='image'),
        image2=result_corr2.get_map('logp_level-cluster_corr-FWE_method-permutation', return_type='image'))
    assert isinstance(sub_meta.results, nimare.base.MetaResult)
