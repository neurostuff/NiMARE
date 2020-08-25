"""
Test nimare.meta.ale (ALE/SCALE meta-analytic algorithms).
"""
import os

import nibabel as nib
import numpy as np

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import ale


def test_ale(testdata_cbma):
    """
    Smoke test for ALE
    """
    out_file = os.path.abspath("file.pkl.gz")
    meta = ale.ALE()
    res = meta.fit(testdata_cbma)
    assert "ale" in res.maps.keys()
    assert "p" in res.maps.keys()
    assert "z" in res.maps.keys()
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(res.get_map("z", return_type="image"), nib.Nifti1Image)
    assert isinstance(res.get_map("z", return_type="array"), np.ndarray)
    res2 = res.copy()
    assert res2 != res
    assert isinstance(res, nimare.results.MetaResult)

    # Test saving
    meta.save(out_file)
    assert os.path.isfile(out_file)

    # Test loading
    meta2 = ale.ALE.load(out_file)
    assert isinstance(meta2, ale.ALE)
    os.remove(out_file)

    # Test MCC methods
    # Monte Carlo FWE
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=-1)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.results.MetaResult)
    assert "z_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "z_level-voxel_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert isinstance(
        cres.get_map("z_level-cluster_corr-FWE_method-montecarlo", return_type="image"),
        nib.Nifti1Image,
    )
    assert isinstance(
        cres.get_map("z_level-cluster_corr-FWE_method-montecarlo", return_type="array"), np.ndarray
    )

    # Bonferroni FWE
    corr = FWECorrector(method="bonferroni")
    cres = corr.transform(res)
    assert isinstance(cres, nimare.results.MetaResult)

    # FDR
    corr = FDRCorrector(method="indep", alpha=0.05)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.results.MetaResult)


def test_ale_subtraction(testdata_cbma):
    """
    Smoke test for ALESubtraction
    """
    out_file = os.path.abspath("file.pkl.gz")
    meta1 = ale.ALE()
    meta1.fit(testdata_cbma)

    meta2 = ale.ALE()
    meta2.fit(testdata_cbma)

    sub_meta = ale.ALESubtraction(n_iters=10)
    sub_meta.fit(meta1, meta2)
    assert isinstance(sub_meta.results, nimare.results.MetaResult)
    assert "z_desc-group1MinusGroup2" in sub_meta.results.maps.keys()

    sub_meta.save(out_file)
    assert os.path.isfile(out_file)
    os.remove(out_file)
