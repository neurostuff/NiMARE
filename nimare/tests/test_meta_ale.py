"""
Test nimare.meta.ale (ALE/SCALE meta-analytic algorithms).
"""
import os
import pickle

import nibabel as nib
import numpy as np
import pytest

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import ale


def test_ale(testdata_cbma, tmp_path_factory):
    """
    Smoke test for ALE
    """
    tmpdir = tmp_path_factory.mktemp("test_ale")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

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

    # Test saving/loading
    meta.save(out_file, compress=True)
    assert os.path.isfile(out_file)
    meta2 = ale.ALE.load(out_file, compressed=True)
    assert isinstance(meta2, ale.ALE)
    with pytest.raises(pickle.UnpicklingError):
        ale.ALE.load(out_file, compressed=False)

    meta.save(out_file, compress=False)
    assert os.path.isfile(out_file)
    meta2 = ale.ALE.load(out_file, compressed=False)
    assert isinstance(meta2, ale.ALE)
    with pytest.raises(OSError):
        ale.ALE.load(out_file, compressed=True)

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


def test_ale_subtraction(testdata_cbma, tmp_path_factory):
    """
    Smoke test for ALESubtraction
    """
    tmpdir = tmp_path_factory.mktemp("test_ale_subtraction")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    sub_meta = ale.ALESubtraction(n_iters=10, low_memory=False)
    sub_meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(sub_meta.results, nimare.results.MetaResult)
    assert "z_desc-group1MinusGroup2" in sub_meta.results.maps.keys()

    sub_meta.save(out_file)
    assert os.path.isfile(out_file)


def test_ale_subtraction_lowmem(testdata_cbma, tmp_path_factory):
    """
    Smoke test for ALESubtraction with low memory settings.
    """
    tmpdir = tmp_path_factory.mktemp("test_ale_subtraction_lowmem")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    sub_meta = ale.ALESubtraction(n_iters=10, low_memory=True)
    sub_meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(sub_meta.results, nimare.results.MetaResult)
    assert "z_desc-group1MinusGroup2" in sub_meta.results.maps.keys()

    sub_meta.save(out_file)
    assert os.path.isfile(out_file)
