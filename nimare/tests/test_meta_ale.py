"""Test nimare.meta.ale (ALE/SCALE meta-analytic algorithms)."""
import logging
import os
import pickle

import nibabel as nib
import numpy as np
import pytest

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import ale


def test_ALE_ma_map_reuse(testdata_cbma, tmp_path_factory, caplog):
    """Test that MA maps are re-used when appropriate."""
    from nimare.meta import kernel

    tmpdir = tmp_path_factory.mktemp("test_ALE_ma_map_reuse")
    testdata_cbma.update_path(tmpdir)

    # ALEKernel cannot extract sample_size from a Dataset,
    # so we need to set it for this kernel and for the later meta-analyses.
    kern = kernel.ALEKernel(sample_size=20)
    dset = kern.transform(testdata_cbma, return_type="dataset")

    # The associated column should be in the new Dataset's images DataFrame
    cols = dset.images.columns.tolist()
    assert any(["ALEKernel" in col for col in cols])

    # The Dataset without the images will generate them from scratch.
    meta = ale.ALE(kernel__sample_size=20)
    with caplog.at_level(logging.DEBUG, logger="nimare.meta.cbma.base"):
        meta.fit(testdata_cbma)
    assert "Loading pre-generated MA maps" not in caplog.text

    # The Dataset with the images will re-use them,
    # as evidenced by the logger message.
    with caplog.at_level(logging.DEBUG, logger="nimare.meta.cbma.base"):
        meta.fit(dset)
    assert "Loading pre-generated MA maps" in caplog.text


def test_ALESubtraction_ma_map_reuse(testdata_cbma, tmp_path_factory, caplog):
    """Test that MA maps are re-used when appropriate."""
    from nimare.meta import kernel

    tmpdir = tmp_path_factory.mktemp("test_ALESubtraction_ma_map_reuse")
    testdata_cbma.update_path(tmpdir)

    # ALEKernel cannot extract sample_size from a Dataset,
    # so we need to set it for this kernel and for the later meta-analyses.
    kern = kernel.ALEKernel(sample_size=20)
    dset = kern.transform(testdata_cbma, return_type="dataset")

    # The Dataset without the images will generate them from scratch.
    sub_meta = ale.ALESubtraction(n_iters=10, kernel__sample_size=20)

    with caplog.at_level(logging.DEBUG, logger="nimare.meta.cbma.base"):
        sub_meta.fit(testdata_cbma, testdata_cbma)
    assert "Loading pre-generated MA maps" not in caplog.text

    # The Dataset with the images will re-use them,
    # as evidenced by the logger message.
    with caplog.at_level(logging.DEBUG, logger="nimare.meta.cbma.base"):
        sub_meta.fit(dset, dset)
    assert "Loading pre-generated MA maps" in caplog.text


def test_ALE_approximate_null_unit(testdata_cbma, tmp_path_factory):
    """Unit test for ALE with approximate null_method."""
    tmpdir = tmp_path_factory.mktemp("test_ALE_approximate_null_unit")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    meta = ale.ALE(null_method="approximate")
    res = meta.fit(testdata_cbma)
    assert "stat" in res.maps.keys()
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
    assert isinstance(
        cres.get_map("z_corr-FWE_method-bonferroni", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        cres.get_map("z_corr-FWE_method-bonferroni", return_type="array"), np.ndarray
    )

    # FDR
    corr = FDRCorrector(method="indep", alpha=0.05)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.results.MetaResult)
    assert isinstance(
        cres.get_map("z_corr-FDR_method-indep", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(cres.get_map("z_corr-FDR_method-indep", return_type="array"), np.ndarray)


def test_ALE_montecarlo_null_unit(testdata_cbma, tmp_path_factory):
    """Unit test for ALE with an montecarlo null_method.

    This test is run with low-memory kernel transformation as well.
    """
    tmpdir = tmp_path_factory.mktemp("test_ALE_montecarlo_null_unit")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    meta = ale.ALE(null_method="montecarlo", n_iters=10, kernel__low_memory=True)
    res = meta.fit(testdata_cbma)
    assert "stat" in res.maps.keys()
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
    assert isinstance(
        cres.get_map("z_corr-FWE_method-bonferroni", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        cres.get_map("z_corr-FWE_method-bonferroni", return_type="array"), np.ndarray
    )

    # FDR
    corr = FDRCorrector(method="indep", alpha=0.05)
    cres = corr.transform(meta.results)
    assert isinstance(cres, nimare.results.MetaResult)
    assert isinstance(
        cres.get_map("z_corr-FDR_method-indep", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(cres.get_map("z_corr-FDR_method-indep", return_type="array"), np.ndarray)


def test_ALESubtraction_smoke(testdata_cbma, tmp_path_factory):
    """Smoke test for ALESubtraction."""
    tmpdir = tmp_path_factory.mktemp("test_ALESubtraction_smoke")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    sub_meta = ale.ALESubtraction(n_iters=10, low_memory=False)
    sub_meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(sub_meta.results, nimare.results.MetaResult)
    assert "z_desc-group1MinusGroup2" in sub_meta.results.maps.keys()
    assert isinstance(
        sub_meta.results.get_map("z_desc-group1MinusGroup2", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        sub_meta.results.get_map("z_desc-group1MinusGroup2", return_type="array"), np.ndarray
    )

    sub_meta.save(out_file)
    assert os.path.isfile(out_file)


def test_ALESubtraction_smoke_lowmem(testdata_cbma, tmp_path_factory):
    """Smoke test for ALESubtraction with low memory settings."""
    tmpdir = tmp_path_factory.mktemp("test_ALESubtraction_smoke_lowmem")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    sub_meta = ale.ALESubtraction(n_iters=10, low_memory=True)
    sub_meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(sub_meta.results, nimare.results.MetaResult)
    assert "z_desc-group1MinusGroup2" in sub_meta.results.maps.keys()
    assert isinstance(
        sub_meta.results.get_map("z_desc-group1MinusGroup2", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        sub_meta.results.get_map("z_desc-group1MinusGroup2", return_type="array"), np.ndarray
    )

    sub_meta.save(out_file)
    assert os.path.isfile(out_file)


def test_SCALE_smoke(testdata_cbma):
    """Smoke test for SCALE."""
    dset = testdata_cbma.slice(testdata_cbma.ids[:3])
    ijk = np.vstack(np.where(testdata_cbma.masker.mask_img.get_fdata())).T
    ijk = ijk[:, :20]
    meta = ale.SCALE(n_iters=5, n_cores=1, ijk=ijk)
    res = meta.fit(dset)
    assert isinstance(res, nimare.results.MetaResult)
    assert "z" in res.maps.keys()
    assert isinstance(res.get_map("z", return_type="image"), nib.Nifti1Image)
    assert isinstance(res.get_map("z", return_type="array"), np.ndarray)


def test_SCALE_smoke_lowmem(testdata_cbma):
    """Smoke test for SCALE with low memory settings."""
    dset = testdata_cbma.slice(testdata_cbma.ids[:3])
    ijk = np.vstack(np.where(testdata_cbma.masker.mask_img.get_fdata())).T
    ijk = ijk[:, :20]
    meta = ale.SCALE(n_iters=5, n_cores=1, ijk=ijk, low_memory=True)
    res = meta.fit(dset)
    assert isinstance(res, nimare.results.MetaResult)
    assert "z" in res.maps.keys()
    assert isinstance(res.get_map("z", return_type="image"), nib.Nifti1Image)
    assert isinstance(res.get_map("z", return_type="array"), np.ndarray)
