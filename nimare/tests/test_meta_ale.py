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
from nimare.utils import vox2mm


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
    # If drop_invalid is False, then there should be an Exception, since two studies in the test
    # dataset are missing coordinates.
    meta = ale.ALE(kernel__sample_size=20)
    with pytest.raises(Exception):
        meta.fit(testdata_cbma, drop_invalid=False)

    with caplog.at_level(logging.DEBUG, logger="nimare.meta.cbma.base"):
        meta.fit(testdata_cbma)
    assert "Loading pre-generated MA maps" not in caplog.text

    # The Dataset with the images will re-use them, as evidenced by the logger message.
    with caplog.at_level(logging.DEBUG, logger="nimare.meta.cbma.base"):
        meta.fit(dset)
    assert "Loading pre-generated MA maps" in caplog.text

    # If there is a memory limit along with pre-generated images, then we should still see the
    # logger message.
    meta = ale.ALE(kernel__sample_size=20)
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
    cres = corr.transform(res)
    assert isinstance(cres, nimare.results.MetaResult)
    assert "z_desc-size_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "z_desc-mass_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "z_level-voxel_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert isinstance(
        cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="image"),
        nib.Nifti1Image,
    )
    assert isinstance(
        cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"),
        np.ndarray,
    )
    assert isinstance(
        cres.get_map("z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="image"),
        nib.Nifti1Image,
    )
    assert isinstance(
        cres.get_map("z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"),
        np.ndarray,
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
    cres = corr.transform(res)
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

    meta = ale.ALE(null_method="montecarlo", n_iters=10)
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
    cres = corr.transform(res)
    assert isinstance(cres, nimare.results.MetaResult)
    assert "z_desc-size_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "z_desc-mass_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "z_level-voxel_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in cres.maps.keys()
    assert isinstance(
        cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="image"),
        nib.Nifti1Image,
    )
    assert isinstance(
        cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"),
        np.ndarray,
    )
    assert isinstance(
        cres.get_map("z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="image"),
        nib.Nifti1Image,
    )
    assert isinstance(
        cres.get_map("z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"),
        np.ndarray,
    )

    # Check that the updated null distribution is in the corrected MetaResult's Estimator.
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        in cres.estimator.null_distributions_.keys()
    )
    # The updated null distribution should *not* be in the original Estimator, nor in the
    # uncorrected MetaResult's Estimator.
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        not in meta.null_distributions_.keys()
    )
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        not in res.estimator.null_distributions_.keys()
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
    cres = corr.transform(res)
    assert isinstance(cres, nimare.results.MetaResult)
    assert isinstance(
        cres.get_map("z_corr-FDR_method-indep", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(cres.get_map("z_corr-FDR_method-indep", return_type="array"), np.ndarray)


def test_ALESubtraction_smoke(testdata_cbma, tmp_path_factory):
    """Smoke test for ALESubtraction."""
    tmpdir = tmp_path_factory.mktemp("test_ALESubtraction_smoke")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    sub_meta = ale.ALESubtraction(n_iters=10, n_cores=2)
    results = sub_meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)
    assert "z_desc-group1MinusGroup2" in results.maps.keys()
    assert isinstance(
        results.get_map("z_desc-group1MinusGroup2", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(results.get_map("z_desc-group1MinusGroup2", return_type="array"), np.ndarray)

    sub_meta.save(out_file)
    assert os.path.isfile(out_file)


def test_SCALE_smoke(testdata_cbma, tmp_path_factory):
    """Smoke test for SCALE."""
    tmpdir = tmp_path_factory.mktemp("test_SCALE_smoke")
    out_file = os.path.join(tmpdir, "file.pkl.gz")
    dset = testdata_cbma.slice(testdata_cbma.ids[:3])

    with pytest.raises(TypeError):
        ale.SCALE(xyz="dog", n_iters=5, n_cores=1)

    with pytest.raises(ValueError):
        ale.SCALE(xyz=np.random.random((5, 3, 1)), n_iters=5, n_cores=1)

    with pytest.raises(ValueError):
        ale.SCALE(xyz=np.random.random((3, 10)), n_iters=5, n_cores=1)

    xyz = vox2mm(
        np.vstack(np.where(testdata_cbma.masker.mask_img.get_fdata())).T,
        testdata_cbma.masker.mask_img.affine,
    )
    xyz = xyz[:20, :]
    meta = ale.SCALE(xyz, n_iters=5, n_cores=1)
    res = meta.fit(dset)
    assert isinstance(res, nimare.results.MetaResult)
    assert "z" in res.maps.keys()
    assert isinstance(res.get_map("z", return_type="image"), nib.Nifti1Image)
    assert isinstance(res.get_map("z", return_type="array"), np.ndarray)

    meta.save(out_file)
    assert os.path.isfile(out_file)
