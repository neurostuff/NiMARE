"""Test nimare.meta.ale (ALE/SCALE meta-analytic algorithms)."""

import os
import pickle

import nibabel as nib
import numpy as np
import pytest
from nilearn.input_data import NiftiLabelsMasker

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import ale
from nimare.results import MetaResult
from nimare.tests.utils import get_test_data_path
from nimare.utils import vox2mm


def test_ALE_approximate_null_unit(testdata_cbma, tmp_path_factory):
    """Unit test for ALE with approximate null_method."""
    tmpdir = tmp_path_factory.mktemp("test_ALE_approximate_null_unit")
    est_out_file = os.path.join(tmpdir, "est_file.pkl.gz")
    res_out_file = os.path.join(tmpdir, "res_file.pkl.gz")

    meta = ale.ALE(null_method="approximate")
    results = meta.fit(testdata_cbma)
    assert "stat" in results.maps.keys()
    assert "p" in results.maps.keys()
    assert "z" in results.maps.keys()
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.get_map("z", return_type="image"), nib.Nifti1Image)
    assert isinstance(results.get_map("z", return_type="array"), np.ndarray)
    results_copy = results.copy()
    assert results_copy != results
    assert isinstance(results, nimare.results.MetaResult)

    # Test saving/loading estimator
    for compress in [True, False]:
        meta.save(est_out_file, compress=compress)
        assert os.path.isfile(est_out_file)
        meta2 = ale.ALE.load(est_out_file, compressed=compress)
        assert isinstance(meta2, ale.ALE)
        if compress:
            with pytest.raises(pickle.UnpicklingError):
                ale.ALE.load(est_out_file, compressed=(not compress))
        else:
            with pytest.raises(OSError):
                ale.ALE.load(est_out_file, compressed=(not compress))

    # Test saving/loading MetaResult object
    for compress in [True, False]:
        results.save(res_out_file, compress=compress)
        assert os.path.isfile(res_out_file)
        res2 = MetaResult.load(res_out_file, compressed=compress)
        assert isinstance(res2, MetaResult)
        if compress:
            with pytest.raises(pickle.UnpicklingError):
                MetaResult.load(res_out_file, compressed=(not compress))
        else:
            with pytest.raises(OSError):
                MetaResult.load(res_out_file, compressed=(not compress))

    # Test MCC methods
    # Monte Carlo FWE
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=-1)
    corr_results = corr.transform(results)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)
    assert "z_desc-size_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "z_desc-mass_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "z_level-voxel_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert isinstance(
        corr_results.get_map(
            "z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="image"
        ),
        nib.Nifti1Image,
    )
    assert isinstance(
        corr_results.get_map(
            "z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ),
        np.ndarray,
    )
    assert isinstance(
        corr_results.get_map(
            "z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="image"
        ),
        nib.Nifti1Image,
    )
    assert isinstance(
        corr_results.get_map(
            "z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ),
        np.ndarray,
    )

    # Bonferroni FWE
    corr = FWECorrector(method="bonferroni")
    corr_results = corr.transform(results)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)
    assert isinstance(
        corr_results.get_map("z_corr-FWE_method-bonferroni", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        corr_results.get_map("z_corr-FWE_method-bonferroni", return_type="array"), np.ndarray
    )

    # FDR
    corr = FDRCorrector(method="indep", alpha=0.05)
    corr_results = corr.transform(results)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)
    assert isinstance(
        corr_results.get_map("z_corr-FDR_method-indep", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        corr_results.get_map("z_corr-FDR_method-indep", return_type="array"), np.ndarray
    )


def test_ALE_montecarlo_null_unit(testdata_cbma, tmp_path_factory):
    """Unit test for ALE with an montecarlo null_method.

    This test is run with low-memory kernel transformation as well.
    """
    tmpdir = tmp_path_factory.mktemp("test_ALE_montecarlo_null_unit")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    meta = ale.ALE(null_method="montecarlo", n_iters=10)
    results = meta.fit(testdata_cbma)
    assert isinstance(results.description_, str)
    assert "stat" in results.maps.keys()
    assert "p" in results.maps.keys()
    assert "z" in results.maps.keys()
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.get_map("z", return_type="image"), nib.Nifti1Image)
    assert isinstance(results.get_map("z", return_type="array"), np.ndarray)
    results_copy = results.copy()
    assert results_copy != results
    assert isinstance(results, nimare.results.MetaResult)

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
    corr_results = corr.transform(results)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)
    assert "z_desc-size_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "z_desc-mass_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "z_level-voxel_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in corr_results.maps.keys()
    assert isinstance(
        corr_results.get_map(
            "z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="image"
        ),
        nib.Nifti1Image,
    )
    assert isinstance(
        corr_results.get_map(
            "z_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ),
        np.ndarray,
    )
    assert isinstance(
        corr_results.get_map(
            "z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="image"
        ),
        nib.Nifti1Image,
    )
    assert isinstance(
        corr_results.get_map(
            "z_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ),
        np.ndarray,
    )

    # Check that the updated null distribution is in the corrected MetaResult's Estimator.
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        in corr_results.estimator.null_distributions_.keys()
    )
    # The updated null distribution should *not* be in the original Estimator, nor in the
    # uncorrected MetaResult's Estimator.
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        not in meta.null_distributions_.keys()
    )
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        not in results.estimator.null_distributions_.keys()
    )

    # Bonferroni FWE
    corr = FWECorrector(method="bonferroni")
    corr_results = corr.transform(results)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(
        corr_results.get_map("z_corr-FWE_method-bonferroni", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        corr_results.get_map("z_corr-FWE_method-bonferroni", return_type="array"), np.ndarray
    )

    # FDR
    corr = FDRCorrector(method="indep", alpha=0.05)
    corr_results = corr.transform(results)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(
        corr_results.get_map("z_corr-FDR_method-indep", return_type="image"), nib.Nifti1Image
    )
    assert isinstance(
        corr_results.get_map("z_corr-FDR_method-indep", return_type="array"), np.ndarray
    )


def test_ALESubtraction_smoke(testdata_cbma, tmp_path_factory):
    """Smoke test for ALESubtraction."""
    tmpdir = tmp_path_factory.mktemp("test_ALESubtraction_smoke")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    sub_meta = ale.ALESubtraction(n_iters=10, n_cores=2)
    results = sub_meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
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
    results = meta.fit(dset)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert "z" in results.maps.keys()
    assert isinstance(results.get_map("z", return_type="image"), nib.Nifti1Image)
    assert isinstance(results.get_map("z", return_type="array"), np.ndarray)

    meta.save(out_file)
    assert os.path.isfile(out_file)


def test_ALE_non_nifti_masker(testdata_cbma):
    """Unit test for ALE with non-NiftiMasker.

    CBMA estimators don't work with non-NiftiMasker (e.g., a NiftiLabelsMasker).
    """
    atlas = os.path.join(get_test_data_path(), "test_pain_dataset", "atlas.nii.gz")
    masker = NiftiLabelsMasker(atlas)
    meta = ale.ALE(mask=masker, n_iters=10)

    with pytest.raises(ValueError):
        meta.fit(testdata_cbma)
