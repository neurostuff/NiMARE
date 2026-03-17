"""Test nimare.meta.ale (ALE/SCALE meta-analytic algorithms)."""

import copy
import os
import pickle

import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiLabelsMasker

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.generate import create_coordinate_dataset
from nimare.meta import ale
from nimare.results import MetaResult
from nimare.tests.utils import get_test_data_path
from nimare.utils import vox2mm


def test_ALE_missing_sample_sizes_raises_informative_error(testdata_cbma_full):
    """Raise a helpful error listing ids when sample sizes are missing."""
    dset = copy.deepcopy(testdata_cbma_full)
    bad_id = dset.coordinates["id"].iloc[0]
    dset.metadata.loc[dset.metadata["id"] == bad_id, "sample_sizes"] = None

    with pytest.raises(ValueError) as excinfo:
        ale.ALE(null_method="approximate").fit(dset)

    msg = str(excinfo.value).lower()
    assert "sample size" in msg
    assert bad_id.lower() in msg


def test_cbma_raises_without_masker():
    """CBMA estimators require a masker to run."""
    dset_dict = {
        "study1": {
            "contrasts": {"contrast1": {"coords": {"space": "MNI", "x": [0], "y": [0], "z": [0]}}}
        }
    }
    dset = nimare.dataset.Dataset(dset_dict, target=None, mask=None)

    with pytest.raises(ValueError, match=r"masker is required"):
        ale.ALE(null_method="approximate").fit(dset)


def test_cbma_raises_on_mixed_coordinate_spaces(mni_mask):
    """CBMA estimators reject datasets with mixed coordinate spaces."""
    dset_dict = {
        "study1": {
            "contrasts": {"contrast1": {"coords": {"space": "MNI", "x": [0], "y": [0], "z": [0]}}}
        },
        "study2": {
            "contrasts": {"contrast1": {"coords": {"space": "TAL", "x": [0], "y": [0], "z": [0]}}}
        },
    }
    dset = nimare.dataset.Dataset(dset_dict, target=None, mask=None)

    with pytest.raises(ValueError, match=r"Mixed coordinate spaces detected"):
        ale.ALE(null_method="approximate", mask=mni_mask).fit(dset)


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


def test_ALE_montecarlo_histogram_reduction_matches_batch():
    """Streaming Monte Carlo histogram reduction should match batch reduction semantics."""
    _, dset = create_coordinate_dataset(
        foci=3,
        fwhm=10.0,
        n_studies=4,
        sample_size=30,
        n_noise_foci=5,
        seed=7,
    )
    meta = ale.ALE(null_method="montecarlo", n_iters=3, n_cores=1)
    meta.masker = dset.masker
    meta._collect_inputs(dset)
    meta._preprocess_input(dset)
    meta.null_distributions_ = {"histogram_bins": np.array([0.0, 0.5, 1.0])}

    counts_seq = [
        np.array([3, 0, 1], dtype=np.int32),
        np.array([1, 2, 0], dtype=np.int32),
        np.array([0, 1, 3], dtype=np.int32),
    ]
    expected_uncorr = np.sum(counts_seq, axis=0)
    expected_vfwe = np.zeros_like(expected_uncorr)
    for idx in (2, 1, 2):
        expected_vfwe[idx] += 1

    counter = iter(counts_seq)

    def fake_permutation(iter_xyz, iter_df, bin_edges=None):
        return next(counter).copy()

    meta._compute_null_montecarlo_permutation = fake_permutation
    meta._compute_null_montecarlo(n_iters=3, n_cores=1)

    np.testing.assert_array_equal(
        meta.null_distributions_["histweights_corr-none_method-montecarlo"],
        expected_uncorr,
    )
    np.testing.assert_array_equal(
        meta.null_distributions_["histweights_level-voxel_corr-fwe_method-montecarlo"],
        expected_vfwe,
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
    assert (
        "values_level-voxel_corr-fwe_method-montecarlo"
        in results.estimator.null_distributions_.keys()
    )
    assert (
        "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
        not in results.estimator.null_distributions_.keys()
    )
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        not in results.estimator.null_distributions_.keys()
    )

    sub_meta.save(out_file)
    assert os.path.isfile(out_file)


def test_ALESubtraction_init_vfwe_voxel_thresh_logic():
    """Verify ALESubtraction init validation for vfwe_only/voxel_thresh."""
    # Default init should work and keep default vfwe_only behavior.
    sub_meta = ale.ALESubtraction()
    assert sub_meta.vfwe_only is True
    assert sub_meta.voxel_thresh == 0.001

    # If cluster nulls are requested, voxel_thresh is required.
    with pytest.raises(ValueError, match="voxel_thresh must be provided"):
        ale.ALESubtraction(vfwe_only=False, voxel_thresh=None)

    # If cluster nulls are requested, voxel_thresh must be numeric.
    with pytest.raises(TypeError, match="voxel_thresh must be a scalar numeric value"):
        ale.ALESubtraction(vfwe_only=False, voxel_thresh="not_a_number")

    # If cluster nulls are requested, voxel_thresh must be in (0, 1).
    with pytest.raises(ValueError, match="between 0 and 1"):
        ale.ALESubtraction(vfwe_only=False, voxel_thresh=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ale.ALESubtraction(vfwe_only=False, voxel_thresh=1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ale.ALESubtraction(vfwe_only=False, voxel_thresh=-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ale.ALESubtraction(vfwe_only=False, voxel_thresh=1.1)

    # Numeric-like strings are accepted by float conversion.
    sub_meta = ale.ALESubtraction(vfwe_only=False, voxel_thresh="0.01")
    assert sub_meta.vfwe_only is False


def test_ALESubtraction_cluster_nulls(testdata_cbma):
    """Verify optional cluster nulls are computed for ALESubtraction."""
    sub_meta = ale.ALESubtraction(n_iters=2, n_cores=1, vfwe_only=False, voxel_thresh=0.05)
    results = sub_meta.fit(testdata_cbma, testdata_cbma)

    assert (
        "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
        in results.estimator.null_distributions_.keys()
    )
    assert (
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
        in results.estimator.null_distributions_.keys()
    )


def test_ALESubtraction_masked_sparse_summary_matches_coo():
    """Masked sparse ALE summaries should match the original COO path."""
    _, dset = create_coordinate_dataset(
        foci=3,
        fwhm=10.0,
        n_studies=6,
        sample_size=30,
        n_noise_foci=5,
        seed=12,
    )
    dset1 = dset.slice(dset.ids[:3])

    sub_meta = ale.ALESubtraction(n_iters=2, n_cores=1)
    sub_meta.masker = dset1.masker
    sub_meta._collect_inputs(dset1)
    sub_meta._preprocess_input(dset1)
    sub_meta.inputs_["id1"] = sub_meta.inputs_.pop("id")
    sub_meta.inputs_["coordinates1"] = sub_meta.inputs_.pop("coordinates")

    ma_maps = sub_meta._collect_ma_maps(coords_key="coordinates1")
    masked_ma_maps = sub_meta._ma_maps_to_masked_matrix(ma_maps)

    coo_summary = sub_meta._compute_summarystat_est(ma_maps)
    masked_summary = sub_meta._compute_summarystat_est(masked_ma_maps)

    np.testing.assert_allclose(masked_summary, coo_summary, rtol=1e-5, atol=2e-7)


def test_ALESubtraction_chunked_pvalues_match_scalar_path():
    """Chunked p-value conversion should match the scalar implementation."""
    rng = np.random.default_rng(4)
    sub_meta = ale.ALESubtraction(n_iters=5, n_cores=1)

    stat_values = rng.normal(size=17).astype(np.float32)
    iter_diff_values = rng.normal(size=(13, 17)).astype(np.float32)

    scalar_p = np.array(
        [
            sub_meta._alediff_to_p_voxel(
                i_voxel, stat_values[i_voxel], iter_diff_values[:, i_voxel]
            )[0]
            for i_voxel in range(stat_values.shape[0])
        ]
    ).reshape(-1)
    scalar_sign = np.sign(stat_values - np.median(iter_diff_values, axis=0))

    chunked_p, chunked_sign = sub_meta._alediff_to_p_values(
        stat_values, iter_diff_values, chunk_size=4
    )

    np.testing.assert_allclose(chunked_p, scalar_p)
    np.testing.assert_array_equal(chunked_sign, scalar_sign)


def test_ALESubtraction_fwe_description_branches(testdata_cbma):
    """Verify ALESubtraction Monte Carlo FWE descriptions match correction mode."""
    # Voxel-only branch.
    sub_meta_vfwe = ale.ALESubtraction(n_iters=2, n_cores=1, vfwe_only=True)
    results_vfwe = sub_meta_vfwe.fit(testdata_cbma, testdata_cbma)
    corr_vfwe = FWECorrector(method="montecarlo", n_iters=2, n_cores=1, vfwe_only=True)
    corr_results_vfwe = corr_vfwe.transform(results_vfwe)

    assert (
        "voxel-level Monte Carlo procedure for ALE subtraction" in corr_results_vfwe.description_
    )
    assert "cluster sizes, and cluster masses" not in corr_results_vfwe.description_

    # Voxel + cluster branch.
    sub_meta_cluster = ale.ALESubtraction(n_iters=2, n_cores=1, vfwe_only=False, voxel_thresh=0.05)
    results_cluster = sub_meta_cluster.fit(testdata_cbma, testdata_cbma)
    corr_cluster = FWECorrector(
        method="montecarlo", n_iters=2, n_cores=1, vfwe_only=False, voxel_thresh=0.05
    )
    corr_results_cluster = corr_cluster.transform(results_cluster)

    assert "cluster sizes, and cluster masses" in corr_results_cluster.description_
    assert "face-wise connectivity" in corr_results_cluster.description_
    assert "p < 0.05" in corr_results_cluster.description_


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


def test_SCALE_chunked_pvalues_match_scalar_path():
    """Chunked SCALE p-value conversion should match the scalar implementation."""
    rng = np.random.default_rng(7)
    meta = ale.SCALE(xyz=np.zeros((1, 3)), n_iters=5, n_cores=1)
    meta.null_distributions_ = {"histogram_bins": np.linspace(0, 1, 11)}

    stat_values = rng.uniform(0.0, 0.8, size=9).astype(np.float32)
    scale_values = rng.uniform(0.0, 0.8, size=(13, 9)).astype(np.float32)
    scale_values[scale_values < 0.2] = 0

    scalar_p = np.array(
        [
            meta._scale_to_p_voxel(i_voxel, stat_values[i_voxel], scale_values[:, i_voxel].copy())[
                0
            ]
            for i_voxel in range(stat_values.shape[0])
        ]
    ).reshape(-1)

    chunked_p = meta._scale_to_p_values(stat_values, scale_values, chunk_size=4)

    np.testing.assert_allclose(chunked_p, scalar_p)


def test_ALE_non_nifti_masker(testdata_cbma):
    """Unit test for ALE with non-NiftiMasker.

    CBMA estimators don't work with non-NiftiMasker (e.g., a NiftiLabelsMasker).
    """
    atlas = os.path.join(get_test_data_path(), "test_pain_dataset", "atlas.nii.gz")
    masker = NiftiLabelsMasker(atlas)
    meta = ale.ALE(mask=masker, n_iters=10)

    with pytest.raises(ValueError):
        meta.fit(testdata_cbma)
