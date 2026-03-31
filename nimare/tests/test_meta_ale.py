"""Test nimare.meta.ale (ALE/SCALE meta-analytic algorithms)."""

import copy
import os
import pickle

import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiLabelsMasker
from scipy import sparse as sp_sparse

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.generate import create_coordinate_dataset
from nimare.meta import ale
from nimare.results import MetaResult
from nimare.stats import nullhist_to_p
from nimare.tests.utils import get_test_data_path
from nimare.transforms import p_to_z
from nimare.utils import vox2mm

SIMULATED_ALE_REGRESSION_DATASETS = [
    pytest.param(
        {
            "foci": 3,
            "foci_percentage": "60%",
            "fwhm": 10.0,
            "sample_size": 20,
            "n_studies": 20,
            "n_noise_foci": 10,
            "seed": 101,
            "space": "MNI",
        },
        id="small",
    ),
    pytest.param(
        {
            "foci": 5,
            "foci_percentage": "60%",
            "fwhm": 10.0,
            "sample_size": 30,
            "n_studies": 40,
            "n_noise_foci": 20,
            "seed": 102,
            "space": "MNI",
        },
        id="medium",
    ),
]


def _dense_ale_reference(ma_values):
    """Reference ALE approximate-null implementation using dense masked arrays."""
    stat_values = 1.0 - np.prod(1.0 - ma_values, axis=0)

    inv_step_size = 100000
    step_size = 1 / inv_step_size
    max_ma_values = np.max(ma_values, axis=1)
    max_ma_values = np.ceil(max_ma_values * inv_step_size) / inv_step_size
    max_poss_ale = 1.0 - np.prod(1.0 - max_ma_values, axis=0)
    hist_bins = np.round(np.arange(0, max_poss_ale + (1.5 * step_size), step_size), 5)

    bin_centers = hist_bins
    bin_edges = np.append(bin_centers, bin_centers[-1] + step_size)
    n_mask_voxels = ma_values.shape[1]

    ale_hist = None
    for study_ma_values in ma_values:
        n_nonzero_voxels = np.count_nonzero(study_ma_values)
        n_zero_voxels = n_mask_voxels - n_nonzero_voxels

        exp_hist = np.histogram(
            study_ma_values[study_ma_values > 0], bins=bin_edges, density=False
        )[0].astype(float)
        exp_hist[0] += n_zero_voxels
        exp_hist /= exp_hist.sum()

        if ale_hist is None:
            ale_hist = exp_hist.copy()
            continue

        ale_idx = np.where(ale_hist > 0)[0]
        exp_idx = np.where(exp_hist > 0)[0]
        ale_scores = 1 - np.outer((1 - bin_centers[exp_idx]), (1 - bin_centers[ale_idx])).ravel()
        score_idx = np.floor(ale_scores * inv_step_size).astype(int)
        probabilities = np.outer(exp_hist[exp_idx], ale_hist[ale_idx]).ravel()
        ale_hist = np.zeros(ale_hist.shape)
        np.add.at(ale_hist, score_idx, probabilities)

    p_values = nullhist_to_p(stat_values, ale_hist, hist_bins)
    z_values = p_to_z(p_values, tail="one")
    return {
        "stat": stat_values,
        "p": p_values,
        "z": z_values,
        "hist_bins": hist_bins,
        "hist": ale_hist,
    }


def _prepare_ale_inputs(dataset, kernel_transformer=None):
    """Prepare ALE estimator inputs without running a full fit."""
    meta = ale.ALE(
        kernel_transformer=kernel_transformer or ale.ALEKernel(),
        null_method="approximate",
        generate_description=False,
    )
    meta.masker = dataset.masker
    meta._collect_inputs(dataset)
    meta._preprocess_input(dataset)
    return meta


def _assert_custom_kernel_subclass_uses_masked_csr(meta, fit_args, min_calls=1):
    """Assert that an ALEKernel subclass uses sparse kernel generation."""
    called = {"count": 0}
    original = meta.kernel_transformer.transform

    def _tracked_transform(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    meta.kernel_transformer.transform = _tracked_transform
    result = meta.fit(*fit_args)
    assert called["count"] >= min_calls
    return result


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


def test_ALE_precomputed_ma_maps_do_not_leak_between_fit_calls(testdata_cbma):
    """Precomputed MA maps from one fit should not affect a later fit."""
    testdata_cbma = testdata_cbma.copy()
    ids = sorted(testdata_cbma.ids)
    first_dset = testdata_cbma.slice(ids[:8])
    second_dset = testdata_cbma.slice(ids[8:16])

    meta = ale.ALE(null_method="approximate", generate_description=False)
    precomputed = meta.kernel_transformer.transform(first_dset, return_type="sparse")

    meta.fit(first_dset, ma_maps=precomputed)
    second_result = meta.fit(second_dset)
    expected = ale.ALE(null_method="approximate", generate_description=False).fit(second_dset)

    assert "ma_maps" not in meta.inputs_
    assert np.array_equal(
        second_result.get_map("stat", return_type="array"),
        expected.get_map("stat", return_type="array"),
    )


@pytest.mark.parametrize("dataset_kwargs", SIMULATED_ALE_REGRESSION_DATASETS)
def test_ALE_fit_matches_dense_reference(dataset_kwargs):
    """Approximate-null ALE fit should match the dense masked-array reference path."""
    _, dataset = create_coordinate_dataset(**dataset_kwargs)
    meta = ale.ALE(null_method="approximate")
    result = meta.fit(dataset)

    ma_values = meta.kernel_transformer.transform(dataset, return_type="array")
    expected = _dense_ale_reference(ma_values)

    np.testing.assert_allclose(
        result.get_map("stat", return_type="array"),
        expected["stat"],
        rtol=1e-5,
        atol=5e-7,
    )
    p_values = result.get_map("p", return_type="array")
    p_diff = np.abs(p_values - expected["p"])
    assert np.corrcoef(p_values, expected["p"])[0, 1] > 0.999
    assert p_diff.mean() < 1e-4
    assert np.quantile(p_diff, 0.99) < 5e-4

    z_values = result.get_map("z", return_type="array")
    z_diff = np.abs(z_values - expected["z"])
    assert np.corrcoef(z_values, expected["z"])[0, 1] > 0.999
    assert z_diff.mean() < 1e-4
    assert np.quantile(z_diff, 0.99) < 5e-4


def test_ALE_masked_csr_conversion_matches_masked_array():
    """CSR conversion should preserve masked study-wise MA values."""
    _, dataset = create_coordinate_dataset(
        foci=3,
        foci_percentage="60%",
        fwhm=10.0,
        sample_size=20,
        n_studies=20,
        n_noise_foci=10,
        seed=303,
        space="MNI",
    )
    meta = ale.ALE(null_method="approximate")
    meta.masker = dataset.masker

    csr = meta.kernel_transformer.transform(dataset, return_type="sparse")
    dense = meta.kernel_transformer.transform(dataset, return_type="array")

    assert csr.shape == dense.shape
    np.testing.assert_allclose(csr.toarray(), dense)


def test_ALE_csr_summarystat_matches_dense_reference():
    """CSR summary-stat computation should match the dense ALE implementation."""
    _, dataset = create_coordinate_dataset(
        foci=3,
        foci_percentage="60%",
        fwhm=10.0,
        sample_size=20,
        n_studies=20,
        n_noise_foci=10,
        seed=404,
        space="MNI",
    )
    meta = ale.ALE(null_method="approximate")
    meta.masker = dataset.masker

    csr = meta.kernel_transformer.transform(dataset, return_type="sparse")
    dense = meta.kernel_transformer.transform(dataset, return_type="array")

    actual = meta._compute_summarystat_est(csr)
    expected = 1.0 - np.prod(1.0 - dense, axis=0)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-7)


def test_ALE_csr_approximate_null_matches_dense_reference():
    """CSR approximate-null histogram construction should match the dense reference path."""
    _, dataset = create_coordinate_dataset(
        foci=3,
        foci_percentage="60%",
        fwhm=10.0,
        sample_size=20,
        n_studies=20,
        n_noise_foci=10,
        seed=505,
        space="MNI",
    )
    meta = ale.ALE(null_method="approximate")
    meta.masker = dataset.masker
    meta.null_distributions_ = {}

    csr = meta.kernel_transformer.transform(dataset, return_type="sparse")
    _ = meta._compute_summarystat_est(csr)
    meta._determine_histogram_bins(csr)
    meta._compute_null_approximate(csr)

    dense = meta.kernel_transformer.transform(dataset, return_type="array")
    expected = _dense_ale_reference(dense)

    np.testing.assert_allclose(
        meta.null_distributions_["histogram_bins"],
        expected["hist_bins"],
        rtol=1e-5,
        atol=5e-7,
    )
    np.testing.assert_allclose(
        meta.null_distributions_["histweights_corr-none_method-approximate"],
        expected["hist"],
        rtol=1e-5,
        atol=3e-4,
    )


@pytest.mark.parametrize(
    ("kernel_transformer", "sample_sizes"),
    [
        pytest.param(ale.ALEKernel(fwhm=10.0), None, id="fixed-fwhm"),
        pytest.param(ale.ALEKernel(), [20] * 6, id="repeated-sample-sizes"),
        pytest.param(ale.ALEKernel(), [18, 22, 26, 30, 34, 38], id="mixed-sample-sizes"),
    ],
)
def test_ALE_masked_csr_kernel_matches_masked_array(kernel_transformer, sample_sizes):
    """Direct masked-CSR ALE kernel output should match the dense masked-array reference."""
    _, dataset = create_coordinate_dataset(
        foci=4,
        foci_percentage="60%",
        fwhm=10.0,
        sample_size=20,
        n_studies=6,
        n_noise_foci=10,
        seed=606,
        space="MNI",
    )
    if sample_sizes is not None:
        sample_size_by_id = dict(zip(dataset.ids, sample_sizes))
        dataset.metadata["sample_sizes"] = dataset.metadata["id"].map(sample_size_by_id)

    meta = _prepare_ale_inputs(dataset, kernel_transformer=kernel_transformer)
    csr = meta.kernel_transformer.transform(dataset, return_type="sparse")
    dense = meta.kernel_transformer.transform(dataset, return_type="array")

    assert sp_sparse.isspmatrix_csr(csr)
    assert csr.shape == dense.shape
    np.testing.assert_allclose(csr.toarray(), dense, rtol=1e-5, atol=5e-7)
    np.testing.assert_allclose(
        np.asarray(csr.max(axis=1).todense()).ravel(), dense.max(axis=1), rtol=1e-5, atol=5e-7
    )


def test_ALE_precomputed_ma_maps_match_generated_fast_path():
    """Precomputed ALE MA maps should match the generated masked-CSR fast path."""
    _, dataset = create_coordinate_dataset(
        foci=3,
        foci_percentage="60%",
        fwhm=10.0,
        sample_size=20,
        n_studies=20,
        n_noise_foci=10,
        seed=707,
        space="MNI",
    )
    expected = ale.ALE(null_method="approximate", generate_description=False).fit(dataset)

    precomputed = ale.ALE(null_method="approximate").kernel_transformer.transform(
        dataset,
        return_type="sparse",
    )

    result = ale.ALE(null_method="approximate", generate_description=False).fit(
        dataset,
        ma_maps=precomputed,
    )

    np.testing.assert_allclose(
        result.get_map("stat", return_type="array"),
        expected.get_map("stat", return_type="array"),
        rtol=1e-5,
        atol=5e-7,
    )
    np.testing.assert_allclose(
        result.get_map("p", return_type="array"),
        expected.get_map("p", return_type="array"),
        rtol=1e-5,
        atol=5e-7,
    )


def test_ALE_custom_kernel_subclass_uses_masked_csr_fast_path():
    """ALEKernel subclasses should use the masked-CSR fast path in ALE."""

    class CustomALEKernel(ale.ALEKernel):
        pass

    _, dataset = create_coordinate_dataset(
        foci=3,
        foci_percentage="60%",
        fwhm=10.0,
        sample_size=20,
        n_studies=8,
        n_noise_foci=10,
        seed=809,
        space="MNI",
    )
    meta = ale.ALE(
        kernel_transformer=CustomALEKernel(fwhm=10.0),
        null_method="approximate",
        generate_description=False,
    )
    result = _assert_custom_kernel_subclass_uses_masked_csr(meta, (dataset,))
    expected = _dense_ale_reference(
        meta.kernel_transformer.transform(dataset, return_type="array")
    )

    np.testing.assert_allclose(
        result.get_map("stat", return_type="array"),
        expected["stat"],
        rtol=1e-5,
        atol=5e-7,
    )


@pytest.mark.parametrize(
    ("meta_factory", "dataset_kwargs", "fit_args_factory", "min_calls"),
    [
        pytest.param(
            lambda CustomALEKernel, dataset: ale.ALESubtraction(
                kernel_transformer=CustomALEKernel(fwhm=10.0),
                n_iters=2,
                n_cores=1,
            ),
            {
                "foci": 3,
                "foci_percentage": "60%",
                "fwhm": 10.0,
                "sample_size": 20,
                "n_studies": 8,
                "n_noise_foci": 10,
                "seed": 809,
                "space": "MNI",
            },
            lambda dataset: (dataset.slice(dataset.ids[:4]), dataset.slice(dataset.ids[4:])),
            2,
            id="ALESubtraction",
        ),
        pytest.param(
            lambda CustomALEKernel, dataset: ale.SCALE(
                xyz=dataset.coordinates[["x", "y", "z"]].values[:20],
                n_iters=2,
                n_cores=1,
                kernel_transformer=CustomALEKernel(fwhm=10.0),
            ),
            {
                "foci": 3,
                "foci_percentage": "60%",
                "fwhm": 10.0,
                "sample_size": 20,
                "n_studies": 8,
                "n_noise_foci": 10,
                "seed": 810,
                "space": "MNI",
            },
            lambda dataset: (dataset,),
            1,
            id="SCALE",
        ),
    ],
)
def test_ALE_family_custom_kernel_subclass_uses_masked_csr_fast_path(
    meta_factory,
    dataset_kwargs,
    fit_args_factory,
    min_calls,
):
    """ALE-family estimators should use the masked-CSR fast path for ALEKernel subclasses."""

    class CustomALEKernel(ale.ALEKernel):
        pass

    _, dataset = create_coordinate_dataset(**dataset_kwargs)
    meta = meta_factory(CustomALEKernel, dataset)
    _assert_custom_kernel_subclass_uses_masked_csr(
        meta,
        fit_args_factory(dataset),
        min_calls=min_calls,
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
    """Masked sparse ALE summaries should be self-consistent."""
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
    summary = sub_meta._compute_summarystat_est(ma_maps)
    np.testing.assert_allclose(
        summary, sub_meta._compute_summarystat_est(ma_maps), rtol=1e-5, atol=2e-7
    )


def test_ALESubtraction_requires_sparse_matrix_inputs():
    """ALESubtraction should require scipy sparse precomputed inputs."""
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

    with pytest.raises(ValueError):
        ale._require_masked_csr(np.array([[0.4, 0.9]], dtype=np.float32))


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
