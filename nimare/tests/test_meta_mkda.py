"""Test nimare.meta.mkda (KDA-based meta-analytic algorithms)."""
import logging

import numpy as np

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import KDA, MKDAChi2, MKDADensity, MKDAKernel


def test_MKDADensity_kernel_instance_with_kwargs(testdata_cbma):
    """Smoke test for MKDADensity with a kernel transformer object.

    With kernel arguments provided, which should result in a warning, but the original
    object's parameters should remain untouched.
    """
    kern = MKDAKernel(r=2)
    meta = MKDADensity(kern, kernel__r=6, null_method="montecarlo", n_iters=10)

    assert meta.kernel_transformer.get_params().get("r") == 2


def test_MKDADensity_kernel_class(testdata_cbma):
    """Smoke test for MKDADensity with a kernel transformer class."""
    meta = MKDADensity(MKDAKernel, kernel__r=5, null_method="montecarlo", n_iters=10)
    res = meta.fit(testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)


def test_MKDADensity_kernel_instance(testdata_cbma):
    """Smoke test for MKDADensity with a kernel transformer object."""
    kern = MKDAKernel(r=5)
    meta = MKDADensity(kern, null_method="montecarlo", n_iters=10)
    res = meta.fit(testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)


def test_MKDADensity_approximate_null(testdata_cbma_full, caplog):
    """Smoke test for MKDADensity with the "approximate" null_method."""
    meta = MKDADensity(null="approximate")
    res = meta.fit(testdata_cbma_full)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)

    # Check that the vfwe_only option does not work
    corr2 = FWECorrector(
        method="montecarlo",
        voxel_thresh=0.001,
        n_iters=5,
        n_cores=1,
        vfwe_only=True,
    )
    with caplog.at_level(logging.WARNING):
        cres2 = corr2.transform(res)

    assert "Running permutations from scratch." in caplog.text

    assert isinstance(cres2, nimare.results.MetaResult)
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in cres2.maps
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" not in cres2.maps


def test_MKDADensity_montecarlo_null(testdata_cbma):
    """Smoke test for MKDADensity with the "montecarlo" null_method."""
    meta = MKDADensity(null_method="montecarlo", n_iters=10)
    res = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)

    # Check that the vfwe_only option works
    corr2 = FWECorrector(
        method="montecarlo",
        voxel_thresh=0.001,
        n_iters=5,
        n_cores=1,
        vfwe_only=True,
    )
    cres2 = corr2.transform(res)
    assert isinstance(cres2, nimare.results.MetaResult)
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in cres2.maps
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" not in cres2.maps


def test_MKDAChi2_fdr(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    corr = FDRCorrector(method="indep", alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)

    methods = FDRCorrector.inspect(res)
    assert methods == ["indep", "negcorr"]


def test_MKDAChi2_fwe_1core(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    valid_methods = FWECorrector.inspect(res)
    assert "montecarlo" in valid_methods

    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)
    assert (
        "values_desc-pFgA_level-voxel_corr-fwe_method-montecarlo"
        in cres.estimator.null_distributions_.keys()
    )
    assert (
        "values_desc-pAgF_level-voxel_corr-fwe_method-montecarlo"
        in cres.estimator.null_distributions_.keys()
    )


def test_MKDAChi2_fwe_2core(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)
    corr_2core = FWECorrector(method="montecarlo", n_iters=5, n_cores=2)
    cres_2core = corr_2core.transform(res)
    assert isinstance(cres_2core, nimare.results.MetaResult)


def test_KDA_approximate_null(testdata_cbma):
    """Smoke test for KDA with approximate null and FWE correction."""
    meta = KDA(null_method="approximate")
    res = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert res.get_map("p", return_type="array").dtype == np.float64
    assert isinstance(cres, nimare.results.MetaResult)
    assert (
        cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo", return_type="array").dtype
        == np.float64
    )
    assert (
        cres.get_map(
            "logp_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float64
    )
    assert (
        cres.get_map(
            "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float64
    )


def test_KDA_fwe_1core(testdata_cbma):
    """Smoke test for KDA with montecarlo null and FWE correction."""
    meta = KDA(null_method="montecarlo", n_iters=10)
    res = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert res.get_map("p", return_type="array").dtype == np.float64
    assert isinstance(cres, nimare.results.MetaResult)
    assert (
        cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo", return_type="array").dtype
        == np.float64
    )
    assert (
        cres.get_map(
            "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float64
    )
    assert (
        cres.get_map(
            "logp_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float64
    )


def test_MKDADensity_approximate_montecarlo_convergence(testdata_cbma_full):
    """Evaluate convergence between approximate and montecarlo null methods in MKDA."""
    est_a = MKDADensity(null_method="approximate")
    n_iters = 10
    est_e = MKDADensity(null_method="montecarlo", n_iters=n_iters)
    res_a = est_a.fit(testdata_cbma_full)
    res_e = est_e.fit(testdata_cbma_full)
    # Get smallest p-value above 0 from the montecarlo estimator; above this,
    # the two should converge reasonably closely.
    min_p = 1 / n_iters
    p_idx = res_e.maps["p"] > min_p
    p_approximate = res_a.maps["p"][p_idx]
    p_montecarlo = res_e.maps["p"][p_idx]
    # Correlation must be near unity and mean difference should be tiny
    assert np.corrcoef(p_approximate, p_montecarlo)[0, 1] > 0.98
    assert (p_approximate - p_montecarlo).mean() < 1e-3


def test_KDA_approximate_montecarlo_convergence(testdata_cbma_full):
    """Evaluate convergence between approximate and montecarlo null methods in KDA."""
    est_a = KDA(null_method="approximate")
    n_iters = 10
    est_e = KDA(null_method="montecarlo", n_iters=n_iters)
    res_a = est_a.fit(testdata_cbma_full)
    res_e = est_e.fit(testdata_cbma_full)
    # Get smallest p-value above 0 from the montecarlo estimator; above this,
    # the two should converge reasonably closely.
    min_p = 1 / n_iters
    p_idx = res_e.maps["p"] > min_p
    p_approximate = res_a.maps["p"][p_idx]
    p_montecarlo = res_e.maps["p"][p_idx]
    # Correlation must be near unity and mean difference should be tiny
    assert np.corrcoef(p_approximate, p_montecarlo)[0, 1] > 0.98
    assert (p_approximate - p_montecarlo).mean() < 1e-3
