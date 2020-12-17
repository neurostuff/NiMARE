"""
Test nimare.meta.mkda (KDA-based meta-analytic algorithms).
"""
import numpy as np

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.meta import MKDAKernel, MKDADensity, MKDAChi2, KDA


def test_mkda_density_kernel_instance_with_kwargs(testdata_cbma):
    """
    Smoke test for MKDADensity with a kernel transformer object, with kernel
    arguments provided, which should result in a warning, but the original
    object's parameters should remain untouched.
    """
    kern = MKDAKernel(r=2)
    meta = MKDADensity(kern, kernel__r=6, null_method="empirical", n_iters=10)

    assert meta.kernel_transformer.get_params().get("r") == 2


def test_mkda_density_kernel_class(testdata_cbma):
    """
    Smoke test for MKDADensity with a kernel transformer class.
    """
    meta = MKDADensity(MKDAKernel, kernel__r=5, null_method="empirical", n_iters=10)
    res = meta.fit(testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)


def test_mkda_density_kernel_instance(testdata_cbma):
    """
    Smoke test for MKDADensity with a kernel transformer object.
    """
    kern = MKDAKernel(r=5)
    meta = MKDADensity(kern, null_method="empirical", n_iters=10)
    res = meta.fit(testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)


def test_mkda_density_analytic_null(testdata_cbma_full):
    """
    Smoke test for MKDADensity
    """
    meta = MKDADensity(null="analytic")
    res = meta.fit(testdata_cbma_full)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=1, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_mkda_density(testdata_cbma):
    """
    Smoke test for MKDADensity
    """
    meta = MKDADensity(null_method="empirical", n_iters=10)
    res = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_mkda_chi2_fdr(testdata_cbma):
    """
    Smoke test for MKDAChi2
    """
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    corr = FDRCorrector(method="bh", alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_mkda_chi2_fwe_1core(testdata_cbma):
    """
    Smoke test for MKDAChi2
    """
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_mkda_chi2_fwe_2core(testdata_cbma):
    """
    Smoke test for MKDAChi2
    """
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)
    corr_2core = FWECorrector(method="montecarlo", n_iters=5, n_cores=2)
    cres_2core = corr_2core.transform(res)
    assert isinstance(cres_2core, nimare.results.MetaResult)


def test_kda_density_analytic_null(testdata_cbma):
    """
    Smoke test for KDA with analytical null and FWE correction.
    """
    meta = KDA(null_method="analytic")
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


def test_kda_density_fwe_1core(testdata_cbma):
    """
    Smoke test for KDA with empirical null and FWE correction.
    """
    meta = KDA(null_method="empirical", n_iters=10)
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


def test_mkda_analytic_empirical_convergence(testdata_cbma_full):
    est_a = MKDADensity(null_method="analytic")
    n_iters = 10
    est_e = MKDADensity(null_method="empirical", n_iters=n_iters)
    res_a = est_a.fit(testdata_cbma_full)
    res_e = est_e.fit(testdata_cbma_full)
    # Get smallest p-value above 0 from the empirical estimator; above this,
    # the two should converge reasonably closely.
    min_p = 1 / n_iters
    p_idx = res_e.maps["p"] > min_p
    p_analytical = res_a.maps["p"][p_idx]
    p_empirical = res_e.maps["p"][p_idx]
    # Correlation must be near unity and mean difference should be tiny
    assert np.corrcoef(p_analytical, p_empirical)[0, 1] > 0.98
    assert (p_analytical - p_empirical).mean() < 1e-3


def test_kda_analytic_empirical_convergence(testdata_cbma_full):
    est_a = KDA(null_method="analytic")
    n_iters = 10
    est_e = KDA(null_method="empirical", n_iters=n_iters)
    res_a = est_a.fit(testdata_cbma_full)
    res_e = est_e.fit(testdata_cbma_full)
    # Get smallest p-value above 0 from the empirical estimator; above this,
    # the two should converge reasonably closely.
    min_p = 1 / n_iters
    p_idx = res_e.maps["p"] > min_p
    p_analytical = res_a.maps["p"][p_idx]
    p_empirical = res_e.maps["p"][p_idx]
    # Correlation must be near unity and mean difference should be tiny
    assert np.corrcoef(p_analytical, p_empirical)[0, 1] > 0.98
    assert (p_analytical - p_empirical).mean() < 1e-3
