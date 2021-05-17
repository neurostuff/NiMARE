"""Test nimare.meta.mkda (KDA-based meta-analytic algorithms)."""
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


def test_MKDADensity_approximate_null(testdata_cbma_full):
    """Smoke test for MKDADensity."""
    meta = MKDADensity(null="approximate")
    res = meta.fit(testdata_cbma_full)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=1, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_MKDADensity(testdata_cbma):
    """Smoke test for MKDADensity."""
    meta = MKDADensity(null_method="montecarlo", n_iters=10)
    res = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_MKDADensity_low_memory(testdata_cbma):
    """Smoke test for MKDADensity with low_memory option."""
    meta = MKDADensity(null_method="montecarlo", n_iters=10, low_memory=True)
    res = meta.fit(testdata_cbma)
    assert meta.low_memory
    assert not meta.kernel_transformer.low_memory
    assert isinstance(res, nimare.results.MetaResult)


def test_MKDAChi2_fdr(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    corr = FDRCorrector(method="bh", alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_MKDAChi2_fwe_1core(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(cres, nimare.results.MetaResult)


def test_MKDAChi2_fwe_2core(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    res = meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)
    corr_2core = FWECorrector(method="montecarlo", n_iters=5, n_cores=2)
    cres_2core = corr_2core.transform(res)
    assert isinstance(cres_2core, nimare.results.MetaResult)


def test_MKDAChi2_low_memory(testdata_cbma):
    """Smoke test for MKDAChi2 with low_memory option."""
    meta = MKDAChi2(low_memory=True)
    res = meta.fit(testdata_cbma, testdata_cbma)
    assert meta.low_memory
    assert not meta.kernel_transformer.low_memory
    assert isinstance(res, nimare.results.MetaResult)


def test_MKDAChi2_low_memory_reuse(testdata_cbma, tmp_path_factory):
    """Smoke test for MKDAChi2 with low_memory option, in which a memory-mapped array is used."""
    tmpdir = tmp_path_factory.mktemp("test_MKDAChi2_low_memory_reuse")

    # Generate MKDAKernel MA maps as files in the Dataset
    testdata_cbma.update_path(tmpdir)
    kern = MKDAKernel()
    dset = kern.transform(testdata_cbma, return_type="dataset")

    # Reuse the MA files, loading them as a memory-mapped array
    meta = MKDAChi2(low_memory=True)
    res = meta.fit(dset, dset)
    assert meta.low_memory
    assert not meta.kernel_transformer.low_memory
    assert isinstance(res, nimare.results.MetaResult)


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
