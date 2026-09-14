"""Test nimare.meta.mkda (KDA-based meta-analytic algorithms)."""

import logging

import numpy as np
import pytest
from scipy import sparse as sp_sparse
from scipy import special

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
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)


def test_MKDADensity_kernel_instance(testdata_cbma):
    """Smoke test for MKDADensity with a kernel transformer object."""
    kern = MKDAKernel(r=5)
    meta = MKDADensity(kern, null_method="montecarlo", n_iters=10)
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)


def test_MKDADensity_approximate_null(testdata_cbma_full, caplog):
    """Smoke test for MKDADensity with the "approximate" null_method."""
    meta = MKDADensity(null="approximate")
    results = meta.fit(testdata_cbma_full)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=1)
    corr_results = corr.transform(results)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)

    # Check that the vfwe_only option does not work
    corr2 = FWECorrector(
        method="montecarlo",
        voxel_thresh=0.001,
        n_iters=5,
        n_cores=1,
        vfwe_only=True,
    )
    with caplog.at_level(logging.WARNING):
        corr_results2 = corr2.transform(results)

    assert "Running permutations from scratch." in caplog.text

    assert isinstance(corr_results2, nimare.results.MetaResult)
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in corr_results2.maps
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" not in corr_results2.maps


def test_MKDADensity_montecarlo_null(testdata_cbma):
    """Smoke test for MKDADensity with the "montecarlo" null_method."""
    meta = MKDADensity(null_method="montecarlo", n_iters=10)
    results = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5, n_cores=1)
    corr_results = corr.transform(results)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)

    # Check that the vfwe_only option works
    corr2 = FWECorrector(
        method="montecarlo",
        voxel_thresh=0.001,
        n_iters=5,
        n_cores=1,
        vfwe_only=True,
    )
    corr_results2 = corr2.transform(results)
    assert isinstance(corr_results2, nimare.results.MetaResult)
    assert "logp_level-voxel_corr-FWE_method-montecarlo" in corr_results2.maps
    assert "logp_desc-size_level-cluster_corr-FWE_method-montecarlo" not in corr_results2.maps


def test_MKDAChi2_fdr(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    results = meta.fit(testdata_cbma, testdata_cbma)
    assert "z_desc-group1" in results.maps
    assert "z_desc-group2" in results.maps
    assert "p_desc-group1" in results.maps
    assert "p_desc-group2" in results.maps
    corr = FDRCorrector(method="indep", alpha=0.001)
    corr_results = corr.transform(results)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)
    assert "z_desc-group1_level-voxel_corr-FDR_method-indep" in corr_results.maps
    assert "z_desc-group2_level-voxel_corr-FDR_method-indep" in corr_results.maps

    methods = FDRCorrector.inspect(results)
    assert methods == ["indep", "negcorr"]


def test_MKDAChi2_directional_inference_maps_gate_association_results(testdata_cbma_full):
    """Directional inference maps should zero unsupported MKDAChi2 association voxels."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:20])

    baseline = MKDAChi2(generate_description=False).fit(dset1, dset2)
    baseline_z = baseline.get_map("z_desc-association", return_type="array")
    pos_map = (baseline_z > 0).astype(np.int8)
    neg_map = (baseline_z < 0).astype(np.int8)

    masked = MKDAChi2(generate_description=False).fit(
        dset1,
        dset2,
        inference_map1=pos_map,
        inference_map2=neg_map,
    )

    z_values = masked.get_map("z_desc-association", return_type="array")
    p_values = masked.get_map("p_desc-association", return_type="array")
    union = (pos_map > 0) | (neg_map > 0)

    assert np.all(z_values[~union] == 0)
    np.testing.assert_allclose(p_values[~union], 1.0)
    assert np.all(z_values[pos_map <= 0] <= 0)
    assert np.all(z_values[neg_map <= 0] >= 0)


def test_MKDAChi2_fwe_1core(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    results = meta.fit(testdata_cbma, testdata_cbma)
    valid_methods = FWECorrector.inspect(results)
    assert "montecarlo" in valid_methods

    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    corr_results = corr.transform(results)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)
    assert (
        "values_desc-pFgA_level-voxel_corr-fwe_method-montecarlo"
        in corr_results.estimator.null_distributions_.keys()
    )
    assert (
        "values_desc-pAgF_level-voxel_corr-fwe_method-montecarlo"
        in corr_results.estimator.null_distributions_.keys()
    )
    assert (
        "values_desc-group2_level-voxel_corr-fwe_method-montecarlo"
        in corr_results.estimator.null_distributions_.keys()
    )
    assert "z_desc-group1_level-voxel_corr-FWE_method-montecarlo" in corr_results.maps
    assert "z_desc-group2_level-voxel_corr-FWE_method-montecarlo" in corr_results.maps


def test_MKDAChi2_fwe_null_method_default_and_label_permutation(testdata_cbma):
    """MKDAChi2 should default to label-permutation FWE and run the label-permutation path."""
    meta = MKDAChi2()
    assert meta.fwe_null_method == "label-permutation"

    results = meta.fit(testdata_cbma, testdata_cbma)
    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    corr_results = corr.transform(results)

    assert (
        "values_desc-pAgF_level-voxel_corr-fwe_method-montecarlo"
        in corr_results.estimator.null_distributions_.keys()
    )
    assert (
        "values_desc-pFgA_level-voxel_corr-fwe_method-montecarlo"
        in corr_results.estimator.null_distributions_.keys()
    )


def test_MKDAChi2_invalid_fwe_null_method():
    """MKDAChi2 should reject unsupported FWE null methods."""
    with pytest.raises(
        ValueError, match="fwe_null_method must be 'label-permutation' or 'random-foci'"
    ):
        MKDAChi2(fwe_null_method="invalid")


def test_MKDAChi2_precomputed_ma_maps_do_not_leak_between_fit_calls(testdata_cbma):
    """Precomputed pairwise MA maps should not affect a later fit."""
    testdata_cbma = testdata_cbma.copy()
    ids = sorted(testdata_cbma.ids)
    first_dset1 = testdata_cbma.slice(ids[:4])
    first_dset2 = testdata_cbma.slice(ids[4:8])
    second_dset1 = testdata_cbma.slice(ids[8:12])
    second_dset2 = testdata_cbma.slice(ids[12:16])

    meta = MKDAChi2(generate_description=False)
    precomputed1 = meta.kernel_transformer.transform(first_dset1, return_type="sparse")
    precomputed2 = meta.kernel_transformer.transform(first_dset2, return_type="sparse")

    meta.fit(first_dset1, first_dset2, ma_maps1=precomputed1, ma_maps2=precomputed2)
    second_result = meta.fit(second_dset1, second_dset2)
    expected = MKDAChi2(generate_description=False).fit(second_dset1, second_dset2)

    assert "ma_maps1" not in meta.inputs_
    assert "ma_maps2" not in meta.inputs_
    assert np.array_equal(
        second_result.get_map("chi2_desc-association", return_type="array"),
        expected.get_map("chi2_desc-association", return_type="array"),
    )


def test_MKDAChi2_fwe_2core(testdata_cbma):
    """Smoke test for MKDAChi2."""
    meta = MKDAChi2()
    results = meta.fit(testdata_cbma, testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)
    corr_2core = FWECorrector(method="montecarlo", n_iters=5, n_cores=2)
    cres_2core = corr_2core.transform(results)
    assert isinstance(cres_2core, nimare.results.MetaResult)


def test_KDA_approximate_null(testdata_cbma):
    """Smoke test for KDA with approximate null and FWE correction."""
    meta = KDA(null_method="approximate")
    results = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    corr_results = corr.transform(results)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert results.get_map("p", return_type="array").dtype == np.float32
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert isinstance(corr_results.description_, str)
    assert (
        corr_results.get_map(
            "logp_level-voxel_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float32
    )
    assert (
        corr_results.get_map(
            "logp_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float32
    )
    assert (
        corr_results.get_map(
            "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float32
    )


def test_KDA_fwe_1core(testdata_cbma):
    """Smoke test for KDA with montecarlo null and FWE correction."""
    meta = KDA(null_method="montecarlo", n_iters=10)
    results = meta.fit(testdata_cbma)
    corr = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    corr_results = corr.transform(results)
    assert isinstance(results, nimare.results.MetaResult)
    assert results.get_map("p", return_type="array").dtype == np.float32
    assert isinstance(corr_results, nimare.results.MetaResult)
    assert (
        corr_results.get_map(
            "logp_level-voxel_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float32
    )
    assert (
        corr_results.get_map(
            "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float32
    )
    assert (
        corr_results.get_map(
            "logp_desc-size_level-cluster_corr-FWE_method-montecarlo", return_type="array"
        ).dtype
        == np.float32
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


def test_MKDADensity_masked_csr_kernel_matches_masked_array(testdata_cbma):
    """Direct masked-CSR MKDA kernel output should match the dense masked array."""
    meta = MKDADensity(generate_description=False)
    csr = meta.kernel_transformer.transform(testdata_cbma, return_type="sparse")
    dense = meta.kernel_transformer.transform(testdata_cbma, return_type="array")

    assert sp_sparse.isspmatrix_csr(csr)
    np.testing.assert_allclose(csr.toarray(), dense)


def test_MKDADensity_csr_summarystat_matches_dense(testdata_cbma):
    """Masked-CSR MKDA summary stats should match the dense path."""
    meta = MKDADensity(generate_description=False)
    meta.masker = testdata_cbma.masker
    meta._collect_inputs(testdata_cbma)
    meta._preprocess_input(testdata_cbma)

    ma_maps = meta.kernel_transformer.transform(testdata_cbma, return_type="sparse")
    dense = meta.kernel_transformer.transform(testdata_cbma, return_type="array")
    meta.weight_vec_ = meta._compute_weights(ma_maps)

    csr_summary = meta._compute_summarystat_est(ma_maps)
    dense_summary = meta._compute_summarystat_est(dense)
    np.testing.assert_allclose(csr_summary, dense_summary)


def test_MKDADensity_precomputed_masked_csr_matches_generated_fast_path(testdata_cbma):
    """MKDADensity should accept precomputed masked-CSR MA maps."""
    baseline = MKDADensity(null_method="approximate", generate_description=False).fit(
        testdata_cbma
    )

    meta = MKDADensity(null_method="approximate", generate_description=False)
    meta.masker = testdata_cbma.masker
    meta._collect_inputs(testdata_cbma)
    meta._preprocess_input(testdata_cbma)
    precomputed = meta.kernel_transformer.transform(testdata_cbma, return_type="sparse")

    result = MKDADensity(null_method="approximate", generate_description=False).fit(
        testdata_cbma,
        ma_maps=precomputed,
    )

    np.testing.assert_allclose(
        result.get_map("stat", return_type="array"),
        baseline.get_map("stat", return_type="array"),
    )
    np.testing.assert_allclose(
        result.get_map("p", return_type="array"),
        baseline.get_map("p", return_type="array"),
    )


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


def test_MKDAChi2_logp_maps_follow_the_chi_squared_tail(testdata_cbma_full):
    """The reported -log10(p) must track the chi-squared statistic, not a floor.

    The old code floored the p-value at machine epsilon, capping every -log10(p) at 15.65
    from a chi-squared of 71 on, and then stored it through a float32 p-value, capping it
    again at 44.85.
    """
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])

    results = MKDAChi2(generate_description=False).fit(dset1, dset2)

    for name in ("uniformity", "association", "group2"):
        chi2_values = results.get_map(f"chi2_desc-{name}", return_type="array")
        logp_values = results.get_map(f"logp_desc-{name}", return_type="array")
        # The exact one-dof upper tail, computed independently of the implementation.
        expected = -(np.log(2.0) + special.log_ndtr(-np.sqrt(chi2_values))) / np.log(10.0)
        assert np.allclose(logp_values, expected, rtol=1e-5), name

    deepest = results.get_map("logp_desc-group2", return_type="array").max()
    assert deepest > 44.85, "used to be capped, first at 15.65 and then at 44.85"
    # The p map cannot follow it there, which is why the logp map exists.
    assert results.get_map("p_desc-group2", return_type="array").min() == np.float32(1e-45)


def test_MKDAChi2_z_maps_are_unchanged_by_the_log_space_tail(testdata_cbma_full):
    """z is sqrt(chi2) with a sign and never went through a p-value, so it must not move."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])

    results = MKDAChi2(generate_description=False).fit(dset1, dset2)

    for name in ("uniformity", "association", "group2"):
        z_values = results.get_map(f"z_desc-{name}", return_type="array")
        chi2_values = results.get_map(f"chi2_desc-{name}", return_type="array")
        # Voxels with no directional evidence get a sign of zero, so compare where there is one.
        signed = z_values != 0
        assert signed.any(), name
        assert np.allclose(np.abs(z_values[signed]), np.sqrt(chi2_values[signed]), rtol=1e-6), name


def test_nullhist_to_summarystat_accumulates_tail():
    """A null-histogram threshold must respond to p, not to the first empty bin."""
    from nimare.meta.cbma.base import _nullhist_to_summarystat

    # Upper-tail probabilities are [1, .35, .069, .0092, .00088, .00006, 0, 0].
    counts = np.array([7435009, 3197643, 686419, 95013, 9392, 651, 23, 0], dtype=np.int64)
    bins = np.arange(counts.size)

    assert _nullhist_to_summarystat(counts, bins, 0.05) == 2
    assert _nullhist_to_summarystat(counts, bins, 0.01) == 2
    assert _nullhist_to_summarystat(counts, bins, 0.001) == 3
    assert _nullhist_to_summarystat(counts, bins, 0.0001) == 4

    # Degenerate inputs resolve to an end of the range rather than raising.
    assert _nullhist_to_summarystat(np.zeros(4), np.arange(4), 0.001) == 0
    assert _nullhist_to_summarystat(np.array([1.0, 1.0]), np.arange(2), 1.0) == 0


def test_p_to_summarystat_is_monotone_in_p(testdata_cbma):
    """Lowering p must never lower the cluster-forming threshold, for either null."""
    approximate = MKDADensity(null_method="approximate", generate_description=False)
    approximate.fit(testdata_cbma)
    montecarlo = MKDADensity(null_method="montecarlo", n_iters=50, generate_description=False)
    montecarlo.fit(testdata_cbma)

    p_values = [0.05, 0.01, 0.005, 0.001]
    for estimator, null_method in ((approximate, "approximate"), (montecarlo, "montecarlo")):
        thresholds = [estimator._p_to_summarystat(p, null_method=null_method) for p in p_values]
        assert np.all(np.diff(thresholds) >= 0), (null_method, thresholds)
        # A threshold pinned at the first never-observed bin would sit at the top of
        # the range and be flat in p; this is the regression that guards it.
        assert thresholds[0] < len(estimator.inputs_["id"])
        assert thresholds[0] < thresholds[-1]


def _sample_size_studyset(sample_sizes):
    """Build a Studyset whose analyses carry the given sample sizes."""
    from nimare.studyset import Studyset

    rng = np.random.RandomState(7)
    studies = []
    for i_study, n in enumerate(sample_sizes):
        points = [
            {"space": "MNI", "coordinates": list(rng.randint(-40, 40, size=3).astype(float))}
            for _ in range(3)
        ]
        studies.append(
            {
                "id": f"S{i_study}",
                "name": f"study {i_study}",
                "analyses": [
                    {
                        "id": f"A{i_study}",
                        "metadata": {"sample_sizes": [n]},
                        "points": points,
                    }
                ],
            }
        )
    return Studyset({"id": "mkda-weighting", "studies": studies})


def test_MKDADensity_weighting_matches_published_formula(testdata_cbma):
    """Weights are delta * sqrt(N), renormalised to sum to the number of contrasts."""
    meta = MKDADensity(weighting="sample_size", generate_description=False)
    meta.fit(testdata_cbma)

    study_ids = np.unique(meta.inputs_["coordinates"]["id"].values)
    sample_sizes = np.array(
        [np.mean(n) for n in testdata_cbma.get_metadata(field="sample_sizes", ids=study_ids)],
        dtype=float,
    )
    expected = np.sqrt(sample_sizes)
    expected = len(study_ids) * expected / expected.sum()

    np.testing.assert_allclose(meta.weight_vec_.ravel(), expected)
    assert meta.weight_vec_.sum() == pytest.approx(len(study_ids))


def test_MKDADensity_weighting_is_off_by_default(testdata_cbma):
    """Weighting must be opt-in, so existing analyses do not silently change."""
    meta = MKDADensity(generate_description=False)
    meta.fit(testdata_cbma)
    np.testing.assert_array_equal(meta.weight_vec_.ravel(), np.ones(len(meta.inputs_["id"])))


def test_MKDADensity_equal_sample_sizes_match_unweighted():
    """With constant N, a weighted fit is identical to an unweighted one, bit for bit."""
    studyset = _sample_size_studyset([20] * 8)

    unweighted = MKDADensity(generate_description=False).fit(studyset)
    weighted = MKDADensity(weighting="sample_size", generate_description=False).fit(studyset)

    for key in ("stat", "p", "z"):
        np.testing.assert_array_equal(unweighted.maps[key], weighted.maps[key])
    np.testing.assert_array_equal(
        unweighted.estimator.null_distributions_["histogram_bins"],
        weighted.estimator.null_distributions_["histogram_bins"],
    )


def test_MKDADensity_weighted_null_matches_independent_computation():
    """The fitted null is the weighted Poisson binomial of the fitted weights."""
    from nimare.meta.cbma.mkda import _weighted_histogram_bins, _weighted_null_histogram

    studyset = _sample_size_studyset([8, 12, 20, 45, 90, 150])
    meta = MKDADensity(weighting="sample_size", generate_description=False)
    meta.fit(studyset)

    weights = meta.weight_vec_.ravel()
    prop_active = meta.null_distributions_["histogram_means"]
    bins = _weighted_histogram_bins(weights, meta.n_histogram_bins)
    expected = _weighted_null_histogram(weights, prop_active, bins)

    np.testing.assert_allclose(meta.null_distributions_["histogram_bins"], bins)
    np.testing.assert_allclose(
        meta.null_distributions_["histweights_corr-none_method-approximate"], expected
    )
    # The null is a probability distribution over the grid, and the grid spans the
    # attainable range of the statistic.
    assert expected.sum() == pytest.approx(1.0)
    assert bins[-1] >= weights.sum()
    assert bins[-1] == pytest.approx(weights.sum(), rel=1e-4)


def test_MKDADensity_weighted_null_is_not_the_unweighted_one():
    """Guards the actual defect: a weighted statistic scored on a unit-weight null."""
    studyset = _sample_size_studyset([8, 12, 20, 45, 90, 150])
    weighted = MKDADensity(weighting="sample_size", generate_description=False).fit(studyset)
    unweighted = MKDADensity(generate_description=False).fit(studyset)

    weighted_bins = weighted.estimator.null_distributions_["histogram_bins"]
    unweighted_bins = unweighted.estimator.null_distributions_["histogram_bins"]
    assert weighted_bins.size > unweighted_bins.size
    assert not np.array_equal(weighted.maps["z"], unweighted.maps["z"])


def test_MKDADensity_weights_align_by_id_not_position():
    """A weight must follow its own contrast, whatever order the ids arrive in."""
    sample_sizes = [8, 12, 20, 45, 90, 150]
    studyset = _sample_size_studyset(sample_sizes)

    meta = MKDADensity(weighting="sample_size", generate_description=False)
    meta.fit(studyset)

    row_ids = np.unique(meta.inputs_["coordinates"]["id"].values)
    by_id = dict(zip(studyset.ids, sample_sizes))
    expected = np.sqrt([by_id[study_id] for study_id in row_ids])
    expected = len(row_ids) * expected / expected.sum()
    np.testing.assert_allclose(meta.weight_vec_.ravel(), expected)

    # Reversing the caller's id order must not permute the weights off their contrasts.
    reversed_meta = MKDADensity(weighting="sample_size", generate_description=False)
    reversed_meta.fit(studyset.filter_ids(list(studyset.ids)[::-1]))
    np.testing.assert_allclose(reversed_meta.weight_vec_.ravel(), meta.weight_vec_.ravel())


def test_MKDADensity_weights_renormalise_over_a_subset():
    """Leave-one-out renormalises, as CANlab's Meta_Select_Contrasts does."""
    sample_sizes = [8, 12, 20, 45, 90, 150]
    studyset = _sample_size_studyset(sample_sizes)

    meta = MKDADensity(weighting="sample_size", generate_description=False)
    meta.fit(studyset)
    ma_maps = meta._collect_ma_maps()

    row_ids = list(np.unique(meta.inputs_["coordinates"]["id"].values))
    kept = row_ids[1:]
    meta._prepare_subsample_null(ma_maps[1:, :], subset_study_ids=np.array(kept))

    by_id = dict(zip(studyset.ids, sample_sizes))
    expected = np.sqrt([by_id[study_id] for study_id in kept])
    expected = len(kept) * expected / expected.sum()
    np.testing.assert_allclose(meta.weight_vec_.ravel(), expected)
    assert meta.weight_vec_.sum() == pytest.approx(len(kept))


def test_MKDADensity_weighted_montecarlo_null(testdata_cbma):
    """The Monte Carlo null bins onto the weighted grid and still sums to n_iters * voxels."""
    meta = MKDADensity(
        null_method="montecarlo",
        n_iters=20,
        weighting="sample_size",
        generate_description=False,
    )
    result = meta.fit(testdata_cbma)

    bins = meta.null_distributions_["histogram_bins"]
    counts = meta.null_distributions_["histweights_corr-none_method-montecarlo"]
    assert counts.shape == bins.shape
    assert counts.sum() == 20 * len(result.maps["stat"])
    assert np.all(np.isfinite(result.maps["z"]))


def test_MKDADensity_weighted_description_reports_the_scheme(testdata_cbma):
    """A weighted analysis has to say so, and say how concentrated the weights are."""
    meta = MKDADensity(weighting="sample_size")
    result = meta.fit(testdata_cbma)
    description = result.description_

    assert "square root of the sample size" in description
    assert "effective sample size" in description
    assert "wager2009evaluating" in description


def test_MKDADensity_unweighted_description_is_silent_about_weights(testdata_cbma):
    """An unweighted analysis should not advertise a weighting scheme."""
    meta = MKDADensity()
    result = meta.fit(testdata_cbma)
    assert "square root of the sample size" not in result.description_


@pytest.mark.parametrize(
    "weights",
    [
        [1e-9, 5.0],  # a negligible weight must not set the resolution for the whole grid
        [1.0, 1.0],
        [0.001, 5.0],
        list(np.sqrt(np.arange(4, 400.0))),
    ],
)
def test_weighted_null_grid_is_bounded_and_normalised(weights):
    """The grid stays within n_bins + k bins and the null stays a distribution."""
    from nimare.meta.cbma.mkda import _weighted_histogram_bins, _weighted_null_histogram

    weights = np.asarray(weights, dtype=float)
    n_bins = 100_000
    bins = _weighted_histogram_bins(weights, n_bins)

    assert bins.size <= n_bins + weights.size
    assert bins[0] == 0
    assert bins[-1] >= weights.sum()
    np.testing.assert_allclose(np.diff(bins), bins[1] - bins[0])

    hist = _weighted_null_histogram(weights, np.full(weights.size, 0.01), bins)
    assert hist.shape == bins.shape
    assert hist.sum() == pytest.approx(1.0)
    assert np.all(hist >= 0)


def test_weighted_null_matches_brute_force_enumeration():
    """The weighted Poisson binomial must agree with enumerating all 2**k outcomes."""
    import itertools

    from nimare.meta.cbma.mkda import _weighted_histogram_bins, _weighted_null_histogram

    rng = np.random.RandomState(7)
    k = 12
    sample_sizes = rng.randint(6, 300, size=k).astype(float)
    weights = np.sqrt(sample_sizes)
    weights = k * weights / weights.sum()
    prop_active = rng.uniform(0.002, 0.10, size=k)

    bins = _weighted_histogram_bins(weights, 100_000)
    hist = _weighted_null_histogram(weights, prop_active, bins)

    values, probabilities = [], []
    for bits in itertools.product([0, 1], repeat=k):
        indicator = np.array(bits)
        values.append(weights @ indicator)
        probabilities.append(np.prod(np.where(indicator == 1, prop_active, 1 - prop_active)))
    values, probabilities = np.array(values), np.array(probabilities)

    for quantile in (0.5, 0.9, 0.99, 0.999):
        threshold = np.quantile(values, quantile)
        exact = probabilities[values >= threshold - 1e-12].sum()
        approximate = hist[bins >= threshold - 1e-12].sum()
        assert approximate == pytest.approx(exact, abs=1e-6)

    # The mean is an independent check that the mass sits in the right place.
    assert (hist * bins).sum() == pytest.approx((weights * prop_active).sum(), abs=1e-3)


def test_uniform_bin_counts_matches_rounding_used_for_observed_map():
    """Permuted maps must land in the same bin the observed map would."""
    from nimare.meta.cbma.mkda import _uniform_bin_counts

    step = 0.25
    n_bins = 9
    values = np.array([0.0, 0.12, 0.13, 0.24, 0.26, 1.99, 2.0, 5.0, -1.0])
    counts = _uniform_bin_counts(values, n_bins=n_bins, step=step)

    expected = np.zeros(n_bins, dtype=int)
    for value in values:
        expected[int(np.clip(np.rint(value / step), 0, n_bins - 1))] += 1
    np.testing.assert_array_equal(counts, expected)
    assert counts.sum() == values.size
