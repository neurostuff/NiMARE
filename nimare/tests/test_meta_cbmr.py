"""Tests for CBMR meta-analytic methods."""

import logging
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import scipy
import scipy.sparse

try:
    import torch
except ImportError:
    warnings.warn("Torch not installed. CBMR tests will be skipped.")
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta import cbmr as cbmr_module
    from nimare.meta import models
    from nimare.meta.cbmr import (
        CBMREstimator,
        CBMRInference,
        CBMRResult,
        DEFAULT_GROUP_NAME,
    )
    from nimare.meta.utils import fit_spatial_cbmr_approximate

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.transforms import StandardizeField

# numba has a lot of debug messages that are not useful for testing
logging.getLogger("numba").setLevel(logging.WARNING)
# indexed_gzip has a few debug messages that are not useful for testing
logging.getLogger("indexed_gzip").setLevel(logging.WARNING)


if TORCH_INSTALLED:

    @pytest.fixture(
        scope="session",
        params=[
            pytest.param(models.PoissonEstimator, id="Poisson"),
            pytest.param(models.NegativeBinomialEstimator, id="NegativeBinomial"),
            pytest.param(
                models.ClusteredNegativeBinomialEstimator, id="ClusteredNegativeBinomial"
            ),
        ],
    )
    def model(request):
        """CBMR models."""
        return request.param

else:
    model = None


@pytest.fixture(scope="session")
def cbmr_result(testdata_cbmr_simulated, model):
    """Test CBMR estimator."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        spline_spacing=100,
        model=model,
        generate_description=False,
        penalty=False,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cpu",
        random_state=100,
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(res.description_, str)

    return res


@pytest.fixture(scope="session")
def inference_results(testdata_cbmr_simulated, cbmr_result):
    """Test inference results for CBMR estimator."""
    return cbmr_result.infer(
        group_contrasts=[("DepressionYes", "DepressionNo")],
        moderator_contrasts=[("standardized_sample_sizes", "standardized_avg_age")],
    )


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(FWECorrector(method="bonferroni"), id="bonferroni"),
        pytest.param(FDRCorrector(method="indep"), id="indep"),
        pytest.param(FDRCorrector(method="negcorr"), id="negcorr"),
    ],
)
def corrector(request):
    """Corrector classes."""
    return request.param


def test_cbmr_estimator(cbmr_result):
    """Unit test for CBMR estimator."""
    assert isinstance(cbmr_result, CBMRResult)
    assert isinstance(cbmr_result, nimare.results.MetaResult)
    assert cbmr_result.groups


def test_cbmr_result_interface_lists_inference_inputs(cbmr_result):
    """CBMR results should expose discoverable group and moderator names."""
    description = cbmr_result.describe_inference_inputs()

    assert description["groups"] == cbmr_result.groups
    assert description["moderators"] == cbmr_result.moderators


def test_cbmr_default_group_is_used_when_group_categories_none(testdata_cbmr_simulated):
    """CBMR should create a single default group when no grouping columns are supplied."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )
    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=None,
        moderators=["standardized_sample_sizes"],
        spline_spacing=100,
        model=models.PoissonEstimator,
        generate_description=False,
        penalty=False,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cpu",
        random_state=100,
    )

    result = cbmr.fit(dataset=dset)
    homogeneity_result = result.test_groups()

    assert result.groups == (DEFAULT_GROUP_NAME,)
    assert cbmr.inputs_["ids_by_group"][DEFAULT_GROUP_NAME]
    assert f"z_group-{DEFAULT_GROUP_NAME}" in homogeneity_result.maps


def test_cbmr_result_helpers_run_inference(cbmr_result):
    """Verify that CBMRResult supports a result-centered inference workflow."""
    homogeneity_result = cbmr_result.test_groups()
    comparison_result = cbmr_result.compare_groups([("DepressionYes", "DepressionNo")])
    moderator_result = cbmr_result.test_moderators()
    moderator_comparison_result = cbmr_result.compare_moderators(
        [("standardized_sample_sizes", "standardized_avg_age")]
    )

    assert isinstance(homogeneity_result, CBMRResult)
    assert "z_group-DepressionYes" in homogeneity_result.maps
    assert "z_group-DepressionYes-DepressionNo" in comparison_result.maps
    assert "p_standardized_sample_sizes" in moderator_result.tables
    assert "p_standardized_sample_sizes-standardized_avg_age" in moderator_comparison_result.tables


def test_cbmr_fit_is_repeatable(testdata_cbmr_simulated):
    """Repeated CBMR fits on the same dataset should return identical results."""
    dset = testdata_cbmr_simulated.slice(testdata_cbmr_simulated.ids[:60])
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        dset
    )

    def _fit_once():
        cbmr = CBMREstimator(
            moderator_effect="global",
            group_categories=["diagnosis", "drug_status"],
            moderators=[
                "standardized_sample_sizes",
                "standardized_avg_age",
                "schizophrenia_subtype",
            ],
            spline_spacing=100,
            model=models.PoissonEstimator,
            generate_description=False,
            penalty=False,
            n_iter=200,
            lr=1,
            tol=1e4,
            device="cpu",
            random_state=100,
        )
        return cbmr.fit(dataset=dset)

    first_result = _fit_once()
    second_result = _fit_once()

    assert first_result.maps.keys() == second_result.maps.keys()
    assert first_result.tables.keys() == second_result.tables.keys()

    for map_name in first_result.maps:
        np.testing.assert_array_equal(first_result.maps[map_name], second_result.maps[map_name])

    for table_name in first_result.tables:
        pd.testing.assert_frame_equal(
            first_result.tables[table_name], second_result.tables[table_name]
        )


@pytest.mark.skipif(
    not TORCH_INSTALLED or not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_cbmr_cuda_fit_and_inference_run(testdata_cbmr_simulated):
    """CBMR fit and inference should run end to end on CUDA."""
    dset = testdata_cbmr_simulated.slice(testdata_cbmr_simulated.ids[:60])
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        dset
    )

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        spline_spacing=100,
        model=models.PoissonEstimator,
        generate_description=False,
        penalty=False,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cuda",
        random_state=100,
    )
    result = cbmr.fit(dataset=dset)

    assert cbmr.model.spatial_coef_linears[cbmr.groups[0]].weight.device.type == "cuda"
    if cbmr.model.moderators_coef_dim:
        assert cbmr.model.moderators_linear.weight.device.type == "cuda"

    inference = CBMRInference(device="cuda", moderator_effect="global")
    inference.fit(result)
    transformed = inference.transform(
        t_con_groups=inference.create_contrast(["DepressionYes-DepressionNo"], source="groups"),
        t_con_moderators=inference.create_contrast(
            ["standardized_sample_sizes-standardized_avg_age"],
            source="moderators",
        ),
    )

    first_group = inference.groups[0]
    assert inference.estimator.model.spatial_coef_linears[first_group].weight.device.type == "cuda"
    assert isinstance(transformed, nimare.results.MetaResult)
    for map_name, map_values in transformed.maps.items():
        assert np.all(np.isfinite(map_values)), map_name


def test_cbmr_description_generation(testdata_cbmr_simulated):
    """CBMR should still generate a description when requested."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        spline_spacing=100,
        model=models.PoissonEstimator,
        generate_description=True,
        penalty=False,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cpu",
        random_state=100,
    )
    res = cbmr.fit(dataset=dset)

    assert isinstance(res.description_, str)
    assert res.description_


def test_cbmr_summary_tables_match_legacy_from_dict_construction(cbmr_result):
    """Optimized summary table construction should match the legacy DataFrame builders."""
    model = cbmr_result.estimator.model
    legacy_tables = {
        "spatial_regression_coef": pd.DataFrame.from_dict(
            model.spatial_regression_coef, orient="index"
        ),
        "spatial_regression_coef_se": pd.DataFrame.from_dict(
            model.spatial_regression_coef_se, orient="index"
        ),
        "log_spatial_intensity_se": pd.DataFrame.from_dict(
            model.log_spatial_intensity_se, orient="index"
        ),
        "spatial_intensity_se": pd.DataFrame.from_dict(model.spatial_intensity_se, orient="index"),
    }
    if model.moderators_coef_dim:
        legacy_tables["moderators_regression_coef"] = pd.DataFrame(
            data=model.moderators_coef, columns=model.moderators
        )
        legacy_tables["moderators_effect"] = pd.DataFrame.from_dict(
            model.moderators_effect, orient="index"
        )
        legacy_tables["moderators_regression_se"] = pd.DataFrame(
            data=model.se_moderators, columns=model.moderators
        )

    for table_name, legacy_table in legacy_tables.items():
        pd.testing.assert_frame_equal(cbmr_result.tables[table_name], legacy_table)


def test_cbmr_inference(inference_results):
    """Unit test for CBMR inference."""
    assert isinstance(inference_results, CBMRResult)
    assert isinstance(inference_results, nimare.results.MetaResult)


def test_cbmr_inference_does_not_mutate_input_result(cbmr_result):
    """Inference should append new outputs without mutating the input MetaResult."""
    original_map_keys = set(cbmr_result.maps)
    original_table_keys = set(cbmr_result.tables)

    inference = CBMRInference(device="cpu", moderator_effect="global")
    inference.fit(cbmr_result)
    t_con_groups = inference.create_contrast(
        [
            "DepressionYes-DepressionNo",
        ],
        source="groups",
    )
    t_con_moderators = inference.create_contrast(
        ["standardized_sample_sizes-standardized_avg_age"],
        source="moderators",
    )
    contrast_result = inference.transform(
        t_con_groups=t_con_groups, t_con_moderators=t_con_moderators
    )

    assert set(cbmr_result.maps) == original_map_keys
    assert set(cbmr_result.tables) == original_table_keys
    assert original_map_keys < set(contrast_result.maps)
    assert original_table_keys < set(contrast_result.tables)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_chi_square_log_intensity_matches_legacy_loop():
    """Vectorized voxel-wise chi-square computation should match the legacy loop."""
    inference = CBMRInference(device="cpu", moderator_effect="global")
    simp_con_group = np.array([[1.0, -1.0], [0.5, 0.5]])
    contrast_log_intensity = np.array(
        [[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]],
        dtype=float,
    )
    cov_log_intensity = np.array(
        [
            [2.0, 1.5, 1.2],
            [0.3, 0.2, 0.1],
            [0.3, 0.2, 0.1],
            [1.8, 1.1, 1.6],
        ],
        dtype=float,
    )

    expected = []
    for voxel_index in range(contrast_log_intensity.shape[1]):
        contrast_vector = contrast_log_intensity[:, voxel_index].reshape(2, 1)
        covariance = cov_log_intensity[:, voxel_index].reshape(2, 2)
        projected_covariance = simp_con_group @ covariance @ simp_con_group.T
        expected.append(
            (contrast_vector.T @ np.linalg.inv(projected_covariance) @ contrast_vector).item()
        )

    actual = inference._chi_square_log_intensity(
        m=2,
        n_brain_voxel=contrast_log_intensity.shape[1],
        n_con_group_involved=2,
        simp_con_group=simp_con_group,
        cov_log_intensity=cov_log_intensity,
        contrast_log_intensity=contrast_log_intensity,
    )

    np.testing.assert_allclose(actual, np.asarray(expected))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_group_glh_uses_stable_chi_square_survival_function():
    """Extreme chi-square statistics should retain nonzero p-values when representable."""
    inference = CBMRInference(device="cpu", moderator_effect="global")

    chi_square, z_values, p_values = inference._compute_group_glh_statistics(
        simp_con_group=np.array([[1.0]]),
        involved_groups=["group"],
        cov_spatial_coef=np.array([[1.0]]),
        contrast_log_intensity=np.array([[10.0]]),
        X=np.array([[1.0]]),
        spatial_coef_dim=1,
        n_brain_voxel=1,
        is_homogeneity_test=False,
    )

    np.testing.assert_allclose(chi_square, [100.0])
    np.testing.assert_allclose(p_values, scipy.stats.chi2.sf(100.0, df=1), rtol=1e-6, atol=0)
    assert p_values[0] > 0
    assert np.isfinite(z_values[0])


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_poisson_multigroup_log_likelihood_matches_legacy_loop():
    """Refactored multigroup Poisson likelihood should match the legacy formulation."""
    model = models.PoissonEstimator(device="cpu")
    coef_spline_bases = torch.tensor(
        [[0.2, 1.0], [1.2, -0.4], [0.5, 0.3]],
        dtype=torch.float64,
    )
    spatial_coef = torch.tensor(
        [[[0.1], [0.3]], [[-0.2], [0.4]]],
        dtype=torch.float64,
    )
    moderator_coef = torch.tensor(
        [[0.2], [-0.1]],
        dtype=torch.float64,
    )
    foci_per_voxel = [
        torch.tensor([[1.0], [0.0], [2.0]], dtype=torch.float64),
        torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float64),
    ]
    foci_per_experiment = [
        torch.tensor([[1.0], [3.0]], dtype=torch.float64),
        torch.tensor([[2.0], [1.0], [0.0]], dtype=torch.float64),
    ]
    moderators = [
        torch.tensor([[0.1, 1.0], [1.2, -0.2]], dtype=torch.float64),
        torch.tensor([[0.5, -1.0], [0.3, 0.7], [1.1, 0.4]], dtype=torch.float64),
    ]

    expected = 0.0
    for group_index in range(spatial_coef.shape[0]):
        group_log_spatial = torch.matmul(
            coef_spline_bases, spatial_coef[group_index, :, :]
        ).reshape(-1)
        group_spatial = torch.exp(group_log_spatial)
        group_log_moderator = torch.matmul(moderators[group_index], moderator_coef).reshape(-1)
        group_moderator = torch.exp(group_log_moderator)
        expected += (
            torch.dot(foci_per_voxel[group_index].reshape(-1), group_log_spatial)
            + torch.dot(foci_per_experiment[group_index].reshape(-1), group_log_moderator)
            - torch.sum(group_spatial) * torch.sum(group_moderator)
        )

    actual = model._log_likelihood_mult_group(
        spatial_coef=spatial_coef,
        moderator_coef=moderator_coef,
        coef_spline_bases=coef_spline_bases,
        foci_per_voxel=foci_per_voxel,
        foci_per_experiment=foci_per_experiment,
        moderators=moderators,
        device="cpu",
    )

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_poisson_analytic_fisher_matches_generic_hessian():
    """Analytic Poisson Fisher matrices should match the generic Hessian route."""
    model = models.PoissonEstimator(device="cpu")
    groups = ["A", "B"]
    moderators = ["m1", "m2"]
    model.init_weights(
        groups=groups,
        moderators=moderators,
        spatial_coef_dim=2,
        moderators_coef_dim=2,
    )

    with torch.no_grad():
        model.spatial_coef_linears["A"].weight[:] = torch.tensor([[0.1, 0.3]], dtype=torch.float64)
        model.spatial_coef_linears["B"].weight[:] = torch.tensor(
            [[-0.2, 0.4]], dtype=torch.float64
        )
        model.moderators_linear.weight[:] = torch.tensor([[0.2, -0.1]], dtype=torch.float64)

    model.moderators_coef = model.moderators_linear.weight.detach().cpu().numpy()

    coef_spline_bases = np.array([[0.2, 1.0], [1.2, -0.4], [0.5, 0.3]], dtype=float)
    moderators_by_group = {
        "A": np.array([[0.1, 1.0], [1.2, -0.2]], dtype=float),
        "B": np.array([[0.5, -1.0], [0.3, 0.7], [1.1, 0.4]], dtype=float),
    }
    foci_per_voxel = {
        "A": np.array([[1.0], [0.0], [2.0]], dtype=float),
        "B": np.array([[0.0], [1.0], [1.0]], dtype=float),
    }
    foci_per_experiment = {
        "A": np.array([[1.0], [3.0]], dtype=float),
        "B": np.array([[2.0], [1.0], [0.0]], dtype=float),
    }

    analytic_spatial = model.fisher_info_multiple_group_spatial(
        ["A", "B"],
        coef_spline_bases,
        moderators_by_group,
        foci_per_voxel,
        foci_per_experiment,
    )
    generic_spatial = models.GeneralLinearModelEstimator.fisher_info_multiple_group_spatial(
        model,
        ["A", "B"],
        coef_spline_bases,
        moderators_by_group,
        foci_per_voxel,
        foci_per_experiment,
    )
    np.testing.assert_allclose(analytic_spatial, generic_spatial, rtol=1e-6, atol=1e-8)

    analytic_moderator = model.fisher_info_multiple_group_moderator(
        coef_spline_bases,
        moderators_by_group,
        foci_per_voxel,
        foci_per_experiment,
    )
    generic_moderator = models.GeneralLinearModelEstimator.fisher_info_multiple_group_moderator(
        model,
        coef_spline_bases,
        moderators_by_group,
        foci_per_voxel,
        foci_per_experiment,
    )
    np.testing.assert_allclose(analytic_moderator, generic_moderator, rtol=1e-6, atol=1e-8)


def test_cbmr_inference_multi_contrast_matches_individual_transforms(cbmr_result):
    """Cached multi-contrast inference should match independent contrast evaluations."""
    inference = CBMRInference(device="cpu", moderator_effect="global")
    inference.fit(cbmr_result)
    groups = inference.groups
    moderators = inference.moderators

    group_contrast_names = [
        f"{groups[0]}-{groups[1]}",
        f"{groups[1]}-{groups[0]}",
    ]
    moderator_contrast_names = [
        moderators[0],
        f"{moderators[0]}-{moderators[1]}",
    ]

    multi_group_contrasts = inference.create_contrast(group_contrast_names, source="groups")
    multi_moderator_contrasts = inference.create_contrast(
        moderator_contrast_names,
        source="moderators",
    )
    multi_result = inference.transform(
        t_con_groups=multi_group_contrasts,
        t_con_moderators=multi_moderator_contrasts,
    )

    for contrast_name in group_contrast_names:
        single_inference = CBMRInference(device="cpu", moderator_effect="global")
        single_inference.fit(cbmr_result)
        single_result = single_inference.transform(
            t_con_groups=single_inference.create_contrast([contrast_name], source="groups"),
        )
        np.testing.assert_allclose(
            multi_result.maps[f"p_group-{contrast_name}"],
            single_result.maps[f"p_group-{contrast_name}"],
        )
        np.testing.assert_allclose(
            multi_result.maps[f"z_group-{contrast_name}"],
            single_result.maps[f"z_group-{contrast_name}"],
        )

    for contrast_name in moderator_contrast_names:
        single_inference = CBMRInference(device="cpu", moderator_effect="global")
        single_inference.fit(cbmr_result)
        single_result = single_inference.transform(
            t_con_moderators=single_inference.create_contrast(
                [contrast_name], source="moderators"
            ),
        )
        pd.testing.assert_frame_equal(
            multi_result.tables[f"p_{contrast_name}"],
            single_result.tables[f"p_{contrast_name}"],
        )
        pd.testing.assert_frame_equal(
            multi_result.tables[f"z_{contrast_name}"],
            single_result.tables[f"z_{contrast_name}"],
        )


def test_cbmr_correctors(inference_results, corrector):
    """Unit test for Correctors that work with CBMR."""
    corrected_results = corrector.transform(inference_results)
    assert isinstance(corrected_results, nimare.results.MetaResult)


def test_firth_penalty(testdata_cbmr_simulated):
    """Unit test for Firth penalty."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )
    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        spline_spacing=100,
        model=models.PoissonEstimator,
        generate_description=False,
        penalty=True,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cpu",
        random_state=100,
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)


def test_moderators_none(testdata_cbmr_simulated):
    """Unit test for Firth penalty."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )
    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
        moderators=None,
        spline_spacing=100,
        model=models.PoissonEstimator,
        generate_description=False,
        penalty=False,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cpu",
        random_state=100,
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)
    inference = CBMRInference(device="cpu", moderator_effect="global")
    inference.fit(res)

    t_con_groups = inference.create_contrast(
        [
            "DepressionYes",
        ],
        source="groups",
    )
    inference_results = inference.transform(t_con_groups=t_con_groups)

    assert isinstance(inference_results, nimare.results.MetaResult)


def test_CBMREstimator_update(testdata_cbmr_simulated):
    """Unit test for CBMR estimator update function."""
    testdata_cbmr_simulated = StandardizeField(
        fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]
    ).transform(testdata_cbmr_simulated)
    cbmr = CBMREstimator(
        moderator_effect="global",
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        model=models.PoissonEstimator,
        generate_description=False,
        lr=1,
        random_state=100,
    )

    cbmr._collect_inputs(testdata_cbmr_simulated, drop_invalid=True)
    cbmr._preprocess_input(testdata_cbmr_simulated)

    # fit the model
    init_weight_kwargs = {
        "groups": cbmr.groups,
        "moderators": cbmr.moderators,
        "spatial_coef_dim": cbmr.inputs_["coef_spline_bases"].shape[1],
        "moderators_coef_dim": len(cbmr.moderators) if cbmr.moderators else None,
    }

    cbmr.model.init_weights(**init_weight_kwargs)
    optimizer = torch.optim.LBFGS(cbmr.model.parameters(), cbmr.lr)
    # load dataset info to torch.tensor
    if cbmr.moderators:
        moderators_by_group_tensor = dict()
        for group in cbmr.model.groups:
            moderators_tensor = torch.tensor(
                cbmr.inputs_["moderators_by_group"][group],
                dtype=torch.float64,
                device=cbmr.device,
            )
            moderators_by_group_tensor[group] = moderators_tensor
    else:
        moderators_by_group_tensor = None
    foci_per_voxel_tensor, foci_per_experiment_tensor = dict(), dict()
    for group in cbmr.model.groups:
        group_foci_per_voxel_tensor = torch.tensor(
            cbmr.inputs_["foci_per_voxel"][group], dtype=torch.float64, device=cbmr.device
        )
        group_foci_per_experiment_tensor = torch.tensor(
            cbmr.inputs_["foci_per_experiment"][group], dtype=torch.float64, device=cbmr.device
        )
        foci_per_voxel_tensor[group] = group_foci_per_voxel_tensor
        foci_per_experiment_tensor[group] = group_foci_per_experiment_tensor

    prev_loss = torch.tensor(float("inf"))  # initialization loss difference

    cbmr.model._update(
        optimizer,
        torch.tensor(cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device),
        moderators_by_group_tensor,
        foci_per_voxel_tensor,
        foci_per_experiment_tensor,
        prev_loss,
    )
    # deliberately set the first spatial coefficient to nan
    for group in cbmr.model.groups:
        nan_coef = torch.tensor(cbmr.model.spatial_coef_linears[group].weight)
        nan_coef[:, 0] = float("nan")
        cbmr.model.spatial_coef_linears[group].weight = torch.nn.Parameter(nan_coef)

    # Expect exceptions when one of the spatial coefficients is nan.
    with pytest.raises(ValueError):
        cbmr.model._update(
            optimizer,
            torch.tensor(
                cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device
            ),
            moderators_by_group_tensor,
            foci_per_voxel_tensor,
            foci_per_experiment_tensor,
            prev_loss,
        )


def test_cbmr_group_arrays_remain_aligned_when_experiment_has_no_in_mask_foci(
    testdata_cbmr_simulated,
):
    """Experiment-level arrays should stay aligned after focus filtering removes all foci."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )
    target_id = dset.annotations.iloc[0]["id"]
    dset.coordinates.loc[
        dset.coordinates["id"] == target_id,
        ["x", "y", "z"],
    ] = 10_000

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        model=models.PoissonEstimator,
        generate_description=False,
        lr=1,
        random_state=100,
    )

    cbmr._collect_inputs(dset, drop_invalid=True)
    cbmr._preprocess_input(dset)

    for group in cbmr.groups:
        n_experiments = len(cbmr.inputs_["ids_by_group"][group])
        assert scipy.sparse.isspmatrix_csr(cbmr.inputs_["foci_by_experiment"][group])
        assert cbmr.inputs_["foci_by_experiment"][group].shape == (
            n_experiments,
            cbmr.inputs_["foci_per_voxel"][group].shape[0],
        )
        assert cbmr.inputs_["foci_per_experiment"][group].shape == (n_experiments, 1)
        assert cbmr.inputs_["moderators_by_group"][group].shape[0] == n_experiments

    target_group = next(
        group
        for group, group_ids in cbmr.inputs_["ids_by_group"].items()
        if target_id in group_ids
    )
    target_id_index = cbmr.inputs_["ids_by_group"][target_group].index(target_id)
    assert cbmr.inputs_["foci_per_experiment"][target_group][target_id_index, 0] == 0


def test_cbmr_groups_full_experiment_ids_instead_of_collapsing_study_ids():
    """CBMR should follow ALE/MKDA and treat each experiment id as its own observation."""
    dset = nimare.dataset.Dataset(
        {
            "study-0": {
                "contrasts": {
                    "1": {"coords": {"space": "MNI", "x": [0], "y": [0], "z": [0]}},
                    "2": {"coords": {"space": "MNI", "x": [20], "y": [0], "z": [0]}},
                }
            },
            "study-1": {
                "contrasts": {
                    "1": {"coords": {"space": "MNI", "x": [-20], "y": [0], "z": [0]}},
                }
            },
        }
    )
    dset.annotations["diagnosis"] = ["schizophrenia", "depression", "depression"]
    dset.annotations["drug_status"] = ["Yes", "No", "No"]

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
        model=models.PoissonEstimator,
        generate_description=False,
        lr=1,
        random_state=100,
    )

    cbmr._collect_inputs(dset, drop_invalid=True)
    cbmr._preprocess_input(dset)

    grouped_ids = [id_ for group_ids in cbmr.inputs_["ids_by_group"].values() for id_ in group_ids]
    assert sorted(grouped_ids) == sorted(list(cbmr.inputs_["id"]))
    assert "study-0-1" in cbmr.inputs_["ids_by_group"]["SchizophreniaYes"]
    assert "study-0-2" in cbmr.inputs_["ids_by_group"]["DepressionNo"]


def test_StandardizeField(testdata_cbmr_simulated):
    """Unit test for StandardizeField."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age"]).transform(testdata_cbmr_simulated)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert "standardized_sample_sizes" in dset.annotations
    assert "standardized_avg_age" in dset.annotations
    assert dset.annotations["standardized_sample_sizes"].mean() == pytest.approx(0.0, abs=1e-3)
    assert dset.annotations["standardized_sample_sizes"].std(ddof=0) == pytest.approx(
        1.0, abs=1e-3
    )
    assert dset.annotations["standardized_avg_age"].mean() == pytest.approx(0.0, abs=1e-3)
    assert dset.annotations["standardized_avg_age"].std(ddof=0) == pytest.approx(1.0, abs=1e-3)


def test_meta_package_defers_cbmr_import():
    """Importing nimare.meta should not eagerly import optional CBMR modules."""
    import importlib
    import sys

    nimare.__dict__.pop("meta", None)
    sys.modules.pop("nimare.meta", None)
    sys.modules.pop("nimare.meta.cbmr", None)
    sys.modules.pop("nimare.meta.models", None)

    meta = importlib.import_module("nimare.meta")

    assert "nimare.meta.cbmr" not in sys.modules
    assert "nimare.meta.models" not in sys.modules
    assert hasattr(meta, "ALE")


@pytest.mark.cbmr_importerror
def test_cbmr_importerror():
    """Test that ImportErrors are raised when torch is not installed."""
    if TORCH_INSTALLED:
        pytest.skip("torch is installed in this test environment")

    with pytest.raises(ImportError):
        from nimare.meta.cbmr import CBMREstimator

        CBMREstimator()

    with pytest.raises(ImportError):
        from nimare.meta.cbmr import CBMRInference

        CBMRInference()

    with pytest.raises(ImportError):
        from nimare.meta.models import GeneralLinearModelEstimator

        GeneralLinearModelEstimator()

    with pytest.raises(ImportError):
        from nimare.meta.models import PoissonEstimator

        PoissonEstimator()

    with pytest.raises(ImportError):
        from nimare.meta.models import NegativeBinomialEstimator

        NegativeBinomialEstimator()

    with pytest.raises(ImportError):
        from nimare.meta.models import ClusteredNegativeBinomialEstimator

        ClusteredNegativeBinomialEstimator()


def _mask_img():
    """Return a small valid mask image for MetaResult construction."""
    return nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))


def _spatial_result_for_inference():
    """Return a small fitted voxelwise CBMR result suitable for inference tests."""
    estimator = CBMREstimator(moderators=["age"], backend="approximate")
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "moderators_by_group": {"Default": np.array([[2.0], [4.0]])},
        "foci_by_experiment_voxel": {
            "Default": scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        },
    }
    estimator.voxelwise_coef = {"Default": np.array([[0.1], [0.2], [0.3], [0.4]])}
    maps = {}
    tables = {}
    estimator._add_approximate_results(
        maps,
        tables,
        "Default",
        estimator.inputs_["moderators_by_group"]["Default"],
        estimator.voxelwise_coef["Default"],
    )
    return CBMRResult(
        estimator=estimator,
        mask=_mask_img(),
        maps=maps,
        tables=tables,
        description="spatial inference test",
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
@pytest.mark.parametrize(
    ("estimator_kwargs", "inference_kwargs", "expected_moderator_effect"),
    [
        ({}, {}, "voxelwise"),
        ({"moderator_effect": "global"}, {"moderator_effect": "global"}, "global"),
    ],
)
def test_cbmr_public_api_dispatches_by_moderator_effect(
    estimator_kwargs,
    inference_kwargs,
    expected_moderator_effect,
):
    """The public CBMR API should dispatch to parallel global/voxelwise pipelines."""
    estimator = CBMREstimator(**estimator_kwargs)
    inference = CBMRInference(**inference_kwargs)

    assert type(estimator) is CBMREstimator
    assert type(inference) is CBMRInference
    assert estimator.moderator_effect == expected_moderator_effect
    assert inference.moderator_effect == expected_moderator_effect


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_named_pairwise_contrasts_normalize_tuples():
    """Tuple shorthand should normalize to the named contrast strings used by inference."""
    assert cbmr_module._normalize_named_pairwise_contrasts(("A", "B")) == ["A-B"]
    assert cbmr_module._normalize_named_pairwise_contrasts([("A", "B"), "C-D"]) == [
        "A-B",
        "C-D",
    ]


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_mask_lookup_and_coordinate_filtering_use_mask_indices():
    """Coordinate filtering should drop out-of-mask foci and keep masked-space indices."""
    estimator = CBMREstimator(moderator_effect="global")
    mask_data = np.array(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), np.eye(4))
    mask_lookup, n_mask_voxels = estimator._build_mask_lookup(mask_data)
    coordinates = pd.DataFrame(
        {
            "id": ["kept-0", "not-in-mask", "kept-1", "out-of-bounds"],
            "x": [0, 0, 1, 10],
            "y": [0, 1, 1, 0],
            "z": [0, 0, 0, 0],
        }
    )

    filtered = estimator._filter_coordinates_to_mask(
        coordinates,
        mask_img,
        mask_data,
        mask_lookup,
    )

    assert n_mask_voxels == 4
    assert filtered["id"].tolist() == ["kept-0", "kept-1"]
    expected_flat_indices = np.ravel_multi_index(
        np.array([[0, 1], [0, 1], [0, 0]]),
        mask_data.shape,
    )
    np.testing.assert_array_equal(
        filtered["_cbmr_mask_index"].to_numpy(),
        mask_lookup[expected_flat_indices],
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_group_label_validation_and_multicolumn_formatting():
    """Group label preprocessing should format valid labels and reject missing columns."""
    annotations = pd.DataFrame(
        {
            "id": ["study-1", "study-2"],
            "diagnosis": ["depression", "healthy"],
            "drug_status": ["yes", "no"],
        }
    )
    estimator = CBMREstimator(
        moderator_effect="global",
        group_categories=["diagnosis", "drug_status"],
    )

    grouped = estimator._assign_group_labels(annotations.copy())

    assert grouped[estimator._group_column].tolist() == ["DepressionYes", "HealthyNo"]
    with pytest.raises(ValueError, match="does not exist"):
        CBMREstimator(moderator_effect="global", group_categories="missing")._assign_group_labels(
            annotations.copy()
        )
    with pytest.raises(ValueError, match="do/does not exist"):
        CBMREstimator(
            moderator_effect="global",
            group_categories=["diagnosis", "missing"],
        )._assign_group_labels(annotations.copy())


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_inference_forwards_voxelwise_options_to_voxelwise_pipeline():
    """Voxelwise-only inference options should be preserved by public dispatch."""
    inference = CBMRInference(
        method="FI",
        sandwich_meat="iid",
        sandwich_correction="hc0",
        ridge=1e-3,
    )

    assert isinstance(inference, CBMRInference)
    assert inference.moderator_effect == "voxelwise"
    assert inference.method == "FI"
    assert inference.sandwich_meat == "iid"
    assert inference.sandwich_correction == "hc0"
    assert inference.ridge == pytest.approx(1e-3)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_inference_rejects_voxelwise_options_for_global_pipeline():
    """Global inference should reject options that only apply to voxelwise inference."""
    with pytest.raises(ValueError, match="method is only supported"):
        CBMRInference(moderator_effect="global", method="FI")
    with pytest.raises(ValueError, match="only supported for voxelwise"):
        CBMRInference(moderator_effect="global", sandwich_meat="iid")
    with pytest.raises(ValueError, match="only supported for voxelwise"):
        CBMRInference(moderator_effect="global", sandwich_correction="hc0")
    with pytest.raises(ValueError, match="only supported for voxelwise"):
        CBMRInference(moderator_effect="global", ridge=1e-3)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_moderator_effect_validation_is_shared_by_estimator_and_inference():
    """Both public entry points should reject unsupported moderator-effect values."""
    with pytest.raises(ValueError, match="moderator_effect must be one of"):
        CBMREstimator(moderator_effect="bad-effect")
    with pytest.raises(ValueError, match="moderator_effect must be one of"):
        CBMRInference(moderator_effect="bad-effect")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_invalid_backend_raises():
    """The estimator should validate backend names at initialization."""
    with pytest.raises(ValueError, match="backend must be one of"):
        CBMREstimator(backend="bad-backend")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_build_group_foci_matrices_counts_foci():
    """Experiment-by-voxel matrices should preserve ordered experiment IDs and focus counts."""
    coordinates = pd.DataFrame(
        {
            "id": ["study-1", "study-1", "study-2", "study-3"],
            "_cbmr_mask_index": [0, 2, 2, 1],
        }
    )
    ids_by_group = {"A": ["study-1", "study-2"], "B": ["study-3", "study-4"]}

    matrices = CBMREstimator._build_group_foci_matrices(
        coordinates,
        ids_by_group,
        n_mask_voxels=4,
    )

    assert set(matrices) == {"A", "B"}
    assert all(scipy.sparse.isspmatrix_csr(matrix) for matrix in matrices.values())
    np.testing.assert_array_equal(
        matrices["A"].toarray(),
        np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        matrices["B"].toarray(),
        np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_build_group_foci_matrices_handles_empty_coordinates():
    """Groups with no retained foci should still receive correctly shaped sparse matrices."""
    coordinates = pd.DataFrame({"id": [], "_cbmr_mask_index": []})
    ids_by_group = {"Default": ["study-1", "study-2"]}

    matrices = CBMREstimator._build_group_foci_matrices(
        coordinates,
        ids_by_group,
        n_mask_voxels=3,
    )

    assert matrices["Default"].shape == (2, 3)
    assert matrices["Default"].nnz == 0


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_prepare_torch_inputs_densifies_sparse_matrices():
    """Torch backend inputs should be float64 tensors on the estimator device."""
    estimator = CBMREstimator(moderators=["age"], device="cpu")
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.5, 0.5]]),
        "moderators_by_group": {"Default": np.array([[0.2], [1.0]])},
        "foci_by_experiment_voxel": {
            "Default": scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        },
    }

    bases, moderators_by_group, foci_by_experiment_voxel = estimator._prepare_torch_inputs()

    assert bases.dtype == torch.float64
    assert moderators_by_group["Default"].dtype == torch.float64
    assert foci_by_experiment_voxel["Default"].dtype == torch.float64
    assert foci_by_experiment_voxel["Default"].device.type == "cpu"
    np.testing.assert_array_equal(
        foci_by_experiment_voxel["Default"].cpu().numpy(),
        np.array([[1.0, 0.0], [0.0, 2.0]]),
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_result_helpers_preserve_result_type_and_summarize_maps():
    """The result should retain CBMRResult behavior plus spatial helper methods."""
    result = CBMRResult(
        estimator=object(),
        mask=_mask_img(),
        maps={
            "voxelwiseModeratorEffect_age_group-Default": np.array([1.0, 2.0, 3.0]),
            "spatialIntensity_group-Default": np.array([4.0, 5.0, 6.0]),
        },
        tables={"spatial_regression_coef": pd.DataFrame([[1.0, 2.0]], index=["Default"])},
        description="spatial test",
    )

    copied = result.copy()

    assert isinstance(result, CBMRResult)
    assert isinstance(result, nimare.results.MetaResult)
    assert copied is not result
    assert isinstance(copied, CBMRResult)
    assert result.voxelwise_moderator_effect_map_names == (
        "voxelwiseModeratorEffect_age_group-Default",
    )
    assert result.describe_voxelwise_moderator_effect_maps()[
        "voxelwiseModeratorEffect_age_group-Default"
    ] == (1.0, 2.0, 3.0)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_output_tables_match_cbmr_group_table_convention():
    """Spatial CBMR summaries should expose aggregate CBMR-style output tables."""
    tables = {}

    CBMREstimator._add_spatial_coef_table(tables, "A", np.array([1.0, 2.0]))
    CBMREstimator._add_spatial_coef_table(tables, "B", np.array([3.0, 4.0]))
    CBMREstimator._add_moderator_table(
        tables,
        "A",
        ["age", "sample_size"],
        np.array([[0.1, 0.2], [0.3, 0.4]]),
    )

    expected_spatial = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["A", "B"],
        columns=["basis_0", "basis_1"],
    )
    pd.testing.assert_frame_equal(tables["spatial_regression_coef"], expected_spatial)
    assert "voxelwise_moderator_effect_regression_coef_group-A" in tables
    assert "voxelwise_moderator_effects_regression_coef" in tables
    assert tables["voxelwise_moderator_effects_regression_coef"].index.names == [
        "group",
        "moderator",
    ]
    assert ("A", "age") in tables["voxelwise_moderator_effects_regression_coef"].index


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_add_approximate_results_creates_expected_maps_and_tables():
    """Approximate backend result extraction should create finite maps and CBMR-style tables."""
    estimator = CBMREstimator(moderators=["age"], backend="approximate")
    estimator.inputs_ = {"coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]])}
    maps = {}
    tables = {}
    moderators = np.array([[2.0], [4.0]])
    coefficient = np.array([[0.1], [0.2], [0.3], [0.4]])

    estimator._add_approximate_results(maps, tables, "Default", moderators, coefficient)

    assert set(maps) == {
        "spatialIntensity_group-Default",
        "voxelwiseModeratorEffect_age_group-Default",
        "voxelwiseModeratorEffectTotal_group-Default",
    }
    assert np.all(np.isfinite(maps["spatialIntensity_group-Default"]))
    np.testing.assert_allclose(maps["spatialIntensity_group-Default"], np.exp([0.3, 0.4]))
    np.testing.assert_allclose(maps["voxelwiseModeratorEffect_age_group-Default"], [0.3, 0.6])
    pd.testing.assert_frame_equal(
        tables["spatial_regression_coef"],
        pd.DataFrame([[0.3, 0.4]], index=["Default"], columns=["basis_0", "basis_1"]),
    )
    assert ("Default", "age") in tables["voxelwise_moderator_effects_regression_coef"].index


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_torch_result_extraction_uses_model_weights():
    """Full backend result extraction should project fitted torch weights into maps and tables."""
    estimator = CBMREstimator(moderators=["age"], backend="full", device="cpu")
    estimator.groups = ["Default"]
    estimator.inputs_ = {"coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]])}
    estimator.voxelwise_model = models.SpatialCBMRModel(
        groups=estimator.groups,
        spatial_coef_dim=2,
        moderators_coef_dim=1,
        device="cpu",
    )
    with torch.no_grad():
        estimator.voxelwise_model.spatial_coef_linears["Default"].weight[:] = torch.tensor(
            [[0.3, 0.4]], dtype=torch.float64
        )
        estimator.voxelwise_model.moderator_coef_linears["Default"].weight[:] = torch.tensor(
            [[0.1, 0.2]], dtype=torch.float64
        )
    moderators_by_group = {"Default": torch.tensor([[2.0], [4.0]], dtype=torch.float64)}

    maps, tables = estimator._extract_torch_results(moderators_by_group)

    np.testing.assert_allclose(maps["spatialIntensity_group-Default"], np.exp([0.3, 0.4]))
    np.testing.assert_allclose(maps["voxelwiseModeratorEffect_age_group-Default"], [0.3, 0.6])
    pd.testing.assert_frame_equal(
        tables["spatial_regression_coef"],
        pd.DataFrame([[0.3, 0.4]], index=["Default"], columns=["basis_0", "basis_1"]),
    )
    assert ("Default", "age") in tables["voxelwise_moderator_effects_regression_coef"].index


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_exposes_approximate_solver_as_property():
    """The approximate solver accessor should behave like an attribute, not a method."""
    estimator = CBMREstimator(moderators=["age"], backend="approximate")

    assert (
        estimator._voxelwise_cbmr_approximate_solver is cbmr_module.fit_voxelwise_cbmr_approximate
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_fit_dispatches_to_approximate_backend(monkeypatch):
    """The backend option should route CBMREstimator through the approximate solver."""
    estimator = CBMREstimator(
        moderators=["age"],
        backend="approximate",
        n_iter=3,
        tol=1e-4,
        alpha=0.5,
        damping=0.01,
        compute_nll=True,
    )
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "moderators_by_group": {"Default": np.array([[2.0], [4.0]])},
        "foci_by_experiment_voxel": {
            "Default": scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        },
    }
    calls = []

    def fake_fit_voxelwise_cbmr_approximate(
        moderators, bases, foci, tol, max_iter, alpha, damping, compute_nll
    ):
        calls.append(
            {
                "moderators": moderators,
                "bases": bases,
                "foci": foci,
                "tol": tol,
                "max_iter": max_iter,
                "alpha": alpha,
                "damping": damping,
                "compute_nll": compute_nll,
            }
        )
        return np.array([[0.1], [0.2], [0.3], [0.4]])

    monkeypatch.setattr(
        cbmr_module,
        "fit_voxelwise_cbmr_approximate",
        fake_fit_voxelwise_cbmr_approximate,
    )

    maps, tables, description = estimator._fit(dataset=None)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["moderators"], np.array([[2.0, 1.0], [4.0, 1.0]]))
    assert calls[0]["tol"] == estimator.tol
    assert calls[0]["max_iter"] == estimator.n_iter
    assert calls[0]["alpha"] == estimator.alpha
    assert calls[0]["damping"] == estimator.damping
    assert calls[0]["compute_nll"] is True
    assert "spatialIntensity_group-Default" in maps
    assert "spatial_regression_coef" in tables
    assert "approximate" in description


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_fit_dispatches_to_full_backend(monkeypatch):
    """The full backend should route through the torch fitting implementation."""
    estimator = CBMREstimator(moderators=["age"], backend="full")
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "moderators_by_group": {"Default": np.array([[2.0], [4.0]])},
        "foci_by_experiment_voxel": {
            "Default": scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        },
    }
    calls = []

    def fake_fit_full(dataset):
        calls.append(dataset)
        return {"map": np.array([1.0])}, {"table": pd.DataFrame([1.0])}, "full backend"

    monkeypatch.setattr(estimator, "_fit_full", fake_fit_full)

    maps, tables, description = estimator._fit(dataset="dataset")

    assert calls == ["dataset"]
    assert maps["map"].tolist() == [1.0]
    assert "table" in tables
    assert description == "full backend"


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_approximate_solver_returns_finite_coefficients():
    """The approximate solver should return finite coefficient vectors."""
    moderators = np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    bases = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    foci = scipy.sparse.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))

    coefficient = fit_spatial_cbmr_approximate(
        moderators,
        bases,
        foci,
        max_iter=2,
        alpha=1e-3,
        damping=1.0,
    )

    assert coefficient.shape == (moderators.shape[1] * bases.shape[1], 1)
    assert np.all(np.isfinite(coefficient))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_result_get_inference_returns_fitted_engine():
    """The result should expose a result-centered inference workflow."""
    result = _spatial_result_for_inference()

    inference = result.get_inference()

    assert isinstance(inference, CBMRInference)
    assert inference.result is not result
    assert inference.groups == ["Default"]
    assert inference.moderators == ["age"]
    assert inference.method == "sandwich"


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_result_helpers_allow_fisher_information_method():
    """Result-centered inference should let users request FI standard errors."""
    result = _spatial_result_for_inference()

    inference = result.get_inference(method="FI")
    transformed = result.test_groups(method="FI")

    assert inference.method == "FI"
    assert transformed.metadata["voxelwise_cbmr_inference_method"] == "FI"
    assert "z_group-Default" in transformed.maps


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_result_helpers_run_inference():
    """The result should support CBMRResult-style inference helpers."""
    result = _spatial_result_for_inference()

    group_result = result.test_groups()
    moderator_result = result.test_moderators()

    assert isinstance(group_result, CBMRResult)
    assert isinstance(moderator_result, CBMRResult)
    assert "z_group-Default" in group_result.maps
    assert "p_voxelwiseModeratorEffect_age_group-Default" in moderator_result.maps


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_requires_fit_and_spatial_result():
    """The inference object should validate fit state and result type like CBMRInference."""
    inference = CBMRInference(device="cpu")

    with pytest.raises(ValueError, match="has not been fit"):
        inference.create_contrast("Default", source="groups")
    with pytest.raises(TypeError, match="requires a CBMRResult"):
        inference.fit(object())


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_validates_standard_error_options():
    """The inference object should validate sandwich and FI method options."""
    with pytest.raises(ValueError, match="method must be one of"):
        CBMRInference(method="bad-method")
    with pytest.raises(ValueError, match="sandwich_meat"):
        CBMRInference(sandwich_meat="bad-meat")
    with pytest.raises(ValueError, match="sandwich_correction"):
        CBMRInference(sandwich_correction="bad-correction")
    with pytest.raises(ValueError, match="ridge"):
        CBMRInference(ridge=-1.0)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_create_contrast():
    """The inference object should parse named group and moderator contrasts like CBMR."""
    inference = CBMRInference(device="cpu")
    inference.fit(_spatial_result_for_inference())

    group_contrast = inference.create_contrast("Default", source="groups")
    moderator_contrast = inference.create_contrast("age", source="moderators")

    np.testing.assert_array_equal(group_contrast["Default"], np.array([1.0]))
    np.testing.assert_array_equal(moderator_contrast["age"], np.array([1.0]))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_preprocesses_raw_contrasts_like_cbmr():
    """Raw contrast arrays should be two-dimensional, standardized, and deduplicated."""
    inference = CBMRInference(device="cpu")
    inference.fit(_spatial_result_for_inference())
    inference.t_con_moderators = [np.array([2.0]), np.array([2.0])]

    contrasts, names = inference._preprocess_t_con_regressor(source="moderators")

    assert names is None
    assert len(contrasts) == 1
    np.testing.assert_array_equal(contrasts[0], np.array([[1.0]]))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_rejects_wrong_contrast_shape():
    """The inference object should reject contrasts with incorrect regressor width."""
    inference = CBMRInference(device="cpu")
    inference.fit(_spatial_result_for_inference())
    inference.t_con_moderators = [np.array([1.0, 0.0])]

    with pytest.raises(ValueError, match="doesn't match with moderators"):
        inference._preprocess_t_con_regressor(source="moderators")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_transform_adds_maps_without_mutating_input():
    """The inference object should append maps to a copy."""
    result = _spatial_result_for_inference()
    original_map_keys = set(result.maps)
    inference = CBMRInference(device="cpu", ridge=1e-3)

    transformed = inference.fit_transform(
        result,
        t_con_groups="Default",
        t_con_moderators="age",
    )

    assert isinstance(transformed, CBMRResult)
    assert set(result.maps) == original_map_keys
    assert "p_group-Default" in transformed.maps
    assert "z_group-Default" in transformed.maps
    assert "p_voxelwiseModeratorEffect_age_group-Default" in transformed.maps
    assert "z_voxelwiseModeratorEffect_age_group-Default" in transformed.maps
    assert transformed.metadata["voxelwise_cbmr_inference_method"] == "sandwich"
    assert transformed.metadata["voxelwise_cbmr_sandwich_meat"] == "cluster"
    transformed_fi = inference.transform(t_con_groups="Default", method="FI")
    assert transformed_fi.metadata["voxelwise_cbmr_inference_method"] == "FI"
    assert "voxelwise_cbmr_sandwich_meat" not in transformed_fi.metadata
    for map_name in [
        "p_group-Default",
        "z_group-Default",
        "p_voxelwiseModeratorEffect_age_group-Default",
        "z_voxelwiseModeratorEffect_age_group-Default",
    ]:
        assert np.all(np.isfinite(transformed.maps[map_name])), map_name


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_fisher_information_matches_explicit_kron():
    """Kronecker Fisher information should match explicit weighted design construction."""
    moderators = np.array([[1.0, 0.5], [1.0, -0.2], [1.0, 1.2]])
    bases = np.array([[1.0, 0.0], [0.25, 0.75]])
    mean = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.75]])

    expected_rows = []
    expected_weights = []
    for experiment_index in range(moderators.shape[0]):
        for voxel_index in range(bases.shape[0]):
            expected_rows.append(np.kron(moderators[experiment_index], bases[voxel_index]))
            expected_weights.append(mean[experiment_index, voxel_index])
    explicit_design = np.vstack(expected_rows)
    explicit_weights = np.asarray(expected_weights)
    expected = explicit_design.T @ (explicit_design * explicit_weights[:, None])

    actual = CBMRInference._compute_fisher_information(moderators, bases, mean)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_sandwich_covariance_matches_explicit_kron():
    """Sandwich covariance should match explicit Kronecker-design calculations."""
    moderators = np.array([[1.0, 0.5], [1.0, -0.2], [1.0, 1.2]])
    bases = np.array([[1.0, 0.0], [0.25, 0.75]])
    mean = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.75]])
    foci = np.array([[0.0, 2.0], [1.0, 1.0], [3.0, 0.0]])
    ridge = 1e-4

    explicit_rows = []
    explicit_weights = []
    explicit_residuals = []
    for experiment_index in range(moderators.shape[0]):
        for voxel_index in range(bases.shape[0]):
            explicit_rows.append(np.kron(moderators[experiment_index], bases[voxel_index]))
            explicit_weights.append(mean[experiment_index, voxel_index])
            explicit_residuals.append(
                foci[experiment_index, voxel_index] - mean[experiment_index, voxel_index]
            )
    explicit_design = np.vstack(explicit_rows)
    explicit_weights = np.asarray(explicit_weights)
    explicit_residuals = np.asarray(explicit_residuals)
    bread = explicit_design.T @ (explicit_design * explicit_weights[:, None])
    bread_inverse = np.linalg.pinv(bread + ridge * np.eye(bread.shape[0]))
    meat = explicit_design.T @ (explicit_design * explicit_residuals[:, None] ** 2)
    expected = bread_inverse @ meat @ bread_inverse

    actual = CBMRInference._compute_sandwich_covariance(
        moderators,
        bases,
        scipy.sparse.csr_matrix(foci),
        mean,
        ridge=ridge,
        meat="iid",
        correction="hc0",
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
@pytest.mark.parametrize("meat", ["cluster", "iid"])
@pytest.mark.parametrize("correction", ["hc0", "hc1", "hc3"])
def test_spatial_cbmr_sparse_sandwich_covariance_matches_dense(meat, correction):
    """Sparse-response sandwich covariance should match the dense-response path."""
    moderators = np.array([[1.0, 0.5], [1.0, -0.2], [1.0, 1.2]])
    bases = np.array([[1.0, 0.0], [0.25, 0.75], [0.5, 0.5]])
    mean = np.array([[1.0, 2.0, 0.8], [0.5, 1.5, 1.2], [2.0, 0.75, 0.4]])
    foci = np.array([[0.0, 2.0, 0.0], [1.0, 1.0, 0.0], [3.0, 0.0, 1.0]])
    kwargs = {"ridge": 1e-4, "meat": meat, "correction": correction}

    dense = CBMRInference._compute_sandwich_covariance(moderators, bases, foci, mean, **kwargs)
    sparse = CBMRInference._compute_sandwich_covariance(
        moderators,
        bases,
        scipy.sparse.csr_matrix(foci),
        mean,
        **kwargs,
    )

    np.testing.assert_allclose(sparse, dense)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_sandwich_helpers_handle_hc_corrections():
    """Sandwich helpers should apply HC corrections."""
    moderators = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    bases = np.array([[1.0, 0.0], [0.5, 0.5]])
    mean = np.full((3, 2), 0.5)
    residuals = np.array([[0.5, -0.5], [0.2, -0.2], [0.1, -0.1]])
    fisher_info = CBMRInference._compute_fisher_information(moderators, bases, mean)
    bread_inverse = CBMRInference._sandwich_bread_inverse(fisher_info, ridge=1e-4)

    hc1_residuals, hc1_factor = CBMRInference._apply_sandwich_correction(
        "hc1",
        bread_inverse,
        moderators,
        bases,
        mean,
        residuals,
    )
    hc3_residuals, hc3_factor = CBMRInference._apply_sandwich_correction(
        "hc3",
        bread_inverse,
        moderators,
        bases,
        mean,
        residuals,
    )

    np.testing.assert_array_equal(hc1_residuals, residuals)
    assert hc1_factor == pytest.approx(3.0)
    assert hc3_factor == pytest.approx(1.0)
    assert hc3_residuals.shape == residuals.shape
    assert np.all(np.isfinite(hc3_residuals))
    with pytest.raises(ValueError, match="HC1 sandwich correction requires more experiments"):
        CBMRInference._apply_sandwich_correction(
            "hc1",
            bread_inverse,
            moderators[:2],
            bases,
            mean[:2],
            residuals[:2],
        )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_spatial_statistics_match_manual_wald():
    """Voxel-wise single-contrast Wald statistics should match a manual calculation."""
    coefficient = np.array([[0.1, 0.2], [0.3, 0.4]])
    covariance = np.diag([0.5, 0.25, 0.75, 0.5])
    contrast = np.array([[1.0, 0.0]])
    bases = np.array([[1.0, 0.0], [0.5, 0.5]])

    stats = CBMRInference._compute_spatial_coefficient_statistics(
        coefficient,
        covariance,
        contrast,
        bases,
    )

    expected_eta = contrast @ coefficient @ bases.T
    expected_var = np.array([0.5, 0.1875])
    expected_z = expected_eta.ravel() / np.sqrt(expected_var)
    expected_p = 2.0 * scipy.stats.norm.sf(np.abs(expected_z))

    assert stats["chi_square"] is None
    np.testing.assert_allclose(stats["z"], expected_z)
    np.testing.assert_allclose(stats["p"], expected_p)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_spatial_cbmr_inference_chi_square_log_intensity_matches_legacy_loop():
    """Vectorized group GLH chi-square statistics should match a direct voxel loop."""
    simp_con_group = np.array([[1.0, -1.0], [0.5, 0.5]])
    contrast_log_intensity = np.array(
        [[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]],
        dtype=float,
    )
    cov_log_intensity = np.array(
        [
            [2.0, 1.5, 1.2],
            [0.3, 0.2, 0.1],
            [0.3, 0.2, 0.1],
            [1.8, 1.1, 1.6],
        ],
        dtype=float,
    )

    expected = []
    for voxel_index in range(contrast_log_intensity.shape[1]):
        contrast_vector = contrast_log_intensity[:, voxel_index].reshape(2, 1)
        covariance = cov_log_intensity[:, voxel_index].reshape(2, 2)
        projected_covariance = simp_con_group @ covariance @ simp_con_group.T
        expected.append(
            (contrast_vector.T @ np.linalg.inv(projected_covariance) @ contrast_vector).item()
        )

    actual = CBMRInference._chi_square_log_intensity(
        n_voxels=contrast_log_intensity.shape[1],
        n_involved_groups=2,
        simp_con_group=simp_con_group,
        cov_log_intensity=cov_log_intensity,
        contrast_log_intensity=contrast_log_intensity,
    )

    np.testing.assert_allclose(actual, np.asarray(expected))
