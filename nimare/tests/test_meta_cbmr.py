"""Tests for CBMR meta-analytic methods."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    warnings.warn("Torch not installed. CBMR tests will be skipped.")
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta import models
    from nimare.meta.cbmr import CBMREstimator, CBMRInference, CBMRResult

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
    assert cbmr_result.available_groups()


def test_cbmr_result_interface_lists_inference_inputs(cbmr_result):
    """CBMR results should expose discoverable group and moderator names."""
    description = cbmr_result.describe_inference_inputs()

    assert description["groups"] == cbmr_result.available_groups()
    assert description["moderators"] == cbmr_result.available_moderators()


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
    )
    result = cbmr.fit(dataset=dset)

    assert cbmr.model.spatial_coef_linears[cbmr.groups[0]].weight.device.type == "cuda"
    if cbmr.model.moderators_coef_dim:
        assert cbmr.model.moderators_linear.weight.device.type == "cuda"

    inference = CBMRInference(device="cuda")
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

    inference = CBMRInference(device="cpu")
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
    inference = CBMRInference(device="cpu")
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
    np.testing.assert_allclose(analytic_spatial, generic_spatial)

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
    np.testing.assert_allclose(analytic_moderator, generic_moderator)


def test_cbmr_inference_multi_contrast_matches_individual_transforms(cbmr_result):
    """Cached multi-contrast inference should match independent contrast evaluations."""
    inference = CBMRInference(device="cpu")
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
        single_inference = CBMRInference(device="cpu")
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
        single_inference = CBMRInference(device="cpu")
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
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)


def test_moderators_none(testdata_cbmr_simulated):
    """Unit test for Firth penalty."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )
    cbmr = CBMREstimator(
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
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)
    inference = CBMRInference(device="cpu")
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
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        model=models.PoissonEstimator,
        generate_description=False,
        lr=1,
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

    if cbmr.iter == 0:
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
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        model=models.PoissonEstimator,
        generate_description=False,
        lr=1,
    )

    cbmr._collect_inputs(dset, drop_invalid=True)
    cbmr._preprocess_input(dset)

    for group in cbmr.groups:
        n_experiments = len(cbmr.inputs_["ids_by_group"][group])
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
        group_categories=["diagnosis", "drug_status"],
        model=models.PoissonEstimator,
        generate_description=False,
        lr=1,
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
