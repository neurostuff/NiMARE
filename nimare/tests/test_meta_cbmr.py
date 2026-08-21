"""Tests for CBMR meta-analytic methods."""

import logging
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import scipy
import scipy.sparse
import scipy.special

try:
    import torch
except ImportError:
    warnings.warn("Torch not installed. CBMR tests will be skipped.", stacklevel=2)
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
    from nimare.meta.utils import fit_voxelwise_cbmr_approximate

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.transforms import StandardizeField

# numba has a lot of debug messages that are not useful for testing
logging.getLogger("numba").setLevel(logging.WARNING)
# indexed_gzip has a few debug messages that are not useful for testing
logging.getLogger("indexed_gzip").setLevel(logging.WARNING)

CBMR_STANDARDIZE_FIELDS = ("sample_sizes", "avg_age", "schizophrenia_subtype")
CBMR_GROUP_CATEGORIES = ("diagnosis", "drug_status")
CBMR_MODERATORS = (
    "standardized_sample_sizes",
    "standardized_avg_age",
    "schizophrenia_subtype",
)


def _standardize_cbmr_dataset(dataset):
    """Standardize the fields used by most CBMR tests."""
    return StandardizeField(fields=list(CBMR_STANDARDIZE_FIELDS)).transform(dataset)


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
    dset = _standardize_cbmr_dataset(testdata_cbmr_simulated)

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=list(CBMR_GROUP_CATEGORIES),
        moderators=list(CBMR_MODERATORS),
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


def _assert_frame_equal_by_block(left, right):
    """Assert two frames match, comparing their values as one block.

    Every check :func:`pandas.testing.assert_frame_equal` makes is made below -- axes,
    axis names and dtypes, then the values -- but it makes them column by column. These
    frames carry one row per group and one column per voxel, a few hundred thousand of them,
    so that walk costs about thirty seconds per table: several times more than fitting the
    model the table summarizes. The values are compared exactly rather than within a
    tolerance, so this is the stricter of the two.
    """
    for axis_name, left_axis, right_axis in (
        ("index", left.index, right.index),
        ("columns", left.columns, right.columns),
    ):
        assert left_axis.name == right_axis.name, axis_name
        assert left_axis.dtype == right_axis.dtype, axis_name
        assert left_axis.equals(right_axis), axis_name

    assert left.dtypes.equals(right.dtypes)
    np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())


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
    dset = _standardize_cbmr_dataset(testdata_cbmr_simulated)
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


def test_cbmr_result_get_inference_inherits_incidence_threshold(cbmr_result, monkeypatch):
    """Result-centered inference should inherit the fitted estimator's incidence threshold."""

    class DummyInference:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.incidence_threshold = kwargs["incidence_threshold"]

        def fit(self, result):
            self.fit_result = result

    # CBMRResult.get_inference imports this at call time to avoid an import cycle, so the
    # patch has to land on the defining module.
    monkeypatch.setattr("nimare.meta.cbmr.inference.CBMRInference", DummyInference)

    result = cbmr_result.copy()
    result.estimator.incidence_threshold = None

    inherited = result.get_inference()
    overridden = result.get_inference(incidence_threshold=0.0)

    assert inherited.incidence_threshold is None
    assert inherited.fit_result is result
    assert overridden.incidence_threshold == 0.0


def test_cbmr_inference_fit_inherits_incidence_threshold(cbmr_result):
    """Direct CBMRInference.fit should inherit the fitted estimator's incidence threshold."""
    result = cbmr_result.copy()
    result.estimator.incidence_threshold = None

    inherited = CBMRInference(device="cpu", moderator_effect="global")
    inherited.fit(result)
    overridden = CBMRInference(
        device="cpu",
        moderator_effect="global",
        incidence_threshold=0.0,
    )

    assert inherited.incidence_threshold is None
    assert overridden.incidence_threshold == 0.0


def test_cbmr_fit_is_repeatable(testdata_cbmr_simulated):
    """Repeated CBMR fits on the same dataset should return identical results."""
    dset = testdata_cbmr_simulated.slice(testdata_cbmr_simulated.ids[:60])
    dset = _standardize_cbmr_dataset(dset)

    def _fit_once():
        cbmr = CBMREstimator(
            moderator_effect="global",
            group_categories=list(CBMR_GROUP_CATEGORIES),
            moderators=list(CBMR_MODERATORS),
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
    dset = _standardize_cbmr_dataset(dset)

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=list(CBMR_GROUP_CATEGORIES),
        moderators=list(CBMR_MODERATORS),
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
    dset = _standardize_cbmr_dataset(testdata_cbmr_simulated)

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=list(CBMR_GROUP_CATEGORIES),
        moderators=list(CBMR_MODERATORS),
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
        _assert_frame_equal_by_block(cbmr_result.tables[table_name], legacy_table)


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
@pytest.mark.parametrize("statistic", [10.0, np.sqrt(5000.0)])
def test_cbmr_group_glh_evaluates_the_chi_square_tail_in_log_space(statistic):
    """The GLH tail must stay finite past the chi-square whose p-value underflows.

    ``chi2.sf`` is exactly zero from a chi-square of about 1416 on, which used to give a
    p-value of zero and a z-statistic clamped to 10.
    """
    inference = CBMRInference(device="cpu", moderator_effect="global")

    chi_square, z_values, nlogp_values = inference._compute_group_glh_statistics(
        simp_con_group=np.array([[1.0]]),
        involved_groups=["group"],
        cov_spatial_coef=np.array([[1.0]]),
        contrast_log_intensity=np.array([[statistic]]),
        X=np.array([[1.0]]),
        spatial_coef_dim=1,
        n_brain_voxel=1,
        is_homogeneity_test=False,
    )

    np.testing.assert_allclose(chi_square, [statistic**2])
    # The exact one-dof upper tail, computed independently of the implementation.
    expected = np.log(2.0) + scipy.special.log_ndtr(-np.abs(statistic))
    np.testing.assert_allclose(nlogp_values, expected, rtol=1e-6)
    assert np.isfinite(z_values[0])
    if statistic > 30:
        assert z_values[0] > 38.5, "z used to be clamped to 10 here"


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
    dset = _standardize_cbmr_dataset(testdata_cbmr_simulated)
    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=list(CBMR_GROUP_CATEGORIES),
        moderators=list(CBMR_MODERATORS),
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
    """CBMR should fit and run group inference when moderators are omitted."""
    dset = _standardize_cbmr_dataset(testdata_cbmr_simulated)
    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=list(CBMR_GROUP_CATEGORIES),
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


def test_cbmr_estimator_update(testdata_cbmr_simulated):
    """Unit test for CBMR estimator update function."""
    testdata_cbmr_simulated = _standardize_cbmr_dataset(testdata_cbmr_simulated)
    cbmr = CBMREstimator(
        moderator_effect="global",
        moderators=list(CBMR_MODERATORS),
        model=models.PoissonEstimator,
        generate_description=False,
        lr=1,
        random_state=100,
    )

    cbmr._collect_inputs(testdata_cbmr_simulated, drop_invalid=True)
    cbmr._preprocess_input(testdata_cbmr_simulated)

    # Fit the model.
    init_weight_kwargs = {
        "groups": cbmr.groups,
        "moderators": cbmr.moderators,
        "spatial_coef_dim": cbmr.inputs_["coef_spline_bases"].shape[1],
        "moderators_coef_dim": len(cbmr.moderators) if cbmr.moderators else None,
    }

    cbmr.model.init_weights(**init_weight_kwargs)
    optimizer = torch.optim.LBFGS(cbmr.model.parameters(), cbmr.lr)
    # Load dataset information into tensors.
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

    prev_loss = torch.tensor(float("inf"))

    cbmr.model._update(
        optimizer,
        torch.tensor(cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device),
        moderators_by_group_tensor,
        foci_per_voxel_tensor,
        foci_per_experiment_tensor,
        prev_loss,
    )
    # Deliberately set the first spatial coefficient to NaN.
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
    dset = _standardize_cbmr_dataset(testdata_cbmr_simulated)
    target_id = dset.annotations.iloc[0]["id"]
    dset.coordinates.loc[
        dset.coordinates["id"] == target_id,
        ["x", "y", "z"],
    ] = 10_000

    cbmr = CBMREstimator(
        moderator_effect="global",
        group_categories=list(CBMR_GROUP_CATEGORIES),
        moderators=list(CBMR_MODERATORS),
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
        group_categories=list(CBMR_GROUP_CATEGORIES),
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


def test_standardize_field(testdata_cbmr_simulated):
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

    # Clear by prefix, not by name. cbmr is a package, so leaving its submodules behind
    # would let a later import bind a fresh parent to stale children -- which breaks any
    # monkeypatch that addresses those submodules by dotted path.
    stale = [
        name
        for name in sys.modules
        if name == "nimare.meta" or name.startswith(("nimare.meta.cbmr", "nimare.meta.models"))
    ]
    saved = {name: sys.modules[name] for name in stale}
    saved_attr = nimare.__dict__.pop("meta", None)
    for name in stale:
        del sys.modules[name]

    try:
        meta = importlib.import_module("nimare.meta")

        assert not [n for n in sys.modules if n.startswith("nimare.meta.cbmr")]
        assert "nimare.meta.models" not in sys.modules
        assert hasattr(meta, "ALE")
    finally:
        # Put the original module objects back, so tests that already hold references to
        # classes from them keep patching and comparing against the same objects.
        for name in [n for n in sys.modules if n.startswith("nimare.meta")]:
            if name not in saved:
                del sys.modules[name]
        sys.modules.update(saved)
        if saved_attr is not None:
            nimare.__dict__["meta"] = saved_attr


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


def _voxelwise_result_for_inference():
    """Return a small fitted voxelwise CBMR result suitable for inference tests."""
    estimator = CBMREstimator(
        moderator_effect="voxelwise",
        moderators=["age"],
        backend="approximate",
    )
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
        mask=nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4)),
        maps=maps,
        tables=tables,
        description="voxelwise inference test",
    )


def _mixed_result_for_inference():
    """Return a small fitted mixed CBMR result suitable for inference tests."""
    estimator = CBMREstimator(
        moderator_effect="mixed",
        global_moderators=["sample_size"],
        voxelwise_moderators=["age"],
    )
    estimator.groups = ["Default"]
    estimator.moderators = ["sample_size", "age"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "global_moderators_by_group": {"Default": np.array([[1.0], [2.0]])},
        "voxelwise_moderators_by_group": {"Default": np.array([[2.0], [4.0]])},
        "foci_by_experiment": {
            "Default": scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        },
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
        estimator.inputs_["voxelwise_moderators_by_group"]["Default"],
        estimator.voxelwise_coef["Default"],
    )
    tables["global_moderators_regression_coef"] = pd.DataFrame(
        [[0.05]],
        columns=["sample_size"],
    )
    return CBMRResult(
        estimator=estimator,
        mask=nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4)),
        maps=maps,
        tables=tables,
        description="mixed inference test",
    )


def _global_result_for_sandwich_inference():
    """Return a small fitted global Poisson CBMR result suitable for sandwich tests."""
    estimator = CBMREstimator(
        moderator_effect="global",
        moderators=["age"],
        model=models.PoissonEstimator,
    )
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "moderators_by_group": {"Default": np.array([[0.0], [1.0], [2.0]])},
        "foci_per_voxel": {"Default": np.array([3.0, 5.0])},
        "foci_per_experiment": {"Default": np.array([2.0, 3.0, 3.0])},
    }
    maps = {"spatialIntensity_group-Default": np.array([1.0, 1.5])}
    tables = {
        "spatial_regression_coef": pd.DataFrame([[0.0, np.log(1.5)]], index=["Default"]),
        "moderators_regression_coef": pd.DataFrame([[0.1]], columns=["age"]),
    }
    return CBMRResult(
        estimator=estimator,
        mask=nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4)),
        maps=maps,
        tables=tables,
        description="global sandwich inference test",
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
@pytest.mark.parametrize(
    ("estimator_kwargs", "inference_kwargs", "expected_moderator_effect"),
    [
        ({}, {}, "global"),
        ({"moderator_effect": "global"}, {"moderator_effect": "global"}, "global"),
        (
            {
                "moderator_effect": "mixed",
                "global_moderators": ["sample_size"],
                "voxelwise_moderators": ["age"],
            },
            {"moderator_effect": "mixed"},
            "mixed",
        ),
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
def test_mixed_cbmr_api_separates_global_and_voxelwise_moderators():
    """Mixed CBMR should expose distinct global and voxelwise moderator sets."""
    estimator = CBMREstimator(
        moderator_effect="mixed",
        global_moderators=["sample_size"],
        voxelwise_moderators=["age"],
    )

    assert estimator.global_moderators == ["sample_size"]
    assert estimator.voxelwise_moderators == ["age"]
    assert estimator.moderators == ["sample_size", "age"]

    result = _mixed_result_for_inference()
    description = result.describe_inference_inputs()

    assert description["moderator_effect"] == "mixed"
    assert description["global_moderators"] == ("sample_size",)
    assert description["voxelwise_moderators"] == ("age",)

    with pytest.raises(ValueError, match="both global and voxelwise"):
        CBMREstimator(
            moderator_effect="mixed",
            global_moderators=["age"],
            voxelwise_moderators=["age"],
        )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_mixed_cbmr_inference_uses_joint_covariance_blocks():
    """Mixed inference should slice global and voxelwise blocks from one joint covariance."""
    result = _mixed_result_for_inference()
    inference = CBMRInference(moderator_effect="mixed", method="FI")
    inference.fit(result)

    joint_covariance = inference._get_mixed_joint_covariance()
    global_covariance, _ = inference._get_moderator_covariance()
    group_covariance = inference._get_group_covariance("Default")
    global_slice, group_slices, _, _, _ = inference._mixed_joint_parameter_layout()
    group_slice = group_slices["Default"]

    np.testing.assert_allclose(global_covariance, joint_covariance[global_slice, global_slice])
    np.testing.assert_allclose(group_covariance, joint_covariance[group_slice, group_slice])
    assert np.any(np.abs(joint_covariance[global_slice, group_slice]) > 0)

    bases = result.estimator.inputs_["coef_spline_bases"]
    local_moderators = inference._get_group_augmented_moderators("Default")
    mean = inference._get_group_mean("Default")
    independent_covariance = inference._sandwich_bread_inverse(
        inference._compute_fisher_information(local_moderators, bases, mean),
        inference.ridge,
    )
    assert not np.allclose(group_covariance, independent_covariance)

    sandwich_inference = CBMRInference(
        moderator_effect="mixed",
        method="sandwich",
        sandwich_correction="hc0",
    )
    sandwich_inference.fit(result)
    sandwich_joint_covariance = sandwich_inference._get_mixed_joint_covariance()
    assert sandwich_joint_covariance.shape == joint_covariance.shape
    sandwich_result = sandwich_inference.transform(t_con_moderators=["sample_size", "age"])
    assert "p_sample_size" in sandwich_result.tables
    assert "p_voxelwiseModeratorEffect_age_group-Default" in sandwich_result.maps


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_mode_specific_moderators_are_not_silently_ignored():
    """Mode-specific moderator arguments should be honored or rejected explicitly."""
    global_estimator = CBMREstimator(global_moderators=["sample_size"])
    voxelwise_estimator = CBMREstimator(
        moderator_effect="voxelwise",
        voxelwise_moderators=["age"],
    )

    assert global_estimator.moderators == ["sample_size"]
    assert voxelwise_estimator.moderators == ["age"]

    with pytest.raises(ValueError, match="voxelwise_moderators"):
        CBMREstimator(voxelwise_moderators=["age"])
    with pytest.raises(ValueError, match="global_moderators"):
        CBMREstimator(moderator_effect="voxelwise", global_moderators=["sample_size"])
    with pytest.raises(ValueError, match="moderators or global_moderators"):
        CBMREstimator(moderators=["age"], global_moderators=["sample_size"])
    with pytest.raises(ValueError, match="moderators or voxelwise_moderators"):
        CBMREstimator(
            moderator_effect="voxelwise",
            moderators=["sample_size"],
            voxelwise_moderators=["age"],
        )


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
def test_cbmr_incidence_threshold_drops_low_incidence_voxels():
    """Incidence filtering should drop voxels whose empirical rate is <= threshold."""
    estimator = CBMREstimator(moderator_effect="global", incidence_threshold=0.25)
    estimator.inputs_ = {}
    mask_data = np.ones((3, 1, 1), dtype=bool)
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), np.eye(4))
    filtered_coordinates = pd.DataFrame(
        {
            "id": ["study-1", "study-2", "study-3"],
            "_cbmr_mask_index": [0, 0, 1],
        }
    )
    ids_by_group = {"Default": ["study-1", "study-2", "study-3", "study-4"]}

    thresholded_img = estimator._threshold_mask_by_incidence(
        mask_img,
        mask_data,
        filtered_coordinates,
        ids_by_group,
        n_mask_voxels=3,
    )

    np.testing.assert_array_equal(np.asanyarray(thresholded_img.dataobj).ravel(), [1, 0, 0])
    np.testing.assert_allclose(estimator.inputs_["empirical_incidence_rate_roi"], [0.5, 0.25, 0.0])
    np.testing.assert_allclose(estimator.inputs_["empirical_incidence_rate"], [0.5])


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_inference_can_restrict_fitted_result_by_incidence():
    """Inference-level incidence filtering should subset maps, masks, and voxel inputs."""
    estimator = CBMREstimator(
        moderator_effect="voxelwise",
        moderators=["age"],
        backend="approximate",
    )
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "empirical_incidence_rate": np.array([0.002, 0.001]),
        "moderators_by_group": {"Default": np.array([[2.0], [4.0]])},
        "foci_by_experiment_voxel": {
            "Default": scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        },
    }
    result = CBMRResult(
        estimator=estimator,
        mask=nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4)),
        maps={"spatialIntensity_group-Default": np.array([1.0, 2.0])},
        tables={"spatial_regression_coef": pd.DataFrame([[0.0, 0.0]], index=["Default"])},
        description="incidence inference test",
    )

    inference = CBMRInference(moderator_effect="voxelwise", incidence_threshold=0.001)
    inference.fit(result)

    np.testing.assert_array_equal(inference.result.maps["spatialIntensity_group-Default"], [1.0])
    assert int(np.asanyarray(inference.result.masker.mask_img.dataobj).sum()) == 1
    assert inference.estimator.inputs_["coef_spline_bases"].shape == (1, 2)
    assert inference.estimator.inputs_["foci_by_experiment_voxel"]["Default"].shape == (2, 1)


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
        group_categories=list(CBMR_GROUP_CATEGORIES),
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
        moderator_effect="voxelwise",
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
def test_cbmr_inference_accepts_sandwich_options_for_global_pipeline():
    """Global inference should expose sandwich covariance options."""
    inference = CBMRInference(
        moderator_effect="global",
        method="sandwich",
        sandwich_meat="iid",
        sandwich_correction="hc0",
        ridge=1e-3,
    )

    assert inference.moderator_effect == "global"
    assert inference.method == "sandwich"
    assert inference.sandwich_meat == "iid"
    assert inference.sandwich_correction == "hc0"
    assert inference.ridge == pytest.approx(1e-3)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_transform_method_switch_preserves_fitted_coefficients():
    """Changing covariance methods should not erase fitted coefficient tables."""
    result = _global_result_for_sandwich_inference()
    inference = CBMRInference(moderator_effect="global", method="FI")
    inference.fit(result)

    np.testing.assert_allclose(inference._moderator_coef_table, [[0.1]])

    inference.transform(t_con_groups=False, t_con_moderators=True, method="sandwich")

    assert inference.method == "sandwich"
    np.testing.assert_allclose(inference._moderator_coef_table, [[0.1]])
    assert "z_age" in inference.result.tables


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_cbmr_moderator_effect_validation_is_shared_by_estimator_and_inference():
    """Both public entry points should reject unsupported moderator-effect values."""
    with pytest.raises(ValueError, match="moderator_effect must be one of"):
        CBMREstimator(moderator_effect="bad-effect")
    with pytest.raises(ValueError, match="moderator_effect must be one of"):
        CBMRInference(moderator_effect="bad-effect")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_invalid_backend_raises():
    """The estimator should validate backend names at initialization."""
    with pytest.raises(ValueError, match="backend must be one of"):
        CBMREstimator(backend="bad-backend")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_build_group_foci_matrices_counts_foci():
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
def test_voxelwise_cbmr_build_group_foci_matrices_handles_empty_coordinates():
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
def test_voxelwise_cbmr_prepare_torch_inputs_densifies_sparse_matrices():
    """Torch backend inputs should be float64 tensors on the estimator device."""
    estimator = CBMREstimator(moderator_effect="voxelwise", moderators=["age"], device="cpu")
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.5, 0.5]]),
        "moderators_by_group": {"Default": np.array([[0.2], [1.0]])},
        "foci_by_experiment_voxel": {
            "Default": scipy.sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
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
def test_voxelwise_cbmr_result_helpers_preserve_result_type_and_summarize_maps():
    """The result should retain CBMRResult behavior plus voxelwise helper methods."""
    result = CBMRResult(
        estimator=object(),
        mask=_mask_img(),
        maps={
            "voxelwiseModeratorEffect_age_group-Default": np.array([1.0, 2.0, 3.0]),
            "spatialIntensity_group-Default": np.array([4.0, 5.0, 6.0]),
        },
        tables={"spatial_regression_coef": pd.DataFrame([[1.0, 2.0]], index=["Default"])},
        description="voxelwise test",
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
def test_voxelwise_cbmr_output_tables_match_cbmr_group_table_convention():
    """Voxelwise CBMR summaries should expose aggregate CBMR-style output tables."""
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
def test_voxelwise_cbmr_add_approximate_results_creates_expected_maps_and_tables():
    """Approximate backend result extraction should create finite maps and CBMR-style tables."""
    estimator = CBMREstimator(
        moderator_effect="voxelwise",
        moderators=["age"],
        backend="approximate",
    )
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
def test_voxelwise_cbmr_torch_result_extraction_uses_model_weights():
    """Full backend result extraction should project fitted torch weights into maps and tables."""
    estimator = CBMREstimator(
        moderator_effect="voxelwise",
        moderators=["age"],
        backend="full",
        device="cpu",
    )
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
def test_voxelwise_cbmr_exposes_approximate_solver_as_property():
    """The approximate solver accessor should behave like an attribute, not a method."""
    estimator = CBMREstimator(
        moderator_effect="voxelwise",
        moderators=["age"],
        backend="approximate",
    )

    assert (
        estimator._voxelwise_cbmr_approximate_solver is cbmr_module.fit_voxelwise_cbmr_approximate
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_fit_dispatches_to_approximate_backend(monkeypatch):
    """The backend option should route CBMREstimator through the approximate solver."""
    estimator = CBMREstimator(
        moderator_effect="voxelwise",
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

    # Patch where the estimator looks the solver up, not the package that re-exports it.
    monkeypatch.setattr(
        "nimare.meta.cbmr.estimator.fit_voxelwise_cbmr_approximate",
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
def test_voxelwise_cbmr_fit_dispatches_to_full_backend(monkeypatch):
    """The full backend should route through the torch fitting implementation."""
    estimator = CBMREstimator(moderator_effect="voxelwise", moderators=["age"], backend="full")
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
def test_voxelwise_cbmr_approximate_solver_returns_finite_coefficients():
    """The approximate solver should return finite coefficient vectors."""
    moderators = np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    bases = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    foci = scipy.sparse.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))

    coefficient = fit_voxelwise_cbmr_approximate(
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
def test_voxelwise_cbmr_result_get_inference_returns_fitted_engine():
    """The result should expose a result-centered inference workflow."""
    result = _voxelwise_result_for_inference()

    inference = result.get_inference()

    assert isinstance(inference, CBMRInference)
    assert inference.result is not result
    assert inference.groups == ["Default"]
    assert inference.moderators == ["age"]
    assert inference.method == "sandwich"


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_result_helpers_allow_fisher_information_method():
    """Result-centered inference should let users request FI standard errors."""
    result = _voxelwise_result_for_inference()

    inference = result.get_inference(method="FI")
    transformed = result.test_groups(method="FI")

    assert inference.method == "FI"
    assert transformed.metadata["voxelwise_cbmr_inference_method"] == "FI"
    assert "z_group-Default" in transformed.maps


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_result_helpers_allow_sandwich_method_options():
    """Result-centered inference should let users request voxelwise sandwich standard errors."""
    result = _voxelwise_result_for_inference()

    transformed = result.test_moderators(
        method="sandwich",
        sandwich_meat="iid",
        sandwich_correction="hc0",
        ridge=1e-4,
    )

    assert transformed.metadata["voxelwise_cbmr_inference_method"] == "sandwich"
    assert transformed.metadata["voxelwise_cbmr_sandwich_meat"] == "iid"
    assert transformed.metadata["voxelwise_cbmr_sandwich_correction"] == "hc0"
    assert "z_voxelwiseModeratorEffect_age_group-Default" in transformed.maps


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_global_cbmr_result_helpers_allow_sandwich_method_options():
    """Result-centered inference should let users request global sandwich standard errors."""
    result = _global_result_for_sandwich_inference()

    group_result = result.test_groups(
        method="sandwich",
        sandwich_meat="iid",
        sandwich_correction="hc0",
        ridge=1e-4,
    )
    moderator_result = result.test_moderators(
        method="sandwich",
        sandwich_meat="iid",
        sandwich_correction="hc0",
        ridge=1e-4,
    )

    assert group_result.metadata["global_cbmr_inference_method"] == "sandwich"
    assert group_result.metadata["global_cbmr_sandwich_meat"] == "iid"
    assert group_result.metadata["global_cbmr_sandwich_correction"] == "hc0"
    assert "z_group-Default" in group_result.maps
    assert "z_age" in moderator_result.tables
    assert np.all(np.isfinite(group_result.maps["z_group-Default"]))
    assert np.all(np.isfinite(moderator_result.tables["z_age"].to_numpy()))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_global_cbmr_sandwich_requires_poisson_model():
    """Global sandwich should reject models with different score equations."""
    result = _global_result_for_sandwich_inference()
    result.estimator.model = models.NegativeBinomialEstimator()
    inference = CBMRInference(moderator_effect="global", method="sandwich")

    with pytest.raises(ValueError, match="Global sandwich inference"):
        inference.fit(result)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_result_helpers_run_inference():
    """The result should support CBMRResult-style inference helpers."""
    result = _voxelwise_result_for_inference()

    group_result = result.test_groups()
    moderator_result = result.test_moderators()

    assert isinstance(group_result, CBMRResult)
    assert isinstance(moderator_result, CBMRResult)
    assert "z_group-Default" in group_result.maps
    assert "p_voxelwiseModeratorEffect_age_group-Default" in moderator_result.maps


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_mixed_cbmr_inference_routes_moderator_types_separately():
    """Mixed inference should return tables for global and maps for voxelwise moderators."""
    result = _mixed_result_for_inference()

    inference = CBMRInference(moderator_effect="mixed", method="FI", ridge=1e-3)
    transformed = inference.fit_transform(
        result,
        t_con_moderators=["sample_size", "age"],
    )

    assert "z_sample_size" in transformed.tables
    assert "p_sample_size" in transformed.tables
    assert "z_voxelwiseModeratorEffect_age_group-Default" in transformed.maps
    assert "p_voxelwiseModeratorEffect_age_group-Default" in transformed.maps
    assert transformed.metadata["global_cbmr_inference_method"] == "FI"
    assert transformed.metadata["voxelwise_cbmr_inference_method"] == "FI"

    helper_result = result.test_moderators(method="FI", ridge=1e-3)
    assert "p_sample_size" in helper_result.tables
    assert "p_voxelwiseModeratorEffect_age_group-Default" in helper_result.maps

    with pytest.raises(ValueError, match="cannot combine global and voxelwise"):
        inference.transform(t_con_moderators=[np.array([1.0, 1.0])])


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_requires_fit_and_cbmr_result():
    """The inference object should validate fit state and result type like CBMRInference."""
    inference = CBMRInference(device="cpu", moderator_effect="voxelwise")

    with pytest.raises(ValueError, match="has not been fit"):
        inference.create_contrast("Default", source="groups")
    with pytest.raises(TypeError, match="requires a CBMRResult"):
        inference.fit(object())


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_validates_standard_error_options():
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
def test_voxelwise_cbmr_inference_create_contrast():
    """The inference object should parse named group and moderator contrasts like CBMR."""
    inference = CBMRInference(device="cpu", moderator_effect="voxelwise")
    inference.fit(_voxelwise_result_for_inference())

    group_contrast = inference.create_contrast("Default", source="groups")
    moderator_contrast = inference.create_contrast("age", source="moderators")

    np.testing.assert_array_equal(group_contrast["Default"], np.array([1.0]))
    np.testing.assert_array_equal(moderator_contrast["age"], np.array([1.0]))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_preprocesses_raw_contrasts_like_cbmr():
    """Raw contrast arrays should be two-dimensional, standardized, and deduplicated."""
    inference = CBMRInference(device="cpu", moderator_effect="voxelwise")
    inference.fit(_voxelwise_result_for_inference())
    inference.t_con_moderators = [np.array([2.0]), np.array([2.0])]

    contrasts, names = inference._preprocess_t_con_regressor(source="moderators")

    assert names is None
    assert len(contrasts) == 1
    np.testing.assert_array_equal(contrasts[0], np.array([[1.0]]))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_rejects_wrong_contrast_shape():
    """The inference object should reject contrasts with incorrect regressor width."""
    inference = CBMRInference(device="cpu", moderator_effect="voxelwise")
    inference.fit(_voxelwise_result_for_inference())
    inference.t_con_moderators = [np.array([1.0, 0.0])]

    with pytest.raises(ValueError, match="doesn't match with moderators"):
        inference._preprocess_t_con_regressor(source="moderators")


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_generates_moderator_effect_diagnostic_maps():
    """Voxelwise moderator-effect diagnostics should generate per-unit RI and ID maps."""
    inference = CBMRInference(device="cpu", moderator_effect="voxelwise")
    inference.fit(_voxelwise_result_for_inference())

    diagnostic_result = inference.generate_voxelwise_moderator_effect_maps(
        moderators="age",
        groups="Default",
        unit_change=2.0,
    )

    relative_key = "relativeIntensity_voxelwiseModeratorEffect_age_unit-2_group-Default"
    difference_key = "intensityDifference_voxelwiseModeratorEffect_age_unit-2_group-Default"
    expected_relative = np.exp([0.2, 0.4])
    expected_difference = np.exp([0.3, 0.4]) * (expected_relative - 1.0)

    assert relative_key in diagnostic_result.maps
    assert difference_key in diagnostic_result.maps
    np.testing.assert_allclose(diagnostic_result.maps[relative_key], expected_relative)
    np.testing.assert_allclose(diagnostic_result.maps[difference_key], expected_difference)
    assert diagnostic_result.metadata["voxelwise_moderator_effect_diagnostic_unit_change"] == 2.0


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_result_exposes_moderator_effect_diagnostic_workflow():
    """Users should be able to run RI/ID diagnostics from a fitted voxelwise result."""
    result = _voxelwise_result_for_inference()
    inference = result.get_inference(method="FI")

    diagnostic_result = inference.generate_voxelwise_moderator_effect_maps(
        moderators=["age"],
        groups="Default",
        unit_change=1.0,
    )

    assert inference.method == "FI"
    assert "relativeIntensity_voxelwiseModeratorEffect_age_unit-1_group-Default" in (
        diagnostic_result.metadata["voxelwise_moderator_effect_diagnostic_maps"]
    )
    assert "intensityDifference_voxelwiseModeratorEffect_age_unit-1_group-Default" in (
        diagnostic_result.metadata["voxelwise_moderator_effect_diagnostic_maps"]
    )


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_plots_moderator_effect_diagnostics():
    """Voxelwise moderator-effect diagnostics should plot RI within an ID-thresholded ROI."""
    import matplotlib.pyplot as plt

    inference = CBMRInference(device="cpu", moderator_effect="voxelwise")
    inference.fit(_voxelwise_result_for_inference())

    figure = inference.plot_voxelwise_moderator_effects(
        moderators="age",
        groups="Default",
        cut_coords=[0],
        display_mode="x",
    )

    assert figure is not None
    assert len(figure.axes) >= 1
    plt.close(figure)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_masks_ri_to_default_id_roi():
    """The default ROI should keep voxels above the median absolute ID value."""
    relative_intensity = np.array([1.1, 1.2, 1.3, 1.4])
    intensity_difference = np.array([0.1, -0.4, 0.2, -0.8])

    masked, threshold = CBMRInference._mask_relative_intensity_to_id_roi(
        relative_intensity,
        intensity_difference,
    )

    assert threshold == pytest.approx(0.3)
    np.testing.assert_allclose(masked, np.array([0.0, 1.2, 0.0, 1.4]))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_masks_ri_to_custom_id_roi():
    """A user-provided ID threshold should define the RI display ROI."""
    relative_intensity = np.array([1.1, 1.2, 1.3])
    intensity_difference = np.array([0.1, 0.5, -0.9])

    masked, threshold = CBMRInference._mask_relative_intensity_to_id_roi(
        relative_intensity,
        intensity_difference,
        id_threshold=0.5,
    )

    assert threshold == pytest.approx(0.5)
    np.testing.assert_allclose(masked, np.array([0.0, 1.2, 1.3]))


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_validates_moderator_effect_diagnostics():
    """Voxelwise moderator-effect diagnostics should reject unavailable inputs."""
    inference = CBMRInference(device="cpu", moderator_effect="voxelwise")
    inference.fit(_voxelwise_result_for_inference())

    with pytest.raises(ValueError, match="Unknown moderators"):
        inference.generate_voxelwise_moderator_effect_maps(moderators="sample_size")
    with pytest.raises(ValueError, match="unit_change"):
        inference.generate_voxelwise_moderator_effect_maps(unit_change=np.inf)
    with pytest.raises(ValueError, match="id_threshold"):
        inference.plot_voxelwise_moderator_effects(id_threshold=-1)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_transform_adds_maps_without_mutating_input():
    """The inference object should append maps to a copy."""
    result = _voxelwise_result_for_inference()
    original_map_keys = set(result.maps)
    inference = CBMRInference(device="cpu", moderator_effect="voxelwise", ridge=1e-3)

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
def test_voxelwise_cbmr_inference_fisher_information_matches_explicit_kron():
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
@pytest.mark.parametrize("meat", ["iid", "cluster"])
def test_voxelwise_cbmr_inference_sandwich_covariance_matches_explicit_kron(meat):
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
    if meat == "iid":
        meat_matrix = explicit_design.T @ (explicit_design * explicit_residuals[:, None] ** 2)
    else:
        cluster_scores = []
        for experiment_index in range(moderators.shape[0]):
            start = experiment_index * bases.shape[0]
            stop = start + bases.shape[0]
            cluster_scores.append(explicit_design[start:stop].T @ explicit_residuals[start:stop])
        cluster_scores = np.column_stack(cluster_scores)
        meat_matrix = cluster_scores @ cluster_scores.T
    expected = bread_inverse @ meat_matrix @ bread_inverse

    actual = CBMRInference._compute_sandwich_covariance(
        moderators,
        bases,
        scipy.sparse.csr_matrix(foci),
        mean,
        ridge=ridge,
        meat=meat,
        correction="hc0",
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_global_cbmr_inference_sandwich_covariance_matches_explicit_glm():
    """Global sandwich covariance should match explicit marginal GLM calculations."""
    design = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    mean = np.array([1.0, 1.5, 2.0])
    foci = np.array([0.0, 2.0, 3.0])
    ridge = 1e-4

    bread = design.T @ (design * mean[:, None])
    bread_inverse = np.linalg.pinv(bread + ridge * np.eye(bread.shape[0]))
    residuals = foci - mean
    meat_matrix = design.T @ (design * residuals[:, None] ** 2)
    expected = bread_inverse @ meat_matrix @ bread_inverse

    actual = CBMRInference._compute_glm_sandwich_covariance(
        design,
        foci,
        mean,
        ridge=ridge,
        correction="hc0",
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
@pytest.mark.parametrize("meat", ["cluster", "iid"])
@pytest.mark.parametrize("correction", ["hc0", "hc1", "hc3"])
def test_voxelwise_cbmr_sparse_sandwich_covariance_matches_dense(meat, correction):
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
        scipy.sparse.csc_matrix(foci),
        mean,
        **kwargs,
    )

    np.testing.assert_allclose(sparse, dense)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_voxelwise_cbmr_inference_sandwich_helpers_handle_hc_corrections():
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
def test_voxelwise_cbmr_inference_spatial_statistics_match_manual_wald():
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
def test_voxelwise_cbmr_inference_chi_square_log_intensity_matches_legacy_loop():
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


def _reference_b_spline_bases(mask, spacing, margin=10):
    """Build the tensor-product basis the slow way: full product, then filter.

    Mirrors what :func:`~nimare.utils.b_spline_bases` did before it started walking the
    product one basis plane at a time, and exists so the optimized version can be pinned
    against it.
    """
    from nimare.utils import coef_spline_bases

    mask = np.asanyarray(mask).astype(bool, copy=False)
    xx = np.where(mask.sum(axis=(1, 2)) > 0)[0]
    yy = np.where(mask.sum(axis=(0, 2)) > 0)[0]
    zz = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    x_spline = coef_spline_bases(xx, spacing, margin)
    y_spline = coef_spline_bases(yy, spacing, margin)
    z_spline = coef_spline_bases(zz, spacing, margin)

    cropped = mask[xx.min() : xx.max() + 1, yy.min() : yy.max() + 1, zz.min() : zz.max() + 1]
    coords = np.argwhere(cropped)
    x_rows = x_spline[coords[:, 0]]
    y_rows = y_spline[coords[:, 1]]
    z_rows = z_spline[coords[:, 2]]
    xy_rows = (x_rows[:, :, None] * y_rows[:, None, :]).reshape(coords.shape[0], -1)
    full = (xy_rows[:, :, None] * z_rows[:, None, :]).reshape(coords.shape[0], -1)
    return full[:, np.max(full, axis=0) >= 0.1]


@pytest.mark.parametrize("spacing", [8, 12])
def test_b_spline_bases_matches_the_unfiltered_tensor_product(spacing):
    """Pruning basis planes early must not change a single value.

    :func:`~nimare.utils.b_spline_bases` skips whole (i, j) planes whose peak falls under the
    support threshold rather than building the entire tensor product and discarding most of
    it afterwards, which at spacing=5 on the 2 mm mask cut peak memory from 13.1 GB to
    5.7 GB. The saving is only worth having if the result is untouched, so this pins it
    against the full-product reference, values and column order alike.
    """
    from nimare.utils import b_spline_bases

    rng = np.random.default_rng(0)
    grid = np.zeros((26, 24, 22), dtype=bool)
    grid[4:22, 3:21, 5:18] = True
    # Punch out a few voxels so the mask is not a perfect box, which is what leaves some
    # tensor-product bases unsupported and gives the pruning something to do.
    holes = rng.integers([4, 3, 5], [22, 21, 18], size=(40, 3))
    grid[holes[:, 0], holes[:, 1], holes[:, 2]] = False

    actual = b_spline_bases(masker_voxels=grid, spacing=spacing)
    expected = _reference_b_spline_bases(grid, spacing)

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_b_spline_bases_prunes_unsupported_bases():
    """The pruning must actually discard columns, or the equality test proves nothing."""
    from nimare.utils import b_spline_bases, coef_spline_bases

    grid = np.zeros((26, 24, 22), dtype=bool)
    grid[4:22, 3:21, 5:18] = True

    kept = b_spline_bases(masker_voxels=grid, spacing=8).shape[1]
    n_axis_bases = [
        coef_spline_bases(np.where(grid.sum(axis=axes) > 0)[0], 8, 10).shape[1]
        for axes in ((1, 2), (0, 2), (0, 1))
    ]
    built = n_axis_bases[0] * n_axis_bases[1] * n_axis_bases[2]

    assert kept < built, "no bases were pruned; the equality test would be vacuous"
