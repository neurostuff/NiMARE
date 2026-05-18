"""Tests for spatially varying CBMR meta-analytic methods."""

import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

try:
    import torch
except ImportError:
    warnings.warn("Torch not installed. Spatial CBMR tests will be skipped.")
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta import models
    from nimare.meta import spatial_cbmr as spatial_cbmr_module
    from nimare.meta.spatial_cbmr import (
        SpatialCBMREstimator,
        SpatialCBMRInference,
        SpatialCBMRResult,
    )
    from nimare.meta.utils import fit_spatial_cbmr_approximate

import nimare


def _mask_img():
    """Return a small valid mask image for MetaResult construction."""
    return nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))


def _spatial_result_for_inference():
    """Return a small fitted SpatialCBMRResult suitable for inference tests."""
    estimator = SpatialCBMREstimator(moderators=["age"], backend="approximate")
    estimator.groups = ["Default"]
    estimator.inputs_ = {
        "coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "moderators_by_group": {"Default": np.array([[2.0], [4.0]])},
        "foci_by_experiment_voxel": {
            "Default": scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        },
    }
    estimator.spatial_varying_coef = {"Default": np.array([[0.1], [0.2], [0.3], [0.4]])}
    maps = {}
    tables = {}
    estimator._add_approximate_results(
        maps,
        tables,
        "Default",
        estimator.inputs_["moderators_by_group"]["Default"],
        estimator.spatial_varying_coef["Default"],
    )
    return SpatialCBMRResult(
        estimator=estimator,
        mask=_mask_img(),
        maps=maps,
        tables=tables,
        description="spatial inference test",
    )


pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")


def test_spatial_cbmr_invalid_backend_raises():
    """The estimator should validate backend names at initialization."""
    with pytest.raises(ValueError, match="backend must be one of"):
        SpatialCBMREstimator(backend="bad-backend")


def test_spatial_cbmr_build_group_foci_matrices_counts_foci():
    """Experiment-by-voxel matrices should preserve ordered experiment IDs and focus counts."""
    coordinates = pd.DataFrame(
        {
            "id": ["study-1", "study-1", "study-2", "study-3"],
            "_cbmr_mask_index": [0, 2, 2, 1],
        }
    )
    ids_by_group = {"A": ["study-1", "study-2"], "B": ["study-3", "study-4"]}

    matrices = SpatialCBMREstimator._build_group_foci_matrices(
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


def test_spatial_cbmr_build_group_foci_matrices_handles_empty_coordinates():
    """Groups with no retained foci should still receive correctly shaped sparse matrices."""
    coordinates = pd.DataFrame({"id": [], "_cbmr_mask_index": []})
    ids_by_group = {"Default": ["study-1", "study-2"]}

    matrices = SpatialCBMREstimator._build_group_foci_matrices(
        coordinates,
        ids_by_group,
        n_mask_voxels=3,
    )

    assert matrices["Default"].shape == (2, 3)
    assert matrices["Default"].nnz == 0


def test_spatial_cbmr_prepare_torch_inputs_densifies_sparse_matrices():
    """Torch backend inputs should be float64 tensors on the estimator device."""
    estimator = SpatialCBMREstimator(moderators=["age"], device="cpu")
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


def test_spatial_cbmr_result_helpers_preserve_result_type_and_summarize_maps():
    """The result should retain CBMRResult behavior plus spatial helper methods."""
    result = SpatialCBMRResult(
        estimator=object(),
        mask=_mask_img(),
        maps={
            "svModerator_age_group-Default": np.array([1.0, 2.0, 3.0]),
            "spatialIntensity_group-Default": np.array([4.0, 5.0, 6.0]),
        },
        tables={"spatial_regression_coef": pd.DataFrame([[1.0, 2.0]], index=["Default"])},
        description="spatial test",
    )

    copied = result.copy()

    assert isinstance(result, SpatialCBMRResult)
    assert isinstance(result, nimare.results.MetaResult)
    assert copied is not result
    assert isinstance(copied, SpatialCBMRResult)
    assert result.sv_moderator_names == ("svModerator_age_group-Default",)
    assert result.describe_sv_effects()["svModerator_age_group-Default"] == (1.0, 2.0, 3.0)


def test_spatial_cbmr_output_tables_match_cbmr_group_table_convention():
    """Spatial CBMR summaries should expose aggregate CBMR-style output tables."""
    tables = {}

    SpatialCBMREstimator._add_spatial_coef_table(tables, "A", np.array([1.0, 2.0]))
    SpatialCBMREstimator._add_spatial_coef_table(tables, "B", np.array([3.0, 4.0]))
    SpatialCBMREstimator._add_moderator_table(
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
    assert "sv_moderator_regression_coef_group-A" in tables
    assert "sv_moderators_regression_coef" in tables
    assert tables["sv_moderators_regression_coef"].index.names == ["group", "moderator"]
    assert ("A", "age") in tables["sv_moderators_regression_coef"].index


def test_spatial_cbmr_add_approximate_results_creates_expected_maps_and_tables():
    """Approximate backend result extraction should create finite maps and CBMR-style tables."""
    estimator = SpatialCBMREstimator(moderators=["age"], backend="approximate")
    estimator.inputs_ = {"coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]])}
    maps = {}
    tables = {}
    moderators = np.array([[2.0], [4.0]])
    coefficient = np.array([[0.1], [0.2], [0.3], [0.4]])

    estimator._add_approximate_results(maps, tables, "Default", moderators, coefficient)

    assert set(maps) == {
        "spatialIntensity_group-Default",
        "svModerator_age_group-Default",
        "svModeratorTotal_group-Default",
    }
    assert np.all(np.isfinite(maps["spatialIntensity_group-Default"]))
    np.testing.assert_allclose(maps["spatialIntensity_group-Default"], np.exp([0.3, 0.4]))
    np.testing.assert_allclose(maps["svModerator_age_group-Default"], [0.3, 0.6])
    pd.testing.assert_frame_equal(
        tables["spatial_regression_coef"],
        pd.DataFrame([[0.3, 0.4]], index=["Default"], columns=["basis_0", "basis_1"]),
    )
    assert ("Default", "age") in tables["sv_moderators_regression_coef"].index


def test_spatial_cbmr_torch_result_extraction_uses_model_weights():
    """Full backend result extraction should project fitted torch weights into maps and tables."""
    estimator = SpatialCBMREstimator(moderators=["age"], backend="full", device="cpu")
    estimator.groups = ["Default"]
    estimator.inputs_ = {"coef_spline_bases": np.array([[1.0, 0.0], [0.0, 1.0]])}
    estimator.spatial_varying_model = models.SpatialCBMRModel(
        groups=estimator.groups,
        spatial_coef_dim=2,
        moderators_coef_dim=1,
        device="cpu",
    )
    with torch.no_grad():
        estimator.spatial_varying_model.spatial_coef_linears["Default"].weight[:] = torch.tensor(
            [[0.3, 0.4]], dtype=torch.float64
        )
        estimator.spatial_varying_model.moderator_coef_linears["Default"].weight[:] = torch.tensor(
            [[0.1, 0.2]], dtype=torch.float64
        )
    moderators_by_group = {"Default": torch.tensor([[2.0], [4.0]], dtype=torch.float64)}

    maps, tables = estimator._extract_torch_results(moderators_by_group)

    np.testing.assert_allclose(maps["spatialIntensity_group-Default"], np.exp([0.3, 0.4]))
    np.testing.assert_allclose(maps["svModerator_age_group-Default"], [0.3, 0.6])
    pd.testing.assert_frame_equal(
        tables["spatial_regression_coef"],
        pd.DataFrame([[0.3, 0.4]], index=["Default"], columns=["basis_0", "basis_1"]),
    )
    assert ("Default", "age") in tables["sv_moderators_regression_coef"].index


def test_spatial_cbmr_fit_dispatches_to_approximate_backend(monkeypatch):
    """The backend option should route SpatialCBMREstimator through the approximate solver."""
    estimator = SpatialCBMREstimator(
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

    def fake_fit_spatial_cbmr_approximate(
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
        spatial_cbmr_module,
        "fit_spatial_cbmr_approximate",
        fake_fit_spatial_cbmr_approximate,
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


def test_spatial_cbmr_result_get_inference_returns_fitted_engine():
    """The result should expose a result-centered inference workflow."""
    result = _spatial_result_for_inference()

    inference = result.get_inference()

    assert isinstance(inference, SpatialCBMRInference)
    assert inference.result is not result
    assert inference.groups == ["Default"]
    assert inference.moderators == ["age"]
    assert inference.method == "sandwich"


def test_spatial_cbmr_result_helpers_allow_fisher_information_method():
    """Result-centered inference should let users request FI standard errors."""
    result = _spatial_result_for_inference()

    inference = result.get_inference(method="FI")
    transformed = result.test_groups(method="FI")

    assert inference.method == "FI"
    assert transformed.metadata["spatial_cbmr_inference_method"] == "FI"
    assert "z_group-Default" in transformed.maps


def test_spatial_cbmr_result_helpers_run_inference():
    """The result should support CBMRResult-style inference helpers."""
    result = _spatial_result_for_inference()

    group_result = result.test_groups()
    moderator_result = result.test_moderators()

    assert isinstance(group_result, SpatialCBMRResult)
    assert isinstance(moderator_result, SpatialCBMRResult)
    assert "z_group-Default" in group_result.maps
    assert "p_svModerator_age_group-Default" in moderator_result.maps


def test_spatial_cbmr_inference_requires_fit_and_spatial_result():
    """The inference object should validate fit state and result type like CBMRInference."""
    inference = SpatialCBMRInference(device="cpu")

    with pytest.raises(ValueError, match="has not been fit"):
        inference.create_contrast("Default", source="groups")
    with pytest.raises(TypeError, match="requires a SpatialCBMRResult"):
        inference.fit(object())


def test_spatial_cbmr_inference_validates_standard_error_options():
    """The inference object should validate sandwich and FI method options."""
    with pytest.raises(ValueError, match="method must be one of"):
        SpatialCBMRInference(method="bad-method")
    with pytest.raises(ValueError, match="sandwich_meat"):
        SpatialCBMRInference(sandwich_meat="bad-meat")
    with pytest.raises(ValueError, match="sandwich_correction"):
        SpatialCBMRInference(sandwich_correction="bad-correction")
    with pytest.raises(ValueError, match="ridge"):
        SpatialCBMRInference(ridge=-1.0)


def test_spatial_cbmr_inference_create_contrast():
    """The inference object should parse named group and moderator contrasts like CBMR."""
    inference = SpatialCBMRInference(device="cpu")
    inference.fit(_spatial_result_for_inference())

    group_contrast = inference.create_contrast("Default", source="groups")
    moderator_contrast = inference.create_contrast("age", source="moderators")

    np.testing.assert_array_equal(group_contrast["Default"], np.array([1.0]))
    np.testing.assert_array_equal(moderator_contrast["age"], np.array([1.0]))


def test_spatial_cbmr_inference_preprocesses_raw_contrasts_like_cbmr():
    """Raw contrast arrays should be two-dimensional, standardized, and deduplicated."""
    inference = SpatialCBMRInference(device="cpu")
    inference.fit(_spatial_result_for_inference())
    inference.t_con_moderators = [np.array([2.0]), np.array([2.0])]

    contrasts, names = inference._preprocess_t_con_regressor(source="moderators")

    assert names is None
    assert len(contrasts) == 1
    np.testing.assert_array_equal(contrasts[0], np.array([[1.0]]))


def test_spatial_cbmr_inference_rejects_wrong_contrast_shape():
    """The inference object should reject contrasts with incorrect regressor width."""
    inference = SpatialCBMRInference(device="cpu")
    inference.fit(_spatial_result_for_inference())
    inference.t_con_moderators = [np.array([1.0, 0.0])]

    with pytest.raises(ValueError, match="doesn't match with moderators"):
        inference._preprocess_t_con_regressor(source="moderators")


def test_spatial_cbmr_inference_transform_adds_maps_without_mutating_input():
    """The inference object should append maps to a copy."""
    result = _spatial_result_for_inference()
    original_map_keys = set(result.maps)
    inference = SpatialCBMRInference(device="cpu", ridge=1e-3)

    transformed = inference.fit_transform(
        result,
        t_con_groups="Default",
        t_con_moderators="age",
    )

    assert isinstance(transformed, SpatialCBMRResult)
    assert set(result.maps) == original_map_keys
    assert "p_group-Default" in transformed.maps
    assert "z_group-Default" in transformed.maps
    assert "p_svModerator_age_group-Default" in transformed.maps
    assert "z_svModerator_age_group-Default" in transformed.maps
    assert transformed.metadata["spatial_cbmr_inference_method"] == "sandwich"
    assert transformed.metadata["spatial_cbmr_sandwich_meat"] == "cluster"
    transformed_fi = inference.transform(t_con_groups="Default", method="FI")
    assert transformed_fi.metadata["spatial_cbmr_inference_method"] == "FI"
    assert "spatial_cbmr_sandwich_meat" not in transformed_fi.metadata
    for map_name in [
        "p_group-Default",
        "z_group-Default",
        "p_svModerator_age_group-Default",
        "z_svModerator_age_group-Default",
    ]:
        assert np.all(np.isfinite(transformed.maps[map_name])), map_name


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

    actual = SpatialCBMRInference._compute_fisher_information(moderators, bases, mean)

    np.testing.assert_allclose(actual, expected)


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

    actual = SpatialCBMRInference._compute_sandwich_covariance(
        moderators,
        bases,
        scipy.sparse.csr_matrix(foci),
        mean,
        ridge=ridge,
        meat="iid",
        correction="hc0",
    )

    np.testing.assert_allclose(actual, expected)


def test_spatial_cbmr_inference_spatial_statistics_match_manual_wald():
    """Voxel-wise single-contrast Wald statistics should match a manual calculation."""
    coefficient = np.array([[0.1, 0.2], [0.3, 0.4]])
    covariance = np.diag([0.5, 0.25, 0.75, 0.5])
    contrast = np.array([[1.0, 0.0]])
    bases = np.array([[1.0, 0.0], [0.5, 0.5]])

    stats = SpatialCBMRInference._compute_spatial_coefficient_statistics(
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

    actual = SpatialCBMRInference._chi_square_log_intensity(
        n_voxels=contrast_log_intensity.shape[1],
        n_involved_groups=2,
        simp_con_group=simp_con_group,
        cov_log_intensity=cov_log_intensity,
        contrast_log_intensity=contrast_log_intensity,
    )

    np.testing.assert_allclose(actual, np.asarray(expected))
