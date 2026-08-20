"""Tests for nimare.correct module."""

import numpy as np

from nimare.correct import FDRCorrector, FWECorrector
from nimare.results import MetaResult
from nimare.transforms import p_to_z
from nimare.utils import _minimum_positive_float


class DummyEstimator:
    """Minimal estimator stand-in for correction tests."""


def _masked_values(mask_img, fill_value=0.5):
    mask_data = np.asanyarray(mask_img.dataobj).astype(bool)
    return np.full(mask_data.sum(), fill_value, dtype=np.float32)


def test_FWECorrector_montecarlo_default_parameters():
    """FWECorrector(montecarlo) should not override estimator defaults with None."""
    corr = FWECorrector(method="montecarlo")

    # If n_iters was not provided, it should not appear in parameters,
    # allowing the estimator's default to be used.
    assert "n_iters" not in corr.parameters
    # n_cores should still be passed through with its default.
    assert corr.parameters["n_cores"] == 1


def test_FWECorrector_montecarlo_custom_parameters():
    """FWECorrector(montecarlo) should propagate explicit n_iters and n_cores."""
    corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=2)

    assert corr.parameters["n_iters"] == 10
    assert corr.parameters["n_cores"] == 2


def test_MetaResult_clips_tiny_analytical_p_values_to_float32_floor(mni_mask):
    """Analytical p-values below float32 resolution should be censored, not zeroed."""
    p_values = _masked_values(mni_mask)
    p_values[:2] = [0, 1e-50]
    z_values = p_to_z(p_values, tail="one")

    result = MetaResult(
        DummyEstimator(),
        mask=mni_mask,
        maps={
            "p": p_values,
            "z": z_values,
            "stat": np.arange(p_values.size, dtype=np.float32),
        },
    )

    assert result.maps["p"].dtype == np.float32
    assert np.all(result.maps["p"][:2] == _minimum_positive_float())
    assert np.all(np.isfinite(result.maps["z"][:2]))


def test_FDRCorrector_clips_tiny_p_values_and_finite_secondary_maps(mni_mask):
    """FDR correction should not generate infinite z/logp maps from tiny p-values."""
    p_values = _masked_values(mni_mask)
    p_values[:3] = [0, 1e-50, 0.01]
    z_values = p_to_z(p_values)
    with np.errstate(divide="ignore"):
        logp_values = -np.log10(p_values)

    result = MetaResult(
        DummyEstimator(),
        mask=mni_mask,
        maps={
            "p": p_values,
            "z": z_values,
            "logp": logp_values,
        },
    )

    assert np.all(np.isfinite(result.maps["logp"][:3]))
    corr_result = FDRCorrector(method="indep").transform(result)

    assert corr_result.maps["p_corr-FDR_method-indep"].dtype == np.float32
    assert np.all(corr_result.maps["p_corr-FDR_method-indep"][:2] > 0)
    assert np.all(np.isfinite(corr_result.maps["z_corr-FDR_method-indep"][:3]))
    assert np.all(np.isfinite(corr_result.maps["logp_corr-FDR_method-indep"][:3]))


def test_p_to_z_uses_float32_probability_floor():
    """Zero p-values should map to the finite float32-resolution z ceiling."""
    z_values = p_to_z(np.array([0, 0], dtype=np.float32), tail="two")

    assert np.all(np.isfinite(z_values))
    assert np.all(z_values > 0)


def test_corrector_leaves_nan_p_values_as_nan():
    """Voxels with no p value keep it, rather than picking one up from the output buffer."""

    class _Result:
        def __init__(self, p):
            self.maps = {"p": p, "z": np.ones_like(p), "logp": np.ones_like(p)}
            self.estimator = None
            self.tables = {}

    p = np.random.RandomState(0).uniform(0.001, 1.0, size=1000)
    uncovered = np.zeros(p.size, dtype=bool)
    uncovered[::7] = True
    p[uncovered] = np.nan

    corr_maps, _, _ = FDRCorrector(alpha=0.05)._transform(_Result(p), "correct_fdr_indep")

    for name in ("p", "z", "logp"):
        assert np.isnan(corr_maps[name][uncovered]).all(), name
    # The voxels that do have p values are still corrected.
    assert np.isfinite(corr_maps["p"][~uncovered]).all()
