"""Tests for nimare.correct module."""

import numpy as np
import pytest
from pymare.stats import fdr

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


def test_p_to_z_bottoms_out_at_the_double_precision_floor():
    """A p-value of zero maps to the largest z a representable p-value describes.

    The floor used to be the smallest positive float32, i.e. a z of 14.12.
    """
    z_values = p_to_z(np.array([0, 0], dtype=np.float32), tail="two")

    assert np.all(np.isfinite(z_values))
    assert np.allclose(z_values, 38.485, atol=1e-3)


def test_corrector_leaves_nan_p_values_as_nan():
    """Voxels with no p value keep it, rather than picking one up from the output buffer."""

    class _Result:
        def __init__(self, p):
            with np.errstate(divide="ignore", invalid="ignore"):
                self.maps = {"p": p, "z": np.ones_like(p), "logp": -np.log10(p)}
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


@pytest.mark.parametrize(
    "corrector",
    [
        FWECorrector(method="bonferroni"),
        FDRCorrector(method="indep"),
        FDRCorrector(method="negcorr"),
    ],
)
def test_correction_carries_a_tail_the_p_map_cannot_hold(mni_mask, corrector):
    """Correcting off the logp map must not re-truncate the tail at the float32 p floor.

    The stored p map floors at 1e-45, so a correction read from it could never report a
    -log10(p) past 44.85 or a z past 14.12, however deep the uncorrected tail went.
    """
    logp_values = _masked_values(mni_mask) * 0.0 + 1.0
    logp_values[0] = 600.0  # a p of 1e-600, which no float can represent
    p_values = np.power(10.0, -np.minimum(logp_values, 45.0))

    result = MetaResult(
        DummyEstimator(),
        mask=mni_mask,
        maps={"p": p_values, "z": np.ones_like(p_values), "logp": logp_values},
    )
    corrected = corrector.transform(result)

    suffix = corrector._name_suffix
    assert corrected.maps[f"logp{suffix}"][0] > 590.0
    assert corrected.maps[f"z{suffix}"][0] > 50.0
    # And the p map still reports the floor, since that is all a float32 holds.
    assert corrected.maps[f"p{suffix}"][0] == _minimum_positive_float()


@pytest.mark.parametrize("method", ["indep", "negcorr"])
@pytest.mark.parametrize("from_logp", [True, False], ids=["from_logp", "from_p"])
def test_fdr_correction_matches_the_p_space_procedure(mni_mask, method, from_logp):
    """Moving the step-up into log space must not move the significance boundary.

    Covers both inputs the corrector accepts: the logp map where the estimator produced one,
    and the p map where it did not.
    """
    rng = np.random.default_rng(0)
    p_values = rng.uniform(1e-12, 1.0, size=_masked_values(mni_mask).size)

    maps = {"p": p_values, "z": np.ones_like(p_values)}
    if from_logp:
        maps["logp"] = -np.log10(p_values)
    result = MetaResult(DummyEstimator(), mask=mni_mask, maps=maps)
    corrected = FDRCorrector(method=method).transform(result)

    expected = fdr(p_values.copy(), method="bh" if method == "indep" else "by")
    got = corrected.maps[f"p_corr-FDR_method-{method}"]
    assert np.array_equal(got <= 0.05, expected <= 0.05)
    assert np.allclose(got, expected.astype(np.float32), rtol=1e-5)
