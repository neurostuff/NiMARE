"""Tests for named hypothesis tests on a term-based CBMR model."""

import numpy as np
import pandas as pd
import pytest

statsmodels_api = pytest.importorskip("statsmodels.api")

try:
    import torch
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr.contrasts import ContrastError, evaluate_hypotheses
    from nimare.meta.cbmr.distributions import Poisson
    from nimare.meta.cbmr.model import CBMRModel
    from nimare.meta.cbmr.predictor import CBMRPredictor
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_LEVEL = 10
N_VOXELS = 20
N_BASES = 4
LEVELS = ("a", "b", "c")


@pytest.fixture(scope="module")
def fitted():
    """Fit a three-level factor plus a scalar moderator on interior-MLE counts."""
    rng = np.random.default_rng(71)
    n_experiments = N_PER_LEVEL * len(LEVELS)
    annotations = pd.DataFrame(
        {
            "dx": [level for level in LEVELS for _ in range(N_PER_LEVEL)],
            "n": rng.normal(size=n_experiments),
        }
    )
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    bases = raw / raw.sum(axis=1, keepdims=True)

    predictor = CBMRPredictor(bind(Design.from_formula("~ s(dx) + n"), annotations), bases)
    spatial = torch.tensor(
        rng.normal(1.3, 0.25, (predictor.n_spatial_columns, N_BASES)), dtype=torch.float64
    )
    global_coef = torch.tensor(rng.normal(0.0, 0.15, 1), dtype=torch.float64)
    with torch.no_grad():
        eta = predictor.linear_predictor(spatial, global_coef).numpy()
    foci = rng.poisson(np.exp(eta)).astype(float)

    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=2000, tol=1e-12)
    return model, foci


def test_pairwise_spatial_contrast_gives_a_z_map(fitted):
    """Comparing two levels of a spatial factor must yield per-voxel statistics."""
    model, foci = fitted
    result = evaluate_hypotheses(model, "dx[a] = dx[b]", foci, name="a-b")

    for prefix in ("z", "p", "logp"):
        assert result["maps"][f"{prefix}_a-b"].shape == (N_VOXELS,)
    assert np.all(np.isfinite(result["maps"]["z_a-b"]))
    assert np.all(result["maps"]["p_a-b"] > 0) and np.all(result["maps"]["p_a-b"] <= 1)


def test_spatial_contrast_matches_statsmodels_wald(fitted):
    """A named contrast must equal the Wald statistic an independent GLM would report.

    Ties the contrast machinery to the external oracle: the same contrast vector is applied to
    statsmodels' coefficients and covariance on the materialized design.
    """
    model, foci = fitted
    predictor = model.predictor
    n_experiments = predictor.patterns.n_experiments

    spatial = np.einsum("ic,vb->ivcb", predictor.spatial_block, predictor.bases).reshape(
        n_experiments * N_VOXELS, -1
    )
    design = np.hstack([spatial, np.repeat(predictor.global_block, N_VOXELS, axis=0)])
    expected_fit = statsmodels_api.GLM(
        foci.reshape(-1), design, family=statsmodels_api.families.Poisson()
    ).fit()

    actual = evaluate_hypotheses(model, "dx[a] = dx[b]", foci, name="t")["maps"]["z_t"]

    # Contrast for level a minus level b, per voxel, in the flat parameter space.
    names = list(predictor.design.blocks[0].column_names)
    index_a, index_b = names.index("dx[a]"), names.index("dx[b]")
    covariance = expected_fit.cov_params()
    expected = np.empty(N_VOXELS)
    for voxel in range(N_VOXELS):
        weights = np.zeros(design.shape[1])
        block = predictor.bases[voxel]
        weights[index_a * N_BASES : (index_a + 1) * N_BASES] = block
        weights[index_b * N_BASES : (index_b + 1) * N_BASES] = -block
        estimate = weights @ expected_fit.params
        expected[voxel] = estimate / np.sqrt(weights @ covariance @ weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def test_joint_hypothesis_gives_a_chi_square_map(fitted):
    """Several statements at once should be a generalized linear hypothesis, not a loop."""
    model, foci = fitted
    result = evaluate_hypotheses(model, ["dx[a] = dx[b]", "dx[b] = dx[c]"], foci, name="joint")

    chi_square = result["maps"]["chiSquare_joint"]
    assert chi_square.shape == (N_VOXELS,)
    assert np.all(chi_square >= 0)
    assert np.all(np.isfinite(result["maps"]["z_joint"]))


def test_a_level_can_be_tested_against_zero(fitted):
    """``= 0`` should be accepted as the right-hand side."""
    model, foci = fitted
    result = evaluate_hypotheses(model, "dx[a] = 0", foci, name="a")
    assert np.all(np.isfinite(result["maps"]["z_a"]))


def test_scalar_term_contrast_gives_a_table(fitted):
    """A non-spatial term has one coefficient, so its test is a row not a map."""
    model, foci = fitted
    result = evaluate_hypotheses(model, "n = 0", foci, name="n")

    assert not result["maps"]
    table = result["tables"]["contrast_n"]
    assert set(["estimate", "standard_error", "z", "p", "logp"]) <= set(table.columns)
    assert np.isfinite(table["z"].iloc[0])


def test_scalar_contrast_matches_statsmodels(fitted):
    """The scalar path must agree with the oracle too."""
    model, foci = fitted
    predictor = model.predictor
    n_experiments = predictor.patterns.n_experiments
    spatial = np.einsum("ic,vb->ivcb", predictor.spatial_block, predictor.bases).reshape(
        n_experiments * N_VOXELS, -1
    )
    design = np.hstack([spatial, np.repeat(predictor.global_block, N_VOXELS, axis=0)])
    expected_fit = statsmodels_api.GLM(
        foci.reshape(-1), design, family=statsmodels_api.families.Poisson()
    ).fit()

    table = evaluate_hypotheses(model, "n = 0", foci, name="n")["tables"]["contrast_n"]

    np.testing.assert_allclose(table["estimate"].iloc[0], expected_fit.params[-1], rtol=1e-4)
    np.testing.assert_allclose(table["standard_error"].iloc[0], expected_fit.bse[-1], rtol=1e-4)
    np.testing.assert_allclose(
        table["z"].iloc[0], expected_fit.params[-1] / expected_fit.bse[-1], rtol=1e-4
    )


def test_bare_levels_are_accepted(fitted):
    """Map labels use bare levels, so hypotheses should accept them too."""
    model, foci = fitted
    from_bare = evaluate_hypotheses(model, "a = b", foci, name="x")["maps"]["z_x"]
    from_full = evaluate_hypotheses(model, "dx[a] = dx[b]", foci, name="y")["maps"]["z_y"]
    np.testing.assert_allclose(from_bare, from_full)


def test_reversing_a_contrast_flips_its_sign(fitted):
    """A sanity check that the contrast direction means what it reads as."""
    model, foci = fitted
    forward = evaluate_hypotheses(model, "dx[a] = dx[b]", foci, name="f")["maps"]["z_f"]
    reverse = evaluate_hypotheses(model, "dx[b] = dx[a]", foci, name="r")["maps"]["z_r"]
    np.testing.assert_allclose(forward, -reverse, rtol=1e-10)


def test_unknown_coefficient_lists_the_options(fitted):
    """A typo should say what could have been meant."""
    model, foci = fitted
    with pytest.raises(ContrastError, match="Available columns"):
        evaluate_hypotheses(model, "dx[z] = dx[a]", foci)


def test_cross_term_contrasts_are_refused(fitted):
    """Comparing a map to a scalar has no single scale, so it is refused not guessed."""
    model, foci = fitted
    with pytest.raises(ContrastError, match="must stay within one term"):
        evaluate_hypotheses(model, "dx[a] = n", foci)


def test_malformed_hypotheses_are_refused(fitted):
    """Syntax problems should surface as ContrastError."""
    model, foci = fitted
    for bad in ["dx[a]", "dx[a] = dx[b] = dx[c]", "= dx[a]"]:
        with pytest.raises(ContrastError):
            evaluate_hypotheses(model, bad, foci)
    with pytest.raises(ContrastError, match="No hypotheses"):
        evaluate_hypotheses(model, [], foci)
