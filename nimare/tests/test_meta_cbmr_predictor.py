"""Tests for the CBMR linear predictor and its marginal-based Poisson likelihood."""

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
    from nimare.meta.cbmr.predictor import (
        CBMRPredictor,
        SpatialPatterns,
        poisson_log_likelihood,
    )
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_EXPERIMENTS = 12
N_VOXELS = 30
N_BASES = 5


@pytest.fixture
def annotations():
    """Experiment annotations with a two-level factor and two covariates."""
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "diagnosis": ["schiz", "dep"] * (N_EXPERIMENTS // 2),
            "drug": (["yes", "yes", "no", "no"] * N_EXPERIMENTS)[:N_EXPERIMENTS],
            "n": rng.normal(size=N_EXPERIMENTS),
            "age": rng.normal(size=N_EXPERIMENTS),
        }
    )


@pytest.fixture
def bases():
    """Return a partition-of-unity basis, as a real cubic B-spline basis is."""
    rng = np.random.default_rng(5)
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    return raw / raw.sum(axis=1, keepdims=True)


def _predictor(formula, annotations, bases):
    return CBMRPredictor(bind(Design.from_formula(formula), annotations), bases)


def _random_coefficients(predictor, seed=6):
    rng = np.random.default_rng(seed)
    spatial = torch.tensor(
        rng.normal(0.5, 0.3, (predictor.n_spatial_columns, predictor.n_bases)),
        dtype=torch.float64,
        requires_grad=True,
    )
    if predictor.n_global_columns:
        global_coef = torch.tensor(
            rng.normal(0.0, 0.2, predictor.n_global_columns),
            dtype=torch.float64,
            requires_grad=True,
        )
    else:
        global_coef = None
    return spatial, global_coef


def _poisson_counts(predictor, spatial, global_coef, seed=7):
    with torch.no_grad():
        eta = predictor.linear_predictor(spatial, global_coef).numpy()
    return np.random.default_rng(seed).poisson(np.exp(eta)).astype(float)


@pytest.mark.parametrize(
    "formula,expected_patterns",
    [
        ("~ 1", 1),  # one shared map
        ("~ s(diagnosis)", 2),  # one per diagnosis
        ("~ s(diagnosis:drug)", 4),  # one per cell
        ("~ s(diagnosis) + n", 2),  # a scalar moderator does not split patterns
        ("~ s(n)", N_EXPERIMENTS),  # a spatial covariate gives everyone their own map
    ],
)
def test_pattern_count_follows_the_design(annotations, bases, formula, expected_patterns):
    """Distinct spatial loadings, not a user-selected mode, decide the cost of a fit.

    A group-only design collapses onto one pattern per group, which is the historical fast
    path. A spatial covariate gives every experiment its own pattern, which is the historical
    general path. Both fall out of the same code.
    """
    predictor = _predictor(formula, annotations, bases)
    assert predictor.patterns.n_patterns == expected_patterns


def test_a_spatial_covariate_is_reported_as_degenerate(annotations, bases):
    """The expensive end of the range should be identifiable, not silent."""
    assert _predictor("~ s(n)", annotations, bases).patterns.is_degenerate
    assert not _predictor("~ s(diagnosis)", annotations, bases).patterns.is_degenerate


@pytest.mark.parametrize(
    "formula",
    ["~ 1", "~ s(diagnosis)", "~ s(diagnosis) + n", "~ s(diagnosis) + n + s(age)", "~ s(n)"],
)
def test_marginal_likelihood_equals_the_elementwise_form(annotations, bases, formula):
    """The marginal collapse must be exact, not an approximation.

    This is the whole justification for never materializing the (experiment x voxel) array.
    If it drifted, every fit would be optimizing a subtly different objective than the one
    documented.
    """
    predictor = _predictor(formula, annotations, bases)
    spatial, global_coef = _random_coefficients(predictor)
    foci = _poisson_counts(predictor, spatial, global_coef)

    actual = poisson_log_likelihood(predictor, spatial, global_coef, foci)

    eta = predictor.linear_predictor(spatial, global_coef)
    expected = torch.sum(torch.as_tensor(foci, dtype=eta.dtype) * eta - torch.exp(eta))

    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-8), (
        f"{formula}: marginal form {float(actual):.10f} != elementwise " f"{float(expected):.10f}"
    )


@pytest.mark.parametrize("formula", ["~ s(diagnosis)", "~ s(diagnosis) + n", "~ s(n)"])
def test_likelihood_matches_statsmodels_on_the_materialized_design(annotations, bases, formula):
    """The predictor must agree with an independent GLM built from the same design.

    Ties the new term machinery to the external oracle rather than only to itself: the design
    is materialized column by column and handed to statsmodels, whose log-likelihood must match
    once the dropped -sum(log(y!)) constant is added back.
    """
    from scipy.special import gammaln

    predictor = _predictor(formula, annotations, bases)
    spatial, global_coef = _random_coefficients(predictor)
    foci = _poisson_counts(predictor, spatial, global_coef)

    # One column per (spatial column, basis) pair, plus the non-spatial columns.
    spatial_design = np.einsum("ic,vb->ivcb", predictor.spatial_block, predictor.bases).reshape(
        N_EXPERIMENTS * N_VOXELS, -1
    )
    columns = [spatial_design]
    parameters = [spatial.detach().numpy().reshape(-1)]
    if predictor.global_block is not None:
        columns.append(np.repeat(predictor.global_block, N_VOXELS, axis=0))
        parameters.append(global_coef.detach().numpy().reshape(-1))
    design = np.hstack(columns)
    coefficients = np.concatenate(parameters)

    model = statsmodels_api.GLM(
        foci.reshape(-1), design, family=statsmodels_api.families.Poisson()
    )
    expected = float(model.loglike(coefficients))
    actual = float(poisson_log_likelihood(predictor, spatial, global_coef, foci).detach())
    actual -= float(gammaln(foci + 1).sum())

    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_gradients_flow_to_every_coefficient(annotations, bases):
    """Fitting needs autograd through the marginal form, for spatial and global terms alike."""
    predictor = _predictor("~ s(diagnosis) + n + s(age)", annotations, bases)
    spatial, global_coef = _random_coefficients(predictor)
    foci = _poisson_counts(predictor, spatial, global_coef)

    poisson_log_likelihood(predictor, spatial, global_coef, foci).backward()

    assert spatial.grad is not None and torch.all(torch.isfinite(spatial.grad))
    assert global_coef.grad is not None and torch.all(torch.isfinite(global_coef.grad))
    assert torch.any(spatial.grad != 0) and torch.any(global_coef.grad != 0)


def test_sparse_and_dense_foci_agree(annotations, bases):
    """Foci arrive as sparse matrices from the estimator, so both must be accepted."""
    import scipy.sparse

    predictor = _predictor("~ s(diagnosis) + n", annotations, bases)
    spatial, global_coef = _random_coefficients(predictor)
    foci = _poisson_counts(predictor, spatial, global_coef)

    dense = poisson_log_likelihood(predictor, spatial, global_coef, foci)
    sparse = poisson_log_likelihood(predictor, spatial, global_coef, scipy.sparse.csr_matrix(foci))
    assert torch.allclose(dense, sparse, rtol=1e-12)


def test_marginal_by_pattern_sums_the_right_experiments():
    """Pattern marginals must aggregate exactly the experiments sharing that loading."""
    patterns = SpatialPatterns(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    foci = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])

    marginal = patterns.marginal_by_pattern(foci)

    assert marginal.shape == (2, 2)
    # np.unique sorts rows, so [0, 1] comes before [1, 0].
    index_for_second_experiment = patterns.assignment[1]
    np.testing.assert_array_equal(marginal[index_for_second_experiment], [10.0, 20.0])
    other = 1 - index_for_second_experiment
    np.testing.assert_array_equal(marginal[other], [101.0, 202.0])


def test_a_design_without_a_spatial_term_is_rejected(annotations, bases):
    """CBMR estimates a spatial intensity; a design with no spatial term is not a CBMR model."""
    bound = bind(Design.from_formula("~ n"), annotations)
    # bind() inserts the baseline, so strip it to construct the invalid case deliberately.
    stripped = type(bound)(
        blocks=tuple(b for b in bound.blocks if not b.term.spatial), design=bound.design
    )
    with pytest.raises(ValueError, match="at least one spatial term"):
        CBMRPredictor(stripped, bases)


def test_mismatched_foci_shape_is_reported(annotations, bases):
    """A foci matrix from the wrong studyset should fail with a comprehensible message."""
    predictor = _predictor("~ s(diagnosis)", annotations, bases)
    with pytest.raises(ValueError, match="but the design covers"):
        predictor.patterns.marginal_by_pattern(np.zeros((N_EXPERIMENTS + 3, N_VOXELS)))
