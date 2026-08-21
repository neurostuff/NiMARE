"""Tests for fitting and covariance over a term-based CBMR design.

Checked against statsmodels rather than against CBMR itself. Once the design is materialized
column by column, a CBMR model *is* a Poisson GLM, so an independent implementation can fit the
same thing with a different optimizer and different covariance code. Agreement on standard
errors is the interesting part: the older code kept coefficients in per-group and per-moderator
containers and inverted each block separately, which dropped the cross terms between them.
"""

import numpy as np
import pandas as pd
import pytest

statsmodels_api = pytest.importorskip("statsmodels.api")

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr.distributions import NegativeBinomial, Poisson
    from nimare.meta.cbmr.model import CBMRModel
    from nimare.meta.cbmr.predictor import CBMRPredictor
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_GROUP = 14
N_VOXELS = 32
N_BASES = 5


@pytest.fixture(scope="module")
def data():
    """Simulate counts with an interior maximum, so a converged fit is meaningful.

    Coordinate-based foci give 0/1 per experiment-voxel cell, for which the Poisson maximum sits
    at infinity; see test_meta_cbmr_glm_equivalence. Rates here average a few counts per cell.
    """
    rng = np.random.default_rng(23)
    n_experiments = 2 * N_PER_GROUP
    annotations = pd.DataFrame(
        {
            "diagnosis": ["schiz"] * N_PER_GROUP + ["dep"] * N_PER_GROUP,
            "n": rng.normal(size=n_experiments),
            "age": rng.normal(size=n_experiments),
        }
    )
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    bases = raw / raw.sum(axis=1, keepdims=True)
    return annotations, bases, rng


def _build(formula, annotations, bases):
    return CBMRPredictor(bind(Design.from_formula(formula), annotations), bases)


def _simulate(predictor, rng, scale=1.3):
    """Draw Poisson counts from a known coefficient vector."""
    spatial = torch.tensor(
        rng.normal(scale, 0.25, (predictor.n_spatial_columns, predictor.n_bases)),
        dtype=torch.float64,
    )
    global_coef = (
        torch.tensor(rng.normal(0.0, 0.15, predictor.n_global_columns), dtype=torch.float64)
        if predictor.n_global_columns
        else None
    )
    with torch.no_grad():
        eta = predictor.linear_predictor(spatial, global_coef).numpy()
    return rng.poisson(np.exp(eta)).astype(float)


def _materialize(predictor):
    """Return the GLM design matrix the predictor implies, in flat-parameter order."""
    n_experiments = predictor.patterns.n_experiments
    spatial = np.einsum("ic,vb->ivcb", predictor.spatial_block, predictor.bases).reshape(
        n_experiments * predictor.n_voxels, -1
    )
    if predictor.global_block is None:
        return spatial
    return np.hstack([spatial, np.repeat(predictor.global_block, predictor.n_voxels, axis=0)])


def _statsmodels_fit(predictor, foci):
    return statsmodels_api.GLM(
        foci.reshape(-1),
        _materialize(predictor),
        family=statsmodels_api.families.Poisson(),
    ).fit()


@pytest.mark.parametrize(
    "formula",
    [
        "~ s(diagnosis)",
        "~ s(diagnosis) + n",
        "~ s(diagnosis) + n + age",
        "~ s(n)",  # a spatial covariate: impossible to express in the old API
        "~ s(diagnosis) + s(n)",  # and a mixed design, likewise
        "~ sz(diagnosis)",  # sum-to-zero reparameterization
        "~ sz(diagnosis) + n",
    ],
)
def test_fitted_coefficients_match_statsmodels(data, formula):
    """L-BFGS over the flat vector must reach the same optimum as statsmodels' IRLS."""
    annotations, bases, rng = data
    predictor = _build(formula, annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(31))

    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=2000, tol=1e-12)
    expected = _statsmodels_fit(predictor, foci)

    np.testing.assert_allclose(
        model.coefficients.detach().numpy(), expected.params, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    "formula", ["~ s(diagnosis) + n", "~ s(diagnosis) + s(n)", "~ sz(diagnosis) + n"]
)
def test_standard_errors_match_statsmodels(data, formula):
    """Covariance must be a block of the joint inverse, not the inverse of a block.

    This is what the flat layout buys. Pooled moderator coefficients are correlated with every
    group's spatial coefficients, so inverting their block alone understates their variance --
    which is precisely the defect the older per-container layout invited.
    """
    annotations, bases, rng = data
    predictor = _build(formula, annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(37))

    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=2000, tol=1e-12)
    expected = _statsmodels_fit(predictor, foci)

    errors = np.sqrt(np.diag(model.covariance(foci)))
    np.testing.assert_allclose(errors, expected.bse, rtol=1e-5, atol=1e-8)


def test_information_matrix_has_nonzero_cross_blocks(data):
    """The cross terms the old layout dropped must actually be there.

    A guard against a regression that would silently restore block independence: if these were
    zero, inverting blocks separately would be harmless and the joint matrix pointless.
    """
    annotations, bases, _ = data
    predictor = _build("~ s(diagnosis) + n", annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(41))
    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=500)

    information = model.information_matrix(foci)
    slices = predictor.design.parameter_slices(predictor.n_bases)
    cross = information[slices["s(diagnosis)"], slices["n"]]

    assert cross.shape == (model.n_spatial, model.n_global)
    assert np.abs(cross).max() > 0, "spatial and moderator coefficients must be correlated"


def test_standard_errors_are_reported_per_term(data):
    """Errors come back keyed by term, in each term's own shape."""
    annotations, bases, _ = data
    predictor = _build("~ s(diagnosis) + n", annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(43))
    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=500)

    errors = model.standard_errors(foci)
    assert set(errors) == {"s(diagnosis)", "n"}
    assert errors["s(diagnosis)"].shape == (2, N_BASES)
    assert errors["n"].shape == (1,)
    assert all(np.all(np.isfinite(v)) for v in errors.values())


def test_fitted_coefficients_are_reported_per_term(data):
    """Coefficients use the same per-term layout as the errors."""
    annotations, bases, _ = data
    predictor = _build("~ s(diagnosis) + n", annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(47))
    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=500)

    coefficients = model.fitted_coefficients()
    assert coefficients["s(diagnosis)"].shape == (2, N_BASES)
    assert coefficients["n"].shape == (1,)


def test_negative_binomial_fits_a_grouped_design(data):
    """Overdispersion must still fit where it is defined, with its parameters moving."""
    annotations, bases, _ = data
    predictor = _build("~ s(diagnosis) + n", annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(53))

    model = CBMRModel(predictor, NegativeBinomial()).fit(foci, n_iter=500, lr=1e-2)

    assert model.nuisance is not None
    assert model.nuisance.shape == (predictor.patterns.n_patterns,)
    assert torch.all(torch.isfinite(model.nuisance))
    assert torch.isfinite(model.log_likelihood(foci))

    # Reported on the statistical scale and strictly positive, whatever the optimizer did to
    # the unconstrained parameter it actually moves.
    overdispersion = model.overdispersion()
    assert overdispersion.shape == (predictor.patterns.n_patterns,)
    assert np.all(overdispersion > 0)


def test_poisson_has_no_overdispersion(data):
    """Poisson owns no nuisance parameters, so there is nothing to report."""
    annotations, bases, _ = data
    predictor = _build("~ s(diagnosis)", annotations, bases)
    model = CBMRModel(predictor, Poisson())
    assert model.nuisance is None
    assert model.overdispersion() is None


def test_negative_binomial_refuses_a_design_it_cannot_support(data):
    """Constructing the model is where the distribution's design constraint is enforced."""
    from nimare.meta.cbmr.distributions import DistributionError

    annotations, bases, _ = data
    predictor = _build("~ s(n)", annotations, bases)
    with pytest.raises(DistributionError, match="overdispersion"):
        CBMRModel(predictor, NegativeBinomial())


def test_log_intensity_has_one_row_per_pattern(data):
    """The fitted intensity is reported per distinct spatial map, not per experiment."""
    annotations, bases, _ = data
    predictor = _build("~ s(diagnosis) + n", annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(59))
    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=500)

    assert model.log_intensity().shape == (2, N_VOXELS)


def test_ill_conditioned_information_warns(data, caplog):
    """Standard errors past what double precision can invert should say so."""
    annotations, bases, _ = data
    # Duplicated basis columns make the information exactly singular.
    degenerate = np.hstack([bases, bases])
    predictor = _build("~ s(diagnosis)", annotations, degenerate)
    foci = _simulate(predictor, np.random.default_rng(61))
    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=200)

    with caplog.at_level("WARNING", logger="nimare.meta.cbmr.model"):
        try:
            model.covariance(foci)
        except np.linalg.LinAlgError:
            pass  # exactly singular is also an acceptable outcome
    assert any("condition number" in message for message in caplog.messages)


def test_additive_sum_to_zero_factors_are_fittable(data):
    """``sz(a) + sz(b)`` must fit, where ``s(a) + s(b)`` is rank deficient by a basis width.

    The design is only identified because each factor's coefficients are constrained to sum to
    zero across levels, so a full-rank check on the materialized design is the thing to assert.
    """
    annotations, bases, _ = data
    annotations = annotations.assign(drug=["yes", "no"] * N_PER_GROUP)
    predictor = _build("~ sz(diagnosis) + sz(drug)", annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(67))

    design = _materialize(predictor)
    assert np.linalg.matrix_rank(design) == design.shape[1], "design should be full rank"

    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=2000, tol=1e-12)
    expected = _statsmodels_fit(predictor, foci)

    np.testing.assert_allclose(
        model.coefficients.detach().numpy(), expected.params, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        np.sqrt(np.diag(model.covariance(foci))), expected.bse, rtol=1e-5, atol=1e-8
    )


def test_sum_to_zero_coefficients_sum_to_zero(data):
    """The constraint must hold in the fitted coefficients, not just in the parameterization.

    Recovering the per-level effects means undoing the reparameterization: ``Z gamma`` has to sum
    to zero across levels for every basis function, which is what makes the term a set of
    deviations from the baseline rather than a competing baseline.
    """
    from nimare.meta.cbmr.terms import sum_to_zero_basis

    annotations, bases, _ = data
    predictor = _build("~ sz(diagnosis) + n", annotations, bases)
    foci = _simulate(predictor, np.random.default_rng(71))
    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=1000, tol=1e-10)

    gamma = model.fitted_coefficients()["sz(diagnosis)"]
    per_level = sum_to_zero_basis(2) @ gamma

    np.testing.assert_allclose(per_level.sum(axis=0), 0.0, atol=1e-10)
