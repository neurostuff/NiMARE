"""Tests for ``exposure()``, which conditions CBMR on a per-experiment count.

Two things are being checked, and they are different in kind. That the exposure reaches every
consumer -- the likelihood, the closed-form information, the sandwich, the overdispersed
marginals -- is arithmetic, and is checked by agreement with a form that already worked. That
conditioning changes what the model can estimate is not arithmetic, and is checked by asserting
the designs it makes meaningless are refused rather than fitted.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr.distributions import DistributionError
    from nimare.meta.cbmr.information import closed_form_information
    from nimare.meta.cbmr.model import CBMRModel
    from nimare.meta.cbmr.predictor import CBMRPredictor
    from nimare.meta.cbmr.terms import (
        DERIVED_EXPOSURE_COLUMN,
        Design,
        FormulaError,
        Term,
        bind,
    )

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_GROUP = 40
N_VOXELS = 30
N_BASES = 4


# --------------------------------------------------------------------------------------------
# Parsing and validation, which need neither torch nor data.
# --------------------------------------------------------------------------------------------


def test_exposure_with_no_argument_means_the_derived_total():
    """``exposure()`` names the estimator's generated column, so no bare word can shadow it."""
    design = Design.from_formula("~ s(diagnosis) + exposure()")
    (term,) = design.exposure_terms
    assert term.expr == DERIVED_EXPOSURE_COLUMN
    assert term.is_derived_exposure
    assert str(design) == "~ s(diagnosis) + exposure()"


def test_exposure_with_an_argument_is_an_ordinary_annotation():
    """A user exposure resolves like any other term and is not the derived total."""
    (term,) = Design.from_formula("~ s(diagnosis) + exposure(scan_volume)").exposure_terms
    assert term.expr == "scan_volume"
    assert not term.is_derived_exposure


def test_offset_is_refused_and_points_at_exposure():
    """The two differ by a scale, so the wrong one has to say which is meant."""
    with pytest.raises(FormulaError, match="exposure"):
        Design.from_formula("~ s(diagnosis) + offset(log(n))")


@pytest.mark.parametrize(
    "formula",
    ["~ s(exposure(n))", "~ sz(exposure())", "~ s(diagnosis) + exposure(a, b)"],
)
def test_malformed_exposures_are_refused(formula):
    """An exposure has no basis to vary over and takes at most one expression."""
    with pytest.raises(FormulaError):
        Design.from_formula(formula)


def test_two_exposures_are_refused():
    """Two exposures would multiply, which one ``exposure(a * b)`` says more clearly."""
    with pytest.raises(FormulaError, match="at most one exposure"):
        Design.from_formula("~ s(diagnosis) + exposure(a) + exposure(b)")


def test_a_column_can_be_both_a_moderator_and_an_exposure():
    """``~ s(g) + n + exposure(n)`` is how a user asks whether the coefficient really is 1.

    It reads as a duplicate unless the deduplication key knows the two terms differ.
    """
    design = Design.from_formula("~ s(diagnosis) + n + exposure(n)")
    assert [str(term) for term in design.terms] == ["s(diagnosis)", "n", "exposure(n)"]


def test_a_spatial_exposure_is_refused_at_the_term_level():
    """Constructed directly, not only through the formula parser."""
    with pytest.raises(FormulaError, match="cannot also be spatial"):
        Term(expr="n", spatial=True, exposure=True)


# --------------------------------------------------------------------------------------------
# Everything below needs data.
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    """Simulate counts from ``mu_iv = E_i p_g(v)``, so each group's shape is a distribution."""
    rng = np.random.default_rng(23)
    n_experiments = 2 * N_PER_GROUP
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    bases = raw / raw.sum(axis=1, keepdims=True)

    coefficients = rng.normal(0.0, 0.6, (2, N_BASES))
    shape = np.exp(coefficients @ bases.T)
    shape /= shape.sum(axis=1, keepdims=True)
    group = np.repeat([0, 1], N_PER_GROUP)
    # Reporting volume varies a lot between studies, which is the situation an exposure is for.
    volume = np.exp(np.log(150.0) + 0.8 * rng.normal(size=n_experiments))
    foci = rng.poisson(volume[:, None] * shape[group]).astype(float)

    annotations = pd.DataFrame(
        {
            "diagnosis": ["schiz"] * N_PER_GROUP + ["dep"] * N_PER_GROUP,
            "drug": ["yes", "no"] * N_PER_GROUP,
            "n": rng.normal(size=n_experiments),
            "scan_volume": rng.uniform(1.0, 3.0, size=n_experiments),
            DERIVED_EXPOSURE_COLUMN: foci.sum(axis=1),
        }
    )
    return annotations, bases, foci, shape, group


def _model(formula, data, distribution="poisson", n_iter=400):
    annotations, bases, foci, _, _ = data
    predictor = CBMRPredictor(bind(Design.from_formula(formula), annotations), bases)
    model = CBMRModel(predictor, distribution=distribution)
    model.fit(foci, n_iter=n_iter, tol=1e-10)
    return model


def _log_maps(model):
    """Return each spatial pattern's fitted log intensity."""
    return model.log_intensity()


def test_the_exposure_owns_no_parameters(data):
    """The budget and the coefficient layout are identical with and without it."""
    annotations, bases, _, _, _ = data
    plain = bind(Design.from_formula("~ s(diagnosis)"), annotations)
    exposed = bind(Design.from_formula("~ s(diagnosis) + exposure()"), annotations)

    assert exposed.n_parameters(N_BASES) == plain.n_parameters(N_BASES)
    assert exposed.parameter_slices(N_BASES) == plain.parameter_slices(N_BASES)
    assert "exposure()" not in exposed.parameter_slices(N_BASES)
    assert "fixed at 1" in exposed.describe(N_BASES)


def test_the_exposure_is_not_a_moderator_column(data):
    """It must not reach ``global_block``, which is read as the fitted moderator columns."""
    annotations, bases, _, _, _ = data
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(diagnosis) + exposure()"), annotations), bases
    )
    assert predictor.global_block is None
    assert predictor.n_global_columns == 0
    assert predictor.has_exposure


@pytest.mark.parametrize(
    "expression, match",
    [
        ("exposure(diagnosis)", "single number"),
        ("exposure(negative)", "negative"),
        ("exposure(all_zero)", "zero for every experiment"),
    ],
)
def test_an_unusable_exposure_column_is_refused(data, expression, match):
    """A factor, a negative value and an all-zero column each have no reading as an exposure."""
    annotations, _, _, _, _ = data
    annotations = annotations.assign(negative=-1.0, all_zero=0.0)
    with pytest.raises(FormulaError, match=match):
        bind(Design.from_formula(f"~ s(diagnosis) + {expression}"), annotations)


def test_the_exposure_shifts_a_cell_means_map_by_a_constant_and_nothing_else(data):
    """The sharp statement of what an exposure does to ``~ s(factor)``.

    The exposure enters only through the pattern total ``T_p``, and a cell-means factor gives
    each group its own coefficient block spanning the constant, so the fit can absorb the change
    exactly. Every group's log map must move by ``log(T_p/T_p')`` and by nothing else -- which
    catches a sign or scale slip anywhere along the exposure path, where a summary statistic such
    as the spread of the fitted totals would not.
    """
    annotations, _, _, _, _ = data
    plain_model = _model("~ s(diagnosis)", data)
    plain = _log_maps(plain_model)
    exposed = _log_maps(_model("~ s(diagnosis) + exposure()", data))

    assignment = plain_model.predictor.patterns.assignment
    totals = annotations[DERIVED_EXPOSURE_COLUMN].to_numpy()
    for pattern in range(plain.shape[0]):
        members = assignment == pattern
        # Without an exposure T_p counts the experiments; with one it sums their totals.
        expected = np.log(members.sum() / totals[members].sum())
        difference = exposed[pattern] - plain[pattern]
        assert np.allclose(difference, expected, atol=1e-4)


def test_the_exposure_normalizes_each_pattern(data):
    """Under an exact exposure the score equation makes each spatial term a distribution."""
    model = _model("~ s(diagnosis) + exposure()", data)
    assert np.allclose(np.exp(_log_maps(model)).sum(axis=1), 1.0, atol=1e-3)


def test_the_exposure_recovers_a_shape_the_plain_fit_does_not(data):
    """Counts drawn from ``mu_iv = E_i p_g(v)``: the exposure model recovers ``p_g``."""
    _, _, _, shape, group = data
    model = _model("~ s(diagnosis) + exposure()", data)
    exposed = np.exp(_log_maps(model))
    plain = np.exp(_log_maps(_model("~ s(diagnosis)", data)))

    assignment = model.predictor.patterns.assignment
    for pattern in range(exposed.shape[0]):
        truth = shape[group[assignment == pattern][0]]
        assert np.abs(exposed[pattern] - truth).max() < 0.1 * truth.max()
        # The plain fit estimates a rate, whose scale is the group's mean count, so it is not
        # close to a distribution at all.
        assert np.abs(plain[pattern] - truth).max() > truth.max()


def test_the_exposure_agrees_with_the_fitted_log_form(data):
    """``exposure(w)`` is ``+ log(w)`` with the coefficient fixed, so the fitted one lands at 1."""
    annotations, bases, foci, _, _ = data
    annotations = annotations.assign(log_total=np.log(annotations[DERIVED_EXPOSURE_COLUMN]))
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(diagnosis) + log_total"), annotations), bases
    )
    fitted = CBMRModel(predictor, distribution="poisson")
    fitted.fit(foci, n_iter=600, tol=1e-10)
    coefficient = fitted.fitted_coefficients()["log_total"]
    assert np.abs(coefficient - 1.0).max() < 0.05


def test_a_zero_exposure_experiment_contributes_nothing(data):
    """Carried multiplicatively, ``E_i = 0`` drops out rather than giving ``log(0)``."""
    annotations, bases, foci, _, _ = data
    annotations = annotations.copy()
    with_zero = annotations.copy()
    with_zero.loc[with_zero.index[0], DERIVED_EXPOSURE_COLUMN] = 0.0
    zeroed_foci = foci.copy()
    zeroed_foci[0] = 0.0

    kept = CBMRModel(
        CBMRPredictor(bind(Design.from_formula("~ s(diagnosis) + exposure()"), with_zero), bases),
        distribution="poisson",
    )
    kept.fit(zeroed_foci, n_iter=400, tol=1e-10)

    dropped = CBMRModel(
        CBMRPredictor(
            bind(Design.from_formula("~ s(diagnosis) + exposure()"), with_zero.iloc[1:]), bases
        ),
        distribution="poisson",
    )
    dropped.fit(zeroed_foci[1:], n_iter=400, tol=1e-10)

    kept_values = kept.coefficients.detach().cpu().numpy()
    assert np.all(np.isfinite(kept_values))
    assert np.abs(kept_values - dropped.coefficients.detach().cpu().numpy()).max() < 1e-6


@pytest.mark.parametrize(
    "formula", ["~ s(diagnosis) + exposure()", "~ s(diagnosis) + exposure(scan_volume)"]
)
def test_the_closed_form_information_sees_the_exposure(data, formula):
    """The check that would have caught an exposure applied to the likelihood alone.

    ``_intensity_pieces`` used to rebuild the per-experiment weight instead of asking the
    predictor for it, so an exposure could be present in the fit and absent from every closed
    form while the autodiff fallback stayed correct.
    """
    model = _model(formula, data)

    def negative_log_likelihood(vector):
        return -model.log_likelihood(flat=vector, nuisance=None)

    reference = (
        torch.func.hessian(negative_log_likelihood)(model.coefficients.detach().clone())
        .reshape(model.n_parameters, model.n_parameters)
        .detach()
        .cpu()
        .numpy()
    )
    analytic = closed_form_information(model.distribution)(model)
    assert np.abs(analytic - reference).max() < 1e-8 * np.abs(reference).max()


def test_a_moderator_alongside_the_derived_exposure_is_refused(data):
    """Its estimate would be exactly zero whatever the data, reported as a confident null."""
    from nimare.meta.cbmr.estimator import _reject_moderators_under_a_derived_exposure

    annotations, _, _, _, _ = data
    bound = bind(Design.from_formula("~ s(diagnosis) + n + exposure()"), annotations)
    with pytest.raises(FormulaError, match="no coefficient to estimate"):
        _reject_moderators_under_a_derived_exposure(bound)


def test_a_moderator_alongside_a_user_exposure_is_allowed(data):
    """The refusal is scoped to the derived total, not to the marker.

    An exposure of the user's own does not fit the totals exactly, so a non-spatial term still
    has variation to explain and a coefficient to estimate.
    """
    from nimare.meta.cbmr.estimator import _reject_moderators_under_a_derived_exposure

    annotations, _, _, _, _ = data
    bound = bind(Design.from_formula("~ s(diagnosis) + n + exposure(scan_volume)"), annotations)
    _reject_moderators_under_a_derived_exposure(bound)

    model = _model("~ s(diagnosis) + n + exposure(scan_volume)", data)
    assert np.abs(model.fitted_coefficients()["n"]).max() > 1e-6


@pytest.mark.parametrize("distribution", ["negativebinomial", "clusterednegativebinomial"])
def test_an_overdispersed_family_with_the_derived_exposure_is_refused(data, distribution):
    """Both exist to absorb variation in the totals, which the exposure absorbs first."""
    annotations, bases, _, _, _ = data
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(diagnosis) + exposure()"), annotations), bases
    )
    with pytest.raises(DistributionError, match="exposure"):
        CBMRModel(predictor, distribution=distribution)


@pytest.mark.parametrize("distribution", ["negativebinomial", "clusterednegativebinomial"])
def test_an_overdispersed_family_with_a_user_exposure_is_allowed(data, distribution):
    """Again scoped to the derived total: a user exposure leaves the totals free to vary.

    Fitted to counts drawn overdispersed in the way each family models, for the reason the
    information tests give: a negative binomial fitted to Poisson counts drives its
    overdispersion to zero, and the test would be measuring that instead.
    """
    annotations, bases, foci, _, _ = data
    rng = np.random.default_rng(5)
    if distribution == "clusterednegativebinomial":
        # One factor per experiment, shared across the brain.
        dispersed = rng.poisson(foci * rng.gamma(6.0, 1 / 6.0, size=(foci.shape[0], 1)))
    else:
        # Independent gamma variation at each voxel.
        dispersed = rng.poisson(rng.gamma(6.0, foci / 6.0))
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(diagnosis) + exposure(scan_volume)"), annotations), bases
    )
    model = CBMRModel(predictor, distribution=distribution)
    model.fit(dispersed.astype(float), n_iter=60, tol=1e-8)
    assert np.all(np.isfinite(model.overdispersion()))
    assert model.overdispersion().min() > 1e-3


def test_a_hypothesis_naming_the_exposure_says_so(data):
    """Rather than surfacing as a generic unknown coefficient."""
    from nimare.meta.cbmr.contrasts import ContrastError, build_contrast

    annotations, _, _, _, _ = data
    bound = bind(Design.from_formula("~ s(diagnosis) + exposure(scan_volume)"), annotations)
    with pytest.raises(ContrastError, match="no coefficient to test"):
        build_contrast(bound, "scan_volume = 0")


def test_the_exposure_contrast_equals_the_normalized_plain_contrast(data):
    """The two routes to ``log p_A(v) - log p_B(v)`` agree.

    Normalizing a plain fit afterwards gives the same answer as conditioning during it, which is
    what makes the exposure a reparameterization rather than a different model.
    """
    exposed = _log_maps(_model("~ s(diagnosis) + exposure()", data))
    plain = _log_maps(_model("~ s(diagnosis)", data))
    normalizer = np.log(np.exp(plain).sum(axis=1))

    from_exposure = exposed[0] - exposed[1]
    from_normalizing = (plain[0] - plain[1]) - (normalizer[0] - normalizer[1])
    assert np.abs(from_exposure - from_normalizing).max() < 1e-3
