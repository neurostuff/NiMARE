"""Tests for the CBMR term and design layer."""

import numpy as np
import pandas as pd
import pytest

from nimare.meta.cbmr.terms import (
    Design,
    FormulaError,
    Term,
    _spans_intercept,
    bind,
    formula_to_design,
    sum_to_zero_basis,
)

N_BASES = 457  # what spline_spacing=10 yields on the 2 mm brain mask


@pytest.fixture
def annotations():
    """Experiment annotations with two factors and two covariates."""
    return pd.DataFrame(
        {
            "diagnosis": ["schiz", "schiz", "dep", "dep", "schiz", "dep"],
            "drug": ["yes", "no", "yes", "no", "no", "yes"],
            "n": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "age": [21.0, 34.0, 45.0, 52.0, 29.0, 38.0],
        }
    )


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("~ 1", "~ 1"),
        ("~ s(diagnosis)", "~ s(diagnosis)"),
        ("s(diagnosis)", "~ s(diagnosis)"),  # leading ~ optional
        ("~ s(diagnosis:drug) + n + s(age)", "~ s(diagnosis:drug) + n + s(age)"),
        ("~ spatial(age)", "~ s(age)"),  # alias normalizes
        ("~ s( age )", "~ s(age)"),  # whitespace tolerated
        ("~ s(age, spacing=5)", "~ s(age, spacing=5)"),
    ],
)
def test_formula_round_trips(formula, expected):
    """Parsing then rendering a formula should return a normalized equivalent."""
    assert str(Design.from_formula(formula)) == expected


def test_zero_plus_is_accepted_and_ignored():
    """``0 +`` describes what CBMR already does, so it parses to the same design.

    The B-spline basis is a partition of unity and so already spans the constant; there is
    never a scalar intercept column to suppress. ``0 +`` is the established idiom for this
    basis elsewhere, so it is accepted rather than rejected, but it changes nothing.
    """
    assert Design.from_formula("0 + s(diagnosis)") == Design.from_formula("~ s(diagnosis)")


def test_a_formula_of_only_zero_is_rejected():
    """``~ 0`` asks for no terms at all, which is not a model."""
    with pytest.raises(FormulaError, match="no terms"):
        Design.from_formula("~ 0")


@pytest.mark.parametrize(
    "formula,expected_terms",
    [
        ("~ s(diagnosis) + n", (True, False)),
        ("~ n + s(age)", (False, True)),
        ("~ s(diagnosis) + s(age)", (True, True)),
    ],
)
def test_spatial_marker_is_per_term(formula, expected_terms):
    """``s()`` marks individual terms, not the model.

    The whole point of the redesign: a model can mix a scalar moderator with a spatially
    varying one, which the old ``moderator_effect`` switch could only express by adding a
    third "mixed" mode.
    """
    design = Design.from_formula(formula)
    assert tuple(term.spatial for term in design.terms) == expected_terms


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("~ 1", 457),
        ("~ s(diagnosis)", 914),
        ("~ s(diagnosis:drug)", 1828),
        ("~ s(diagnosis) + n", 915),
        ("~ s(diagnosis) + s(age)", 1371),
        ("~ n", 458),
        ("~ diagnosis:n", 459),
    ],
)
def test_parameter_counts(annotations, formula, expected):
    """Each ``s()`` term costs one basis worth of coefficients per column.

    These numbers are the argument for reporting the budget at fit time: making one moderator
    spatial costs as much as another group's entire baseline map.
    """
    assert bind(Design.from_formula(formula), annotations).n_parameters(N_BASES) == expected


def test_spatial_baseline_is_implicit(annotations):
    """A design with nothing spanning the constant gains a spatial baseline, as R adds one."""
    bound = bind(Design.from_formula("~ n"), annotations)
    assert str(bound.design) == "~ 1 + n"
    assert bound.terms[0].is_intercept and bound.terms[0].spatial


def test_cell_means_factor_absorbs_the_baseline(annotations):
    """A spatial factor already gives every level a map, so no baseline is added."""
    bound = bind(Design.from_formula("~ s(diagnosis)"), annotations)
    assert str(bound.design) == "~ s(diagnosis)"
    assert not any(term.is_intercept for term in bound.terms)


def test_explicit_intercept_with_cell_means_factor_is_rejected(annotations):
    """``~ 1 + s(factor)`` is ambiguous and near-singular, so it must fail loudly.

    The factor's per-level columns sum to the constant, so a separate baseline is collinear
    with them. Because that collinearity is near rather than exact -- the support filter drops
    a little basis mass at the brain edge -- it would not trip a rank check, and the fit would
    degrade silently instead of erroring.
    """
    with pytest.raises(FormulaError, match="cannot be combined"):
        bind(Design.from_formula("~ 1 + s(diagnosis)"), annotations)


def test_explicit_intercept_without_a_factor_is_fine(annotations):
    """``~ 1 + n`` is just the implicit form written out, so it must be accepted."""
    bound = bind(Design.from_formula("~ 1 + n"), annotations)
    assert str(bound.design) == "~ 1 + n"
    assert bound.n_parameters(N_BASES) == 458


def test_non_spatial_factor_does_not_absorb_the_baseline(annotations):
    """A factor only spans the constant when it is crossed with the basis.

    ``~ diagnosis`` asks for scalar per-level offsets on top of a shared spatial baseline, so
    the baseline still has to be supplied.
    """
    bound = bind(Design.from_formula("~ diagnosis"), annotations)
    assert bound.terms[0].is_intercept
    assert bound.n_parameters(N_BASES) == N_BASES + 2


def test_additive_spatial_factors_are_rejected_as_unidentifiable(annotations):
    """``~ s(a) + s(b)`` is rank deficient by a whole basis width, so it must be refused.

    Each cell-means spatial factor's columns sum to the constant, so the two sums are equal and
    their difference is exactly zero -- verified as a rank deficiency of exactly ``n_bases``,
    independent of the data. Identifying it needs sum-to-zero constraints across levels, which
    mgcv supplies via its ``sz`` basis and NiMARE does not implement. Refusing beats fitting a
    singular design and reporting standard errors for it.
    """
    with pytest.raises(FormulaError, match="not jointly identifiable"):
        bind(Design.from_formula("~ s(diagnosis) + s(drug)"), annotations)


def test_one_spatial_factor_plus_a_non_spatial_one_is_fine(annotations):
    """Only *spatial* factors compete for the constant, so this combination is identified."""
    bound = bind(Design.from_formula("~ s(diagnosis) + drug"), annotations)
    assert len(bound.baseline_blocks) == 1


def test_spatial_interaction_is_the_identified_alternative(annotations):
    """The form the error points to must actually work."""
    bound = bind(Design.from_formula("~ s(diagnosis:drug)"), annotations)
    assert len(bound.baseline_blocks) == 1
    assert bound.blocks[0].n_columns == 4


def test_sum_to_zero_basis_spans_the_centered_subspace():
    """The reparameterization basis must be orthonormal and sum to zero down each column.

    This is the construction mgcv documents: a QR decomposition of the all-ones vector with its
    first column dropped. Column signs are pinned because QR fixes the subspace but not the
    orientation, and an unpinned sign would make coefficients flip between LAPACK versions.
    """
    for n_levels in (2, 3, 5):
        basis = sum_to_zero_basis(n_levels)
        assert basis.shape == (n_levels, n_levels - 1)
        np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(basis.T @ basis, np.eye(n_levels - 1), atol=1e-12)
        assert np.all(basis[0] > 0), "column signs should be pinned, not left to QR"

    np.testing.assert_allclose(sum_to_zero_basis(4), sum_to_zero_basis(4))


def test_sum_to_zero_needs_at_least_two_levels():
    """There is nothing to center a single column against."""
    with pytest.raises(FormulaError, match="at least two levels"):
        sum_to_zero_basis(1)


@pytest.mark.parametrize("formula", ["~ sz(diagnosis)", '~ s(diagnosis, bs="sz")'])
def test_sz_is_parsed_and_rendered(formula):
    """``sz(x)`` and mgcv's ``s(x, bs="sz")`` should mean the same term."""
    design = Design.from_formula(formula)
    assert str(design) == "~ sz(diagnosis)"
    assert design.terms[0].is_sum_to_zero


def test_sz_drops_a_column_and_stops_spanning_the_constant(annotations):
    """A constrained factor has one fewer column and no longer competes with the baseline."""
    bound = bind(Design.from_formula("~ sz(diagnosis)"), annotations)
    constrained = next(b for b in bound.blocks if b.term.is_sum_to_zero)

    assert constrained.n_columns == 1, "two levels minus one constraint"
    assert not _spans_intercept(constrained.block)
    assert not constrained.is_baseline
    # A baseline is therefore supplied separately.
    assert str(bound.design) == "~ 1 + sz(diagnosis)"


def test_additive_sz_factors_are_identifiable(annotations):
    """``sz(a) + sz(b)`` is what ``s(a) + s(b)`` should have been."""
    bound = bind(Design.from_formula("~ sz(diagnosis) + sz(drug)"), annotations)
    assert len(bound.baseline_blocks) == 1
    assert str(bound.design) == "~ 1 + sz(diagnosis) + sz(drug)"


def test_additive_sz_is_cheaper_than_the_interaction(annotations):
    """The additive form is the stronger claim, and its parameter count shows it."""
    additive = bind(Design.from_formula("~ sz(diagnosis) + sz(drug)"), annotations)
    interaction = bind(Design.from_formula("~ s(diagnosis:drug)"), annotations)
    assert additive.n_parameters(N_BASES) < interaction.n_parameters(N_BASES)


def test_sz_columns_are_named_as_contrasts_not_levels(annotations):
    """Constrained columns are contrasts among levels, so naming them as levels would misread."""
    bound = bind(Design.from_formula("~ sz(diagnosis)"), annotations)
    constrained = next(b for b in bound.blocks if b.term.is_sum_to_zero)
    assert constrained.column_names == ("diagnosis[sz1]",)


def test_sz_on_a_non_spatial_term_is_rejected():
    """The constraint exists to stop a factor competing with the spatial baseline."""
    with pytest.raises(FormulaError, match="meaningless for the non-spatial term"):
        Term(expr="diagnosis", spatial=False, constraint="sum_to_zero")


def test_unknown_constraint_is_rejected():
    """Only one constraint exists; a typo should say so."""
    with pytest.raises(FormulaError, match="Unknown constraint"):
        Term(expr="diagnosis", spatial=True, constraint="centered")


def test_unknown_basis_is_rejected():
    """``bs=`` only accepts the sum-to-zero basis."""
    with pytest.raises(FormulaError, match="the only one is"):
        Design.from_formula('~ s(diagnosis, bs="tp")')


def test_s_and_sz_of_the_same_factor_are_distinct_terms():
    """They are different parameterizations, so neither should look like a duplicate."""
    design = Design.from_formula("~ s(diagnosis) + sz(drug)")
    assert len(design.terms) == 2


def test_the_rejection_message_points_at_both_alternatives(annotations):
    """A refusal should say what to write instead, in both available forms."""
    with pytest.raises(FormulaError) as caught:
        bind(Design.from_formula("~ s(diagnosis) + s(drug)"), annotations)
    message = str(caught.value)
    assert "sz(diagnosis) + sz(drug)" in message
    assert "s(diagnosis:drug)" in message


def test_spans_intercept_distinguishes_factors_from_covariates():
    """Only cell-means indicator blocks span the constant."""
    assert _spans_intercept(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    assert not _spans_intercept(np.array([[1.0, 2.0], [3.0, 4.0]]))
    # Treatment-coded columns do not sum to one per row.
    assert not _spans_intercept(np.array([[0.0], [1.0], [1.0]]))


def test_per_term_spacing_is_parsed():
    """``s(x, spacing=N)`` overrides the model spacing for that term only."""
    design = Design.from_formula("~ s(diagnosis) + s(age, spacing=5)")
    assert design.terms[0].spacing is None
    assert design.terms[1].spacing == 5


def test_spacing_on_a_non_spatial_term_is_rejected():
    """A term with no basis has no spacing to set."""
    with pytest.raises(FormulaError, match="meaningless"):
        Term(expr="age", spatial=False, spacing=5)


def test_duplicate_terms_are_rejected():
    """Repeating a term would silently double its columns."""
    with pytest.raises(FormulaError, match="more than once"):
        Design.from_formula("~ s(diagnosis) + s(diagnosis)")


def test_a_term_and_its_spatial_version_can_coexist():
    """``n + s(n)`` is a scalar effect plus a spatially varying one, which is a real model."""
    design = Design.from_formula("~ n + s(n)")
    assert tuple(term.spatial for term in design.terms) == (False, True)


def test_unknown_column_names_the_available_ones(annotations):
    """A typo in a formula should say what could have been meant."""
    with pytest.raises(FormulaError, match="Available columns: age, diagnosis, drug, n"):
        bind(Design.from_formula("~ s(diagnosiss)"), annotations)


def test_empty_and_malformed_formulas_are_rejected():
    """Syntax errors should surface as FormulaError, not as something from patsy or re."""
    for bad in ["", "~", "~ s(diagnosis) +", "~ s(", "~ s()"]:
        with pytest.raises(FormulaError):
            Design.from_formula(bad)


def test_non_string_formula_is_rejected():
    """A design object passes through, but other non-strings should not."""
    with pytest.raises(FormulaError, match="must be a string"):
        Design.from_formula(["~ 1"])


def test_formula_to_design_passes_designs_through():
    """Callers may supply either a formula or an already-parsed design."""
    design = Design.from_formula("~ s(diagnosis)")
    assert formula_to_design(design) is design
    assert formula_to_design("~ s(diagnosis)") == design


def test_parameter_slices_tile_the_coefficient_vector(annotations):
    """Slices must be contiguous, disjoint, and cover exactly the parameter vector.

    Covariance estimation indexes coefficient blocks through this layout, so a gap or an
    overlap would silently mix one term's parameters into another's standard errors.
    """
    bound = bind(Design.from_formula("~ s(diagnosis) + n + s(age)"), annotations)
    slices = bound.parameter_slices(N_BASES)

    assert len(slices) == len(bound.terms)
    covered = np.zeros(bound.n_parameters(N_BASES), dtype=int)
    for term_slice in slices.values():
        covered[term_slice] += 1
    assert np.all(covered == 1)

    ordered = [slices[str(term)] for term in bound.terms]
    assert ordered[0].start == 0
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.stop == later.start


def test_describe_reports_the_per_term_budget(annotations):
    """The budget belongs in front of users, since each s() term hides a basis worth of them."""
    bound = bind(Design.from_formula("~ s(diagnosis) + n"), annotations)
    described = bound.describe(N_BASES)

    assert "s(diagnosis)" in described and "914" in described
    assert "total" in described and str(bound.n_parameters(N_BASES)) in described


def test_experiment_blocks_have_one_row_per_experiment(annotations):
    """Every term's block is indexed by experiment, whatever its column count."""
    bound = bind(Design.from_formula("~ s(diagnosis:drug) + n + s(age)"), annotations)
    for block in bound.blocks:
        assert block.block.shape[0] == len(annotations)
        assert block.n_columns == len(block.column_names)


def test_cell_means_coding_gives_one_column_per_level(annotations):
    """A spatial factor must expand to disjoint per-level blocks, not reference contrasts."""
    bound = bind(Design.from_formula("~ s(diagnosis)"), annotations)
    block = bound.blocks[0]

    assert block.n_columns == 2, "expected one column per diagnosis, not a reference contrast"
    np.testing.assert_array_equal(block.block.sum(axis=1), np.ones(len(annotations)))
