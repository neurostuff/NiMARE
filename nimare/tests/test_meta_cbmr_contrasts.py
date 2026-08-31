"""Tests for named hypothesis tests on a term-based CBMR model.

Three things are being held in place here.

*Parsing* is delegated to :meth:`patsy.DesignInfo.linear_constraint`, the same parser statsmodels
uses, so the tests check that its grammar actually reaches users rather than re-testing patsy.

*Translation* is the part that is easy to get silently wrong. A term's coefficients are not
always its levels -- an ``sz()`` factor is reparameterized -- so a hypothesis over levels has to
be pushed through ``level_map``. The invariant that pins it: ``~ s(dx)`` and ``~ sz(dx)`` span
the same column space, so they are the same model written two ways, and the same level contrast
must agree under both.

*Statistics* are checked against statsmodels on the materialized design, not against CBMR itself.
"""

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
    from nimare.meta.cbmr.contrasts import (
        ContrastError,
        build_contrast,
        evaluate_hypotheses,
        generate_hypotheses,
    )
    from nimare.meta.cbmr.distributions import Poisson
    from nimare.meta.cbmr.model import CBMRModel
    from nimare.meta.cbmr.predictor import CBMRPredictor
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_LEVEL = 10
N_VOXELS = 20
N_BASES = 4
LEVELS = ("a", "b", "c")


def _annotations():
    rng = np.random.default_rng(71)
    n = N_PER_LEVEL * len(LEVELS)
    return pd.DataFrame(
        {
            "dx": [level for level in LEVELS for _ in range(N_PER_LEVEL)],
            "n": rng.normal(size=n),
        }
    )


def _bases():
    rng = np.random.default_rng(72)
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    return raw / raw.sum(axis=1, keepdims=True)


def _simulate(formula, annotations, bases, seed=73):
    """Simulate foci from a known model with an interior maximum."""
    predictor = CBMRPredictor(bind(Design.from_formula(formula), annotations), bases)
    rng = np.random.default_rng(seed)
    spatial = torch.tensor(
        rng.normal(1.3, 0.25, (predictor.n_spatial_columns, N_BASES)), dtype=torch.float64
    )
    global_coef = (
        torch.tensor(rng.normal(0.0, 0.15, predictor.n_global_columns), dtype=torch.float64)
        if predictor.n_global_columns
        else None
    )
    with torch.no_grad():
        eta = predictor.linear_predictor(spatial, global_coef).numpy()
    return rng.poisson(np.exp(eta)).astype(float)


def _fit_to(formula, foci, annotations=None, bases=None):
    """Fit one formula to foci that already exist."""
    annotations = _annotations() if annotations is None else annotations
    bases = _bases() if bases is None else bases
    predictor = CBMRPredictor(bind(Design.from_formula(formula), annotations), bases)
    return CBMRModel(predictor, Poisson()).fit(foci, n_iter=3000, tol=1e-13)


def _fit(formula, annotations=None, bases=None, seed=73):
    """Simulate from ``formula`` and fit it back."""
    annotations = _annotations() if annotations is None else annotations
    bases = _bases() if bases is None else bases
    foci = _simulate(formula, annotations, bases, seed=seed)
    return _fit_to(formula, foci, annotations, bases), foci


@pytest.fixture(scope="module")
def fitted():
    """Return a three-level factor plus a scalar moderator, cell-means coded."""
    return _fit("~ s(dx) + n")


def _materialize(predictor):
    n_experiments = predictor.patterns.n_experiments
    spatial = np.einsum("ic,vb->ivcb", predictor.spatial_block, predictor.bases).reshape(
        n_experiments * predictor.n_voxels, -1
    )
    if predictor.global_block is None:
        return spatial
    return np.hstack([spatial, np.repeat(predictor.global_block, predictor.n_voxels, axis=0)])


# --------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "statement,expected_label",
    [
        ("dx[a] = dx[b]", "dx[a]_vs_dx[b]"),
        ("a = b", "a_vs_b"),  # bare labels, as the map keys show them
        ("a - b", "a-b"),  # bare difference expression; the label is what was written
        ("n = 0", "n"),  # against zero
        ("2 * a = b + c", "2*a_vs_b+c"),  # arithmetic
    ],
)
def test_patsy_grammar_reaches_users(fitted, statement, expected_label):
    """Confirm the delegated parser's grammar is usable through ``test``.

    Arithmetic and bare difference expressions come free with
    :meth:`patsy.DesignInfo.linear_constraint`; a hand-rolled parser had neither.
    """
    model, _ = fitted
    result = evaluate_hypotheses(model, statement)
    emitted = list(result["maps"]) + list(result["tables"])
    assert any(name.endswith(expected_label) for name in emitted), emitted


def test_a_non_zero_right_hand_side_shifts_the_estimate(fitted):
    """``a = 1`` must test against one, not against zero."""
    model, _ = fitted
    against_zero = evaluate_hypotheses(model, "a = 0")["maps"]["est_a"]
    against_one = evaluate_hypotheses(model, "a = 1")["maps"]["est_a_vs_1"]

    np.testing.assert_allclose(against_one, against_zero - 1.0, rtol=1e-10)


def test_a_list_is_tested_jointly(fitted):
    """Several statements are one generalized linear hypothesis, not several tests."""
    model, _ = fitted
    result = evaluate_hypotheses(model, ["a = b", "b = c"], name="joint")

    assert "chiSquare_joint" in result["maps"]
    assert np.all(result["maps"]["chiSquare_joint"] >= 0)
    # A joint hypothesis is a statement about a subspace, so there is no single effect size.
    assert "est_joint" not in result["maps"]


def test_unknown_coefficients_list_the_options(fitted):
    """A typo should name what the design actually has."""
    model, _ = fitted
    with pytest.raises(ContrastError, match="coefficient names"):
        evaluate_hypotheses(model, "a = z")


def test_cross_term_contrasts_are_refused(fitted):
    """A map and a number have no common scale, so this is refused rather than guessed."""
    model, _ = fitted
    with pytest.raises(ContrastError, match="must stay within one term"):
        evaluate_hypotheses(model, "a = n")


def test_a_degenerate_contrast_is_refused(fitted):
    """A hypothesis placing no weight anywhere tests nothing.

    Caught by patsy rather than here -- it reports "no variables appear in constraint" -- which is
    why there is no separate check for it in ``build_contrast``.
    """
    model, _ = fitted
    with pytest.raises(ContrastError, match="no variables appear"):
        evaluate_hypotheses(model, "a - a = 0")


# ---------------------------------------------------------------- translation


def test_level_contrasts_work_on_a_reparameterized_term():
    """A hypothesis over levels must work on an ``sz()`` term, whose coefficients are not levels.

    Before ``level_map`` existed this raised: the term's coefficients are named ``dx[sz1]`` and
    ``dx[sz2]``, so a user could only state hypotheses in a space with no interpretation. This is
    exactly the hypothesis-matrix versus contrast-matrix distinction ``hypr`` formalizes.
    """
    model, _ = _fit("~ sz(dx) + n")
    result = evaluate_hypotheses(model, "a = b")

    assert set(result["maps"]) == {
        "est_a_vs_b",
        "se_a_vs_b",
        "z_a_vs_b",
        "p_a_vs_b",
        "logp_a_vs_b",
    }
    assert np.all(np.isfinite(result["maps"]["z_a_vs_b"]))


def test_the_same_contrast_agrees_across_parameterizations():
    """``~ s(dx)`` and ``~ sz(dx)`` are one model written two ways, so a contrast must agree.

    Both span the same column space -- three free spatial maps -- so they fit the same
    log-intensity surface and a level contrast is the same question of both. If ``level_map`` were
    wrong, this is where it would show, and nothing else would notice.
    """
    annotations, bases = _annotations(), _bases()
    # One dataset, fitted twice. Simulating separately would give each parameterization its own
    # truth and so different foci, which would compare nothing.
    foci = _simulate("~ s(dx) + n", annotations, bases)
    cell_means = _fit_to("~ s(dx) + n", foci, annotations, bases)
    constrained = _fit_to("~ sz(dx) + n", foci, annotations, bases)

    for statement in ("a = b", "b = c", "2 * a = b + c"):
        first = evaluate_hypotheses(cell_means, statement)["maps"]
        second = evaluate_hypotheses(constrained, statement)["maps"]
        key = next(k for k in first if k.startswith("z_"))
        np.testing.assert_allclose(
            second[key], first[key], rtol=1e-4, atol=1e-6, err_msg=statement
        )


# ----------------------------------------------------------------- statistics


def test_estimate_and_standard_error_maps_are_emitted(fitted):
    """A contrast must report its effect size, not only its significance.

    NiMARE's own canonical map set is ``["z", "p", "logp", "est", "se", "dof"]`` and nilearn's
    Contrast exposes ``effect_size``/``effect_variance``; emitting only ``z`` would put CBMR at
    odds with both, and with its own scalar path.
    """
    model, _ = fitted
    maps = evaluate_hypotheses(model, "a = b")["maps"]

    assert {"est_a_vs_b", "se_a_vs_b"} <= set(maps)
    assert np.all(maps["se_a_vs_b"] > 0)
    np.testing.assert_allclose(
        maps["z_a_vs_b"], maps["est_a_vs_b"] / maps["se_a_vs_b"], rtol=1e-10
    )


def test_spatial_contrast_matches_statsmodels(fitted):
    """The contrast must equal what an independent GLM reports for the same linear combination."""
    model, foci = fitted
    predictor = model.predictor
    design = _materialize(predictor)
    expected_fit = statsmodels_api.GLM(
        foci.reshape(-1), design, family=statsmodels_api.families.Poisson()
    ).fit()
    covariance = np.asarray(expected_fit.cov_params())

    maps = evaluate_hypotheses(model, "a = b")["maps"]

    names = list(predictor.design.blocks[0].column_names)
    index_a, index_b = names.index("dx[a]"), names.index("dx[b]")
    expected_est = np.empty(N_VOXELS)
    expected_z = np.empty(N_VOXELS)
    for voxel in range(N_VOXELS):
        weights = np.zeros(design.shape[1])
        row = predictor.bases[voxel]
        weights[index_a * N_BASES : (index_a + 1) * N_BASES] = row
        weights[index_b * N_BASES : (index_b + 1) * N_BASES] = -row
        expected_est[voxel] = weights @ expected_fit.params
        expected_z[voxel] = expected_est[voxel] / np.sqrt(weights @ covariance @ weights)

    np.testing.assert_allclose(maps["est_a_vs_b"], expected_est, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(maps["z_a_vs_b"], expected_z, rtol=1e-4, atol=1e-6)


def test_scalar_contrast_matches_statsmodels(fitted):
    """The scalar path reports the same quantities, in a table rather than maps."""
    model, foci = fitted
    expected_fit = statsmodels_api.GLM(
        foci.reshape(-1),
        _materialize(model.predictor),
        family=statsmodels_api.families.Poisson(),
    ).fit()

    table = evaluate_hypotheses(model, "n = 0")["tables"]["contrast_n"]

    assert {"est", "se", "z", "p", "logp"} <= set(table.columns)
    np.testing.assert_allclose(table["est"].iloc[0], expected_fit.params[-1], rtol=1e-4)
    np.testing.assert_allclose(table["se"].iloc[0], expected_fit.bse[-1], rtol=1e-4)


def test_reversing_a_contrast_flips_its_sign(fitted):
    """A sanity check that the direction means what it reads as."""
    model, _ = fitted
    forward = evaluate_hypotheses(model, "a = b")["maps"]
    reverse = evaluate_hypotheses(model, "b = a")["maps"]

    np.testing.assert_allclose(forward["est_a_vs_b"], -reverse["est_b_vs_a"], rtol=1e-10)
    np.testing.assert_allclose(forward["se_a_vs_b"], reverse["se_b_vs_a"], rtol=1e-10)


# ------------------------------------------------------------ named families


@pytest.mark.parametrize(
    "method,expected",
    [
        ("pairwise", ["a_vs_b", "a_vs_c", "b_vs_c"]),
        ("reference", ["b_vs_a", "c_vs_a"]),
        ("consecutive", ["b_vs_a", "c_vs_b"]),
        ("zero", ["a", "b", "c"]),
    ],
)
def test_named_families_enumerate_the_comparisons(fitted, method, expected):
    """Enumerate every comparison, instead of asking the user for one call per pair."""
    model, _ = fitted
    labels = [label for label, _ in generate_hypotheses(model.predictor.design, "dx", method)]
    assert labels == expected


def test_pairwise_emits_one_set_of_maps_per_pair(fitted):
    """Each generated contrast gets its own label, as gratia's difference_smooths does."""
    model, _ = fitted
    maps = evaluate_hypotheses(model, term="dx", method="pairwise")["maps"]

    assert sorted(k for k in maps if k.startswith("z_")) == ["z_a_vs_b", "z_a_vs_c", "z_b_vs_c"]
    assert sorted(k for k in maps if k.startswith("est_")) == [
        "est_a_vs_b",
        "est_a_vs_c",
        "est_b_vs_c",
    ]


def test_a_generated_family_agrees_with_the_hand_written_contrast(fitted):
    """Generating a contrast must not compute anything different from naming it."""
    model, _ = fitted
    generated = evaluate_hypotheses(model, term="dx", method="pairwise")["maps"]
    named = evaluate_hypotheses(model, "a = b")["maps"]

    np.testing.assert_allclose(generated["z_a_vs_b"], named["z_a_vs_b"], rtol=1e-12)


def test_a_single_coefficient_term_cannot_be_compared(fitted):
    """``pairwise`` needs levels to pair; a scalar term has one coefficient."""
    model, foci = fitted
    with pytest.raises(ContrastError, match="only one coefficient"):
        generate_hypotheses(model.predictor.design, "n", "pairwise")


def test_zero_works_on_a_single_coefficient_term(fitted):
    """The alternative the error points at must exist."""
    model, foci = fitted
    assert generate_hypotheses(model.predictor.design, "n", "zero") == [("n", "n = 0")]


def test_unknown_method_and_term_are_reported(fitted):
    """Both halves of the generated form should fail helpfully."""
    model, foci = fitted
    with pytest.raises(ContrastError, match="method must be one of"):
        generate_hypotheses(model.predictor.design, "dx", "tukey")
    with pytest.raises(ContrastError, match="No term named"):
        generate_hypotheses(model.predictor.design, "diagnosis", "pairwise")


def test_the_two_forms_are_mutually_exclusive(fitted):
    """Naming a hypothesis and generating a family are alternatives, not combinable."""
    model, _ = fitted
    with pytest.raises(ContrastError, match="not both"):
        evaluate_hypotheses(model, "a = b", term="dx", method="pairwise")
    with pytest.raises(ContrastError, match="together"):
        evaluate_hypotheses(model, term="dx")
    with pytest.raises(ContrastError, match="Give either"):
        evaluate_hypotheses(model)


def test_build_contrast_pushes_through_the_level_map():
    """The translation is a matrix operation, so assert it directly as well as end to end."""
    annotations = _annotations()
    cell_means = bind(Design.from_formula("~ s(dx)"), annotations)
    constrained = bind(Design.from_formula("~ sz(dx)"), annotations)

    _, plain, _, _ = build_contrast(cell_means, "a = b")
    _, mapped, _, _ = build_contrast(constrained, "a = b")

    np.testing.assert_allclose(plain, [[1.0, -1.0, 0.0]])
    # Same hypothesis, expressed over two sz coefficients instead of three levels.
    block = next(b for b in constrained.blocks if b.term.is_sum_to_zero)
    np.testing.assert_allclose(mapped, np.array([[1.0, -1.0, 0.0]]) @ block.level_map)
    assert mapped.shape == (1, 2)
