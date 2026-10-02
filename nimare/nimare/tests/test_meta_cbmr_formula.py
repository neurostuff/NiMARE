"""End-to-end tests for the formula-specified CBMR estimator."""

import logging
import os
import warnings

import numpy as np
import pytest

statsmodels_api = pytest.importorskip("statsmodels.api")

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr import CBMR
    from nimare.meta.cbmr.distributions import DistributionError
    from nimare.meta.cbmr.terms import FormulaError

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

FIT_KWARGS = dict(
    spline_spacing=100,
    n_iter=100,
    tol=1e2,
    device="cpu",
    random_state=1,
    generate_description=False,
)


@pytest.fixture(scope="module")
def studyset():
    """Return a small simulated Studyset with two factors and two standardized covariates."""
    from nimare.generate import create_coordinate_studyset
    from nimare.transforms import StandardizeField

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, studyset = create_coordinate_studyset(
            foci=10, sample_size=(20, 40), n_studies=40, seed=11
        )
    annotations = studyset.annotations_df.copy()
    n_rows = annotations.shape[0]
    pattern = [
        ("schizophrenia", "Yes"),
        ("schizophrenia", "No"),
        ("depression", "Yes"),
        ("depression", "No"),
    ]
    annotations[["diagnosis", "drug_status"]] = [pattern[i % 4] for i in range(n_rows)]
    annotations["sample_sizes"] = [studyset.metadata.sample_sizes[i][0] for i in range(n_rows)]
    annotations["avg_age"] = np.arange(n_rows, dtype=float)
    # A Studyset is immutable, so the edited frame is attached to a new one.
    studyset = studyset.with_annotations_df(annotations, name="moderators", replace=True)
    return StandardizeField(fields=["sample_sizes", "avg_age"]).transform(studyset)


def _fit(formula, studyset, **overrides):
    return CBMR(formula, **{**FIT_KWARGS, **overrides}).fit(dataset=studyset)


def test_group_design_yields_one_intensity_map_per_cell(studyset):
    """A spatial interaction of two factors should give a map per combination of levels."""
    result = _fit("~ s(diagnosis:drug_status)", studyset)

    expected = {
        f"spatialIntensity_group-{diagnosis}-{drug}"
        for diagnosis in ("schizophrenia", "depression")
        for drug in ("Yes", "No")
    }
    assert expected <= set(result.maps)
    assert all(np.all(np.isfinite(result.maps[name])) for name in expected)
    assert all(np.all(result.maps[name] >= 0) for name in expected), "intensities are rates"


def test_map_labels_come_from_factor_levels(studyset):
    """Labels should read like the levels, not like patsy column names."""
    result = _fit("~ s(diagnosis)", studyset)

    assert "spatialIntensity_group-schizophrenia" in result.maps
    assert not any("[" in name for name in result.maps), "patsy syntax leaked into map names"


def test_scalar_moderator_lands_in_a_table(studyset):
    """A term without ``s()`` has one coefficient, so it belongs in a table not a map."""
    result = _fit("~ s(diagnosis) + standardized_sample_sizes", studyset)

    table = result.tables["moderatorEffect_standardized_sample_sizes"]
    assert list(table.columns) == ["column", "est", "se"]
    assert len(table) == 1
    assert not any("standardized_sample_sizes" in name for name in result.maps)


def test_spatial_moderator_lands_in_a_map(studyset):
    """A term with ``s()`` has a coefficient per voxel, so it belongs in a map.

    The output type follows from the formula, which is the point of marking resolution per term.
    """
    result = _fit("~ s(diagnosis) + s(standardized_avg_age)", studyset)

    assert "voxelwiseModeratorEffect_standardized_avg_age" in result.maps
    assert result.maps["voxelwiseModeratorEffect_standardized_avg_age"].shape == (
        result.maps["spatialIntensity_group-schizophrenia"].shape
    )


def test_mixed_design_splits_moderators_by_marker(studyset):
    """One scalar moderator and one spatial moderator in a single model.

    This is what the old API needed a third ``moderator_effect="mixed"`` mode plus two extra
    keyword arguments to express.
    """
    result = _fit(
        "~ s(diagnosis:drug_status) + standardized_sample_sizes + s(standardized_avg_age)",
        studyset,
    )

    assert "voxelwiseModeratorEffect_standardized_avg_age" in result.maps
    assert "moderatorEffect_standardized_sample_sizes" in result.tables


def test_group_specific_scalar_slope(studyset):
    """``diagnosis:n`` gives one scalar slope per diagnosis, which the old API could not.

    Global moderator coefficients were pooled across groups by construction, since
    ``moderators_linear`` was a single Linear rather than a ModuleDict.
    """
    result = _fit("~ diagnosis:standardized_sample_sizes", studyset)

    table = result.tables["moderatorEffect_diagnosis-standardized_sample_sizes"]
    assert len(table) == 2, "expected one slope per diagnosis"
    assert np.all(np.isfinite(table["est"]))


def test_pooled_spatial_moderator(studyset):
    """``s(n)`` alone is a spatial moderator pooled across groups, also newly expressible.

    The old voxelwise path keyed moderator coefficients by group, so a pooled spatially varying
    moderator had no representation.
    """
    result = _fit("~ s(standardized_sample_sizes)", studyset)

    assert "voxelwiseModeratorEffect_standardized_sample_sizes" in result.maps
    assert "spatialIntensity_group-Default" in result.maps


def test_unidentifiable_additive_spatial_factors_are_refused(studyset):
    """Refused at bind time rather than fitted as a singular design."""
    with pytest.raises(FormulaError, match="not jointly identifiable"):
        _fit("~ s(diagnosis) + s(drug_status)", studyset)


def test_additive_spatial_factors_work_when_constrained(studyset):
    """``sz()`` is the identified way to write additive spatial main effects.

    Each factor's coefficients are constrained to sum to zero across levels, so they measure
    deviations from the baseline that ``1`` supplies rather than competing with it.
    """
    result = _fit("~ sz(diagnosis) + sz(drug_status)", studyset)

    assert "spatialIntensity_group-Default" in result.maps
    assert "spatialFactorEffect_diagnosis-sz1" in result.maps
    assert "spatialFactorEffect_drug_status-sz1" in result.maps
    assert all(np.all(np.isfinite(values)) for values in result.maps.values())


def test_constrained_factor_is_cheaper_than_the_interaction(studyset):
    """The additive claim costs fewer parameters than one free map per cell."""
    additive = CBMR("~ sz(diagnosis) + sz(drug_status)", **FIT_KWARGS)
    interaction = CBMR("~ s(diagnosis:drug_status)", **FIT_KWARGS)
    additive.fit(dataset=studyset)
    interaction.fit(dataset=studyset)

    n_bases = additive.predictor.n_bases
    assert additive.bound_design.n_parameters(n_bases) < interaction.bound_design.n_parameters(
        n_bases
    )


def test_overdispersion_with_a_grouped_design(studyset):
    """Overdispersion should fit where several experiments share a spatial map."""
    result = _fit("~ s(diagnosis)", studyset, distribution="negativebinomial", lr=1e-2)

    overdispersion = result.tables["overdispersion"]["overdispersion"].to_numpy()
    assert len(overdispersion) == 2
    assert np.all(overdispersion > 0)


def test_overdispersion_with_a_spatial_covariate_is_refused(studyset):
    """No two experiments share a map, so there is nothing to estimate overdispersion from."""
    with pytest.raises(DistributionError, match="overdispersion"):
        _fit("~ s(standardized_avg_age)", studyset, distribution="negativebinomial")


def test_parameter_budget_is_logged(studyset, caplog):
    """Each ``s()`` term costs a basis width per column; the user should be told."""
    with caplog.at_level(logging.INFO, logger="nimare.meta.cbmr.estimator"):
        _fit("~ s(diagnosis) + standardized_sample_sizes", studyset)

    logged = "\n".join(caplog.messages)
    assert "s(diagnosis)" in logged and "parameters" in logged
    assert "distinct spatial map" in logged


def test_unknown_column_is_reported_before_fitting(studyset):
    """A typo should fail with the available columns, not deep inside patsy."""
    with pytest.raises(FormulaError, match="Available columns"):
        _fit("~ s(diagnosiss)", studyset)


def test_description_names_the_design(studyset):
    """The generated description should record what was actually fitted."""
    result = CBMR(
        "~ s(diagnosis) + standardized_sample_sizes",
        **{**FIT_KWARGS, "generate_description": True},
    ).fit(dataset=studyset)

    assert "s(diagnosis) + standardized_sample_sizes" in result.description_
    assert "Poisson" in result.description_
    # The parameter budget belongs in the methods paragraph, not only in a log line.
    assert "coefficients" in result.description_


@pytest.mark.parametrize(
    "distribution,citation",
    [
        ("poisson", "eisenberg1966general"),
        ("negativebinomial", "barndorff1969negative"),
        ("clusterednegativebinomial", "geoffroy2001poisson"),
    ],
)
def test_description_cites_the_distribution(studyset, distribution, citation):
    """Each distribution must cite its source, and the key must resolve in the bundled BibTeX.

    An unresolvable key is not merely untidy: NiMARE warns and silently drops it from the
    reference list, so a generated methods section would cite nothing.
    """
    from nimare.utils import get_resource_path

    result = CBMR(
        "~ s(diagnosis)",
        distribution=distribution,
        **{**FIT_KWARGS, "generate_description": True, "lr": 1e-2},
    ).fit(dataset=studyset)

    assert citation in result.description_
    with open(os.path.join(get_resource_path(), "references.bib")) as handle:
        assert f"{{{citation}," in handle.read()


def test_result_tests_hypotheses_by_name(studyset):
    """``result.test()`` replaces the positional, order-dependent contrast matrices.

    The old interface needed group_contrasts=[[[1, -1, 0, 0], ...]], where reordering the levels
    silently changed the hypothesis.
    """
    result = _fit("~ s(diagnosis)", studyset)
    tested = result.test("diagnosis[schizophrenia] = diagnosis[depression]", name="dx")

    for prefix in ("z", "p", "logp"):
        assert f"{prefix}_dx" in tested.maps
    assert np.all(np.isfinite(tested.maps["z_dx"]))
    # Default is a copy, so the original is untouched.
    assert "z_dx" not in result.maps


def test_result_tests_hypotheses_jointly(studyset):
    """A list of statements is one generalized linear hypothesis, not several tests."""
    result = _fit("~ s(diagnosis:drug_status)", studyset)
    tested = result.test(
        [
            "schizophrenia-Yes = schizophrenia-No",
            "depression-Yes = depression-No",
        ],
        name="drug",
    )

    assert "chiSquare_drug" in tested.maps
    assert np.all(tested.maps["chiSquare_drug"] >= 0)


def test_result_tests_a_scalar_moderator(studyset):
    """A non-spatial term's test is a table row, matching where its coefficient lives."""
    result = _fit("~ s(diagnosis) + standardized_sample_sizes", studyset)
    tested = result.test("standardized_sample_sizes = 0", name="n")

    table = tested.tables["contrast_n"]
    assert np.isfinite(table["z"].iloc[0])


def test_result_can_test_in_place(studyset):
    """inplace=True should mutate the result rather than copy it."""
    result = _fit("~ s(diagnosis)", studyset)
    returned = result.test("diagnosis[schizophrenia] = 0", name="s", inplace=True)

    assert returned is result
    assert "z_s" in result.maps


def test_result_reports_the_term_budget(studyset):
    """The parameter budget should be reachable from the result, not just the fit log."""
    result = _fit("~ s(diagnosis) + standardized_sample_sizes", studyset)
    described = result.describe_terms()

    assert "s(diagnosis)" in described and "total" in described


def test_robust_statistics_are_available_from_the_result(studyset):
    """``result.test(cov_type="sandwich")`` should give robust statistics end to end.

    Model-based standard errors assume the Poisson mean-variance relationship. Foci are
    overdispersed and correlated within an experiment, so the clustered sandwich is the safer
    default for real data, and it must be reachable without dropping to the model object.
    """
    result = _fit("~ s(diagnosis) + standardized_sample_sizes", studyset)

    model_based = result.test("diagnosis[schizophrenia] = 0", name="m")
    robust = result.test(
        "diagnosis[schizophrenia] = 0", name="r", cov_type="sandwich", meat="cluster"
    )

    assert np.all(np.isfinite(robust.maps["z_r"]))
    # Different variance estimates, so different statistics for the same contrast.
    assert not np.allclose(model_based.maps["z_m"], robust.maps["z_r"])


def test_robust_scalar_contrast_differs_from_model_based(studyset):
    """The scalar path should honour the covariance option too."""
    result = _fit("~ s(diagnosis) + standardized_sample_sizes", studyset)

    plain = result.test("standardized_sample_sizes = 0", name="p")
    robust = result.test(
        "standardized_sample_sizes = 0", name="q", cov_type="sandwich", meat="cluster"
    )

    assert np.isfinite(robust.tables["contrast_q"]["z"].iloc[0])
    assert plain.tables["contrast_p"]["se"].iloc[0] != robust.tables["contrast_q"]["se"].iloc[0]


def test_moderator_effect_maps_are_added(studyset):
    """RI and ID express a spatial moderator's coefficient on interpretable scales."""
    result = _fit("~ s(diagnosis) + s(standardized_avg_age)", studyset)
    diagnosed = result.moderator_effect_maps()

    relative = [n for n in diagnosed.maps if n.startswith("relativeIntensity_")]
    difference = [n for n in diagnosed.maps if n.startswith("intensityDifference_")]
    assert len(relative) == 1
    # One intensity-difference map per baseline group, since ID is against a baseline.
    assert len(difference) == 2

    assert np.all(diagnosed.maps[relative[0]] > 0), "a ratio of intensities is positive"
    assert "relativeIntensity_standardized_avg_age_unit-1" in diagnosed.maps


def test_intensity_difference_is_the_baseline_times_the_ratio_less_one(studyset):
    """ID must equal baseline * (RI - 1), which is what makes it foci rather than a ratio."""
    result = _fit("~ s(diagnosis) + s(standardized_avg_age)", studyset)
    diagnosed = result.moderator_effect_maps()

    relative = diagnosed.maps["relativeIntensity_standardized_avg_age_unit-1"]
    baseline = diagnosed.maps["spatialIntensity_group-schizophrenia"]
    difference = diagnosed.maps[
        "intensityDifference_standardized_avg_age_unit-1_group-schizophrenia"
    ]

    np.testing.assert_allclose(difference, baseline * (relative - 1.0), rtol=1e-10)


def test_unit_change_scales_the_ratio_exponentially(studyset):
    """RI is exp(unit * coefficient), so doubling the unit squares the ratio."""
    result = _fit("~ s(diagnosis) + s(standardized_avg_age)", studyset)

    one = result.moderator_effect_maps(unit_change=1.0)
    two = result.moderator_effect_maps(unit_change=2.0)

    np.testing.assert_allclose(
        two.maps["relativeIntensity_standardized_avg_age_unit-2"],
        one.maps["relativeIntensity_standardized_avg_age_unit-1"] ** 2,
        rtol=1e-8,
    )


def test_a_design_without_a_spatial_moderator_cannot_be_diagnosed(studyset):
    """There is no coefficient map to express, so say what would give one."""
    result = _fit("~ s(diagnosis) + standardized_sample_sizes", studyset)
    with pytest.raises(ValueError, match="no spatial moderator"):
        result.moderator_effect_maps()


def test_unmatched_moderator_lists_the_options(studyset):
    """A typo should name the spatial terms that exist."""
    result = _fit("~ s(diagnosis) + s(standardized_avg_age)", studyset)
    with pytest.raises(ValueError, match="No spatial moderator matched"):
        result.moderator_effect_maps(moderators="standardized_sample_sizes")


def test_moderator_effects_can_be_plotted(studyset):
    """The plotting helper should pair the two scales, since apart they mislead."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    result = _fit("~ s(diagnosis) + s(standardized_avg_age)", studyset)
    figure = result.plot_moderator_effects(group="schizophrenia")

    assert len(figure.axes) >= 2
    matplotlib.pyplot.close(figure)


def test_plotting_asks_which_group_when_there_are_several(studyset):
    """Ambiguity should be reported with the available groups, not resolved arbitrarily."""
    pytest.importorskip("matplotlib")
    result = _fit("~ s(diagnosis) + s(standardized_avg_age)", studyset)

    with pytest.raises(ValueError, match="one baseline group"):
        result.plot_moderator_effects()


def _materialize_public_design(estimator):
    """Return the GLM design the fitted estimator implies, in flat-parameter order."""
    predictor = estimator.predictor
    n_experiments = predictor.patterns.n_experiments
    spatial = np.einsum("ic,vb->ivcb", predictor.spatial_block, predictor.bases).reshape(
        n_experiments * predictor.n_voxels, -1
    )
    if predictor.global_block is None:
        return spatial
    return np.hstack([spatial, np.repeat(predictor.global_block, predictor.n_voxels, axis=0)])


@pytest.mark.parametrize(
    "formula",
    [
        "~ s(diagnosis)",
        "~ s(diagnosis:drug_status) + standardized_sample_sizes",
        "~ s(diagnosis) + s(standardized_avg_age)",
        "~ sz(diagnosis) + sz(drug_status)",
    ],
)
def test_the_public_fit_optimizes_the_statsmodels_likelihood(studyset, formula):
    """The objective CBMR optimizes must be the Poisson likelihood, checked end to end.

    Every other statsmodels comparison in the suite builds a predictor directly; this one goes
    through ``CBMR.fit`` on a Studyset, so it covers the whole path -- masking, incidence
    filtering, the basis, the foci matrix, the term layout and the flat parameter vector.

    It compares the likelihood *at CBMR's own fitted coefficients* rather than comparing optima.
    That is deliberate. Coordinate foci give 0/1 per experiment-voxel cell, so the Poisson maximum
    sits at infinity and two optimizers would stop at different points while both being correct;
    the objective, on the other hand, is exactly comparable wherever you evaluate it. CBMR drops
    the parameter-free ``-sum(log(y!))`` term, which is added back here.
    """
    from scipy.special import gammaln

    estimator = CBMR(formula, **FIT_KWARGS)
    estimator.fit(dataset=studyset)

    foci = np.asarray(estimator.inputs_["foci"].todense(), dtype=float)
    design = _materialize_public_design(estimator)
    coefficients = estimator.cbmr_model.coefficients.detach().numpy()
    assert design.shape[1] == coefficients.size, "design and parameter vector must agree"

    expected = statsmodels_api.GLM(
        foci.reshape(-1), design, family=statsmodels_api.families.Poisson()
    ).loglike(coefficients)
    actual = float(estimator.cbmr_model.log_likelihood().detach()) - float(gammaln(foci + 1).sum())

    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_the_public_fit_reports_the_design_derived_standard_errors(studyset):
    """Standard errors from the public path must match the design's own Fisher information.

    Evaluated at CBMR's fitted coefficients, not at statsmodels' optimum. Letting statsmodels
    converge would put it at a different parameter point -- coordinate foci have no interior
    maximum, so the two optimizers stop in different places -- and comparing standard errors
    across different points compares nothing. Here the bread is built from the same design and
    the same coefficients, so any disagreement is in the covariance code::

        A = X' diag(mu) X,   Cov = A^-1
    """
    estimator = CBMR("~ 1 + diagnosis", **FIT_KWARGS)
    estimator.fit(dataset=studyset)

    design = _materialize_public_design(estimator)
    coefficients = estimator.cbmr_model.coefficients.detach().numpy()

    mean = np.exp(design @ coefficients)
    information = design.T @ (design * mean[:, None])
    expected = np.sqrt(np.diag(np.linalg.inv(information)))

    actual = np.sqrt(np.diag(estimator.cbmr_model.covariance()))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)


def test_exposure_conditions_the_fit_and_leaves_the_studyset_alone(studyset):
    """End to end: the derived column is generated, used, and not left behind.

    It depends on this fit's mask and incidence_threshold, so a later fit with different
    settings must not find a stale one sitting in the studyset's annotations.
    """
    before = studyset.annotations_df.copy()
    result = _fit("~ s(diagnosis) + exposure()", studyset)

    assert "_cbmr_n_foci" not in studyset.annotations_df.columns
    assert list(studyset.annotations_df.columns) == list(before.columns)
    assert "_cbmr_n_foci" in result.estimator.annotations_.columns

    # An exposure owns no coefficient, so it gets no table and no row anywhere.
    assert not [key for key in result.tables if "exposure" in key]
    assert result.estimator.bound_design.n_parameters(result.estimator.predictor.n_bases) == CBMR(
        "~ s(diagnosis)", **FIT_KWARGS
    ).fit(dataset=studyset).estimator.bound_design.n_parameters(result.estimator.predictor.n_bases)


def test_exposure_makes_each_group_a_distribution(studyset):
    """The score equation normalizes every spatial term without being asked.

    Only approximately, and the approximation is the basis rather than the optimizer: the score
    equation forces ``sum_v (B1)_v exp(s_g) = sum_v (B1)_v y``, and a B-spline basis with its
    first column dropped and its unsupported ones removed is a partition of unity to within a
    percent or so rather than exactly. Fitted to convergence, because the suite's fast settings
    stop far short of it and the level is the last thing to settle.
    """
    converged = dict(tol=1e-4, n_iter=1500)
    exposed = _fit("~ s(diagnosis) + exposure()", studyset, **converged)
    plain = _fit("~ s(diagnosis)", studyset, **converged)

    def group_totals(result):
        return np.array(
            [
                result.maps[key].sum()
                for key in result.maps
                if key.startswith("spatialIntensity_group-")
            ]
        )

    assert np.allclose(group_totals(exposed), 1.0, rtol=0.05)
    # Without it each map is a rate, whose integral is the group's mean foci count.
    assert group_totals(plain).min() > 2.0


def test_a_moderator_alongside_exposure_is_refused_with_advice(studyset):
    """Its estimate would be exactly zero whatever the data."""
    with pytest.raises(FormulaError, match="no coefficient to estimate"):
        _fit("~ s(diagnosis) + standardized_sample_sizes + exposure()", studyset)


def test_a_spatial_moderator_alongside_exposure_is_allowed(studyset):
    """Conditioning removes the volume question and leaves the shape question intact."""
    result = _fit("~ s(diagnosis) + s(standardized_avg_age) + exposure()", studyset)
    assert "voxelwiseModeratorEffect_standardized_avg_age" in result.maps


def test_offset_in_a_public_formula_points_at_exposure(studyset):
    """The name CBMR does not use says which one it does."""
    with pytest.raises(FormulaError, match="exposure"):
        CBMR("~ s(diagnosis) + offset(log(n))", **FIT_KWARGS)
