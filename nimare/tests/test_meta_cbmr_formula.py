"""End-to-end tests for the formula-specified CBMR estimator."""

import logging
import warnings

import numpy as np
import pytest

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
    studyset.annotations_df = annotations
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
    assert list(table.columns) == ["column", "coefficient", "standard_error"]
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
    assert np.all(np.isfinite(table["coefficient"]))


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

    assert "s(diagnosis)" in result.description_
    assert "Poisson" in result.description_


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
