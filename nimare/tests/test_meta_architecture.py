"""Tests for the sections 2-5 architecture.

Channels, general-covariance blocks, spatial borrowing, and reporting sensitivity.

Each test encodes either a claim verified in the companion proofs or a refusal the plan
requires. The refusals matter most: every one of them is a place where a record, a covariance or
an envelope would otherwise be read as licensing more than it does.
"""

import numpy as np
import pytest

from nimare.meta.cbma.channels import (
    ALLOWED_CONTRIBUTIONS,
    EvidenceChannel,
    EvidenceLedger,
    EvidenceRecord,
    degrees_of_freedom,
    hedges_factor,
    standardised_scale,
)
from nimare.meta.cbma.orthant import (
    block_silence_log_probability,
    draw_constrained_block,
    maximum_log_density,
    partition_residual,
    recorded_maximum_log_density,
)
from nimare.meta.cbma.sensitivity import (
    envelope_width_is_data_independent,
    implied_magnitude,
    retention_envelope,
    scenario_average,
)
from nimare.meta.cbma.spatial import (
    between_study_covariance,
    compare_against_equal_regularisation,
    gls_projection,
    projection_bias,
    variance_trace,
)

# --------------------------------------------------------------------- section 2: channels


def test_a_channel_may_not_contribute_what_its_provenance_does_not_support():
    """Unknown coverage licenses nothing, and coordinates license spatial features only."""
    ledger = EvidenceLedger()
    unknown = EvidenceRecord("A", EvidenceChannel.UNKNOWN_COVERAGE)
    coordinates = EvidenceRecord("B", EvidenceChannel.POORLY_ANNOTATED_COORDINATES)

    assert ALLOWED_CONTRIBUTIONS[EvidenceChannel.UNKNOWN_COVERAGE] == frozenset()
    for contribution in ("effect_likelihood", "censoring_interval", "selection_likelihood"):
        with pytest.raises(ValueError, match="may contribute"):
            ledger.require(unknown, contribution)
        with pytest.raises(ValueError, match="may contribute"):
            ledger.require(coordinates, contribution)
    ledger.require(coordinates, "spatial_basis")


def test_a_table_declared_qualified_must_name_its_reporting_event():
    """Refuse a table declared qualified with no reporting event qualifying it.

    Being qualified is the difference between a selection likelihood and a spatial basis.
    """
    with pytest.raises(ValueError, match="no reporting_event"):
        EvidenceRecord("A", EvidenceChannel.QUALIFIED_TABLE)
    record = EvidenceRecord(
        "A", EvidenceChannel.QUALIFIED_TABLE, reporting_event="voxelwise FDR 0.05"
    )
    assert record.reporting_event


def test_an_image_and_its_own_table_are_one_study_not_two():
    """They share every subject, so counting both as effect observations double-counts."""
    ledger = EvidenceLedger(
        [
            EvidenceRecord("A", EvidenceChannel.IMAGE, cohort_id="c1"),
            EvidenceRecord(
                "A", EvidenceChannel.QUALIFIED_TABLE, cohort_id="c1", reporting_event="FDR"
            ),
        ]
    )
    assert ledger.paired_studies() == ["A"]
    with pytest.raises(ValueError, match="already contributes an image"):
        ledger.add(EvidenceRecord("A", EvidenceChannel.IMAGE))


def test_an_unknown_cohort_is_not_assumed_to_be_a_distinct_one():
    """A missing identifier is investigated, not read as independence."""
    ledger = EvidenceLedger(
        [
            EvidenceRecord("A", EvidenceChannel.IMAGE, cohort_id="c1"),
            EvidenceRecord("B", EvidenceChannel.IMAGE),
            EvidenceRecord("C", EvidenceChannel.IMAGE),
        ]
    )
    units = ledger.effective_independent_units()
    assert units["by_study"] == 3
    # Two unknown cohorts pool into one unit rather than counting as two.
    assert units["by_cohort"] == 2
    assert units["unknown_cohort"] == 2


def test_a_design_whose_standardisation_counts_cannot_determine_is_refused():
    """A total sample size does not identify a GLM's or a correlation's conversion."""
    for design in ("glm", "mixed-effects", "correlation", "f-map"):
        with pytest.raises(ValueError, match="must be one of"):
            standardised_scale(design, 40)
    assert float(standardised_scale("one-sample", 25)) == pytest.approx(0.2)
    assert float(standardised_scale("paired", 25)) == pytest.approx(0.2)
    assert float(standardised_scale("two-sample", group_sizes=(20, 30))) == pytest.approx(
        np.sqrt(1 / 20 + 1 / 30)
    )
    with pytest.raises(ValueError, match="needs group_sizes"):
        standardised_scale("two-sample", 50)
    assert float(degrees_of_freedom("two-sample", group_sizes=(20, 30))) == 48.0


def test_the_hedges_factor_is_below_one_and_approaches_it():
    """It corrects an observed estimate; it is not a second multiplier on a planned design."""
    factors = hedges_factor(np.array([5.0, 20.0, 100.0, 1000.0]))
    assert np.all(factors < 1.0)
    assert np.all(np.diff(factors) > 0)
    assert factors[-1] == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------- section 3: general-covariance blocks


def _example_block():
    mean = np.array([0.30, 0.45, 0.20])
    base = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, 0.5], [0.2, 0.5, 1.0]])
    return mean, 0.25**2 * base


def test_the_reporting_events_of_a_non_exchangeable_block_exhaust_the_probability():
    """Silence plus every recorded-location density must sum to one.

    The tolerance is the multivariate normal CDF's own quasi-Monte Carlo error, about 1e-5 at
    this dimension, not a number tuned until the test passed.
    """
    mean, covariance = _example_block()
    result = partition_residual(mean, covariance, 0.5)
    assert result["residual"] < 3e-5
    assert len(result["reported"]) == 3
    assert result["silence"] > 0


def test_a_recorded_location_carries_more_than_the_summed_height_density():
    """Summing over locations discards where the peak was, which tables do record."""
    mean, covariance = _example_block()
    per_location = [
        recorded_maximum_log_density(mean, covariance, index, 0.9) for index in range(3)
    ]
    summed = maximum_log_density(mean, covariance, 0.9)
    # The sum over locations exceeds any single one, so naming the location is informative.
    assert summed > max(per_location)
    assert summed == pytest.approx(np.log(np.sum(np.exp(per_location))), abs=1e-10)


def test_a_one_element_block_reduces_to_the_scalar_normal():
    """With one element the maximum is the value and the orthant is a scalar CDF."""
    from scipy.stats import norm

    silence = block_silence_log_probability([0.3], [[0.04]], 0.5)
    assert silence == pytest.approx(float(norm.logcdf(0.5, 0.3, 0.2)), abs=1e-9)
    density = recorded_maximum_log_density([0.3], [[0.04]], 0, 0.7)
    assert density == pytest.approx(float(norm.logpdf(0.7, 0.3, 0.2)), abs=1e-9)


def test_a_degenerate_or_oversized_block_is_refused():
    """A singular block is a smaller block, and a large one is not a reference at all."""
    with pytest.raises(ValueError, match="positive definite"):
        block_silence_log_probability([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]], 0.5)
    with pytest.raises(ValueError, match="small-block reference"):
        block_silence_log_probability(np.zeros(20), np.eye(20), 0.5)
    with pytest.raises(ValueError, match="symmetric"):
        block_silence_log_probability([0.0, 0.0], [[1.0, 0.5], [0.2, 1.0]], 0.5)


def test_constrained_draws_respect_the_observation_event():
    """Augmentation draws must satisfy the event they condition on.

    They are integration variables inside one model, not data.
    """
    mean, covariance = _example_block()
    generator = np.random.default_rng(4)
    for _ in range(20):
        silent = draw_constrained_block(mean, covariance, 0.5, rng=generator)
        assert np.all(silent < 0.5)
        recorded = draw_constrained_block(
            mean, covariance, 0.5, index=1, height=0.9, rng=generator
        )
        assert recorded[1] == pytest.approx(0.9)
        assert np.all(np.delete(recorded, 1) <= 0.9)
    with pytest.raises(ValueError, match="does not clear its threshold"):
        draw_constrained_block(mean, covariance, 0.5, index=0, height=0.1)


# ------------------------------------------- sections 2.2 and 4: restricted spatial structure


def test_a_dense_between_study_covariance_is_refused_at_a_realistic_study_count():
    """Per-study information about between-study structure is capped whatever the voxel count."""
    with pytest.raises(ValueError, match="independent studies"):
        between_study_covariance("dense", 5, n_studies=8)
    matrix, free = between_study_covariance("scaled", 3, tau2=0.02, reference=np.eye(3))
    assert free == 1
    assert np.allclose(matrix, 0.02 * np.eye(3))
    with pytest.raises(ValueError, match="externally justified"):
        between_study_covariance("scaled", 3, tau2=0.02)
    _, free = between_study_covariance("diagonal", 3, variances=[0.01, 0.02, 0.03])
    assert free == 3


def test_the_projection_relocates_effect_and_its_bias_is_exactly_the_residual():
    """A constant basis moves half a one-voxel effect onto a voxel that had none."""
    bias = projection_bias(np.ones((2, 1)), np.eye(2), [1.0, 0.0])
    assert bias == pytest.approx([-0.5, 0.5])
    projection = gls_projection(np.ones((2, 1)), np.eye(2))
    assert projection @ projection == pytest.approx(projection)


def test_the_variance_gain_is_the_ranks_and_only_the_bias_distinguishes_a_basis():
    """Check in code what the projection proof establishes algebraically.

    Two bases of the same rank have identical variance traces however they were built, so a gain
    measured against an unregularised image-only baseline is attributable to the rank. Only a
    harder comparator can separate them, which is why the comparison is not optional.
    """
    positions = np.arange(40)
    covariance = 0.2**2 * np.eye(40)
    truth = np.exp(-0.5 * ((positions - 12) / 4.0) ** 2) + 0.4 * np.exp(
        -0.5 * ((positions - 30) / 2.5) ** 2
    )

    def bumps(centres, width):
        return np.stack(
            [np.exp(-0.5 * ((positions - centre) / width) ** 2) for centre in centres], axis=1
        )

    informed = bumps([12, 14, 30, 31, 20], 4.0)
    misplaced = bumps([2, 8, 35, 38, 20], 6.0)
    assert variance_trace(informed, covariance) == pytest.approx(
        variance_trace(misplaced, covariance)
    )
    assert variance_trace(informed, covariance) == pytest.approx(5 * 0.2**2)

    good = compare_against_equal_regularisation(
        informed,
        truth[None, :],
        covariance,
        truth=truth,
        rng=np.random.default_rng(1),
        draws=64,
    )
    bad = compare_against_equal_regularisation(
        misplaced,
        truth[None, :],
        covariance,
        truth=truth,
        rng=np.random.default_rng(1),
        draws=64,
    )
    assert good["variance_traces_agree"] and bad["variance_traces_agree"]
    assert good["verdict"] == "basis earns its gain"
    # A misplaced basis beats random directions and loses to generic smoothing, which is the
    # confound the plan warns about: only the harder comparator exposes it.
    assert bad["verdict"].startswith("gain is the rank's")
    assert bad["candidate_squared_bias"] < bad["comparator_squared_bias_min"]["random"]
    assert bad["candidate_squared_bias"] > bad["comparator_squared_bias_min"]["smooth"]


# --------------------------------------------------- section 5: reporting sensitivity


def test_the_envelope_reproduces_the_plans_illustration_from_its_endpoints_alone():
    """m(rho) is decreasing, so the endpoints bracket the interior and two evaluations suffice."""
    envelope = retention_envelope(0.1, 0.6, 0.25, (0.3, 0.8))
    assert envelope.lower == pytest.approx(0.3124, abs=5e-4)
    assert envelope.upper == pytest.approx(0.4923, abs=5e-4)
    interior = implied_magnitude(0.1, 0.5, 0.6, 0.25)
    assert envelope.lower < interior < envelope.upper


def test_the_envelope_width_does_not_move_with_the_number_of_studies():
    """That is the operative difference from a confidence interval, and it is exact."""
    widths = envelope_width_is_data_independent(0.1, 0.6, 0.25, (0.3, 0.8))
    assert len(set(widths.values())) == 1
    envelope = retention_envelope(0.1, 0.6, 0.25, (0.3, 0.8))
    assert envelope.as_dict()["is_confidence_interval"] is False
    assert "not a confidence interval" in envelope.coverage_semantics


def test_the_envelope_diverges_as_the_scenario_set_reaches_the_observed_rate():
    """A corpus can print a little of a lot or all of a little and leave the same rate."""
    widths = [
        retention_envelope(0.1, 0.6, 0.25, (floor, 0.8)).width
        for floor in (0.3, 0.15, 0.11, 0.101)
    ]
    assert widths == sorted(widths)
    assert widths[-1] > 4 * widths[0]
    with pytest.raises(ValueError, match="impossible, not conservative"):
        implied_magnitude(0.1, 0.05, 0.6, 0.25)


def test_a_scenario_average_returns_its_weights_and_refuses_to_be_a_posterior_mean():
    """An average without its weights asserts a prior silently."""
    result = scenario_average(0.1, 0.6, 0.25, [0.3, 0.8])
    assert result["is_posterior_mean"] is False
    assert result["jensen_gap"] != pytest.approx(0.0, abs=1e-6)
    assert result["average"] != pytest.approx(result["at_average_retention"], abs=1e-6)
    assert result["weights"] == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="sum to one"):
        scenario_average(0.1, 0.6, 0.25, [0.3, 0.8], weights=[0.5, 0.2])


def test_an_unasserted_scenario_set_carries_no_coverage_statement_at_all():
    """Completeness is the caller's assertion, never an inference."""
    silent = retention_envelope(0.1, 0.6, 0.25, (0.3, 0.8))
    asserted = retention_envelope(0.1, 0.6, 0.25, (0.3, 0.8), scenario_set_complete=True)
    assert "no coverage statement is available" in silent.coverage_semantics
    assert "Coverage is one if" in asserted.coverage_semantics
    assert silent.width == asserted.width
