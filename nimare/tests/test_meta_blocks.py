"""Tests for the block peak-selection likelihood.

Invariants rather than recorded outputs. Several encode claims verified symbolically in
``proofs/block_peak_selection.py``, and one checks the block model against the scalar reference
in :mod:`nimare.meta.cbma.censored`, which is the strongest available consistency condition: a
one-element block *is* a scalar observation, so the two modules must agree exactly there.
"""

import numpy as np
import pytest

from nimare.meta.cbma.blocks import (
    block_loglik,
    block_report_probability,
    naive_to_exact_se_ratio,
    quadrature_is_converged,
)
from nimare.meta.cbma.censored import (
    ObservationState,
    bounds_from_states,
    censored_loglik,
)


def test_a_one_element_block_agrees_exactly_with_the_scalar_censored_reference():
    """A block of one element is a scalar observation, so the two modules must coincide.

    Both limbs are checked. A reported one-element block is the Gaussian density of the estimate,
    which is what an exactly observed record contributes; a silent one is the left-censored
    probability, which is what a one-sided absence contributes.
    """
    mean, between, within = 0.35, 0.02, 0.05
    threshold, height = 0.6, 0.82

    reported_block = block_loglik(
        mean,
        between,
        np.array([within]),
        heights=np.array([height]),
        thresholds=np.array([threshold]),
        elements=np.array([1]),
    )
    lower, upper = bounds_from_states([ObservationState.IMAGE], values=np.array([height]))
    assert reported_block == pytest.approx(
        censored_loglik(mean, between, lower, upper, np.array([within])), abs=1e-9
    )

    silent_block = block_loglik(
        mean,
        between,
        np.array([within]),
        heights=np.array([np.nan]),
        thresholds=np.array([threshold]),
        elements=np.array([1]),
    )
    # The voxelwise reading of an absent peak is asserted here because for a **one-element**
    # block it is exactly true: the block's maximum is its only value, so "no local maximum
    # cleared the cut" and "the value was below the cut" are the same event. That equality is
    # precisely why this cross-module identity holds, and it is what fails for larger blocks.
    lower, upper = bounds_from_states(
        [ObservationState.ABSENT_COMPLETE_TABLE],
        thresholds=np.array([threshold]),
        signs=np.array([1.0]),
        absence_is_voxelwise=True,
    )
    assert silent_block == pytest.approx(
        censored_loglik(mean, between, lower, upper, np.array([within])), abs=1e-9
    )


def test_the_height_enters_only_through_its_distance_from_the_mean():
    """Shift the mean and every height together and nothing may change.

    Proved as "the reported height is a location family so its mean score is minus its height
    score". The whole tabulation strategy the calibration bed uses rests on it: one curve in
    ``h - m`` serves every (height, candidate mean) pair.

    Bit-identity is asserted only for shifts that are exactly representable, with dyadic
    heights, because otherwise the *shift itself* rounds -- ``(0.9 + 0.17) - (0.4 + 0.17)``
    differs from ``0.9 - 0.4`` in the last bit, and a failure would say nothing about the model.
    A non-dyadic shift is checked separately to a few units in the last place.
    """
    rng = np.random.default_rng(5)
    heights = np.array([0.75, 1.25, np.nan, 1.0, np.nan])
    thresholds = np.full(5, 0.75)
    elements = np.array([9, 9, 9, 4, 16])
    variances = rng.uniform(0.02, 0.06, size=5)

    reference = block_loglik(
        0.5,
        0.0225,
        variances,
        heights=heights,
        thresholds=thresholds,
        elements=elements,
    )
    for shift in (-0.25, 0.25, 0.5, 2.0):
        shifted = block_loglik(
            0.5 + shift,
            0.0225,
            variances,
            heights=heights + shift,
            thresholds=thresholds + shift,
            elements=elements,
        )
        assert shifted == reference

    for shift in (-0.3, 0.17, 1.1):
        shifted = block_loglik(
            0.5 + shift,
            0.0225,
            variances,
            heights=heights + shift,
            thresholds=thresholds + shift,
            elements=elements,
        )
        assert shifted == pytest.approx(reference, rel=1e-13)


def test_the_reporting_outcomes_exhaust_the_probability():
    """A block either reports or is silent, so the two probabilities must sum to one."""
    mean, between, within = 0.4, 0.0225, 0.04
    for elements in (1, 4, 9, 25):
        for threshold in (0.2, 0.75, 1.3):
            silent = np.exp(
                block_loglik(
                    mean,
                    between,
                    np.array([within]),
                    heights=np.array([np.nan]),
                    thresholds=np.array([threshold]),
                    elements=np.array([elements]),
                )
            )
            reporting = block_report_probability(
                mean,
                between,
                np.array([within]),
                thresholds=np.array([threshold]),
                elements=np.array([elements]),
            )[0]
            assert silent + reporting == pytest.approx(1.0, abs=1e-10)


def test_the_report_probability_matches_a_direct_simulation():
    """The quadrature over the study effect must land where sampling the model lands."""
    mean, between, within, elements, threshold = 0.4, 0.0225, 0.04, 9, 0.75
    quadrature = block_report_probability(
        mean,
        between,
        np.array([within]),
        thresholds=np.array([threshold]),
        elements=np.array([elements]),
    )[0]

    rng = np.random.default_rng(11)
    draws = 200_000
    shared = rng.normal(0.0, np.sqrt(between), size=draws)
    peaks = (
        mean + shared[:, None] + rng.normal(0.0, np.sqrt(within), size=(draws, elements))
    ).max(axis=1)
    simulated = float((peaks > threshold).mean())
    # Three binomial standard errors of the simulation, which is the noisy side.
    tolerance = 3 * np.sqrt(simulated * (1 - simulated) / draws)
    assert quadrature == pytest.approx(simulated, abs=tolerance)


def test_the_quadrature_reports_its_own_convergence():
    """A rule that has not converged gives a smooth, plausible, wrong likelihood."""
    converged, gap = quadrature_is_converged(
        0.4,
        0.0225,
        np.array([0.04]),
        heights=np.array([0.9]),
        thresholds=np.array([0.75]),
        elements=np.array([9]),
    )
    assert converged
    assert gap < 1e-9


def test_a_bigger_block_reports_more_readily_at_the_same_mean():
    """More elements mean more chances to clear the threshold, monotonically."""
    probabilities = [
        block_report_probability(
            0.4,
            0.0225,
            np.array([0.04]),
            thresholds=np.array([0.75]),
            elements=np.array([elements]),
        )[0]
        for elements in (1, 2, 4, 9, 20, 50)
    ]
    assert np.all(np.diff(probabilities) > 0)


def test_reading_the_height_makes_the_likelihood_sharper_than_the_indicator_alone():
    """An observation cannot carry less information when more of it is read.

    Measured as the curvature of the log-likelihood at the truth, which is what sets the width
    of an interval. The margin at this threshold is small -- that is the design assessment's
    "modest" -- so the test asserts the direction and not a size.
    """
    mean, between, within, elements, threshold = 0.4, 0.0225, 0.04, 9, 0.75
    rng = np.random.default_rng(23)
    count = 400
    shared = rng.normal(0.0, np.sqrt(between), size=count)
    peaks = (
        mean + shared[:, None] + rng.normal(0.0, np.sqrt(within), size=(count, elements))
    ).max(axis=1)
    heights = np.where(peaks > threshold, peaks, np.nan)

    def curvature(use_heights):
        step = 1e-3
        values = [
            block_loglik(
                candidate,
                between,
                np.full(count, within),
                heights=heights,
                thresholds=np.full(count, threshold),
                elements=np.full(count, elements),
                use_heights=use_heights,
            )
            for candidate in (mean - step, mean, mean + step)
        ]
        return -(values[0] - 2 * values[1] + values[2]) / step**2

    assert curvature(True) > curvature(False) > 0


def test_a_height_below_its_own_threshold_is_refused():
    """A table and a threshold that contradict each other are a data problem, not a clip."""
    with pytest.raises(ValueError, match="below its own"):
        block_loglik(
            0.4,
            0.0225,
            np.array([0.04]),
            heights=np.array([0.5]),
            thresholds=np.array([0.75]),
            elements=np.array([9]),
        )


def test_a_degenerate_block_or_variance_is_refused():
    """Zero elements and non-positive variances have no likelihood to compute."""
    with pytest.raises(ValueError, match="at least one element"):
        block_loglik(
            0.4,
            0.0225,
            np.array([0.04]),
            heights=np.array([np.nan]),
            thresholds=np.array([0.75]),
            elements=np.array([0]),
        )
    with pytest.raises(ValueError, match="variance must be positive"):
        block_loglik(
            0.4,
            0.0225,
            np.array([0.0]),
            heights=np.array([np.nan]),
            thresholds=np.array([0.75]),
            elements=np.array([9]),
        )


def test_zero_heterogeneity_is_a_point_mass_not_a_degenerate_integral():
    """With no between-study variance the quadrature must collapse rather than divide by zero."""
    value = block_loglik(
        0.4,
        0.0,
        np.array([0.04]),
        heights=np.array([0.9]),
        thresholds=np.array([0.75]),
        elements=np.array([9]),
    )
    assert np.isfinite(value)
    # And it must agree with the same thing computed at a negligible heterogeneity.
    nearby = block_loglik(
        0.4,
        1e-10,
        np.array([0.04]),
        heights=np.array([0.9]),
        thresholds=np.array([0.75]),
        elements=np.array([9]),
    )
    assert value == pytest.approx(nearby, abs=1e-6)


def test_grouping_blocks_by_study_changes_the_likelihood_and_lowers_the_information():
    """Blocks of one study share its effect, so integrating it out once is a different model.

    Proved in ``proofs/composite_block_likelihood.py``: the discrepancy is the covariance of the
    block terms under the study effect, and for an all-silent study every term decreases in that
    effect, so the terms are positively associated. The composite version therefore understates
    the likelihood and overstates the information -- it counts one draw as many.
    """
    mean, between, within, elements, threshold = 0.4, 0.0225, 0.04, 9, 0.75
    ratios = []
    for blocks in (1, 2, 4, 10, 50):
        heights = np.full(blocks, np.nan)
        thresholds = np.full(blocks, threshold)
        counts = np.full(blocks, elements)
        variances = np.full(blocks, within)
        one_study = np.zeros(blocks, dtype=int)

        exact = block_loglik(
            mean,
            between,
            variances,
            heights=heights,
            thresholds=thresholds,
            elements=counts,
            study_index=one_study,
        )
        composite = block_loglik(
            mean,
            between,
            variances,
            heights=heights,
            thresholds=thresholds,
            elements=counts,
        )
        if blocks == 1:
            assert exact == composite
        else:
            assert composite < exact

        ratios.append(
            naive_to_exact_se_ratio(
                mean,
                between,
                variances,
                heights=heights,
                thresholds=thresholds,
                elements=counts,
                study_index=one_study,
            )
        )

    assert ratios[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(ratios) < 0)
    assert ratios[-1] < 0.25


def test_one_block_per_study_is_the_same_whether_or_not_it_is_grouped():
    """Grouping can only matter where a group has more than one member."""
    rng = np.random.default_rng(29)
    blocks = 12
    heights = np.where(rng.random(blocks) < 0.5, 0.95, np.nan)
    thresholds = np.full(blocks, 0.75)
    counts = np.full(blocks, 9)
    variances = rng.uniform(0.02, 0.06, size=blocks)
    distinct = np.arange(blocks)

    assert block_loglik(
        0.4,
        0.0225,
        variances,
        heights=heights,
        thresholds=thresholds,
        elements=counts,
        study_index=distinct,
    ) == pytest.approx(
        block_loglik(
            0.4,
            0.0225,
            variances,
            heights=heights,
            thresholds=thresholds,
            elements=counts,
        ),
        abs=1e-9,
    )


def test_a_mismatched_study_index_is_refused():
    """A grouping that does not label every block cannot be applied silently."""
    with pytest.raises(ValueError, match="one entry per block"):
        block_loglik(
            0.4,
            0.0225,
            np.full(3, 0.04),
            heights=np.full(3, np.nan),
            thresholds=np.full(3, 0.75),
            elements=np.full(3, 9),
            study_index=np.array([0, 0]),
        )
    with pytest.raises(ValueError, match="needs study_index"):
        naive_to_exact_se_ratio(
            0.4,
            0.0225,
            np.full(3, 0.04),
            heights=np.full(3, np.nan),
            thresholds=np.full(3, 0.75),
            elements=np.full(3, 9),
        )
