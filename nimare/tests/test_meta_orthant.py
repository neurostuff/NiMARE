"""Tests for the general-covariance block likelihood.

The two strongest available checks are equivalences rather than recorded outputs. Independent
elements plus a shared study effect *is* the exchangeable model, so a diagonal covariance must
reproduce :mod:`nimare.meta.cbma.blocks` exactly; and a one-element block *is* a scalar
observation, so it must reproduce :mod:`nimare.meta.cbma.censored`. Either failing would mean the
general route and the approximation it is meant to referee disagree about the same model.
"""

import numpy as np
import pytest

from nimare.meta.cbma.blocks import block_loglik
from nimare.meta.cbma.censored import (
    ObservationState,
    bounds_from_states,
    censored_loglik,
)
from nimare.meta.cbma.orthant import (
    block_silence_log_probability,
    fit_general_blocks,
    general_block_loglik,
    maximum_log_density,
    partition_residual,
    recorded_maximum_log_density,
)


def test_a_diagonal_covariance_reproduces_the_exchangeable_block():
    """Independent elements plus a shared effect is exactly the exchangeable model."""
    mean, between = 0.4, 0.15**2
    element_variance, threshold, size = 0.2**2, 0.75, 4
    heights = np.array([0.9, np.nan, 1.2])
    blocks = [
        {
            "covariance": np.eye(size) * element_variance,
            "threshold": threshold,
            "height": None if not np.isfinite(height) else height,
            "index": None,
        }
        for height in heights
    ]
    general = general_block_loglik(mean, between, blocks, nodes=61)
    exchangeable = block_loglik(
        mean,
        between,
        np.full(heights.size, element_variance),
        heights=heights,
        thresholds=np.full(heights.size, threshold),
        elements=np.full(heights.size, size),
    )
    assert general == pytest.approx(float(exchangeable), abs=2e-4)


def test_a_one_element_block_reproduces_the_scalar_reference():
    """A single element has no maximum to take, so the block likelihood must collapse.

    The comparison has to be made against a **one-sided** absence. ``NONSIGNIFICANT`` in
    :mod:`~nimare.meta.cbma.censored` is the two-sided event :math:`|Y| < c` that a published
    protocol actually generates, while a block reports when its maximum clears :math:`+c`, so
    the two modules are describing different events and the first version of this test compared
    them anyway. The bounds are therefore built explicitly rather than taken from
    ``bounds_from_states``, and the difference is a convention in each module rather than a
    defect in either.
    """
    mean, variance, threshold = 0.3, 0.09, 0.75
    blocks = [
        {"covariance": [[variance]], "threshold": threshold, "height": 1.1, "index": 0},
        {"covariance": [[variance]], "threshold": threshold},
    ]
    general = general_block_loglik(mean, 0.0, blocks, nodes=1)

    lower = np.array([1.1, -np.inf])
    upper = np.array([1.1, threshold])
    scalar = censored_loglik(mean, 0.0, lower, upper, np.full(2, variance))
    assert general == pytest.approx(scalar, abs=1e-9)


def test_the_two_sided_scalar_absence_is_a_different_event_and_differs():
    """Guard the distinction above, so a future change cannot quietly conflate the two."""
    mean, variance, threshold = 0.3, 0.09, 0.75
    states = [ObservationState.NONSIGNIFICANT]
    two_sided_lower, two_sided_upper = bounds_from_states(states, [np.nan], [threshold])
    two_sided = censored_loglik(mean, 0.0, two_sided_lower, two_sided_upper, np.array([variance]))
    one_sided = censored_loglik(
        mean, 0.0, np.array([-np.inf]), np.array([threshold]), np.array([variance])
    )
    # The one-sided event contains the two-sided one, so it is strictly more probable.
    assert one_sided > two_sided


def test_correlation_makes_silence_more_likely_than_independence():
    """Positively correlated elements produce fewer exceedances, so more silent blocks.

    The direction is the whole reason the exchangeable approximation mis-states silence: it is
    not a numerical detail but the sign of the error.
    """
    size, variance, threshold, mean = 5, 0.04, 0.75, 0.4
    independent = np.eye(size) * variance
    correlated = np.full((size, size), 0.6 * variance) + np.eye(size) * 0.4 * variance
    centre = np.full(size, mean)
    assert block_silence_log_probability(centre, correlated, threshold) > (
        block_silence_log_probability(centre, independent, threshold)
    )


def test_summing_over_locations_exceeds_naming_one():
    """A maximum somewhere is at least as likely as a maximum at a named place."""
    size, mean = 3, 0.4
    covariance = np.full((size, size), 0.01) + np.eye(size) * 0.03
    centre = np.full(size, mean)
    summed = maximum_log_density(centre, covariance, 0.9)
    named = recorded_maximum_log_density(centre, covariance, 1, 0.9)
    assert summed > named


def test_the_pieces_still_exhaust_the_probability():
    """Silence plus every location's report must account for all of it."""
    size = 4
    covariance = np.full((size, size), 0.008) + np.eye(size) * 0.032
    residual = partition_residual(np.full(size, 0.35), covariance, 0.75)
    assert abs(float(residual["residual"])) < 5e-3


def test_indicator_only_discards_the_height_and_is_worse_identified():
    """With heights off, two different heights must give the same log-likelihood."""
    size = 3
    covariance = np.eye(size) * 0.04
    common = {"covariance": covariance, "threshold": 0.75, "index": 0}
    low = general_block_loglik(0.4, 0.0, [{**common, "height": 0.8}], use_heights=False)
    high = general_block_loglik(0.4, 0.0, [{**common, "height": 1.6}], use_heights=False)
    assert low == pytest.approx(high, abs=1e-12)

    with_height_low = general_block_loglik(0.4, 0.0, [{**common, "height": 0.8}])
    with_height_high = general_block_loglik(0.4, 0.0, [{**common, "height": 1.6}])
    assert with_height_low != pytest.approx(with_height_high, abs=1e-6)


def test_an_all_silent_corpus_pushes_the_mean_below_the_threshold():
    """Silence is informative, and its direction is down."""
    size = 4
    blocks = [{"covariance": np.eye(size) * 0.04, "threshold": 0.75} for _ in range(6)]
    out = fit_general_blocks(blocks, between_variance=0.0)
    assert out["converged"]
    assert out["mean"] < 0.75


def test_a_reported_height_pulls_the_mean_towards_it():
    """And a printed peak pulls the other way, so the two limbs are not the same term."""
    size = 4
    covariance = np.eye(size) * 0.04
    silent = [{"covariance": covariance, "threshold": 0.75} for _ in range(4)]
    with_report = silent + [
        {"covariance": covariance, "threshold": 0.75, "height": 1.4, "index": 2}
    ]
    assert fit_general_blocks(with_report)["mean"] > fit_general_blocks(silent)["mean"]


def test_a_quadrature_rule_whose_weights_underflow_is_refused():
    """More nodes is not more accurate, and a broken rule must not return a number."""
    blocks = [{"covariance": np.eye(2) * 0.04, "threshold": 0.75}]
    with pytest.raises(ValueError, match="underflow"):
        general_block_loglik(0.4, 0.02, blocks, nodes=600)
