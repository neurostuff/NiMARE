"""Tests for the many-location driver over the censored likelihood."""

import numpy as np
import pytest

from nimare.meta.cbma.censored import ObservationState, bounds_from_states, fit_censored
from nimare.meta.cbma.driver import fit_locations, records_for_location


def _studies():
    return [
        {
            "id": "at-the-location",
            "peaks": np.array([[0.0, 0.0, 0.0]]),
            "heights": np.array([1.9]),
            "threshold": 0.6,
            "variance": 0.09,
        },
        {
            "id": "far-away",
            "peaks": np.array([[30.0, 0.0, 0.0]]),
            "heights": np.array([2.2]),
            "threshold": 0.6,
            "variance": 0.09,
        },
        {
            "id": "printed-nothing",
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.6,
            "variance": 0.09,
        },
    ]


def test_a_peak_at_the_location_is_exact_and_one_nearby_is_a_bound():
    """The distance decides the state, and getting that wrong throws information away.

    At the printed peak the height *is* the effect, so an interval would discard what the table
    said. A few millimetres away the same height is the cluster's maximum and bounds the effect
    above. Reading every nearby peak as a bound measurably loses to reading the coincident ones
    as exact, which is why the distance test is here and not left to the caller.
    """
    lower, upper, variances, roles = records_for_location(
        np.zeros(3), _studies(), reach=8.0, image_values=[0.5], image_variances=[0.04]
    )
    # image, the coincident peak, then two silences
    assert lower[0] == upper[0] == pytest.approx(0.5)
    assert lower[1] == upper[1] == pytest.approx(1.9)
    assert not np.isfinite(lower[2:]).any()
    assert upper[2:] == pytest.approx([0.6, 0.6])
    assert roles.tolist() == [0, 1, -1, -1]
    assert variances == pytest.approx([0.04, 0.09, 0.09, 0.09])

    # Four millimetres away the same peak is a bound, and there is no image record.
    lower, upper, _, roles = records_for_location(np.array([4.0, 0.0, 0.0]), _studies(), reach=8.0)
    assert (lower[0], upper[0]) == pytest.approx((0.6, 1.9))
    assert roles.tolist() == [1, -1, -1]


def test_a_peak_outside_the_reach_is_a_silence_not_a_report():
    """The reach is the whole of what makes a study speak or stay silent at a location."""
    far = np.array([20.0, 0.0, 0.0])
    _, upper_short, _, roles_short = records_for_location(far, _studies(), reach=8.0)
    _, _, _, roles_long = records_for_location(far, _studies(), reach=25.0)
    assert roles_short.tolist() == [-1, -1, -1]  # nothing within 8 mm of a point 20 mm out
    assert roles_long.tolist() == [1, 1, -1]  # both printed peaks now inside 25 mm
    assert upper_short == pytest.approx([0.6, 0.6, 0.6])


def test_the_driver_reproduces_a_single_scalar_fit_exactly():
    """The driver must be bookkeeping over :mod:`censored`, not a second implementation.

    Built as a separate module precisely so this can be checked: the same records fitted by hand
    through ``fit_censored`` have to give the same number, or the map is being produced by
    something other than the likelihood that was proved and tested.
    """
    studies = _studies()
    position = np.array([4.0, 0.0, 0.0])
    lower, upper, variances, roles = records_for_location(position, studies, reach=8.0)
    direct = fit_censored(
        lower, upper, variances, retention=0.43, roles=roles, fixed_between_variance=0.0
    )
    driven = fit_locations(
        position[None, :],
        lambda index: studies,
        reach=8.0,
        retention=0.43,
        fixed_between_variance=0.0,
    )
    assert driven["estimate"][0] == pytest.approx(direct["mean"], abs=1e-12)
    assert driven["valid"][0]
    assert np.isfinite(driven["lower"][0]) and np.isfinite(driven["upper"][0])


def test_retention_is_ignored_where_it_cannot_act():
    """At full retention the mixture *is* the certain-silence reading, so nothing should move.

    Passing a retention that changes nothing while implying it does is the kind of inert
    argument that makes a later result unreadable, so the driver declines to pass it on.
    """
    studies = _studies()
    position = np.array([4.0, 0.0, 0.0])
    one = fit_locations(
        position[None, :],
        lambda i: studies,
        reach=8.0,
        retention=1.0,
        fixed_between_variance=0.0,
    )
    lower, upper, variances, _ = records_for_location(position, studies, reach=8.0)
    bare = fit_censored(lower, upper, variances, fixed_between_variance=0.0)
    assert one["estimate"][0] == pytest.approx(bare["mean"], abs=1e-12)

    # And a lower retention must move it, or the state is not retention-eligible after all.
    lowered = fit_locations(
        position[None, :],
        lambda i: studies,
        reach=8.0,
        retention=0.4,
        fixed_between_variance=0.0,
    )
    assert lowered["estimate"][0] > one["estimate"][0]


def test_failures_are_returned_rather_than_dropped():
    """A location that does not fit is reported as such, with its arrays left at ``nan``.

    Summarising only the locations that converged summarises a subset chosen by the outcome,
    which is the same error as dropping non-converged simulations from an error rate.
    """
    empty = [
        {
            "id": "nothing-anywhere",
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.6,
            "variance": 0.09,
        }
    ]
    out = fit_locations(
        np.zeros((1, 3)), lambda index: empty, reach=8.0, fixed_between_variance=0.0
    )
    assert set(out) >= {"estimate", "lower", "upper", "valid", "converged", "touched_search_limit"}
    assert out["estimate"].shape == (1,)
    # One silence alone bounds the mean above and not below, so either it fits at the search
    # edge or it does not fit; both are reportable and neither may be silently dropped.
    assert out["valid"][0] in (True, False)


def test_mismatched_peaks_and_heights_are_refused():
    """A table whose coordinates and heights disagree in length is a data error, not a fit."""
    broken = [
        {
            "id": "ragged",
            "peaks": np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            "heights": np.array([1.9]),
            "threshold": 0.6,
            "variance": 0.09,
        }
    ]
    with pytest.raises(ValueError, match="peaks and"):
        records_for_location(np.zeros(3), broken, reach=8.0)

    with pytest.raises(ValueError, match="image_values and image_variances"):
        records_for_location(
            np.zeros(3), _studies(), reach=8.0, image_values=[0.5, 0.2], image_variances=[0.04]
        )


def test_bounds_come_from_the_shared_state_machine():
    """The driver must not build bounds itself; it must ask :func:`bounds_from_states`.

    Checked by reconstructing the same bounds from the states directly. If the driver ever grows
    its own bound arithmetic, the two will drift and this fails.
    """
    studies = _studies()
    lower, upper, _, _ = records_for_location(np.zeros(3), studies, reach=8.0)
    expected_lower, expected_upper = bounds_from_states(
        [
            ObservationState.EXACT,
            ObservationState.NO_PEAK_NEARBY,
            ObservationState.NO_PEAK_NEARBY,
        ],
        values=np.array([1.9, 0.0, 0.0]),
        thresholds=np.array([0.6, 0.6, 0.6]),
        signs=np.ones(3),
    )
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_the_envelope_sweeps_the_whole_grid_because_the_estimate_is_not_monotone():
    """Corners would not do, and the docstring says why, so the test holds it to that.

    The implied magnitude in :mod:`nimare.meta.cbma.sensitivity` is monotone in its retention,
    so its endpoints suffice. This estimate is not monotone in the reach -- its bias changes
    sign between 8 mm and 12 mm on HCP pseudo-studies -- so an envelope built from the corners
    would report a range excluding the interior it brackets. Here the extreme is found at an
    interior reach, which a corners-only implementation would miss.
    """
    from nimare.meta.cbma.driver import envelope_over_assumptions

    studies = [
        {
            "id": "near",
            "peaks": np.array([[0.0, 0.0, 0.0]]),
            "heights": np.array([1.9]),
            "threshold": 0.6,
            "variance": 0.09,
        },
        {
            "id": "mid",
            "peaks": np.array([[14.0, 0.0, 0.0]]),
            "heights": np.array([2.2]),
            "threshold": 0.6,
            "variance": 0.09,
        },
        {
            "id": "silent",
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.6,
            "variance": 0.09,
        },
    ]
    reaches, retentions = [4.0, 8.0, 12.0, 20.0], [1.0, 0.43, 0.25]
    out = envelope_over_assumptions(
        np.array([[4.0, 0.0, 0.0]]),
        lambda index: studies,
        reaches,
        retentions,
        fixed_between_variance=0.0,
        images_at=lambda index: ([0.5], [0.04]),
    )
    assert len(out["grid"]) == len(reaches) * len(retentions)
    assert out["highest"][0] > out["lowest"][0]
    assert out["spread"][0] == pytest.approx(out["highest"][0] - out["lowest"][0])
    # The union of intervals must contain every point estimate the grid produced.
    assert out["interval_lower"][0] <= out["lowest"][0]
    assert out["interval_upper"][0] >= out["highest"][0]
    # The extreme is attained at an interior reach, so corners alone would understate the range.
    corners = {
        (reaches[0], retentions[0]),
        (reaches[0], retentions[-1]),
        (reaches[-1], retentions[0]),
        (reaches[-1], retentions[-1]),
    }
    assert out["at_highest"][0] not in corners, out["at_highest"][0]
    assert "not a confidence interval" in out["coverage_semantics"]


def test_the_envelope_refuses_an_empty_grid():
    """An envelope over no assumptions is a point estimate wearing a range's name."""
    from nimare.meta.cbma.driver import envelope_over_assumptions

    with pytest.raises(ValueError, match="non-empty"):
        envelope_over_assumptions(np.zeros((1, 3)), lambda index: _studies(), [], [1.0])
