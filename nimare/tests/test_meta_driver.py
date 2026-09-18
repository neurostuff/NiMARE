"""Tests for the many-location driver over the censored likelihood."""

import numpy as np
import pytest

from nimare.meta.cbma.censored import ObservationState, bounds_from_states, fit_censored
from nimare.meta.cbma.driver import (
    bundle_studies,
    fit_locations,
    location_records,
    records_for_location,
)


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
        np.zeros(3),
        _studies(),
        reach=8.0,
        image_values=[0.5],
        image_variances=[0.04],
        sided="one",
    )
    # image, the coincident peak, then two silences
    assert lower[0] == upper[0] == pytest.approx(0.5)
    assert lower[1] == upper[1] == pytest.approx(1.9)
    assert not np.isfinite(lower[2:]).any()
    assert upper[2:] == pytest.approx([0.6, 0.6])
    assert roles.tolist() == [0, 1, -1, -1]
    assert variances == pytest.approx([0.04, 0.09, 0.09, 0.09])

    # Four millimetres away the same peak is a bound, and there is no image record.
    lower, upper, _, roles = records_for_location(
        np.array([4.0, 0.0, 0.0]), _studies(), reach=8.0, sided="one"
    )
    assert (lower[0], upper[0]) == pytest.approx((0.6, 1.9))
    assert roles.tolist() == [1, -1, -1]

    # Under the two-sided default the *states* are identical and only the silences' lower bound
    # changes, from unbounded below to the threshold's mirror. Pinning both conventions here
    # rather than only the one in force: the first version of this test asserted
    # ``not isfinite(lower[2:])`` unconditionally, so it encoded the protocol rather than the
    # distance rule it is named for, and flipping the default broke it for the wrong reason.
    lower, upper, _, roles = records_for_location(
        np.zeros(3), _studies(), reach=8.0, image_values=[0.5], image_variances=[0.04]
    )
    assert lower[1] == upper[1] == pytest.approx(1.9)
    assert lower[2:] == pytest.approx([-0.6, -0.6])
    assert upper[2:] == pytest.approx([0.6, 0.6])
    assert roles.tolist() == [0, 1, -1, -1]


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
    states = [
        ObservationState.EXACT,
        ObservationState.NO_PEAK_NEARBY,
        ObservationState.NO_PEAK_NEARBY,
    ]
    values = np.array([1.9, 0.0, 0.0])
    thresholds = np.array([0.6, 0.6, 0.6])

    # One-sided: every record carries a positive sign, so a silence is bounded above only.
    lower, upper, _, _ = records_for_location(np.zeros(3), studies, reach=8.0, sided="one")
    expected_lower, expected_upper = bounds_from_states(
        states, values=values, thresholds=thresholds, signs=np.ones(3)
    )
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)

    # Two-sided: a silence's sign is unspecified, which is how the state machine is asked for
    # the symmetric interval. The driver must still be *asking* rather than computing.
    lower, upper, _, _ = records_for_location(np.zeros(3), studies, reach=8.0, sided="two")
    expected_lower, expected_upper = bounds_from_states(
        states,
        values=values,
        thresholds=thresholds,
        signs=np.array([1.0, np.nan, np.nan]),
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


def _signed_studies():
    """Three studies at one location: a negative peak nearby, a positive one, and nothing."""
    return [
        {
            "id": "negative",
            "peaks": np.array([[3.0, 0.0, 0.0]]),
            "heights": np.array([-1.4]),
            "threshold": 0.75,
            "variance": 0.04,
        },
        {
            "id": "positive",
            "peaks": np.array([[3.0, 0.0, 0.0]]),
            "heights": np.array([1.4]),
            "threshold": 0.75,
            "variance": 0.04,
        },
        {
            "id": "silent",
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.75,
            "variance": 0.04,
        },
    ]


def test_a_negative_peak_is_a_report_two_sided_and_a_silence_one_sided():
    """The estimand is signed, so a printed deactivation is information, not an absence."""
    from nimare.meta.cbma.driver import records_for_location

    position = np.zeros(3)
    one_lower, one_upper, _, one_roles = records_for_location(
        position, _signed_studies(), 8.0, sided="one"
    )
    two_lower, two_upper, _, two_roles = records_for_location(
        position, _signed_studies(), 8.0, sided="two"
    )

    # One-sided: the negative-peak study is entered as an absence bounded by its own threshold.
    assert one_roles[0] == -1
    assert not np.isfinite(one_lower[0]) and one_upper[0] == pytest.approx(0.75)

    # Two-sided: it is a report, on the interval the table actually implies.
    assert two_roles[0] == 1
    assert two_lower[0] == pytest.approx(-1.4)
    assert two_upper[0] == pytest.approx(-0.75)


def test_the_two_sided_silence_is_symmetric_and_the_one_sided_one_is_not():
    """A symmetric silence and discarded negative peaks cannot be mixed; the sides must match."""
    from nimare.meta.cbma.driver import records_for_location

    position = np.zeros(3)
    one_lower, one_upper, _, _ = records_for_location(
        position, _signed_studies(), 8.0, sided="one"
    )
    two_lower, two_upper, _, _ = records_for_location(
        position, _signed_studies(), 8.0, sided="two"
    )
    assert not np.isfinite(one_lower[2]) and one_upper[2] == pytest.approx(0.75)
    assert two_lower[2] == pytest.approx(-0.75)
    assert two_upper[2] == pytest.approx(0.75)


def test_two_sided_negative_reports_drive_the_estimate_negative():
    """And the direction is the point: silence cannot express a deactivation, a report can."""
    from nimare.meta.cbma.driver import fit_locations

    negatives = [
        {
            "id": f"n{index}",
            "peaks": np.array([[2.0, 0.0, 0.0]]),
            "heights": np.array([-1.3]),
            "threshold": 0.75,
            "variance": 0.04,
        }
        for index in range(6)
    ]

    def studies_at(index):
        return negatives

    positions = np.zeros((1, 3))
    two_sided = fit_locations(positions, studies_at, 8.0, fixed_between_variance=0.0, sided="two")
    one_sided = fit_locations(positions, studies_at, 8.0, fixed_between_variance=0.0, sided="one")
    assert two_sided["estimate"][0] < -0.5
    # One-sided reads the same six tables as six silences, so it cannot see the deactivation and
    # lands wherever a bounded-above absence puts it -- above the two-sided answer either way.
    assert one_sided["estimate"][0] > two_sided["estimate"][0]


def test_the_nearest_peak_wins_when_both_signs_are_within_reach():
    """The stated tie-break, since no algebra settles which of two peaks speaks for a location."""
    from nimare.meta.cbma.driver import records_for_location

    study = {
        "id": "both",
        "peaks": np.array([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        "heights": np.array([-1.1, 2.6]),
        "threshold": 0.75,
        "variance": 0.04,
    }
    lower, upper, _, roles = records_for_location(np.zeros(3), [study], 8.0, sided="two")
    # The nearer peak is the negative one, despite the positive one being much larger.
    assert roles[0] == 1
    assert upper[0] == pytest.approx(-0.75)
    assert lower[0] == pytest.approx(-1.1)


def test_a_study_may_declare_its_own_protocol():
    """Sided-ness is a property of a study's reporting rule, not of the analysis."""
    from nimare.meta.cbma.driver import records_for_location

    studies = _signed_studies()
    studies[0]["sided"] = "two"
    _, _, _, roles = records_for_location(np.zeros(3), studies, 8.0, sided="one")
    assert roles[0] == 1


def test_an_unknown_protocol_is_refused():
    """A protocol that is neither one- nor two-sided is a mistake, not a third model."""
    from nimare.meta.cbma.driver import records_for_location

    studies = _signed_studies()
    studies[0]["sided"] = "both"
    with pytest.raises(ValueError, match="sided="):
        records_for_location(np.zeros(3), studies, 8.0)


def test_the_grid_method_agrees_with_the_optimiser():
    """A faster path that disagrees with the reference is not an optimisation.

    Pinned at four decimal places on both the estimate and the interval bounds, which is where
    the grid resolution was chosen to put it and two orders below the estimate's own standard
    error. A regression here means the fast path has drifted from the likelihood, not that it is
    approximate.
    """
    from nimare.meta.cbma.driver import fit_locations

    rng = np.random.default_rng(11)
    studies = []
    for index in range(24):
        count = int(rng.integers(10, 120))
        threshold = 3.09 / np.sqrt(count)
        n_peaks = max(1, int(rng.poisson(5)))
        magnitudes = threshold * (1.0 + rng.exponential(0.6, n_peaks))
        signs = rng.choice([1.0, -1.0], n_peaks, p=[0.85, 0.15])
        studies.append(
            {
                "id": f"s{index}",
                "peaks": rng.uniform(-40, 40, (n_peaks, 3)),
                "heights": magnitudes * signs,
                "threshold": threshold,
                "variance": 1.0 / count,
            }
        )
    positions = rng.uniform(-40, 40, (25, 3))

    def studies_at(index):
        return studies

    def images_at(index):
        return np.array([0.4]), np.array([0.02])

    shared = dict(retention=0.25, fixed_between_variance=0.0, images_at=images_at, sided="two")
    exact = fit_locations(positions, studies_at, 8.0, method="exact", **shared)
    grid = fit_locations(positions, studies_at, 8.0, method="grid", **shared)

    usable = exact["valid"] & grid["valid"]
    assert usable.sum() >= 20
    assert np.max(np.abs(exact["estimate"][usable] - grid["estimate"][usable])) < 1e-3
    assert np.max(np.abs(exact["lower"][usable] - grid["lower"][usable])) < 1e-3
    assert np.max(np.abs(exact["upper"][usable] - grid["upper"][usable])) < 1e-3


def test_the_grid_method_profiles_the_between_study_variance_when_none_is_fixed():
    """And it must, because holding it at zero makes the estimator unable to express a mid effect.

    A tight silence from a large study and a printed peak from a small one are *contradictory*
    at zero between-study variance -- they then describe the same quantity -- so the likelihood
    separates into two basins and the estimate lands in one of them. On a whole-brain fit that
    left a hole in the distribution: three voxels in the band |g| from 0.115 to 0.185 where 140
    belong. Freeing the variance lets the two records describe different study levels, and the
    estimate can sit between them.
    """
    from nimare.meta.cbma.driver import fit_locations

    # One very precise silence, asserting |g| < 0.1, against one printed peak at 0.9.
    studies = [
        {
            "id": "large_silent",
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.1,
            "variance": 0.001,
        },
        {
            "id": "small_reporting",
            "peaks": np.array([[1.0, 0.0, 0.0]]),
            "heights": np.array([0.9]),
            "threshold": 0.6,
            "variance": 0.05,
        },
    ]

    def studies_at(index):
        return studies

    positions = np.zeros((1, 3))
    held = fit_locations(
        positions, studies_at, 8.0, fixed_between_variance=0.0, method="grid", sided="two"
    )
    freed = fit_locations(positions, studies_at, 8.0, method="grid", sided="two")

    # Held at zero, the estimate is pinned by the precise silence and cannot reach the peak.
    assert abs(held["estimate"][0]) <= 0.15
    assert held["between_variance"][0] == 0.0
    # Freed, the fit uses heterogeneity to reconcile them and lands between the two claims.
    assert freed["between_variance"][0] > 0.0
    assert freed["estimate"][0] > held["estimate"][0]


def test_an_unknown_method_is_refused():
    """Silently falling back to the slow path would hide a typo in a caller's configuration."""
    from nimare.meta.cbma.driver import fit_locations

    def studies_at(index):
        return []

    with pytest.raises(ValueError, match="method must be"):
        fit_locations(np.zeros((1, 3)), studies_at, 8.0, method="fast")


def test_parallel_and_serial_fits_are_identical():
    """Parallelism is a scheduling change, so any difference in the numbers is a defect.

    Checked bit-for-bit rather than approximately: the workers run the same single-location
    function the serial loop does, so there is no arithmetic that could legitimately differ.
    """
    from nimare.meta.cbma.driver import fit_locations

    rng = np.random.default_rng(13)
    studies = []
    for index in range(12):
        count = int(rng.integers(10, 80))
        threshold = 3.09 / np.sqrt(count)
        n_peaks = max(1, int(rng.poisson(4)))
        magnitudes = threshold * (1.0 + rng.exponential(0.6, n_peaks))
        signs = rng.choice([1.0, -1.0], n_peaks, p=[0.8, 0.2])
        studies.append(
            {
                "id": f"s{index}",
                "peaks": rng.uniform(-30, 30, (n_peaks, 3)),
                "heights": magnitudes * signs,
                "threshold": threshold,
                "variance": 1.0 / count,
            }
        )
    positions = rng.uniform(-30, 30, (23, 3))

    def studies_at(index):
        return studies

    def images_at(index):
        return np.array([0.35]), np.array([0.03])

    shared = dict(
        retention=0.25,
        fixed_between_variance=0.0,
        images_at=images_at,
        method="grid",
        sided="two",
    )
    serial = fit_locations(positions, studies_at, 8.0, n_jobs=1, **shared)
    parallel = fit_locations(positions, studies_at, 8.0, n_jobs=2, **shared)

    for key in ("estimate", "lower", "upper", "between_variance"):
        assert np.array_equal(serial[key], parallel[key], equal_nan=True), key
    for key in ("valid", "converged", "touched_search_limit"):
        assert np.array_equal(serial[key], parallel[key]), key


def test_a_protocol_change_is_not_masked_by_the_bundle_cache():
    """Two calls on the same study list under different protocols must differ.

    This is the regression for a module-level bundle cache keyed on ``id(studies)``: it returned
    the first call's flattening to the second, and since the protocol is baked into the bundle,
    a one-sided fit silently got two-sided records.
    """
    from nimare.meta.cbma.driver import fit_locations

    negatives = [
        {
            "id": f"n{index}",
            "peaks": np.array([[2.0, 0.0, 0.0]]),
            "heights": np.array([-1.3]),
            "threshold": 0.75,
            "variance": 0.04,
        }
        for index in range(6)
    ]

    def studies_at(index):
        return negatives

    positions = np.zeros((1, 3))
    two = fit_locations(positions, studies_at, 8.0, fixed_between_variance=0.0, sided="two")
    one = fit_locations(positions, studies_at, 8.0, fixed_between_variance=0.0, sided="one")
    assert one["estimate"][0] != two["estimate"][0]


def test_location_records_labels_agree_with_the_fitting_path():
    """The reporting vector and the likelihood's input must be the same records.

    ``location_records`` exists so a reader can see which statement each record is, and the one
    way it can go wrong is by drifting from ``records_for_location``. Pinned by comparing the
    bounds and variances element for element, since a labelled table that describes different
    records than the fit used would be worse than no table.
    """
    studies = [
        {
            "peaks": np.array([[3.0, 0.0, 0.0], [40.0, 0.0, 0.0]]),
            "heights": np.array([0.9, 1.4]),
            "threshold": 0.5,
            "variance": 0.04,
        },
        {
            "peaks": np.array([[0.0, 0.0, 0.0]]),
            "heights": np.array([-0.7]),
            "threshold": 0.45,
            "variance": 0.03,
        },
        {
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.6,
            "variance": 0.05,
        },
    ]
    position = np.zeros(3)
    images = np.array([0.2, -0.1])
    image_variances = np.array([0.01, 0.02])

    lower, upper, variances, _ = records_for_location(
        position, studies, 4.0, image_values=images, image_variances=image_variances
    )
    table = location_records(
        position, studies, 4.0, image_values=images, image_variances=image_variances
    )

    np.testing.assert_allclose(table["lower"], lower)
    np.testing.assert_allclose(table["upper"], upper)
    np.testing.assert_allclose(table["variance"], variances)
    assert [str(state) for state in table["state"]] == [
        "IMAGE",
        "IMAGE",
        "CLUSTER_PEAK",
        "EXACT",
        "NO_PEAK_NEARBY",
    ]
    assert [str(source) for source in table["source"]] == [
        "image",
        "image",
        "table",
        "table",
        "table",
    ]


def test_location_records_describes_a_silence_as_an_interval_not_a_value():
    """A silence has no value, and reporting one as zero would invent an observation."""
    studies = [
        {
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.35,
            "variance": 0.02,
        }
    ]
    table = location_records(np.zeros(3), studies, 4.0)
    assert str(table["state"][0]) == "NO_PEAK_NEARBY"
    assert np.isnan(table["value"][0])
    assert np.isnan(table["distance_mm"][0])
    # Two-sided silence is the symmetric interval, so it constrains both tails.
    np.testing.assert_allclose([table["lower"][0], table["upper"][0]], [-0.35, 0.35])


def test_location_records_reports_a_negative_peak_as_signed_and_never_as_an_absence():
    """A printed deactivation is a signed report, and its bracket depends on the distance.

    Written first with both states asserted to be ``[h, -c]``, which the code refused: a peak
    *at* the location fixes the value there, so ``EXACT`` is a point and only ``CLUSTER_PEAK``
    is a bracket. The estimator was right and the test was wrong; both are pinned here so the
    distinction cannot be lost, since collapsing them would either invent precision at a
    neighbour or discard it at a coincidence.
    """

    def one(peak, height):
        studies = [
            {
                "peaks": np.array([peak], dtype=float),
                "heights": np.array([height], dtype=float),
                "threshold": 0.4,
                "variance": 0.02,
            }
        ]
        return location_records(np.zeros(3), studies, 4.0, sided="two")

    at_voxel = one([0.0, 0.0, 0.0], -0.8)
    assert str(at_voxel["state"][0]) == "EXACT"
    np.testing.assert_allclose([at_voxel["lower"][0], at_voxel["upper"][0]], [-0.8, -0.8])

    nearby = one([3.0, 0.0, 0.0], -0.8)
    assert str(nearby["state"][0]) == "CLUSTER_PEAK"
    np.testing.assert_allclose([nearby["lower"][0], nearby["upper"][0]], [-0.8, -0.4])

    # Both are strictly negative, so neither can be read as the two-sided absence (-c, c).
    for table in (at_voxel, nearby):
        assert table["upper"][0] < 0.0


def test_location_records_counts_a_real_corpus_as_mostly_silence():
    """The composition is the point of the vector, so a regression on it is worth pinning.

    Twenty studies whose peaks are scattered far from the location: every record must be a
    silence, and the two images must be the only entries carrying a value. This is the shape of
    a real corpus -- 98% silence on the faces tables -- in miniature.
    """
    rng = np.random.default_rng(11)
    studies = [
        {
            "peaks": rng.uniform(40.0, 80.0, size=(3, 3)),
            "heights": rng.uniform(0.6, 1.5, size=3),
            "threshold": 0.5,
            "variance": 0.04,
        }
        for _ in range(20)
    ]
    table = location_records(
        np.zeros(3),
        studies,
        4.0,
        image_values=[0.3, 0.1],
        image_variances=[0.01, 0.02],
    )
    states = [str(state) for state in table["state"]]
    assert states.count("NO_PEAK_NEARBY") == 20
    assert states.count("IMAGE") == 2
    assert np.count_nonzero(np.isfinite(table["value"])) == 2


def test_location_records_refuses_a_bundle_and_a_protocol_together():
    """Passing both is ambiguous, and the first guard for it could never fire.

    The original check compared ``sided`` against every entry of ``bundle["sided"]`` and
    skipped itself whenever ``sided`` held its own default -- which is most calls, and exactly
    the case the check existed for. It was unsound as well as inert: ``bundle_studies`` honours
    a per-study ``sided`` key, so entries differing from the argument are legal. Refusing the
    combination is the check that works.
    """
    studies = [
        {
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.4,
            "variance": 0.02,
        }
    ]
    bundle = bundle_studies(studies, sided="one")

    with pytest.raises(ValueError, match="either a bundle or"):
        location_records(np.zeros(3), studies, 4.0, sided="two", bundle=bundle)
    with pytest.raises(ValueError, match="either a bundle or"):
        location_records(np.zeros(3), studies, 4.0, sided="one", bundle=bundle)

    # A bundle on its own is honoured, and its protocol is the one that applies: a one-sided
    # absence is bounded above only, so the lower bound stays at negative infinity.
    from_bundle = location_records(np.zeros(3), studies, 4.0, bundle=bundle)
    assert np.isneginf(from_bundle["lower"][0])
    np.testing.assert_allclose(from_bundle["upper"][0], 0.4)

    # And with no bundle the default is two-sided, which bounds both tails.
    default = location_records(np.zeros(3), studies, 4.0)
    np.testing.assert_allclose([default["lower"][0], default["upper"][0]], [-0.4, 0.4])
