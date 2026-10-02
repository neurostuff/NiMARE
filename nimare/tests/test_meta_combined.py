"""Tests for the two documented entry points.

Invariants rather than recorded numbers. The load-bearing ones are that neither class will invent
a reporting model when it has not been given or estimated one, and that the calibration's two
routes to the retention -- the pooled likelihood profile and the reach's own implied ``1 - beta``
-- are separate measurements that the coupling identity relates.
"""

import numpy as np
import pytest

from nimare.meta.cbma.combined import MixedEffectSize, ReportingCalibration


def _maps(seed=0, n_locations=600, threshold=0.75, separation=6.0):
    """One study's map and the table a reporting operator would print from it.

    The peaks are the locations clearing the threshold, thinned so no two sit within a
    separation, which is what makes a nearby-but-not-here location a false silence rather than a
    coincidence.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-40, 40, (n_locations, 3))
    values = rng.normal(0.35, 0.35, n_locations)
    order = np.argsort(-values)
    kept = []
    for index in order:
        if values[index] < threshold:
            break
        far = (np.linalg.norm(positions[index] - positions[other]) > separation for other in kept)
        if all(far):
            kept.append(index)
    kept = np.asarray(kept, dtype=int)
    return [
        {
            "id": "s0",
            "positions": positions,
            "values": values,
            "peaks": positions[kept] if kept.size else np.zeros((0, 3)),
            "heights": values[kept] if kept.size else np.zeros(0),
            "threshold": threshold,
        }
    ]


def test_the_calibration_finds_a_reach_where_the_two_errors_balance():
    """The calibration finds a reach where the two errors balance."""
    calibration = ReportingCalibration().fit(_maps())
    assert calibration.reach_ is not None
    assert 0.0 < calibration.reach_ <= 24.0
    rates = calibration.rates_
    # Below the balanced reach false silences dominate; above it false reports do. That sign
    # change is the condition, so it must be present rather than merely bracketed.
    below = rates["imbalance"][rates["reaches"] < calibration.reach_]
    above = rates["imbalance"][rates["reaches"] > calibration.reach_]
    assert below.size and above.size
    assert below[-1] <= 0.0 <= above[0]


def test_the_error_rates_move_in_the_directions_nested_balls_force():
    """Alpha cannot fall and beta cannot rise as the reach grows: the balls are nested."""
    calibration = ReportingCalibration().fit(_maps())
    rates = calibration.rates_
    assert np.all(np.diff(rates["alpha"]) >= -1e-12)
    assert np.all(np.diff(rates["beta"]) <= 1e-12)


def test_the_implied_retention_at_the_balanced_reach_is_full():
    r"""The identity :math:`\hat\rho = (1-\beta) + F\alpha/S` gives one where counts balance.

    Checked on the *curve*, which is where the identity lives, rather than on
    ``implied_retention_``, which is ``1 - beta`` alone and is deliberately the other half of it.
    """
    calibration = ReportingCalibration().fit(_maps())
    curve = calibration.curve()
    at_balance = float(np.interp(calibration.reach_, curve["reaches"], curve["retentions"]))
    assert at_balance == pytest.approx(1.0, abs=1e-6)


def test_the_curve_is_not_the_rectangle():
    """A shorter reach implies a retention below one, which is the whole point of the coupling."""
    calibration = ReportingCalibration().fit(_maps())
    curve = calibration.curve()
    shorter = [
        retention
        for reach, retention in zip(curve["reaches"], curve["retentions"])
        if reach < calibration.reach_ / 2
    ]
    assert shorter and min(shorter) < 1.0


def test_a_calibration_whose_errors_never_balance_reports_no_reach():
    """A study whose every location clears its threshold has no false reports to balance."""
    positions = np.zeros((5, 3))
    maps = [
        {
            "positions": positions,
            "values": np.full(5, 2.0),
            "peaks": positions,
            "heights": np.full(5, 2.0),
            "threshold": 0.75,
        }
    ]
    with pytest.raises(ValueError, match="empty set"):
        ReportingCalibration().fit(maps)


def test_an_unbalanced_calibration_cannot_furnish_a_reach():
    """Where the imbalance has no sign change, the estimator must refuse rather than substitute."""
    calibration = ReportingCalibration()
    calibration.rates_ = {"reaches": np.array([0.0, 1.0]), "beta": np.array([1.0, 1.0])}
    calibration.reach_ = None
    with pytest.raises(ValueError, match="never balanced"):
        MixedEffectSize(calibration)


def test_a_map_with_no_positive_peak_is_refused_rather_than_guessed():
    """A map with no positive peak is refused rather than guessed."""
    maps = _maps()
    maps[0]["heights"] = -np.abs(maps[0]["heights"])
    with pytest.raises(ValueError, match="positive peak"):
        ReportingCalibration().fit(maps)


def test_the_estimator_refuses_to_default_the_reporting_model():
    """The estimator refuses to default the reporting model."""
    with pytest.raises(ValueError, match="retention is required"):
        MixedEffectSize(8.0)


def test_the_estimator_takes_its_settings_from_a_fitted_calibration():
    """The estimator takes its settings from a fitted calibration."""
    calibration = ReportingCalibration().fit(_maps())
    estimator = MixedEffectSize(calibration)
    assert estimator.reach == pytest.approx(calibration.reach_)
    assert estimator.retention == pytest.approx(calibration.implied_retention_)


def test_fitting_locations_returns_estimates_intervals_and_failures():
    """Fitting locations returns estimates intervals and failures."""
    rng = np.random.default_rng(3)
    positions = rng.uniform(-10, 10, (6, 3))
    studies = [
        {
            "id": f"s{index}",
            "peaks": rng.uniform(-10, 10, (2, 3)),
            "heights": np.array([0.9, 1.1]),
            "threshold": 0.75,
            "variance": 0.04,
        }
        for index in range(5)
    ]

    def studies_at(index):
        return studies

    def images_at(index):
        return np.array([0.4]), np.array([0.01])

    estimator = MixedEffectSize(8.0, retention=0.5, fixed_between_variance=0.0)
    estimator.fit(positions, studies_at, images_at=images_at)
    assert estimator.estimate_.shape == (6,)
    low, high = estimator.interval_
    finite = np.isfinite(low) & np.isfinite(high) & np.isfinite(estimator.estimate_)
    assert finite.any()
    assert np.all(low[finite] <= estimator.estimate_[finite] + 1e-9)
    assert np.all(estimator.estimate_[finite] <= high[finite] + 1e-9)


def test_a_higher_retention_never_leaves_the_estimate_unchanged_on_silent_locations():
    """Retention is load-bearing: if it were inert, supplying it would not be necessary."""
    rng = np.random.default_rng(4)
    positions = rng.uniform(-10, 10, (4, 3))
    studies = [
        {
            "id": f"s{index}",
            "peaks": np.zeros((0, 3)),
            "heights": np.zeros(0),
            "threshold": 0.75,
            "variance": 0.04,
        }
        for index in range(8)
    ]

    def studies_at(index):
        return studies

    def images_at(index):
        return np.array([0.5]), np.array([0.02])

    low_retention = MixedEffectSize(8.0, retention=0.2, fixed_between_variance=0.0).fit(
        positions, studies_at, images_at=images_at
    )
    high_retention = MixedEffectSize(8.0, retention=1.0, fixed_between_variance=0.0).fit(
        positions, studies_at, images_at=images_at
    )
    # Certain silence pushes the estimate down, so the low-retention arm must sit higher.
    assert np.nanmean(low_retention.estimate_) > np.nanmean(high_retention.estimate_)
