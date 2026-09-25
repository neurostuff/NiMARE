"""Tests for prospective planning from a meta-analytic effect.

Invariants and one external check. The external check matters most: the power function is
compared against simulating the t test it claims to describe, so an error in the noncentrality
convention cannot hide behind self-consistency.
"""

import numpy as np
import pytest
from scipy.stats import nct, norm, ttest_1samp

from nimare.meta.cbma.planning import (
    assurance,
    assurance_ceiling,
    assurance_is_converged,
    design_scale,
    directional_power,
    predictive_standard_deviation,
    required_sample_size,
    roi_standardised_effect,
    two_sided_power,
)


def test_the_power_function_matches_a_direct_simulation_of_the_t_test():
    """Simulate the test the formula describes, so a convention error cannot hide."""
    rng = np.random.default_rng(3)
    draws = 20_000
    for size, effect in ((20, 0.4), (40, 0.25), (12, 0.8)):
        samples = rng.normal(effect, 1.0, size=(draws, size))
        rejected = float((ttest_1samp(samples, 0.0, axis=1).pvalue < 0.05).mean())
        predicted = float(two_sided_power(effect, size))
        tolerance = 3 * np.sqrt(rejected * (1 - rejected) / draws)
        assert predicted == pytest.approx(rejected, abs=tolerance)


def test_assurance_sits_below_plug_in_power_above_the_fifty_percent_effect_and_above_it_below():
    """Jensen's direction flips at the effect giving fifty percent power.

    Proved in ``proofs/assurance_not_plug_in_power.py``: power's curvature is the density times
    minus the non-centrality, so it changes from convex to concave exactly there. This is the
    practical consequence -- plug-in power is optimistic in the regime where a study is being
    designed to be adequately powered.
    """
    for size in (16, 36, 64):
        fifty = float(nct.ppf(0.975, size - 1, 0.0)) / np.sqrt(size)
        spread = 0.2
        below = float(assurance(size, fifty * 0.5, spread)[0])
        above = float(assurance(size, fifty * 1.8, spread)[0])
        assert below > float(two_sided_power(fifty * 0.5, size))
        assert above < float(two_sided_power(fifty * 1.8, size))


def test_assurance_never_reaches_its_ceiling_however_large_the_study():
    """No sample size buys assurance past the chance the new effect has the right sign."""
    mean, spread = 0.3, 0.25
    ceiling = assurance_ceiling(mean, spread)
    values = assurance(np.array([50, 500, 5000, 50000]), mean, spread)
    assert np.all(np.diff(values) > 0)
    assert np.all(values < ceiling)
    assert values[-1] == pytest.approx(ceiling, abs=0.01)

    # And the distinction the ceiling depends on: two-sided assurance has no such ceiling,
    # because a two-sided test rejects for either sign and its power tends to one everywhere.
    two_sided = assurance(np.array([50000]), mean, spread, directional=False)
    assert two_sided[0] > ceiling
    assert assurance_ceiling(mean, spread, directional=False) == 1.0


def test_zero_predictive_spread_is_exactly_plug_in_power():
    """The degenerate predictive distribution must not route through the quadrature.

    Compared against the matching power in each mode. Directional assurance is *not* two-sided
    power -- they differ by the probability of rejecting in the wrong tail, 6e-4 at these
    settings -- and conflating them is how the ceiling error got in.
    """
    sizes = np.array([12, 30, 90])
    assert np.array_equal(
        assurance(sizes, 0.4, 0.0),
        np.asarray(directional_power(0.4, sizes, direction=1.0), dtype=float),
    )
    assert np.array_equal(
        assurance(sizes, 0.4, 0.0, directional=False),
        np.asarray(two_sided_power(0.4, sizes), dtype=float),
    )


def test_power_is_even_in_the_effect_and_monotone_in_both_arguments():
    """A two-sided test cannot care about the sign, and more data cannot hurt."""
    assert float(two_sided_power(0.35, 40)) == float(two_sided_power(-0.35, 40))
    by_size = two_sided_power(0.3, np.array([10, 20, 40, 80, 160]))
    assert np.all(np.diff(by_size) > 0)
    by_effect = two_sided_power(np.array([0.05, 0.2, 0.4, 0.8]), np.full(4, 30))
    assert np.all(np.diff(by_effect) > 0)


def test_power_at_a_null_effect_is_the_test_size():
    """At no effect a valid test rejects at its nominal rate, which anchors the whole curve."""
    for size in (8, 25, 120):
        assert float(two_sided_power(0.0, size, alpha=0.05)) == pytest.approx(0.05, abs=1e-9)
        assert float(two_sided_power(0.0, size, alpha=0.01)) == pytest.approx(0.01, abs=1e-9)


def test_the_sample_size_search_distinguishes_its_three_failure_modes():
    """Unattainable, over budget, and reached are different answers and must not be one None.

    The search also must not step over a feasible size below the budget. The audited defect:
    mean .3, no predictive spread, target .8 and a maximum of 100 returned nothing although
    assurance at 100 is .8439 and 90 suffices, because the bracket doubled past the budget.
    """
    mean, spread = 0.2, 0.3
    ceiling = assurance_ceiling(mean, spread)

    size, status = required_sample_size(min(ceiling + 0.05, 0.999), mean, spread)
    assert size is None and status == "above_asymptotic_limit"

    size, status = required_sample_size(ceiling * 0.95, mean, spread, maximum=10)
    assert size is None and status == "not_reached_within_budget"

    size, status = required_sample_size(ceiling * 0.8, mean, spread)
    assert status == "reached"
    assert float(assurance(size, mean, spread)[0]) >= ceiling * 0.8
    assert float(assurance(size - 1, mean, spread)[0]) < ceiling * 0.8

    # The audited case, exactly.
    assert required_sample_size(0.8, 0.3, 0.0, maximum=100) == (90, "reached")


def test_a_point_mass_at_no_effect_has_the_test_size_as_its_limit_not_one():
    """A degenerate predictive distribution at zero effect never becomes detectable.

    An earlier version returned one for any zero predictive spread, which is right for a point
    mass at a non-zero effect and wrong at zero: the directional rejection probability there is
    alpha/2 at every sample size, including in the limit. Caught by the external audit.
    """
    assert assurance_ceiling(0.0, 0.0) == pytest.approx(0.025)
    assert assurance_ceiling(0.0, 0.0, directional=False) == pytest.approx(0.05)
    assert assurance_ceiling(0.0, 0.0, alpha=0.01) == pytest.approx(0.005)
    # A point mass away from zero is still detectable with certainty in the limit.
    assert assurance_ceiling(0.4, 0.0) == pytest.approx(1.0)
    # An explicit atom inside a continuous mixture is carried through.
    assert assurance_ceiling(0.3, 0.25, null_mass=0.2) == pytest.approx(
        0.8 * float(norm.cdf(0.3 / 0.25)) + 0.025 * 0.2
    )


def test_a_two_sample_design_needs_more_observations_than_a_one_sample_one():
    """The scale factor differs, which the document insists is not a total-count detail."""
    assert float(design_scale(64, "two-sample")) > float(design_scale(64, "one-sample"))
    one, one_status = required_sample_size(0.8, 0.5, 0.1, design="one-sample")
    two, two_status = required_sample_size(0.8, 0.5, 0.1, design="two-sample")
    assert one_status == two_status == "reached"
    assert two > one


def test_an_roi_average_is_not_the_average_of_voxelwise_standardised_effects():
    """The ROI denominator is the root mean of the covariance entries, correlations included."""
    means = np.array([0.4, 0.4])
    variance = 0.04
    average_of_voxelwise = float(np.mean(means / np.sqrt(variance)))
    for correlation in (0.0, 0.5, 0.95):
        covariance = variance * np.array([[1.0, correlation], [correlation, 1.0]])
        roi = roi_standardised_effect(means, covariance)
        if correlation < 1.0:
            assert roi != pytest.approx(average_of_voxelwise, rel=1e-6)
    # Independence makes the ROI average more precise, perfect correlation makes it identical
    # to a single voxel; the two bracket everything between.
    independent = roi_standardised_effect(means, variance * np.array([[1.0, 0.0], [0.0, 1.0]]))
    dependent = roi_standardised_effect(means, variance * np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert independent > average_of_voxelwise > dependent - 1e-12
    assert dependent == pytest.approx(average_of_voxelwise, rel=1e-12)


def test_the_predictive_spread_needs_both_of_its_terms():
    """Heterogeneity and estimation error add in variance; dropping either is a different claim."""
    assert float(predictive_standard_deviation(0.09, 0.0)) == pytest.approx(0.3)
    assert float(predictive_standard_deviation(0.0, 0.3)) == pytest.approx(0.3)
    assert float(predictive_standard_deviation(0.09, 0.4)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="can be negative|cannot be negative"):
        predictive_standard_deviation(-0.1, 0.2)


def test_a_degenerate_design_is_refused():
    """A t design with too few observations, or a bad allocation, has no power to report."""
    with pytest.raises(ValueError, match="at least three"):
        two_sided_power(0.4, 2)
    with pytest.raises(ValueError, match="strictly between zero and one"):
        design_scale(40, "two-sample", allocation=0.0)
    with pytest.raises(ValueError, match="must be one of"):
        design_scale(40, "paired-difference")


def test_the_assurance_integral_reports_its_own_convergence():
    """A rule that has not resolved the power transition returns a plausible wrong number.

    This is not hypothetical. The first version of :func:`assurance` used a fixed Gauss-Hermite
    rule, which cannot resolve a power curve whose transition is a few thousandths wide across a
    predictive distribution several tenths wide; at fifty thousand observations it returned
    0.900 for a quantity whose analytic limit is 0.885. The ceiling test caught it.
    """
    sizes = np.array([50, 500, 5000, 50000, 500000])
    converged, gap = assurance_is_converged(sizes, 0.3, 0.25)
    assert converged
    assert gap < 1e-4

    # And the specific failure: assurance must approach the ceiling from below at every size.
    values = assurance(sizes, 0.3, 0.25)
    ceiling = assurance_ceiling(0.3, 0.25)
    assert np.all(values < ceiling)
    assert values[-1] > 0.98 * ceiling
