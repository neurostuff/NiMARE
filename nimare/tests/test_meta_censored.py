"""Tests for the scalar interval-censored random-effects reference.

These are invariants rather than recorded outputs. Every one of them either encodes a claim
verified symbolically in ``proofs/scalar_censored_reference.py`` or encodes a refusal the design
requires: an unknown table is not a small effect, and an unavailable estimate is not zero.
"""

import numpy as np
import pytest
from scipy.stats import norm

from nimare.meta.cbma.censored import (
    ObservationState,
    bounds_from_states,
    censored_loglik,
    censored_score,
    fit_censored,
    practical_prevalence,
    prevalence_times_conditional_mean_gap,
    profile_interval,
    retention_roles,
)


def _grid_random_effects(values, variances, means, betweens):
    """Independent maximiser of the ordinary random-effects likelihood, by brute force.

    Deliberately does not reuse anything from the module under test: the point is to check the
    censored code against a likelihood written from scratch.
    """
    best = (-np.inf, np.nan, np.nan)
    for mean in means:
        for between in betweens:
            total = float(norm.logpdf(values, loc=mean, scale=np.sqrt(variances + between)).sum())
            if total > best[0]:
                best = (total, mean, between)
    return best


def test_a_fit_of_exact_records_agrees_with_ordinary_random_effects_meta_analysis():
    """A degenerate interval is the Gaussian density, so the two maximisers must coincide.

    This is the identity proved as "a narrow interval carries the density in its leading term"
    together with "no second-order term": the interval width factors out of the log-likelihood
    as an additive constant free of the parameters, so it cannot move the maximum.
    """
    rng = np.random.default_rng(7)
    variances = rng.uniform(0.01, 0.09, size=30)
    values = rng.normal(0.4, np.sqrt(variances + 0.02))

    lower, upper = bounds_from_states([ObservationState.IMAGE] * 30, values=values)
    fit = fit_censored(lower, upper, variances)

    _, grid_mean, grid_between = _grid_random_effects(
        values,
        variances,
        np.linspace(fit["mean"] - 0.05, fit["mean"] + 0.05, 401),
        np.linspace(max(fit["between_variance"] - 0.02, 0.0), fit["between_variance"] + 0.02, 401),
    )
    assert fit["mean"] == pytest.approx(grid_mean, abs=5e-4)
    assert fit["between_variance"] == pytest.approx(grid_between, abs=5e-4)


def test_both_scores_agree_with_finite_differences_on_every_kind_of_record():
    """The gradient the optimiser is handed must be the gradient of the objective it minimises."""
    rng = np.random.default_rng(3)
    states = [
        ObservationState.IMAGE,
        ObservationState.ROUNDED,
        ObservationState.DIRECTION_ONLY,
        ObservationState.NONSIGNIFICANT,
        ObservationState.ABSENT_COMPLETE_TABLE,
        ObservationState.UNKNOWN_COMPLETENESS,
    ]
    lower, upper = bounds_from_states(
        states,
        values=np.array([0.55, 0.62, np.nan, np.nan, np.nan, np.nan]),
        precisions=np.array([np.nan, 0.01, np.nan, np.nan, np.nan, np.nan]),
        thresholds=np.array([np.nan, np.nan, 0.5, 0.5, 0.45, np.nan]),
        signs=np.array([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan]),
        absence_is_voxelwise=True,
    )
    variances = rng.uniform(0.02, 0.08, size=len(states))

    for mean, between in ((0.3, 0.02), (0.0, 0.0), (0.7, 0.15)):
        analytic = censored_score(mean, between, lower, upper, variances)
        step = 1e-6
        numeric_mean = (
            censored_loglik(mean + step, between, lower, upper, variances)
            - censored_loglik(mean - step, between, lower, upper, variances)
        ) / (2 * step)
        numeric_between = (
            censored_loglik(mean, between + step, lower, upper, variances)
            - censored_loglik(mean, max(between - step, 0.0), lower, upper, variances)
        ) / ((between + step) - max(between - step, 0.0))
        assert analytic[0] == pytest.approx(numeric_mean, rel=1e-4, abs=1e-6)
        assert analytic[1] == pytest.approx(numeric_between, rel=1e-3, abs=1e-4)


def test_a_table_of_unknown_completeness_changes_nothing_at_all():
    """Unknown is not null: adding such records must leave the fit bit-identical.

    The strong form is deliberate. A record that contributed even a tiny amount would be a
    record asserting that an unlisted voxel was below threshold, which is exactly the inference
    an incomplete table cannot support.
    """
    rng = np.random.default_rng(19)
    variances = rng.uniform(0.02, 0.06, size=12)
    values = rng.normal(0.35, 0.3, size=12)
    lower, upper = bounds_from_states([ObservationState.IMAGE] * 12, values=values)
    before = fit_censored(lower, upper, variances)

    padded_states = [ObservationState.IMAGE] * 12 + [
        ObservationState.UNKNOWN_COMPLETENESS,
        ObservationState.OUTSIDE_MASK,
    ] * 20
    padded_lower, padded_upper = bounds_from_states(
        padded_states, values=np.concatenate([values, np.zeros(40)])
    )
    padded_variances = np.concatenate([variances, np.full(40, 0.04)])
    after = fit_censored(padded_lower, padded_upper, padded_variances)

    assert after["mean"] == before["mean"]
    assert after["between_variance"] == before["between_variance"]
    assert after["n_informative"] == before["n_informative"] == 12


def test_a_wider_nonsignificance_interval_pulls_the_fit_less():
    """A nonsignificant record is an interval, so widening it must weaken it monotonically.

    If such records were imputed as zeros, widening the threshold would not weaken them at all.
    """
    rng = np.random.default_rng(5)
    observed = rng.normal(0.5, 0.15, size=6)
    exact_lower, exact_upper = bounds_from_states([ObservationState.IMAGE] * 6, values=observed)
    variances = np.full(6, 0.03)
    anchor = fit_censored(exact_lower, exact_upper, variances)["mean"]

    distances = []
    for cut in (0.2, 0.6, 1.8):
        states = [ObservationState.IMAGE] * 6 + [ObservationState.NONSIGNIFICANT] * 6
        lower, upper = bounds_from_states(
            states,
            values=np.concatenate([observed, np.full(6, np.nan)]),
            thresholds=np.concatenate([np.full(6, np.nan), np.full(6, cut)]),
        )
        fit = fit_censored(lower, upper, np.full(12, 0.03))
        distances.append(abs(fit["mean"] - anchor))

    assert distances[0] > distances[1] > distances[2]


def test_nothing_informative_returns_not_a_number_rather_than_zero():
    """An unavailable estimate must not be reported as a precisely estimated zero."""
    states = [ObservationState.UNKNOWN_COMPLETENESS, ObservationState.OUTSIDE_MASK]
    lower, upper = bounds_from_states(states)
    fit = fit_censored(lower, upper, np.full(2, 0.05))
    assert np.isnan(fit["mean"])
    assert fit["valid"] is False
    assert fit["n_informative"] == 0

    interval = profile_interval(lower, upper, np.full(2, 0.05))
    assert np.isnan(interval["lower"]) and np.isnan(interval["upper"])
    assert interval["valid"] is False


def test_an_unsigned_directional_record_is_refused_rather_than_guessed():
    """An unsigned report constrains both tails; it is a union, not an interval."""
    with pytest.raises(ValueError, match="union of two intervals"):
        bounds_from_states([ObservationState.DIRECTION_ONLY], thresholds=[0.5], signs=[0.0])


def test_a_missing_threshold_is_refused_rather_than_defaulted():
    """A silently defaulted threshold is an assumption entering through a gap."""
    with pytest.raises(ValueError, match="need 'thresholds'"):
        bounds_from_states([ObservationState.ABSENT_COMPLETE_TABLE], absence_is_voxelwise=True)
    with pytest.raises(ValueError, match="need 'values'"):
        bounds_from_states([ObservationState.IMAGE], absence_is_voxelwise=True)
    with pytest.raises(ValueError, match="not an observation state"):
        bounds_from_states(["probably_silent"], absence_is_voxelwise=True)


def test_a_one_sided_record_matches_a_two_sided_one_with_a_remote_far_bound():
    """The one-sided limits proved for the likelihood must hold in the code too."""
    rng = np.random.default_rng(23)
    variances = rng.uniform(0.02, 0.05, size=8)
    thresholds = np.full(8, 0.45)

    one_sided_lower, one_sided_upper = bounds_from_states(
        [ObservationState.DIRECTION_ONLY] * 8,
        thresholds=thresholds,
        signs=np.ones(8),
        absence_is_voxelwise=True,
    )
    remote_lower, remote_upper = one_sided_lower.copy(), np.full(8, 1e6)

    for mean, between in ((0.4, 0.02), (0.6, 0.0)):
        assert censored_loglik(
            mean, between, one_sided_lower, one_sided_upper, variances
        ) == pytest.approx(
            censored_loglik(mean, between, remote_lower, remote_upper, variances), abs=1e-9
        )


def test_freezing_heterogeneity_can_never_raise_the_likelihood():
    """The free fit maximises over a superset, so its likelihood bounds every frozen one.

    This is the invariant that makes "borrowed heterogeneity" measurable: any cost of freezing
    shows up here as a likelihood gap rather than being absorbed silently.
    """
    rng = np.random.default_rng(31)
    variances = rng.uniform(0.02, 0.08, size=16)
    values = rng.normal(0.4, np.sqrt(variances + 0.03))
    lower, upper = bounds_from_states([ObservationState.IMAGE] * 16, values=values)

    free = fit_censored(lower, upper, variances)
    for frozen in (0.0, 0.01, 0.05, 0.2):
        held = fit_censored(lower, upper, variances, fixed_between_variance=frozen)
        assert held["loglik"] <= free["loglik"] + 1e-8


def test_the_profile_interval_contains_the_estimate_and_narrows_with_more_studies():
    """A likelihood-ratio interval must bracket the maximum, and more data must tighten it."""
    rng = np.random.default_rng(41)
    widths = []
    for count in (8, 64):
        variances = np.full(count, 0.04)
        values = rng.normal(0.4, np.sqrt(0.04 + 0.02), size=count)
        lower, upper = bounds_from_states([ObservationState.IMAGE] * count, values=values)
        fit = fit_censored(lower, upper, variances)
        interval = profile_interval(lower, upper, variances)
        assert interval["valid"]
        assert interval["lower"] < fit["mean"] < interval["upper"]
        widths.append(interval["upper"] - interval["lower"])
    assert widths[1] < widths[0]


def _retention_setup(reported_flags, threshold=0.6):
    """States, bounds and roles for a table of one-sided reports and absences."""
    states = [
        ObservationState.DIRECTION_ONLY if flag else ObservationState.ABSENT_COMPLETE_TABLE
        for flag in reported_flags
    ]
    count = len(states)
    lower, upper = bounds_from_states(
        states,
        thresholds=np.full(count, threshold),
        signs=np.ones(count),
        absence_is_voxelwise=True,
    )
    return lower, upper, retention_roles(states)


def test_full_retention_is_bit_identical_to_leaving_retention_out():
    """Ignoring retention is exactly the rho = 1 model, so the two must agree to the last bit.

    Proved as "ignoring retention is exactly the model that assumes every exceedance is printed".
    An approximate agreement would mean the retention-aware path is a different likelihood.
    """
    flags = [True, False, False, True, False, False, False, True]
    lower, upper, roles = _retention_setup(flags)
    variances = np.full(len(flags), 0.04)

    for mean, between in ((0.4, 0.0225), (0.0, 0.0), (0.9, 0.3)):
        plain = censored_loglik(mean, between, lower, upper, variances)
        with_full = censored_loglik(
            mean, between, lower, upper, variances, retention=1.0, roles=roles
        )
        assert with_full == plain
        assert np.array_equal(
            censored_score(mean, between, lower, upper, variances),
            censored_score(mean, between, lower, upper, variances, retention=1.0, roles=roles),
        )


def test_the_retention_aware_score_agrees_with_finite_differences():
    """The absence term's derivative carries a factor of rho; check it rather than trust it."""
    flags = [True, False, False, False, True, False]
    lower, upper, roles = _retention_setup(flags)
    variances = np.full(len(flags), 0.04)

    for retention in (0.2, 0.5, 0.95):
        for mean, between in ((0.35, 0.02), (0.6, 0.1)):
            analytic = censored_score(
                mean, between, lower, upper, variances, retention=retention, roles=roles
            )
            step = 1e-6
            numeric_mean = (
                censored_loglik(
                    mean + step,
                    between,
                    lower,
                    upper,
                    variances,
                    retention=retention,
                    roles=roles,
                )
                - censored_loglik(
                    mean - step,
                    between,
                    lower,
                    upper,
                    variances,
                    retention=retention,
                    roles=roles,
                )
            ) / (2 * step)
            numeric_between = (
                censored_loglik(
                    mean,
                    between + step,
                    lower,
                    upper,
                    variances,
                    retention=retention,
                    roles=roles,
                )
                - censored_loglik(
                    mean,
                    between - step,
                    lower,
                    upper,
                    variances,
                    retention=retention,
                    roles=roles,
                )
            ) / (2 * step)
            assert analytic[0] == pytest.approx(numeric_mean, rel=1e-4, abs=1e-6)
            assert analytic[1] == pytest.approx(numeric_between, rel=1e-4, abs=1e-6)


def test_knowing_the_retention_removes_the_bias_that_assuming_full_retention_creates():
    """The rho = 1 fit lands where the algebra says, and the rho-aware fit lands on the truth.

    The misspecified fit's population root satisfies ``S(m_hat) = rho * S(m_0)``, so its target
    is known in closed form. Both arms are checked against their own predicted limits, using
    expected counts rather than a sampled table so that no Monte Carlo noise enters.
    """
    true_mean, threshold, retention = 0.4, 0.6, 0.5
    total_sd = np.sqrt(0.04 + 0.0225)
    survival = norm.sf((threshold - true_mean) / total_sd)

    # A table whose report fraction equals its expectation exactly.
    count = 2000
    n_reported = int(round(retention * survival * count))
    flags = [True] * n_reported + [False] * (count - n_reported)
    lower, upper, roles = _retention_setup(flags, threshold=threshold)
    variances = np.full(count, 0.04)

    assuming_full = fit_censored(lower, upper, variances, fixed_between_variance=0.0225)
    predicted = threshold - total_sd * norm.isf(retention * survival)
    assert assuming_full["mean"] == pytest.approx(predicted, abs=0.01)
    assert assuming_full["mean"] < true_mean - 0.05

    knowing_it = fit_censored(
        lower,
        upper,
        variances,
        retention=retention,
        roles=roles,
        fixed_between_variance=0.0225,
    )
    assert knowing_it["mean"] == pytest.approx(true_mean, abs=0.01)


def test_an_impossible_retention_or_a_missing_role_is_refused():
    """A zero retention prints nothing, and retention without roles has no record to apply to."""
    lower, upper, roles = _retention_setup([True, False])
    variances = np.full(2, 0.04)

    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        censored_loglik(0.4, 0.02, lower, upper, variances, retention=0.0, roles=roles)
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        censored_loglik(0.4, 0.02, lower, upper, variances, retention=1.5, roles=roles)
    with pytest.raises(ValueError, match="requires roles"):
        censored_loglik(0.4, 0.02, lower, upper, variances, retention=0.5)


def test_an_explicit_nonsignificance_is_not_a_retention_event_but_an_absent_row_is():
    """The two absence states differ precisely in whether retention applies to them."""
    roles = retention_roles(
        [
            ObservationState.IMAGE,
            ObservationState.NONSIGNIFICANT,
            ObservationState.ABSENT_COMPLETE_TABLE,
            ObservationState.DIRECTION_ONLY,
            ObservationState.UNKNOWN_COMPLETENESS,
        ]
    )
    assert list(roles) == [0, 0, -1, 1, 0]


def _mixed_corpus(n_images, n_coordinates, thresholds, rng):
    """Build an image cohort and a coordinate cohort with spread sizes and known retention."""
    true_mean, between, retention = 0.4, 0.0225, 0.5
    sizes = np.array([12, 16, 20, 28, 40, 60, 90, 140])

    coordinate_sizes = np.resize(sizes, n_coordinates)
    cuts = np.resize(thresholds, n_coordinates)
    coordinate_variance = 1.0 / coordinate_sizes
    estimates = rng.normal(true_mean, np.sqrt(coordinate_variance + between))
    printed = (estimates > cuts) & (rng.random(n_coordinates) < retention)

    image_sizes = np.resize(sizes, n_images)
    image_variance = 1.0 / image_sizes
    observed = rng.normal(true_mean, np.sqrt(image_variance + between))

    states = [ObservationState.IMAGE] * n_images + [
        ObservationState.DIRECTION_ONLY if flag else ObservationState.ABSENT_COMPLETE_TABLE
        for flag in printed
    ]
    lower, upper = bounds_from_states(
        states,
        values=np.concatenate([observed, np.full(n_coordinates, np.nan)]),
        thresholds=np.concatenate([np.full(n_images, np.nan), cuts]),
        signs=np.concatenate([np.full(n_images, np.nan), np.ones(n_coordinates)]),
        absence_is_voxelwise=True,
    )
    variances = np.concatenate([image_variance, coordinate_variance])
    return lower, upper, variances, retention_roles(states)


def test_estimating_retention_recovers_it_where_images_pin_the_other_two_parameters():
    """Joint estimation must land on the truth, within its own predicted standard error.

    The tolerance is not chosen: the observed-information calculation for this design gives
    se(rho) about 0.19 at eight images and a hundred tables, so 20 replications have a standard
    error on the mean estimate of about 0.04 and three of those is the band used here.
    """
    rng = np.random.default_rng(101)
    spread = np.array([0.35, 0.45, 0.55, 0.6, 0.65, 0.75, 0.85, 0.95])

    retentions, means = [], []
    for _ in range(20):
        lower, upper, variances, roles = _mixed_corpus(8, 100, spread, rng)
        fit = fit_censored(lower, upper, variances, roles=roles, estimate_retention=True)
        assert fit["valid"]
        retentions.append(fit["retention"])
        means.append(fit["mean"])

    assert np.mean(retentions) == pytest.approx(0.5, abs=0.13)
    assert np.mean(means) == pytest.approx(0.4, abs=0.06)


def test_a_corpus_with_one_threshold_and_one_precision_reports_no_identification():
    """Parallel score directions give rank-one information, and that must be visible.

    Proved as "two distinct precisions can carry at most two of the three parameters": the score
    direction depends on a study's threshold and precision alone, so a corpus sharing both
    cannot separate three parameters however many studies it has. With no images to pin the
    other two, the condition number must come back infinite rather than merely large.
    """
    rng = np.random.default_rng(202)
    count = 300
    cuts = np.full(count, 0.6)
    estimates = rng.normal(0.4, np.sqrt(0.04 + 0.0225), size=count)
    printed = (estimates > 0.6) & (rng.random(count) < 0.5)
    states = [
        ObservationState.DIRECTION_ONLY if flag else ObservationState.ABSENT_COMPLETE_TABLE
        for flag in printed
    ]
    lower, upper = bounds_from_states(
        states, thresholds=cuts, signs=np.ones(count), absence_is_voxelwise=True
    )
    fit = fit_censored(
        lower,
        upper,
        np.full(count, 0.04),
        roles=retention_roles(states),
        estimate_retention=True,
    )
    assert not np.isfinite(fit["condition_number"])


def test_estimating_retention_without_roles_is_refused():
    """Refuse a retention fit with no roles: nothing says which records it applies to."""
    lower, upper, roles = _retention_setup([True, False, False])
    with pytest.raises(ValueError, match="requires roles"):
        fit_censored(lower, upper, np.full(3, 0.04), estimate_retention=True)


def test_practical_prevalence_rises_with_heterogeneity_below_the_threshold_and_falls_above_it():
    """Its derivative in the heterogeneity carries the sign of the threshold minus the mean.

    Proved in ``proofs/practical_prevalence.py``. This is the property that makes the quantity
    dangerous to report alone: at a mean that never reaches the threshold, between-study noise
    alone drives it from nothing to two fifths.
    """
    spreads = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    below = practical_prevalence(0.1, spreads**2, 0.3)
    assert np.all(np.diff(below) > 0)
    assert below[0] < 1e-4 and below[-1] > 0.35

    above = practical_prevalence(0.5, spreads**2, 0.3)
    assert np.all(np.diff(above) < 0)

    # At the threshold itself the heterogeneity cannot matter: the answer is a half throughout.
    at_threshold = practical_prevalence(0.3, spreads**2, 0.3)
    assert np.allclose(at_threshold, 0.5)


def test_the_prevalence_product_gap_matches_its_closed_form_and_changes_sign():
    """The gap is tau*phi - (1-pi)*m, which is not signed, and vanishes only in the limit."""
    signs = set()
    for mean in (0.1, 0.3, 0.5):
        for spread in (0.1, 0.2, 0.4):
            for threshold in (0.0, 0.2):
                variance = spread**2
                prevalence = float(practical_prevalence(mean, variance, threshold))
                density = float(norm.pdf((mean - threshold) / spread))
                predicted = spread * density - (1.0 - prevalence) * mean
                assert float(
                    prevalence_times_conditional_mean_gap(mean, variance, threshold)
                ) == pytest.approx(predicted, abs=1e-12)
                signs.add(np.sign(predicted))
    assert signs == {-1.0, 1.0}, "the gap should take both signs over this grid"

    # As the threshold recedes the whole distribution counts as above it and the gap closes.
    receding = [
        abs(float(prevalence_times_conditional_mean_gap(0.3, 0.04, threshold)))
        for threshold in (-0.5, -1.0, -2.0, -4.0)
    ]
    assert receding == sorted(receding, reverse=True)
    assert receding[-1] < 1e-8


def test_a_prevalence_with_no_heterogeneity_is_an_indicator_not_a_division_by_zero():
    """Every study is the mean then, so the share above a threshold is zero or one."""
    assert float(practical_prevalence(0.4, 0.0, 0.2)) == 1.0
    assert float(practical_prevalence(0.1, 0.0, 0.2)) == 0.0
    with pytest.raises(ValueError, match="cannot be negative"):
        practical_prevalence(0.4, -0.01, 0.2)


def test_a_complete_table_is_not_turned_into_a_voxelwise_interval_by_default():
    """The voxelwise reading of an absent peak must be asserted, never assumed.

    A complete table lists every local maximum above the threshold, not every suprathreshold
    voxel: a voxel can clear the cut and simply not be a local maximum. Reading "no peak listed
    here" as "the effect here was below the cut" fabricates an observation, which is exactly the
    inference the module refuses for *unknown* completeness -- and it was making it
    automatically for known completeness. Caught by an external audit.
    """
    with pytest.raises(ValueError, match="absence_is_voxelwise"):
        bounds_from_states(
            [ObservationState.ABSENT_COMPLETE_TABLE],
            thresholds=np.array([0.6]),
            signs=np.array([1.0]),
        )

    # Asserting it explicitly is allowed, and gives the same bounds as before.
    lower, upper = bounds_from_states(
        [ObservationState.ABSENT_COMPLETE_TABLE],
        thresholds=np.array([0.6]),
        signs=np.array([1.0]),
        absence_is_voxelwise=True,
    )
    assert not np.isfinite(lower[0]) and upper[0] == pytest.approx(0.6)

    # NONSIGNIFICANT needs no such assertion: an explicit scalar non-significance really is a
    # censoring interval, which is the whole reason the two states are distinct.
    lower, upper = bounds_from_states(
        [ObservationState.NONSIGNIFICANT], thresholds=np.array([0.6])
    )
    assert (lower[0], upper[0]) == (-0.6, 0.6)


def test_cluster_peak_bounds_the_effect_rather_than_fixing_it():
    """A printed height belongs to the cluster's maximum, so it bounds this location above.

    The interval is ``(c, h]``: the cluster maximum cannot be below the effect at a member
    voxel, and membership of a suprathreshold cluster puts the effect above the cut. Reading the
    same report as ``EXACT`` moves the maximiser from the midpoint to the height itself, which
    is a bias of ``(h - c) / 2`` with a known sign, and on HCP pseudo-studies it is the whole of
    the positive bias at a wide coverage radius.
    """
    lower, upper = bounds_from_states(
        [ObservationState.CLUSTER_PEAK],
        values=np.array([1.9]),
        thresholds=np.array([0.6]),
        signs=np.array([1.0]),
    )
    assert (lower[0], upper[0]) == pytest.approx((0.6, 1.9))

    # The maximiser is the midpoint, from the fitter rather than from the algebra.
    fit = fit_censored(lower, upper, np.array([0.09]), fixed_between_variance=0.0)
    assert fit["mean"] == pytest.approx(1.25, abs=1e-3)

    # A negative report is the mirror image, not an unsigned magnitude.
    lower, upper = bounds_from_states(
        [ObservationState.CLUSTER_PEAK],
        values=np.array([-1.9]),
        thresholds=np.array([0.6]),
        signs=np.array([-1.0]),
    )
    assert (lower[0], upper[0]) == pytest.approx((-1.9, -0.6))


def test_cluster_peak_refuses_a_height_inside_its_own_threshold():
    """A peak below the cut it was selected by is two scales disagreeing, not a value to clip.

    This is the guard that caught a z being divided by the square root of the sample size as
    though it were a t: the converted height fell below the converted threshold, and without the
    refusal it would have been fitted as an ordinary interval and read as estimator bias.
    """
    with pytest.raises(ValueError, match="inside its own threshold"):
        bounds_from_states(
            [ObservationState.CLUSTER_PEAK],
            values=np.array([0.3]),
            thresholds=np.array([0.6]),
            signs=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="sign zero"):
        bounds_from_states(
            [ObservationState.CLUSTER_PEAK],
            values=np.array([1.9]),
            thresholds=np.array([0.6]),
            signs=np.array([0.0]),
        )


def test_no_peak_nearby_is_retention_eligible_where_nonsignificant_is_not():
    """The two states share an interval and differ in what they admit not knowing.

    ``NONSIGNIFICANT`` is a study saying the effect here was small, so its presence is not a
    reporting event. ``NO_PEAK_NEARBY`` is a study saying nothing about this location: a cluster
    it did print may contain the location with its peak beyond the assumed reach. That ambiguity
    is what retention represents, so only the second carries a role.
    """
    states = [ObservationState.NONSIGNIFICANT, ObservationState.NO_PEAK_NEARBY]
    roles = retention_roles(states)
    assert roles.tolist() == [0, -1]

    lower, upper = bounds_from_states(
        states, thresholds=np.array([0.6, 0.6]), signs=np.array([1.0, 1.0])
    )
    assert not np.isfinite(lower).any()
    assert upper == pytest.approx([0.6, 0.6])


def test_the_no_peak_silence_term_is_the_two_branch_mixture():
    """``NO_PEAK_NEARBY`` under retention is exactly ``kappa * S + F`` with ``kappa = 1 - rho``.

    Verified in ``proofs/what_a_silence_says.py`` and checked here numerically, because the
    point of the state is that it needs no new parameter: the retention machinery already
    computes the mixture, and the previous silence term was its ``kappa = 0`` special case. If
    this identity ever fails, the state is no longer the thing the proof describes.
    """
    from scipy.stats import norm

    mean, variance, cut = 0.35, 0.09, 0.6
    states = [ObservationState.NO_PEAK_NEARBY]
    lower, upper = bounds_from_states(states, thresholds=np.array([cut]), signs=np.array([1.0]))
    roles = retention_roles(states)
    below = norm.cdf((cut - mean) / np.sqrt(variance))
    above = 1.0 - below

    for rho in (1.0, 0.9, 0.5, 0.2, 0.05):
        got = censored_loglik(
            mean, 0.0, lower, upper, np.array([variance]), retention=rho, roles=roles
        )
        assert got == pytest.approx(np.log((1.0 - rho) * above + below), abs=1e-12)

    # And the direction the proof gives: a smaller retention means a weaker downward push, so
    # the fitted mean rises monotonically as rho falls.
    fitted = []
    for rho in (1.0, 0.8, 0.6, 0.4):
        records = [ObservationState.IMAGE] + [ObservationState.NO_PEAK_NEARBY] * 8
        bounds = bounds_from_states(
            records,
            values=np.array([0.5] + [0.0] * 8),
            thresholds=np.array([0.0] + [cut] * 8),
            signs=np.ones(9),
        )
        fit = fit_censored(
            *bounds,
            np.full(9, variance),
            retention=rho,
            roles=retention_roles(records),
            fixed_between_variance=0.0,
        )
        fitted.append(fit["mean"])
    assert all(later > earlier for earlier, later in zip(fitted, fitted[1:])), fitted


def test_the_restricted_variance_matches_the_closed_form_it_generalises():
    """In the equal-variance Gaussian case the answer is ``D/(K-1) - s^2``, computed nowhere here.

    That is the whole justification for the numerical integral: the restricted likelihood is
    defined as the likelihood with the mean integrated out under a flat prior, which generalises
    to interval-censored records, and for a Gaussian model that integral must reproduce the
    textbook divisor. Agreement is therefore a check against an independent route rather than a
    restatement of the implementation. Derived in
    ``proofs/small_sample_between_study_variance.py``.
    """
    from nimare.meta.cbma.censored import restricted_between_variance

    rng = np.random.default_rng(0)
    sampling, count = 0.05, 7
    worst = 0.0
    for _ in range(12):
        values = rng.normal(0.0, np.sqrt(sampling + 0.045), size=count)
        lower, upper = bounds_from_states(
            [ObservationState.IMAGE] * count, values=values, thresholds=np.zeros(count)
        )
        variances = np.full(count, sampling)
        got = restricted_between_variance(lower, upper, variances, grid=513)["between_variance"]
        closed_form = max(((values - values.mean()) ** 2).sum() / (count - 1) - sampling, 0.0)
        worst = max(worst, abs(got - closed_form))
    assert worst < 5e-3, worst


def test_the_restricted_variance_exceeds_the_joint_maximiser_at_small_study_counts():
    """The point of it: the joint maximiser is short by one K-th of the *total* variance.

    ``E[tau2_ML] = tau2 - (tau2 + s^2)/K``, so at seven studies the joint fit collapses toward
    zero whenever ``tau2 < s^2/6``. Measured on 21 published pain studies, seven-study fits gave
    a median of .0173 against .0453 from all 21 with 38% at exactly zero. This checks the
    direction on averages rather than on one draw, because either estimator can be the larger on
    a single sample.
    """
    from nimare.meta.cbma.censored import restricted_between_variance

    rng = np.random.default_rng(3)
    sampling, count = 0.05, 7
    joint, restricted = [], []
    for _ in range(60):
        values = rng.normal(0.0, np.sqrt(sampling + 0.045), size=count)
        lower, upper = bounds_from_states(
            [ObservationState.IMAGE] * count, values=values, thresholds=np.zeros(count)
        )
        variances = np.full(count, sampling)
        joint.append(fit_censored(lower, upper, variances)["between_variance"])
        restricted.append(restricted_between_variance(lower, upper, variances)["between_variance"])
    joint, restricted = np.asarray(joint), np.asarray(restricted)
    assert np.median(restricted) > np.median(joint)
    # And it must reduce the pile-up at the boundary, which is the symptom that was measured.
    assert np.mean(restricted <= 1e-9) < np.mean(joint <= 1e-9)


def test_the_restricted_variance_reports_its_boundary_and_refuses_an_empty_record_set():
    """A solution at zero is a fact about the fit and must be visible, not inferred from a value.

    The correction fixes the divisor and not the boundary: for any estimator
    ``E[max(T, 0)] >= E[T]``, so a fifth of seven-study fits still land at zero and a summary
    that quotes these estimates without the boundary share is hiding the part that did not get
    fixed.
    """
    from nimare.meta.cbma.censored import restricted_between_variance

    # Seven identical observations: no dispersion at all, so zero is the right answer and the
    # flag must say so rather than leaving a bare 0.0 to be read as an estimate.
    count = 7
    lower, upper = bounds_from_states(
        [ObservationState.IMAGE] * count,
        values=np.full(count, 0.4),
        thresholds=np.zeros(count),
    )
    out = restricted_between_variance(lower, upper, np.full(count, 0.05))
    assert out["between_variance"] == pytest.approx(0.0)
    assert out["at_zero_boundary"] and out["converged"]

    # Nothing informative: report the failure rather than returning a number.
    uninformative = bounds_from_states([ObservationState.UNKNOWN_COMPLETENESS] * 3)
    empty = restricted_between_variance(*uninformative, np.full(3, 0.05))
    assert not empty["converged"] and np.isnan(empty["between_variance"])


def test_the_restricted_variance_accepts_censored_records_and_retention():
    """The reason it is an integral and not a formula: most records here are intervals.

    A closed-form restricted adjustment is derived for a Gaussian linear model, which a table of
    silences is not. Integrating the mean out is the definition that survives, so the function
    has to accept the mixed record set the estimator actually sees -- and the reporting model has
    to be the same one being fitted, or the variance is estimated under a different model than
    the mean.
    """
    from nimare.meta.cbma.censored import restricted_between_variance

    states = [ObservationState.IMAGE] * 3 + [ObservationState.NO_PEAK_NEARBY] * 6
    lower, upper = bounds_from_states(
        states,
        values=np.array([0.5, 0.2, 0.6] + [0.0] * 6),
        thresholds=np.array([0.0] * 3 + [0.6] * 6),
        signs=np.ones(9),
    )
    variances = np.full(9, 0.05)
    roles = retention_roles(states)
    plain = restricted_between_variance(lower, upper, variances)
    with_retention = restricted_between_variance(
        lower, upper, variances, retention=0.25, roles=roles
    )
    assert plain["converged"] and with_retention["converged"]
    assert np.isfinite(plain["between_variance"])
    assert np.isfinite(with_retention["between_variance"])
    # The two describe different reporting models, so they must not silently coincide.
    assert plain["loglik"] != with_retention["loglik"]
