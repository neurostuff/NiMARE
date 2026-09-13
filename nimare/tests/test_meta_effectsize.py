"""Test nimare.meta.cbma.effectsize (effect-size coordinate-based meta-analysis)."""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import norm

from nimare.correct import FDRCorrector
from nimare.meta.cbma import _selection as sel
from nimare.meta.cbma.effectsize import (
    _MAX_DISTANCE_MATRIX_BYTES,
    CoordinateEffectSize,
    _cluster_foci,
    _dersimonian_laird,
    _g_variance,
    _hedges_g,
    _solve_batch_with_variance,
    _solve_with_variance,
)
from nimare.results import MetaResult

# ---------------------------------------------------------------------------
# Pure numerics: these must be right before anything built on them can be.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("likelihood", ["censored", "conditional"])
def test_score_matches_finite_difference(two_sided, likelihood):
    """The analytic score must be the derivative of the log-likelihood."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        k = int(rng.integers(2, 12))
        g = rng.normal(0.5, 0.6, k)
        s = rng.uniform(0.05, 0.8, k)
        c = rng.uniform(0.0, 2.0, k)
        reported = rng.random(k) < 0.6
        reported[0] = True
        delta, step = rng.uniform(-3, 3), 1e-5

        analytic = sel.score(delta, g, s, c, reported, two_sided, likelihood)
        numeric = (
            sel.loglikelihood(delta + step, g, s, c, reported, two_sided, likelihood)
            - sel.loglikelihood(delta - step, g, s, c, reported, two_sided, likelihood)
        ) / (2 * step)
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("likelihood", ["censored", "conditional"])
def test_information_matches_finite_difference(two_sided, likelihood):
    """The closed-form information must be minus the derivative of the score."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        k = int(rng.integers(2, 12))
        g = rng.normal(0.5, 0.6, k)
        s = rng.uniform(0.2, 0.8, k)
        c = rng.uniform(0.0, 2.0, k)
        reported = rng.random(k) < 0.6
        reported[0] = True
        delta, step = rng.uniform(-2, 2), 1e-4

        analytic = sel.information(delta, g, s, c, reported, two_sided, likelihood)
        numeric = -(
            sel.score(delta + step, g, s, c, reported, two_sided, likelihood)
            - sel.score(delta - step, g, s, c, reported, two_sided, likelihood)
        ) / (2 * step)
        assert analytic == pytest.approx(numeric, rel=1e-3, abs=1e-3)


def test_log_interval_prob_beats_naive_differencing():
    """Deep in the tails and on narrow intervals, CDF differencing fails and we must not."""
    # Narrow interval: the exact value is (b - a) * phi(midpoint) to many digits.
    # Note `b - a` is not exactly 1e-9 in binary, so take the realized width.
    a, b = -0.5, -0.5 + 1e-9
    exact = np.log(b - a) + norm.logpdf(0.5 * (a + b))
    assert float(sel.log_interval_prob(a, b)) == pytest.approx(exact, abs=1e-12)

    # Far tails: ndtr differencing underflows to zero, we stay finite.
    for a, b in [(-41.0, -40.0), (40.0, 41.0)]:
        assert np.isfinite(sel.log_interval_prob(a, b))
    assert norm.cdf(41.0) - norm.cdf(40.0) == 0.0

    # Reflection symmetry.
    assert float(sel.log_interval_prob(2.0, 5.0)) == pytest.approx(
        float(sel.log_interval_prob(-5.0, -2.0))
    )


def test_log_interval_prob_matches_scipy_in_the_easy_regime():
    """Where CDF differencing is trustworthy, we must agree with it."""
    rng = np.random.default_rng(2)
    lower = rng.uniform(-3, 0, 200)
    upper = lower + rng.uniform(0.1, 3, 200)
    expected = np.log(norm.cdf(upper) - norm.cdf(lower))
    np.testing.assert_allclose(sel.log_interval_prob(lower, upper), expected, rtol=1e-10)


def test_score_is_strictly_decreasing():
    """Concavity in delta is what makes the root unique and bisection safe."""
    rng = np.random.default_rng(3)
    for likelihood in ("censored", "conditional"):
        for _ in range(20):
            k = int(rng.integers(3, 10))
            g = rng.normal(0.4, 0.5, k)
            s = rng.uniform(0.1, 0.6, k)
            c = np.full(k, 0.5)
            reported = rng.random(k) < 0.5
            reported[0] = True
            grid = np.linspace(-4, 4, 200)
            scores = [sel.score(d, g, s, c, reported, True, likelihood) for d in grid]
            assert np.all(np.diff(scores) < 0)


def test_uncensored_fit_reduces_to_inverse_variance_mean():
    """With no threshold and no censored study, this must be textbook meta-analysis."""
    rng = np.random.default_rng(4)
    k = 12
    g = rng.normal(0.5, 0.3, k)
    s = rng.uniform(0.1, 0.4, k)
    c = np.zeros(k)
    reported = np.ones(k, dtype=bool)

    delta, se, converged = sel.solve_delta(g, s, c, reported, two_sided=True)
    weights = 1.0 / s**2
    assert converged
    assert delta == pytest.approx(np.sum(g * weights) / np.sum(weights), abs=1e-9)
    assert se == pytest.approx(1.0 / np.sqrt(np.sum(weights)), rel=1e-9)


def test_all_reported_is_unaffected_by_censoring_but_not_by_truncation():
    """L1 ignores the threshold when nobody is censored; L2 never does."""
    rng = np.random.default_rng(5)
    k = 10
    g = np.abs(rng.normal(1.0, 0.2, k)) + 0.6
    s = np.full(k, 0.3)
    c = np.full(k, 0.6)
    reported = np.ones(k, dtype=bool)

    censored_fit, _, _ = sel.solve_delta(g, s, c, reported, True, "censored")
    conditional_fit, _, _ = sel.solve_delta(g, s, c, reported, True, "conditional")
    assert censored_fit == pytest.approx(np.mean(g), abs=1e-8)
    assert conditional_fit < censored_fit


def test_a_lenient_threshold_makes_silence_stronger_evidence_of_absence():
    """Holding the reported set fixed, a lower threshold must pull the estimate down.

    A study that stayed silent despite a permissive reporting threshold is
    telling us a lot about how small its effect was; one that stayed silent
    under a very strict threshold is telling us almost nothing. So the estimate
    must rise monotonically with the threshold, and this is precisely why
    ``threshold="infer"`` -- which can only over-estimate a study's true
    threshold -- biases the result upward.
    """
    rng = np.random.default_rng(6)
    k = 20
    g = rng.normal(0.8, 0.3, k)
    s = np.full(k, 0.25)
    reported = np.zeros(k, dtype=bool)
    reported[:8] = True

    estimates = [
        sel.solve_delta(g, s, np.full(k, c), reported, True, "censored")[0]
        for c in np.linspace(0.05, 1.5, 25)
    ]
    assert np.all(np.diff(estimates) > 0)
    # At a very lax threshold the censored studies stop contributing, so the fit
    # converges on the reported studies alone.
    lax, _, _ = sel.solve_delta(g, s, np.full(k, 50.0), reported, True, "censored")
    assert lax == pytest.approx(g[reported].mean(), abs=1e-3)


def test_no_reported_study_yields_nan():
    """An all-censored region has no estimable effect; say so rather than return zero."""
    k = 5
    delta, se, converged = sel.solve_delta(
        np.zeros(k), np.full(k, 0.3), np.full(k, 0.5), np.zeros(k, dtype=bool)
    )
    assert np.isnan(delta) and not converged and np.isinf(se)


# ---------------------------------------------------------------------------
# The simulation that justifies the whole approach.
# ---------------------------------------------------------------------------


def _simulate_censored_studies(seed, delta_true=0.5, n_studies=60, z_threshold=3.1):
    """Draw studies from a known effect size and censor those below threshold."""
    rng = np.random.default_rng(seed)
    sample_size = rng.integers(15, 61, n_studies).astype(float)
    g = rng.normal(delta_true, np.sqrt(_g_variance(delta_true, sample_size)))
    threshold = np.abs(_hedges_g(np.full(n_studies, z_threshold), sample_size, "z_stat"))
    return g, sample_size, threshold, g > threshold


def test_censored_likelihood_recovers_a_known_effect_size():
    """The premise of the estimator: selection-corrected fitting is unbiased.

    The two comparators bracket it in the expected directions -- keeping only
    the reported studies is inflated, and filling non-reporters with zeros (the
    ES-SDM convention) is attenuated.
    """
    delta_true = 0.5
    censored, observed_only, zero_filled = [], [], []
    for seed in range(150):
        g, n, c, reported = _simulate_censored_studies(seed, delta_true)
        if reported.sum() < 3:
            continue

        fitted, _, converged = _solve_with_variance(g, n, c, reported, 0.0, False, "censored")
        assert converged
        censored.append(fitted)

        var = _g_variance(g[reported].mean(), n)[reported]
        observed_only.append(np.sum(g[reported] / var) / np.sum(1.0 / var))

        filled = np.where(reported, g, 0.0)
        var_all = _g_variance(filled.mean(), n)
        zero_filled.append(np.sum(filled / var_all) / np.sum(1.0 / var_all))

    assert np.mean(censored) == pytest.approx(delta_true, abs=0.02)
    assert np.mean(observed_only) > delta_true + 0.05
    assert np.mean(zero_filled) < delta_true - 0.05


def test_dersimonian_laird_recovers_zero_and_positive_heterogeneity():
    """Sanity-check the moment estimator used for ``tau2='dl'``."""
    rng = np.random.default_rng(8)
    var = np.full(30, 0.04)
    homogeneous = [_dersimonian_laird(rng.normal(0.5, 0.2, 30), var) for _ in range(50)]
    heterogeneous = [
        _dersimonian_laird(rng.normal(0.5, np.sqrt(0.04 + 0.25), 30), var) for _ in range(50)
    ]
    assert np.median(homogeneous) < 0.02
    assert np.median(heterogeneous) == pytest.approx(0.25, rel=0.5)


# ---------------------------------------------------------------------------
# Estimator behaviour.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cbma_with_statistics(testdata_cbma_full):
    """Attach plausible peak Z statistics to the coordinate test dataset."""
    dset = testdata_cbma_full.copy()
    rng = np.random.default_rng(0)
    dset.coordinates = dset.coordinates.copy()
    dset.coordinates["z_stat"] = rng.uniform(3.1, 7.0, len(dset.coordinates))
    return dset


def test_fit_returns_expected_maps_and_tables(cbma_with_statistics):
    """A fit produces voxel maps, a cluster table and an auditable contribution table."""
    result = CoordinateEffectSize(radius=12.0, min_studies=3, tau2="fixed").fit(
        cbma_with_statistics
    )
    assert isinstance(result, MetaResult)
    assert set(result.maps) >= {"est", "se", "z", "p", "logp", "tau2", "label"}

    n_voxels = result.maps["est"].size
    assert all(value.size == n_voxels for value in result.maps.values())
    assert np.isfinite(result.maps["est"]).any()

    clusters = result.tables["clusters"]
    assert isinstance(clusters, pd.DataFrame) and not clusters.empty
    assert {"cluster_id", "x", "y", "z", "g", "se", "z_stat", "p", "p_corr_fdr"} <= set(
        clusters.columns
    )
    # Every cluster must clear the minimum, and the censoring pattern must add up.
    assert (clusters["n_reported"] >= 3).all()

    contributions = result.tables["study_contributions"]
    assert set(contributions["status"]) <= {"reported", "censored", "buffer", "missing"}

    # Every cluster must account for every study on the roster -- that roster is
    # what the censoring denominator is, so a gap here would silently bias the
    # estimate. Note the roster is the studies carrying coordinates, not every
    # study in the collection: NiMARE cannot yet represent a study that reported
    # no foci at all.
    roster_size = cbma_with_statistics.coordinates["id"].nunique()
    assert roster_size < len(cbma_with_statistics.ids)  # guards the caveat above
    per_cluster = contributions.groupby("cluster_id")["status"].value_counts().unstack()
    assert (per_cluster.sum(axis=1) == roster_size).all()


def test_censoring_pulls_the_estimate_below_the_naive_one(cbma_with_statistics):
    """The correction must bite: g < g_naive wherever any study is censored."""
    result = CoordinateEffectSize(radius=12.0, min_studies=3, tau2="fixed").fit(
        cbma_with_statistics
    )
    clusters = result.tables["clusters"]
    censored_present = clusters[clusters["n_censored"] > 0]
    assert not censored_present.empty
    assert (censored_present["g"] < censored_present["g_naive"]).all()


def test_buffer_excludes_rather_than_censors_nearby_studies(cbma_with_statistics):
    """Studies with a focus just outside the region must not be called censored."""
    estimator = CoordinateEffectSize(radius=10.0, buffer=30.0, min_studies=3, tau2="fixed")
    wide = estimator.fit(cbma_with_statistics).tables["study_contributions"]
    narrow = (
        CoordinateEffectSize(radius=10.0, buffer=10.0, min_studies=3, tau2="fixed")
        .fit(cbma_with_statistics)
        .tables["study_contributions"]
    )
    assert (wide["status"] == "buffer").sum() > (narrow["status"] == "buffer").sum()
    assert (narrow["status"] == "buffer").sum() == 0


def test_fdr_correction_runs_over_clusters(cbma_with_statistics):
    """FDR must treat clusters, not voxels, as the tests."""
    result = CoordinateEffectSize(radius=12.0, min_studies=3, tau2="fixed").fit(
        cbma_with_statistics
    )
    corrected = FDRCorrector(method="cluster").transform(result)
    key = "p_corr-FDR_method-cluster"
    assert key in corrected.maps

    # Voxels in one cluster must all carry the same corrected value.
    labels, values = corrected.maps["label"], corrected.maps[key]
    for label in np.unique(labels[np.isfinite(labels)]):
        cluster_values = values[labels == label]
        assert np.nanstd(cluster_values) == pytest.approx(0.0, abs=1e-12)


def test_missing_statistic_column_raises_a_useful_error(testdata_cbma_full):
    """Sleuth-style datasets have no peak statistics; say so plainly."""
    with pytest.raises(ValueError, match="neither a 't_stat' nor a 'z_stat'"):
        CoordinateEffectSize().fit(testdata_cbma_full)


def test_small_samples_are_dropped_with_a_warning(cbma_with_statistics):
    """Hedges' g has no defined variance below N = 4."""
    dset = cbma_with_statistics.copy()
    dset.metadata = dset.metadata.copy()
    dset.metadata["sample_sizes"] = [[3]] * len(dset.metadata)
    with pytest.raises(ValueError, match="usable sample size"):
        with pytest.warns(UserWarning, match="below 4"):
            CoordinateEffectSize().fit(dset)


def test_min_studies_can_exclude_everything(cbma_with_statistics):
    """An impossible min_studies warns and returns empty results rather than crashing."""
    with pytest.warns(UserWarning, match="min_studies"):
        result = CoordinateEffectSize(min_studies=10_000).fit(cbma_with_statistics)
    assert result.tables["clusters"].empty
    assert not np.isfinite(result.maps["est"]).any()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"radius": 0}, "radius must be positive"),
        ({"radius": 10, "buffer": 5}, "must be at least radius"),
        ({"statistic": "f_stat"}, "statistic must be"),
        ({"tail": "three"}, "tail must be"),
        ({"likelihood": "bogus"}, "likelihood must be"),
        ({"tau2": "reml"}, "tau2 must be"),
        ({"tau2_scope": "voxel"}, "tau2_scope must be"),
        ({"within_study": "median"}, "within_study must be"),
        ({"assumed_fwhm": 0}, "assumed_fwhm must be positive"),
        ({"mask_coverage": "wm"}, "mask_coverage must be"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, message):
    """Bad parameters fail at construction, not deep inside a fit."""
    with pytest.raises(ValueError, match=message):
        CoordinateEffectSize(**kwargs)


def test_tail_is_inferred_from_the_sign_of_reported_statistics(cbma_with_statistics):
    """Positive-only data must be treated as one-sided."""
    estimator = CoordinateEffectSize(radius=12.0, min_studies=3, tau2="fixed")
    estimator.fit(cbma_with_statistics)
    assert estimator.two_sided_ is False

    signed = cbma_with_statistics.copy()
    signed.coordinates = signed.coordinates.copy()
    signed.coordinates.loc[signed.coordinates.index[::2], "z_stat"] *= -1
    two_sided_estimator = CoordinateEffectSize(radius=12.0, min_studies=3, tau2="fixed")
    two_sided_estimator.fit(signed)
    assert two_sided_estimator.two_sided_ is True


def test_montecarlo_null_produces_a_cluster_level_p_value(cbma_with_statistics):
    """The false-cluster-discovery null runs and yields usable p-values."""
    result = CoordinateEffectSize(radius=12.0, min_studies=3, tau2="fixed", n_iters=10).fit(
        cbma_with_statistics
    )
    clusters = result.tables["clusters"]
    assert "p_desc-mc" in clusters.columns
    finite = clusters["p_desc-mc"].dropna()
    assert not finite.empty
    assert ((finite >= 0) & (finite <= 1)).all()


# ---------------------------------------------------------------------------
# Performance rework: the optimized paths must agree with the readable ones.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("likelihood", ["censored", "conditional"])
def test_batch_solver_matches_single_cluster_solver(two_sided, likelihood):
    """The vectorized solver must reproduce the single-cluster one exactly.

    Fits are solved for every cluster at once for speed; that optimization is
    only safe if it is indistinguishable from solving them one at a time,
    including the divergence and no-data sentinels.
    """
    rng = np.random.default_rng(11)
    n_fits, width = 80, 30
    g = rng.normal(0.7, 0.5, (n_fits, width))
    s = rng.uniform(0.1, 0.5, (n_fits, width))
    c = rng.uniform(0.2, 1.0, (n_fits, width))
    reported = np.abs(g) > c if two_sided else g > c
    include = rng.random((n_fits, width)) > 0.2
    for row in range(n_fits):
        picks = rng.choice(width, 4, replace=False)
        reported[row, picks] = True
        include[row, picks] = True

    batch_delta, batch_se, batch_ok = sel.solve_delta_batch(
        g, s, c, reported, include, two_sided, likelihood, tol=1e-10
    )
    for row in range(n_fits):
        mask = include[row]
        delta, se, converged = sel.solve_delta(
            g[row][mask], s[row][mask], c[row][mask], reported[row][mask], two_sided, likelihood
        )
        assert batch_ok[row] == converged
        assert batch_delta[row] == pytest.approx(delta, abs=1e-8)
        if np.isfinite(se):
            assert batch_se[row] == pytest.approx(se, abs=1e-8)
        else:
            assert not np.isfinite(batch_se[row])


@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("likelihood", ["censored", "conditional"])
def test_batch_loglikelihood_matches_single(two_sided, likelihood):
    """The batched log-likelihood must match the per-fit one."""
    rng = np.random.default_rng(12)
    n_fits, width = 40, 16
    g = rng.normal(0.6, 0.4, (n_fits, width))
    s = rng.uniform(0.15, 0.5, (n_fits, width))
    c = rng.uniform(0.2, 0.9, (n_fits, width))
    reported = np.abs(g) > c if two_sided else g > c
    include = rng.random((n_fits, width)) > 0.2
    for row in range(n_fits):
        picks = rng.choice(width, 3, replace=False)
        reported[row, picks] = True
        include[row, picks] = True
    delta = rng.normal(0.5, 0.2, n_fits)

    batched = sel.batch_loglikelihood(delta, g, s, c, reported, include, two_sided, likelihood)
    for row in range(n_fits):
        mask = include[row]
        expected = sel.loglikelihood(
            delta[row],
            g[row][mask],
            s[row][mask],
            c[row][mask],
            reported[row][mask],
            two_sided,
            likelihood,
        )
        assert batched[row] == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_variance_fixed_point_is_accurate():
    """Aitken extrapolation must beat plain iteration, not merely run faster.

    The model-based variance depends on delta, and that fixed point converges
    only linearly -- about one decimal digit per pass -- so iterating it to
    convergence is expensive. Extrapolation has to earn its place by landing
    closer to the true optimum than the passes it replaces.
    """
    rng = np.random.default_rng(13)
    n_fits, width = 30, 25
    sample_size = rng.integers(15, 80, (n_fits, width)).astype(float)
    g = rng.normal(0.6, 0.3, (n_fits, width))
    packed = {
        "g": g,
        "sample_size": sample_size,
        "thresh": np.abs(g) * 0.4,
        "reported": np.ones((n_fits, width), dtype=bool),
        "include": np.ones((n_fits, width), dtype=bool),
    }
    packed["reported"][:, 10:] = False

    converged, _, _ = _solve_batch_with_variance(packed, 0.0, False, "censored", n_iter=40)
    accelerated, _, _ = _solve_batch_with_variance(packed, 0.0, False, "censored")
    assert np.nanmax(np.abs(accelerated - converged)) < 1e-6


def test_clustering_uses_one_algorithm_at_every_size():
    """Clustering must not change algorithm with dataset size.

    An earlier version switched to a connectivity-constrained agglomeration
    above 2,500 foci. That is a different algorithm, not an approximation --
    it returned less than half as many clusters, an adjusted Rand index of
    0.08 against average linkage -- so crossing the threshold silently changed
    the meta-analysis result. This pins the behaviour on both sides of where
    that switch used to be.
    """
    rng = np.random.default_rng(14)
    for n_foci in (2_400, 2_600):
        xyz = np.vstack(
            [
                rng.uniform(-60, 60, (n_foci // 2, 3)),
                rng.uniform(-60, 60, (10, 3))[rng.integers(0, 10, n_foci - n_foci // 2)]
                + rng.normal(0, 6, (n_foci - n_foci // 2, 3)),
            ]
        )
        expected = fcluster(linkage(pdist(xyz), method="average"), t=10.0, criterion="distance")
        np.testing.assert_array_equal(_cluster_foci(xyz, 10.0), expected)


def test_clustering_refuses_an_oversized_problem():
    """Past the memory budget, fail up front instead of dying mid-fit."""
    too_many = int(np.sqrt(2 * _MAX_DISTANCE_MATRIX_BYTES / 8)) + 5_000
    with pytest.raises(ValueError, match="distance matrix"):
        # Only the shape is inspected before the guard fires, so this allocates
        # a view rather than the full coordinate array.
        _cluster_foci(np.broadcast_to(np.zeros(3), (too_many, 3)), 10.0)


def test_cluster_assembly_status_counts_are_exhaustive(cbma_with_statistics):
    """Vectorized assembly must still account for every study in every cluster."""
    estimator = CoordinateEffectSize(radius=12.0, min_studies=3, tau2="fixed")
    result = estimator.fit(cbma_with_statistics)
    contributions = result.tables["study_contributions"]
    roster = cbma_with_statistics.coordinates["id"].nunique()
    per_cluster = contributions.groupby("cluster_id")["status"].value_counts().unstack()
    assert (per_cluster.sum(axis=1) == roster).all()
    # A study is reported in a cluster only if it actually has a focus in range.
    reported = contributions[contributions["status"] == "reported"]
    assert reported["g"].notna().all()
    assert contributions[contributions["status"] != "reported"]["g"].isna().all()
