"""Tests for nimare.meta.cbma.marginal (marginal effect size by control variate).

Organised around the algebra the estimator rests on, verified symbolically in
``proofs/marginal_control_variate.py`` of the companion experiments repository: the correction is
unbiased whatever the predictor does, its variance has a two-cohort form with a known optimum,
and that optimum cannot beat a floor the image sample sets. Each test pins one of those, plus the
two ways the construction fails.
"""

import numpy as np
import pytest

from nimare.meta.cbma.marginal import (
    achievable_ratio,
    control_variate_mean,
    optimal_coefficient,
)


def _cohorts(rng, n_images, n_coordinates, n_voxels, correlation, truth, bias=0.0, shift=0.0):
    """Two cohorts whose predictor correlates ``correlation`` with the per-study effect.

    ``bias`` shifts the predictor in both cohorts, which must cancel. ``shift`` moves the
    coordinate cohort's predictor only, which breaks exchangeability and must not cancel.
    """
    total = n_images + n_coordinates
    u = rng.normal(size=(total, n_voxels))
    w = rng.normal(size=(n_images, n_voxels))
    predictions = bias + u
    predictions[n_images:] += shift
    effects = truth + correlation * u[:n_images] + np.sqrt(1.0 - correlation**2) * w
    return effects, predictions[:n_images], predictions[n_images:]


def test_a_badly_biased_predictor_cancels_between_the_cohorts():
    """The predictor's own bias appears in both cohorts and subtracts out.

    This is the whole reason to correct rather than impute: the predictor is never trusted, only
    differenced, so being wrong by a constant costs nothing.
    """
    rng = np.random.default_rng(0)
    truth = 0.4
    effects, f_image, f_coord = _cohorts(rng, 8, 100, 40, 0.6, truth)

    # A constant added to the predictor moves both cohort means by the same amount, so their
    # difference -- the only way the predictor enters -- does not move at all. The two estimates
    # are therefore identical to floating point, which is a stronger statement than both being
    # near the truth and does not depend on how many replications are averaged.
    plain = control_variate_mean(effects, f_image, f_coord, 0.5)["estimate"]
    shifted = control_variate_mean(effects, f_image + 5.0, f_coord + 5.0, 0.5)["estimate"]
    assert np.allclose(plain, shifted, rtol=0, atol=1e-12)

    # And a scaled predictor moves the estimate only through the coefficient, so rescaling the
    # predictor and dividing lambda by the same factor is also a no-op.
    scaled = control_variate_mean(effects, 3.0 * f_image, 3.0 * f_coord, 0.5 / 3.0)["estimate"]
    assert np.allclose(plain, scaled, rtol=0, atol=1e-12)

    # Unbiasedness itself, over replications, with a tolerance set from the estimator's own
    # standard error rather than guessed.
    estimates = []
    for _ in range(400):
        rep, f_i, f_c = _cohorts(rng, 8, 100, 1, 0.6, truth, bias=5.0)
        estimates.append(float(control_variate_mean(rep, f_i, f_c, 0.5)["estimate"][0]))
    tolerance = 3.0 * np.std(estimates, ddof=1) / np.sqrt(len(estimates))
    assert np.mean(estimates) == pytest.approx(truth, abs=tolerance)


def test_a_cohort_that_is_not_exchangeable_biases_the_estimate():
    """The failure mode, demonstrated rather than only warned about.

    If the coordinate-only studies differ from the image studies in what the predictor says
    about them, the difference enters the estimate multiplied by the coefficient. This is the
    non-random image sharing problem and nothing in the fit detects it.
    """
    rng = np.random.default_rng(1)
    truth, shift, lam = 0.4, 1.0, 0.5
    estimates = []
    for _ in range(400):
        effects, f_image, f_coord = _cohorts(rng, 8, 100, 1, 0.6, truth, shift=shift)
        estimates.append(
            float(control_variate_mean(effects, f_image, f_coord, lam)["estimate"][0])
        )
    # The bias is exactly lambda times the shift, which is what makes it undetectable from the
    # estimate alone and visible in ``cohort_shift``.
    assert np.mean(estimates) == pytest.approx(truth + lam * shift, abs=0.03)


def test_the_reported_standard_error_matches_the_spread_of_the_estimate():
    """The two-cohort variance formula, checked against the sampling distribution it describes."""
    rng = np.random.default_rng(2)
    n_images, n_coordinates, lam = 8, 100, 0.5
    estimates, reported = [], []
    for _ in range(600):
        effects, f_image, f_coord = _cohorts(rng, n_images, n_coordinates, 1, 0.6, 0.4)
        out = control_variate_mean(effects, f_image, f_coord, lam)
        estimates.append(float(out["estimate"][0]))
        reported.append(float(out["se"][0]))
    assert np.mean(reported) == pytest.approx(np.std(estimates, ddof=1), rel=0.12)


def test_the_variance_ratio_respects_the_floor_the_images_impose():
    """No coordinate cohort, however large, beats ``1 - rho**2`` of the image-only variance."""
    correlation = 0.8
    floor = 1.0 - correlation**2
    ratios = [achievable_ratio(correlation, 8, n) for n in (10, 100, 10_000, 10_000_000)]
    assert all(r > floor for r in ratios)
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] == pytest.approx(floor, abs=1e-5)


def test_the_optimal_coefficient_is_the_slope_shrunk_by_the_cohort_ratio():
    """``lambda*`` is a regression slope divided by ``1 + n/N``, not the slope itself.

    The shrinkage is what stops a small coordinate cohort from being trusted: correcting with
    few tables adds their noise, so the correction is damped.
    """
    rng = np.random.default_rng(3)
    n_images, n_voxels = 400, 1
    f = rng.normal(size=(n_images, n_voxels))
    y = 0.4 + 0.75 * f + rng.normal(scale=0.1, size=(n_images, n_voxels))
    for n_coordinates in (100, 10_000):
        lam = float(optimal_coefficient(y, f, n_images, n_coordinates, mode="pooled"))
        assert lam == pytest.approx(0.75 / (1 + n_images / n_coordinates), rel=0.05)


def test_one_image_study_reports_no_precision_rather_than_perfect_precision():
    """With a single image there is no covariance to read, and the output says so.

    Returning zero would read as certainty. The estimate is still produced -- it is the image
    plus a correction at whatever coefficient was supplied -- but nothing about its spread is
    knowable from one study.
    """
    rng = np.random.default_rng(4)
    effects, f_image, f_coord = _cohorts(rng, 1, 50, 6, 0.6, 0.4)
    out = control_variate_mean(effects, f_image, f_coord, 0.5)
    assert np.all(np.isinf(out["se"]))
    assert not out["valid"].any()
    assert np.all(np.isfinite(out["estimate"]))
    with pytest.raises(ValueError, match="at least two image studies"):
        optimal_coefficient(effects, f_image, 1, 50)


def test_a_predictor_that_never_varies_leaves_the_image_mean_untouched():
    """A constant predictor carries no information, so the coefficient and the shift are zero."""
    rng = np.random.default_rng(5)
    values = 0.4 + rng.normal(scale=0.3, size=(6, 8))
    flat = np.ones((6, 8))
    lam = optimal_coefficient(values, flat, 6, 40, mode="voxelwise")
    assert np.all(lam == 0.0)
    out = control_variate_mean(values, flat, np.ones((40, 8)), lam)
    assert out["estimate"] == pytest.approx(values.mean(axis=0))
    assert not out["valid"].any()
