"""Tests for nimare.correct module."""

from nimare.correct import FWECorrector


def test_FWECorrector_montecarlo_default_parameters():
    """FWECorrector(montecarlo) should not override estimator defaults with None."""
    corr = FWECorrector(method="montecarlo")

    # If n_iters was not provided, it should not appear in parameters,
    # allowing the estimator's default to be used.
    assert "n_iters" not in corr.parameters
    # n_cores should still be passed through with its default.
    assert corr.parameters["n_cores"] == 1


def test_FWECorrector_montecarlo_custom_parameters():
    """FWECorrector(montecarlo) should propagate explicit n_iters and n_cores."""
    corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=2)

    assert corr.parameters["n_iters"] == 10
    assert corr.parameters["n_cores"] == 2
