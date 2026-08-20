"""Test nimare.stats."""

import math

import numpy as np
import pytest
from pymare.stats import bonferroni, fdr

from nimare.stats import (
    nlogp_bonferroni,
    nlogp_fdr,
    null_to_p,
    nullhist_to_p,
    one_way,
    two_way,
    two_way_counts,
)


def test_null_to_p_float():
    """Test null_to_p with single float input, assuming asymmetric null dist."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]

    # Two-tailed
    assert math.isclose(null_to_p(0, null, "two"), 0.8)
    assert math.isclose(null_to_p(9, null, "two"), 0.1)
    assert math.isclose(null_to_p(10, null, "two"), 0.05)
    assert math.isclose(null_to_p(-9, null, "two"), 0.3)
    assert math.isclose(null_to_p(-10, null, "two"), 0.1)
    # Still 0.05 because minimum valid p-value is 1 / len(null)
    result = null_to_p(20, null, "two")
    assert result == null_to_p(-20, null, "two")
    assert math.isclose(result, 0.05)

    # Left/lower-tailed
    assert math.isclose(null_to_p(9, null, "lower"), 0.95)
    assert math.isclose(null_to_p(-9, null, "lower"), 0.15)
    assert math.isclose(null_to_p(0, null, "lower"), 0.4)

    # Right/upper-tailed
    assert math.isclose(null_to_p(9, null, "upper"), 0.05)
    assert math.isclose(null_to_p(-9, null, "upper"), 0.95)
    assert math.isclose(null_to_p(0, null, "upper"), 0.65)

    # Test that 1/n(null) is preserved with extreme values
    nulldist = np.random.normal(size=10000)
    assert math.isclose(null_to_p(20, nulldist, "two"), 1 / 10000)
    assert math.isclose(null_to_p(20, nulldist, "lower"), 1 - 1 / 10000)


def test_null_to_p_float_symmetric():
    """Test null_to_p with single float input, assuming symmetric null dist."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]

    # Only need to test two-tailed; symmetry is irrelevant for one-tailed
    assert math.isclose(null_to_p(0, null, "two", symmetric=True), 0.95)
    result = null_to_p(9, null, "two", symmetric=True)
    assert result == null_to_p(-9, null, "two", symmetric=True)
    assert math.isclose(result, 0.2)
    result = null_to_p(10, null, "two", symmetric=True)
    assert result == null_to_p(-10, null, "two", symmetric=True)
    assert math.isclose(result, 0.05)
    # Still 0.05 because minimum valid p-value is 1 / len(null)
    result = null_to_p(20, null, "two", symmetric=True)
    assert result == null_to_p(-20, null, "two", symmetric=True)
    assert math.isclose(result, 0.05)


def test_null_to_p_array():
    """Test nimare.stats.null_to_p with 1d array input."""
    N = 10000
    nulldist = np.random.normal(size=N)
    t = np.sort(np.random.normal(size=N))
    p = np.sort(null_to_p(t, nulldist))
    assert p.shape == (N,)
    assert (p < 1).all()
    assert (p > 0).all()
    # Resulting distribution should be roughly uniform
    assert np.abs(p.mean() - 0.5) < 0.02
    assert np.abs(p.var() - 1 / 12) < 0.02


def test_nullhist_to_p():
    """Test nimare.stats.nullhist_to_p."""
    n_voxels = 5

    # Test cross-voxel null distribution
    histogram_bins = np.arange(0, 101, 1)  # 101 bins
    histogram_weights = np.ones(histogram_bins.shape)
    histogram_weights[-1] = 0  # last bin is outside range, so there are 100 bins with values

    # When input is a single value
    assert math.isclose(nullhist_to_p(0, histogram_weights, histogram_bins), 1.0)
    assert math.isclose(nullhist_to_p(1, histogram_weights, histogram_bins), 0.99)
    assert math.isclose(nullhist_to_p(99, histogram_weights, histogram_bins), 0.01)
    assert math.isclose(nullhist_to_p(100, histogram_weights, histogram_bins), 0.01)

    # When input is an array
    assert np.allclose(
        nullhist_to_p([0, 1, 99, 100, 101], histogram_weights, histogram_bins),
        np.array([1.0, 0.99, 0.01, 0.01, 0.01]),
    )

    # Test voxel-wise null distributions
    histogram_weights = np.ones((histogram_bins.shape[0], n_voxels))
    histogram_weights[-1, :] = 0  # last bin is outside range, so there are 100 bins with values

    assert np.allclose(
        nullhist_to_p([0, 1, 99, 100, 101], histogram_weights, histogram_bins),
        np.array([1.0, 0.99, 0.01, 0.01, 0.01]),
    )


def test_one_way_fastpath_matches_reference_formula():
    """one_way should match the legacy one-sample chi-square formula exactly."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 21, size=500, dtype=np.int16)
    term = data.astype("float64")
    no_term = 21 - term
    t_exp = np.mean(term, 0)
    t_exp = np.array([t_exp] * data.shape[0])
    nt_exp = 21 - t_exp
    expected = ((term - t_exp) ** 2 / t_exp) + ((no_term - nt_exp) ** 2 / nt_exp)

    actual = one_way(data, 21)
    assert np.allclose(actual, expected, equal_nan=True)


def test_two_way_counts_matches_two_way_reference():
    """two_way_counts should reproduce two_way for regular and degenerate tables."""
    rng = np.random.default_rng(1)
    selected = rng.integers(0, 21, size=500, dtype=np.int16)
    unselected = rng.integers(0, 17, size=500, dtype=np.int16)

    # Force several degenerate cases where expected cell counts hit zero.
    selected[:4] = [0, 0, 21, 21]
    unselected[:4] = [0, 17, 0, 17]

    cells = np.squeeze(
        np.array(
            [
                [selected, unselected],
                [21 - selected, 17 - unselected],
            ]
        ).T
    )

    expected = two_way(cells)
    actual = two_way_counts(selected, unselected, 21, 17)
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("method", ["bh", "by"])
def test_nlogp_fdr_matches_the_p_space_procedure(method):
    """The step-up procedure must be the same one, only carried out in logs."""
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-12, 1.0, size=5000)
    # Ties, a p of exactly one, and the float32 storage floor all have to behave.
    p[:5] = [1e-8, 1e-8, 0.5, 1.0, np.finfo(np.float32).tiny]

    corrected = np.exp(nlogp_fdr(np.log(p), method=method))
    expected = fdr(p.copy(), method=method)

    assert np.allclose(corrected, expected, rtol=1e-10)
    assert np.array_equal(np.argsort(corrected), np.argsort(expected))
    assert np.array_equal(corrected <= 0.05, expected <= 0.05), "selection must not move"


def test_nlogp_bonferroni_matches_the_p_space_procedure():
    """Multiplying by the number of tests is adding its logarithm."""
    rng = np.random.default_rng(1)
    p = rng.uniform(1e-12, 1.0, size=1000)

    corrected = np.exp(nlogp_bonferroni(np.log(p)))
    expected = bonferroni(p.copy())

    assert np.allclose(corrected, expected, rtol=1e-10)
    assert np.array_equal(corrected <= 0.05, expected <= 0.05)


@pytest.mark.parametrize("correct", [nlogp_bonferroni, nlogp_fdr])
def test_log_corrections_carry_a_tail_no_p_value_could_hold(correct):
    """A corrected p-value below the smallest double must survive as its logarithm."""
    nlogp = np.zeros(1000)
    nlogp[0] = -3000.0  # p = 10 ** -1303, zero in any float

    corrected = correct(nlogp)

    assert np.isfinite(corrected[0])
    # Multiplying by 1000 tests moves the tail by log(1000), and no further.
    assert np.isclose(corrected[0], -3000.0 + np.log(1000.0))
    assert corrected[1] == 0.0, "a p-value of one stays one"


@pytest.mark.parametrize("correct", [nlogp_bonferroni, nlogp_fdr])
def test_log_corrections_reject_plain_p_values(correct):
    """Passing p-values would adjust every one of them to 1, so say so instead."""
    with pytest.raises(ValueError, match="natural logarithms"):
        correct(np.array([0.01, 0.5]))
