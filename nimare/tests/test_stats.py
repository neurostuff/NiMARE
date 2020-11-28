"""
Test nimare.stats
"""
import math

import numpy as np

from nimare import stats


def test_null_to_p():
    """
    Test nimare.stats.null_to_p.
    """
    data = np.arange(1, 101)
    assert math.isclose(stats.null_to_p(0, data, "lower"), 0.01)
    assert math.isclose(stats.null_to_p(0, data, "upper"), 0.99)
    assert math.isclose(stats.null_to_p(0, data, "two"), 0.01)
    assert math.isclose(stats.null_to_p(5.1, data, "lower"), 0.05)
    assert math.isclose(stats.null_to_p(5.1, data, "upper"), 0.95)
    assert math.isclose(stats.null_to_p(5.1, data, "two"), 0.1)
    assert math.isclose(stats.null_to_p(95.1, data, "lower"), 0.95)
    assert math.isclose(stats.null_to_p(95.1, data, "upper"), 0.05)
    assert math.isclose(stats.null_to_p(95.1, data, "two"), 0.1)
    assert math.isclose(stats.null_to_p(101.1, data, "lower"), 0.99)
    assert math.isclose(stats.null_to_p(101.1, data, "upper"), 0.01)
    assert math.isclose(stats.null_to_p(101.1, data, "two"), 0.01)

    # modify data to handle edge case
    data[98] = 100
    assert math.isclose(stats.null_to_p(1, data, "lower"), 0.01)
    assert math.isclose(stats.null_to_p(1, data, "upper"), 0.99)
    assert math.isclose(stats.null_to_p(100.1, data, "lower"), 0.99)
    assert math.isclose(stats.null_to_p(100.1, data, "upper"), 0.01)
    assert math.isclose(stats.null_to_p(100.1, data, "two"), 0.01)


def test_nullhist_to_p():
    """
    Test nimare.stats.nullhist_to_p.
    """
    n_voxels = 5
    histogram_bins = np.arange(0, 101, 1)
    histogram_weights = np.ones(histogram_bins.shape)
    histogram_weights[-1] = 0

    assert math.isclose(
        stats.nullhist_to_p(0, histogram_weights, histogram_bins),
        1 - np.finfo(float).eps
    )
    assert math.isclose(
        stats.nullhist_to_p(1, histogram_weights, histogram_bins),
        0.99
    )
    assert math.isclose(
        stats.nullhist_to_p(99, histogram_weights, histogram_bins),
        0.01
    )
    assert math.isclose(
        stats.nullhist_to_p(100, histogram_weights, histogram_bins),
        np.finfo(float).eps
    )
    histogram_weights = np.ones((histogram_bins.shape[0], n_voxels))
    histogram_weights[-1, :] = 0
    assert np.allclose(
        stats.nullhist_to_p([0, 0, 0, 0, 0], histogram_weights, histogram_bins),
        np.ones(n_voxels) * (1 - np.finfo(float).eps),
    )
    assert np.allclose(
        stats.nullhist_to_p([1, 1, 1, 1, 1], histogram_weights, histogram_bins),
        np.ones(n_voxels) * 0.99
    )
    assert np.allclose(
        stats.nullhist_to_p([99, 99, 99, 99, 99], histogram_weights, histogram_bins),
        np.ones(n_voxels) * 0.01
    )
    assert np.allclose(
        stats.nullhist_to_p([100, 100, 100, 100, 100], histogram_weights, histogram_bins),
        np.ones(n_voxels) * np.finfo(float).eps
    )
