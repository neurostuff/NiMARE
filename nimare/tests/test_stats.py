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
    assert math.isclose(stats.null_to_p(0, data, "lower"), np.finfo(float).eps)
    assert math.isclose(stats.null_to_p(0, data, "upper"), 1.0 - np.finfo(float).eps)
    assert math.isclose(stats.null_to_p(0, data, "two"), np.finfo(float).eps)
    assert math.isclose(stats.null_to_p(5, data, "lower"), 0.05)
    assert math.isclose(stats.null_to_p(5, data, "upper"), 0.95)
    assert math.isclose(stats.null_to_p(5, data, "two"), 0.1)
    assert math.isclose(stats.null_to_p(95, data, "lower"), 0.95)
    assert math.isclose(stats.null_to_p(95, data, "upper"), 0.05)
    assert math.isclose(stats.null_to_p(95, data, "two"), 0.1)
    assert math.isclose(stats.null_to_p(101, data, "lower"), 1.0 - np.finfo(float).eps)
    assert math.isclose(stats.null_to_p(101, data, "upper"), np.finfo(float).eps)
    assert math.isclose(stats.null_to_p(101, data, "two"), np.finfo(float).eps)
