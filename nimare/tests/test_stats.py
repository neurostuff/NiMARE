"""
Test nimare.stats
"""
import os.path as op
import math

import numpy as np
import nibabel as nib

from nimare import stats, utils


def test_null_to_p():
    """
    Test nimare.stats.null_to_p.
    """
    data = np.arange(1, 101)
    assert math.isclose(stats.null_to_p(5, data, 'lower'), 0.05)
    assert math.isclose(stats.null_to_p(5, data, 'upper'), 0.95)
    assert math.isclose(stats.null_to_p(5, data, 'two'), 0.1)
    assert math.isclose(stats.null_to_p(95, data, 'lower'), 0.95)
    assert math.isclose(stats.null_to_p(95, data, 'upper'), 0.05)
    assert math.isclose(stats.null_to_p(95, data, 'two'), 0.1)
