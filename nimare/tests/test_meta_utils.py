"""Tests for nilearn.mass_univariate._utils."""
import numpy as np

from nimare.meta import utils


def test_calculate_tfce():
    """Test calculate_tfce."""
    test_arr4d = np.zeros((10, 10, 10))

    # 10-voxel positive cluster, high intensity
    test_arr4d[:2, :2, :2] = 10
    test_arr4d[0, 2, 0] = 10
    test_arr4d[2, 0, 0] = 10

    # 10-voxel negative cluster, higher intensity
    test_arr4d[3:5, 3:5, 3:5] = -11
    test_arr4d[3, 5, 3] = -11
    test_arr4d[5, 3, 3] = -11

    # One-sided test where positive cluster has the highest TFCE
    true_max_tfce = 5050
    test_tfce_arr2d = utils.calculate_tfce(
        test_arr4d,
        E=1,
        H=1,
        dh='auto',
        two_sided=False,
    )
    assert test_tfce_arr2d.shape == (10, 10, 10)
    assert np.max(np.abs(test_tfce_arr2d)) == true_max_tfce

    # Two-sided test where negative cluster has the highest TFCE
    true_max_tfce = 5555
    test_tfce_arr2d = utils.calculate_tfce(
        test_arr4d,
        E=1,
        H=1,
        dh='auto',
        two_sided=True,
    )
    assert test_tfce_arr2d.shape == (10, 10, 10)
    assert np.max(np.abs(test_tfce_arr2d)) == true_max_tfce

    # One-sided test with preset dh
    true_max_tfce = 550
    test_tfce_arr2d = utils.calculate_tfce(
        test_arr4d,
        E=1,
        H=1,
        dh=1,
        two_sided=False,
    )
    assert test_tfce_arr2d.shape == (10, 10, 10)
    assert np.max(np.abs(test_tfce_arr2d)) == true_max_tfce
