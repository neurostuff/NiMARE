"""Tests for the SDM module."""
import numpy as np

from nimare.meta.cbma import sdm


def test_scale_subject_maps():
    """Run a smoke test for scale_subject_maps."""
    n_voxels = 1000
    sample_size = 10

    study_effect_size_map = np.random.random(n_voxels)
    study_variance_map = np.random.random(n_voxels)
    prelim_subject_maps = np.random.random((sample_size, n_voxels))

    scaled_subject_maps = sdm.scale_subject_maps(
        study_effect_size_map,
        study_variance_map,
        prelim_subject_maps,
    )
    assert np.all(~np.isnan(scaled_subject_maps))
    assert scaled_subject_maps.shape == (sample_size, n_voxels)


def test_simulate_voxel_with_no_neighbors():
    """Run a smoke test for simulate_voxel_with_no_neighbors."""
    n_subjects = 10
    y = sdm.simulate_voxel_with_no_neighbors(n_subjects)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)


def test_simulate_voxel_with_one_neighbor():
    """Run a smoke test for simulate_voxel_with_one_neighbor."""
    n_subjects = 10
    A = np.random.normal(size=n_subjects)
    r_ay = 0.5
    y = sdm.simulate_voxel_with_one_neighbor(A, r_ay)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)


def test_simulate_voxel_with_two_neighbors():
    """Run a smoke test for simulate_voxel_with_two_neighbors."""
    n_subjects = 10
    A = np.random.normal(size=n_subjects)
    B = np.random.normal(size=n_subjects)
    r_ay = 0.5
    r_by = 0.5
    y = sdm.simulate_voxel_with_two_neighbors(A, B, r_ay, r_by)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)


def test_simulate_voxel_with_three_neighbors():
    """Run a smoke test for simulate_voxel_with_three_neighbors."""
    n_subjects = 10
    A = np.random.normal(size=n_subjects)
    B = np.random.normal(size=n_subjects)
    C = np.random.normal(size=n_subjects)
    r_ay = 0.5
    r_by = 0.5
    r_cy = 0.5
    y = sdm.simulate_voxel_with_three_neighbors(A, B, C, r_ay, r_by, r_cy)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)
