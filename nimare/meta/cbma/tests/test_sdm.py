"""Tests for the SDM module."""
import numpy as np

from nimare.meta.cbma import sdm


def test__scale_subject_maps():
    """Run a smoke test for _scale_subject_maps."""
    n_studies = 3
    n_voxels = 1000
    sample_sizes = [10, 15, 10]

    studylevel_effect_size_maps = np.random.random((n_studies, n_voxels))
    studylevel_variance_maps = np.random.random((n_studies, n_voxels))
    prelim_subjectlevel_maps = [np.random.random((ss, n_voxels)) for ss in sample_sizes]

    scaled_subjectlevel_maps = sdm._scale_subject_maps(
        studylevel_effect_size_maps,
        studylevel_variance_maps,
        prelim_subjectlevel_maps,
    )
    assert np.all([np.all(~np.isnan(ssm)) for ssm in scaled_subjectlevel_maps])
    assert all(
        ssm.shape == (sample_sizes[i], n_voxels) for i, ssm in enumerate(scaled_subjectlevel_maps)
    )


def test__simulate_voxel_with_no_neighbors():
    """Run a smoke test for _simulate_voxel_with_no_neighbors."""
    n_subjects = 10
    y = sdm._simulate_voxel_with_no_neighbors(n_subjects)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)


def test__simulate_voxel_with_one_neighbor():
    """Run a smoke test for _simulate_voxel_with_one_neighbor."""
    n_subjects = 10
    A = np.random.normal(size=n_subjects)
    r_ay = 0.5
    y = sdm._simulate_voxel_with_one_neighbor(A, r_ay)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)


def test__simulate_voxel_with_two_neighbors():
    """Run a smoke test for _simulate_voxel_with_two_neighbors."""
    n_subjects = 10
    A = np.random.normal(size=n_subjects)
    B = np.random.normal(size=n_subjects)
    r_ay = 0.5
    r_by = 0.5
    y = sdm._simulate_voxel_with_two_neighbors(A, B, r_ay, r_by)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)


def test__simulate_voxel_with_three_neighbors():
    """Run a smoke test for _simulate_voxel_with_three_neighbors."""
    n_subjects = 10
    A = np.random.normal(size=n_subjects)
    B = np.random.normal(size=n_subjects)
    C = np.random.normal(size=n_subjects)
    r_ay = 0.5
    r_by = 0.5
    r_cy = 0.5
    y = sdm._simulate_voxel_with_three_neighbors(A, B, C, r_ay, r_by, r_cy)
    assert all(~np.isnan(y))
    assert y.shape == (n_subjects,)
