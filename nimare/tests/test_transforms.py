"""Test nimare.transforms."""
import copy

import nibabel as nib
import numpy as np
import pytest

from nimare import transforms, utils


def test_transform_images(testdata_ibma):
    """Smoke test on transforms.transform_images."""
    dset = testdata_ibma
    z_files = dset.images["z"].tolist()
    new_images = transforms.transform_images(
        dset.images, target="z", masker=dset.masker, metadata_df=dset.metadata
    )
    new_z_files = new_images["z"].tolist()
    assert z_files[:-1] == new_z_files[:-1]
    # new z statistic map should have 3 dimensions
    assert len(nib.load(new_z_files[-1]).shape) == 3
    assert all([nzf is not None for nzf in new_z_files])

    varcope_files = dset.images["varcope"].tolist()
    new_images = transforms.transform_images(
        dset.images, target="varcope", masker=dset.masker, metadata_df=dset.metadata
    )
    new_varcope_files = new_images["varcope"].tolist()
    assert not all([isinstance(vf, str) for vf in varcope_files])
    assert all([isinstance(vf, str) for vf in new_varcope_files])


def test_sample_sizes_to_dof():
    """Unit tests for transforms.sample_sizes_to_dof."""
    sample_sizes = [20, 20, 20]
    dof = 57
    assert transforms.sample_sizes_to_dof(sample_sizes) == dof
    sample_sizes = [20]
    dof = 19
    assert transforms.sample_sizes_to_dof(sample_sizes) == dof


def test_sample_sizes_to_sample_size():
    """Unit tests for transforms.sample_sizes_to_sample_size."""
    sample_sizes = [20, 20, 20]
    sample_size = 60
    assert transforms.sample_sizes_to_sample_size(sample_sizes) == sample_size
    sample_sizes = [20]
    sample_size = 20
    assert transforms.sample_sizes_to_sample_size(sample_sizes) == sample_size


def test_t_to_z():
    """Smoke test for transforms.t_to_z."""
    t_arr = np.random.random(100)
    z_arr = transforms.t_to_z(t_arr, dof=20)
    assert z_arr.shape == t_arr.shape
    t_arr2 = transforms.z_to_t(z_arr, dof=20)
    assert np.allclose(t_arr, t_arr2)


@pytest.mark.parametrize(
    "kwargs,drop_data,add_data",
    [
        ({"overwrite": True, "z_threshold": 2.3}, "z", "p"),
        ({"overwrite": True, "z_threshold": 3.1}, None, None),
        ({"overwrite": False}, None, None),
    ],
)
def test_images_to_coordinates(tmp_path, testdata_ibma, kwargs, drop_data, add_data):
    img2coord = transforms.CoordinateGenerator(**kwargs)

    if add_data:
        tst_dset = copy.deepcopy(testdata_ibma)
        tst_dset.images = transforms.transform_images(
            tst_dset.images,
            add_data,
            tst_dset.masker,
            tst_dset.metadata,
            tmp_path,
        )
    else:
        tst_dset = testdata_ibma

    if drop_data:
        tst_dset.images = tst_dset.images.drop(columns=drop_data)

    new_dset = img2coord.transform(tst_dset)

    # since testdata_ibma already has coordinate data for every study
    # this transformation should retain the same number of unique ids.
    assert set(new_dset.coordinates["id"]) == set(tst_dset.coordinates["id"])


@pytest.mark.parametrize(
    "z,tail,expected_p",
    [
        (0.0, "two", 1.0),
        (0.0, "one", 0.5),
        (1.959963, "two", 0.05),
        (1.959963, "one", 0.025),
        (-1.959963, "two", 0.05),
        ([0.0, 1.959963, -1.959963], "two", [1.0, 0.05, 0.05]),
    ],
)
def test_z_to_p(z, tail, expected_p):
    p = transforms.z_to_p(z, tail)

    assert np.all(np.isclose(p, expected_p))
