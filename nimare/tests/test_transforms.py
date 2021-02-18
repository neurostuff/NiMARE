"""
Test nimare.transforms
"""
import numpy as np
import nibabel as nib

from nimare import transforms, utils


def test_transform_images(testdata_ibma):
    """Smoke test on transforms.transform_images"""
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
    """Unit tests for transforms.sample_sizes_to_dof"""
    sample_sizes = [20, 20, 20]
    dof = 57
    assert transforms.sample_sizes_to_dof(sample_sizes) == dof
    sample_sizes = [20]
    dof = 19
    assert transforms.sample_sizes_to_dof(sample_sizes) == dof


def test_sample_sizes_to_sample_size():
    """Unit tests for transforms.sample_sizes_to_sample_size"""
    sample_sizes = [20, 20, 20]
    sample_size = 60
    assert transforms.sample_sizes_to_sample_size(sample_sizes) == sample_size
    sample_sizes = [20]
    sample_size = 20
    assert transforms.sample_sizes_to_sample_size(sample_sizes) == sample_size


def test_t_to_z():
    """Smoke test"""
    t_arr = np.random.random(100)
    z_arr = transforms.t_to_z(t_arr, dof=20)
    assert z_arr.shape == t_arr.shape
    t_arr2 = transforms.z_to_t(z_arr, dof=20)
    assert np.allclose(t_arr, t_arr2)


def test_tal2mni():
    """
    TODO: Get converted coords from official site.
    """
    test = np.array([[-44, 31, 27], [20, -32, 14], [28, -76, 28]])
    true = np.array(
        [
            [-45.83997568, 35.97904559, 23.55194326],
            [22.69248975, -31.34145016, 13.91284087],
            [31.53113226, -76.61685748, 33.22105166],
        ]
    )
    assert np.allclose(transforms.tal2mni(test), true)


def test_mni2tal():
    """
    TODO: Get converted coords from official site.
    """
    test = np.array([[-44, 31, 27], [20, -32, 14], [28, -76, 28]])
    true = np.array(
        [[-42.3176, 26.0594, 29.7364], [17.4781, -32.6076, 14.0009], [24.7353, -75.0184, 23.3283]]
    )
    assert np.allclose(transforms.mni2tal(test), true)


def test_vox2mm():
    """
    Test vox2mm
    """
    test = np.array([[20, 20, 20], [0, 0, 0]])
    true = np.array([[50.0, -86.0, -32.0], [90.0, -126.0, -72.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(transforms.vox2mm(test, aff), true)


def test_mm2vox():
    """
    Test mm2vox
    """
    test = np.array([[20, 20, 20], [0, 0, 0]])
    true = np.array([[35.0, 73.0, 46.0], [45.0, 63.0, 36.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(transforms.mm2vox(test, aff), true)
