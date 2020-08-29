"""
Test nimare.meta.kernel (CBMA kernel estimators).
"""
import shutil
import tempfile

import numpy as np
from scipy.ndimage.measurements import center_of_mass

from nimare.dataset import Dataset
from nimare.meta import kernel


def test_alekernel_smoke(testdata_cbma):
    """
    Smoke test for nimare.meta.kernel.ALEKernel
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    coordinates = testdata_cbma.coordinates.copy()
    coordinates["sample_size"] = 20

    kern = kernel.ALEKernel()
    ma_maps = kern.transform(coordinates, testdata_cbma.masker, return_type="image")
    assert len(ma_maps) == len(testdata_cbma.ids)
    ma_maps = kern.transform(coordinates, testdata_cbma.masker, return_type="array")
    assert ma_maps.shape[0] == len(testdata_cbma.ids)
    # Test set_params
    kern.set_params(fwhm=10, sample_size=None)
    kern2 = kernel.ALEKernel(fwhm=10)
    ma_maps1 = kern.transform(coordinates, testdata_cbma.masker, return_type="array")
    ma_maps2 = kern2.transform(coordinates, testdata_cbma.masker, return_type="array")
    assert ma_maps1.shape[0] == ma_maps2.shape[0] == len(testdata_cbma.ids)
    assert np.array_equal(ma_maps1, ma_maps2)


def test_alekernel_1mm(testdata_cbma):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on 1mm template.
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    coordinates = testdata_cbma.coordinates.copy()
    coordinates["sample_size"] = 20

    id_ = "pain_01.nidm-1"
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(coordinates, testdata_cbma.masker, return_type="image")
    ijk = coordinates.loc[coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = ijk.values.astype(int)
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.where(kern_data == np.max(kern_data))
    max_ijk = np.array(max_idx).T
    assert np.array_equal(ijk, max_ijk)


def test_alekernel_2mm(testdata_cbma):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on 2mm template.
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    coordinates = testdata_cbma.coordinates.copy()
    coordinates["sample_size"] = 20

    id_ = "pain_01.nidm-1"
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(coordinates, masker=testdata_cbma.masker, return_type="image")

    ijk = coordinates.loc[coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_alekernel_inputdataset_returnimages(testdata_cbma):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on Dataset object.
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    testdata_cbma = testdata_cbma.copy()
    coordinates = testdata_cbma.coordinates.copy()
    coordinates["sample_size"] = 20
    testdata_cbma.coordinates = coordinates

    id_ = "pain_01.nidm-1"
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(testdata_cbma, return_type="image")

    ijk = coordinates.loc[coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_alekernel_fwhm(testdata_cbma):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test with explicit FWHM.
    """
    coordinates = testdata_cbma.coordinates.copy()

    id_ = "pain_01.nidm-1"
    kern = kernel.ALEKernel(fwhm=10)
    ma_maps = kern.transform(coordinates, masker=testdata_cbma.masker, return_type="image")

    ijk = coordinates.loc[coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_alekernel_sample_size(testdata_cbma):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test with explicit sample size.
    """
    coordinates = testdata_cbma.coordinates.copy()

    id_ = "pain_01.nidm-1"
    kern = kernel.ALEKernel(sample_size=20)
    ma_maps = kern.transform(coordinates, masker=testdata_cbma.masker, return_type="image")

    ijk = coordinates.loc[coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_alekernel_inputdataset_returndataset(testdata_cbma):
    """
    Check that the different return types produce equivalent results
    (minus the masking element).
    """
    temp_dir = tempfile.mkdtemp()
    testdata_cbma.update_path(temp_dir)
    kern = kernel.ALEKernel(sample_size=20)
    ma_maps = kern.transform(testdata_cbma, return_type="image")
    ma_arr = kern.transform(testdata_cbma, return_type="array")
    dset = kern.transform(testdata_cbma, return_type="dataset")
    ma_maps_from_dset = kern.transform(dset, return_type="image")
    ma_arr_from_dset = kern.transform(dset, return_type="array")
    ma_maps_arr = testdata_cbma.masker.transform(ma_maps)
    ma_maps_from_dset_arr = dset.masker.transform(ma_maps_from_dset)
    dset_from_dset = kern.transform(dset, return_type="dataset")
    ma_maps_dset = testdata_cbma.masker.transform(
        dset.get_images(ids=dset.ids, imtype=kern.image_type)
    )
    assert isinstance(dset_from_dset, Dataset)
    assert np.array_equal(ma_arr, ma_maps_arr)
    assert np.array_equal(ma_arr, ma_maps_dset)
    assert np.array_equal(ma_arr, ma_maps_from_dset_arr)
    assert np.array_equal(ma_arr, ma_arr_from_dset)

    # It would be nice to use a better teardown
    shutil.rmtree(temp_dir)


def test_mkdakernel_smoke(testdata_cbma):
    """
    Smoke test for nimare.meta.kernel.MKDAKernel, using Dataset object.
    """
    kern = kernel.MKDAKernel()
    ma_maps = kern.transform(testdata_cbma, return_type="image")
    assert len(ma_maps) == len(testdata_cbma.ids)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker, return_type="array")
    assert ma_maps.shape[0] == len(testdata_cbma.ids)


def test_mkdakernel_1mm(testdata_cbma):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = "pain_01.nidm-1"
    kern = kernel.MKDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker, return_type="image")

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_mkdakernel_2mm(testdata_cbma):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 2mm template.
    """
    id_ = "pain_01.nidm-1"
    kern = kernel.MKDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker, return_type="image")

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_mkdakernel_inputdataset_returndataset(testdata_cbma):
    """
    Check that the different return types produce equivalent results
    (minus the masking element).
    """
    temp_dir = tempfile.mkdtemp()
    testdata_cbma.update_path(temp_dir)
    kern = kernel.MKDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma, return_type="image")
    ma_arr = kern.transform(testdata_cbma, return_type="array")
    dset = kern.transform(testdata_cbma, return_type="dataset")
    ma_maps_from_dset = kern.transform(dset, return_type="image")
    ma_arr_from_dset = kern.transform(dset, return_type="array")
    dset_from_dset = kern.transform(dset, return_type="dataset")
    ma_maps_arr = testdata_cbma.masker.transform(ma_maps)
    ma_maps_from_dset_arr = dset.masker.transform(ma_maps_from_dset)
    ma_maps_dset = testdata_cbma.masker.transform(
        dset.get_images(ids=dset.ids, imtype=kern.image_type)
    )
    assert isinstance(dset_from_dset, Dataset)
    assert np.array_equal(ma_arr, ma_maps_arr)
    assert np.array_equal(ma_arr, ma_maps_dset)
    assert np.array_equal(ma_arr, ma_maps_from_dset_arr)
    assert np.array_equal(ma_arr, ma_arr_from_dset)

    # It would be nice to use a better teardown
    shutil.rmtree(temp_dir)


def test_kdakernel_smoke(testdata_cbma):
    """
    Smoke test for nimare.meta.kernel.KDAKernel
    """
    kern = kernel.KDAKernel()
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker, return_type="image")
    assert len(ma_maps) == len(testdata_cbma.ids)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker, return_type="array")
    assert ma_maps.shape[0] == len(testdata_cbma.ids)


def test_kdakernel_1mm(testdata_cbma):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = "pain_01.nidm-1"
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker, return_type="image")

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_2mm(testdata_cbma):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 2mm template.
    """
    id_ = "pain_01.nidm-1"
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker, return_type="image")

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_inputdataset_returnimages(testdata_cbma):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on Dataset object.
    """
    id_ = "pain_01.nidm-1"
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma, return_type="image")

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates["id"] == id_, ["i", "j", "k"]]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_inputdataset_returndataset(testdata_cbma):
    """
    Check that the different return types produce equivalent results
    (minus the masking element).
    """
    temp_dir = tempfile.mkdtemp()
    testdata_cbma.update_path(temp_dir)
    kern = kernel.KDAKernel(r=4, value=1)
    # MA map generation from transformer
    ma_maps = kern.transform(testdata_cbma, return_type="image")
    ma_arr = kern.transform(testdata_cbma, return_type="array")
    dset = kern.transform(testdata_cbma, return_type="dataset")
    # Load generated MA maps
    ma_maps_from_dset = kern.transform(dset, return_type="image")
    ma_arr_from_dset = kern.transform(dset, return_type="array")
    dset_from_dset = kern.transform(dset, return_type="dataset")
    ma_maps_arr = testdata_cbma.masker.transform(ma_maps)
    ma_maps_from_dset_arr = dset.masker.transform(ma_maps_from_dset)
    ma_maps_dset = testdata_cbma.masker.transform(
        dset.get_images(ids=dset.ids, imtype=kern.image_type)
    )
    assert isinstance(dset_from_dset, Dataset)
    assert np.array_equal(ma_arr, ma_maps_arr)
    assert np.array_equal(ma_arr, ma_maps_dset)
    assert np.array_equal(ma_arr, ma_maps_from_dset_arr)
    assert np.array_equal(ma_arr, ma_arr_from_dset)

    # It would be nice to use a better teardown
    shutil.rmtree(temp_dir)


def test_kdakernel_transform_attributes(testdata_cbma):
    """
    Check that attributes are added at transform.
    """
    kern = kernel.KDAKernel(r=4, value=1)
    assert not hasattr(kern, "filename_pattern")
    assert not hasattr(kern, "image_type")
    _ = kern.transform(testdata_cbma, return_type="image")
    assert hasattr(kern, "filename_pattern")
    assert hasattr(kern, "image_type")
