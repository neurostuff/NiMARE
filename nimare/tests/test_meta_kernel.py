"""Test nimare.meta.kernel (CBMA kernel estimators)."""
import shutil

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage.measurements import center_of_mass

import nimare
from nimare import extract
from nimare.dataset import Dataset
from nimare.meta import MKDADensity, kernel
from nimare.utils import get_masker, get_template, mm2vox


@pytest.mark.parametrize(
    "kern, res, param, return_type, kwargs",
    [
        (kernel.ALEKernel, 1, "dataset", "dataset", {"sample_size": 20}),
        (kernel.ALEKernel, 2, "dataset", "dataset", {"sample_size": 20}),
        (kernel.ALEKernel, 1, "dataset", "image", {"sample_size": 20}),
        (kernel.ALEKernel, 2, "dataset", "image", {"sample_size": 20}),
        (kernel.ALEKernel, 1, "dataframe", "image", {"sample_size": 20}),
        (kernel.ALEKernel, 2, "dataframe", "image", {"sample_size": 20}),
        (kernel.ALEKernel, 1, "dataset", "array", {"sample_size": 20}),
        (kernel.ALEKernel, 2, "dataset", "array", {"sample_size": 20}),
        (kernel.ALEKernel, 1, "dataframe", "array", {"sample_size": 20}),
        (kernel.ALEKernel, 2, "dataframe", "array", {"sample_size": 20}),
        (kernel.MKDAKernel, 1, "dataset", "image", {"r": 4, "value": 1}),
        (kernel.MKDAKernel, 2, "dataset", "image", {"r": 4, "value": 1}),
        (kernel.MKDAKernel, 1, "dataframe", "image", {"r": 4, "value": 1}),
        (kernel.MKDAKernel, 2, "dataframe", "image", {"r": 4, "value": 1}),
        (kernel.KDAKernel, 1, "dataset", "image", {"r": 4, "value": 1}),
        (kernel.KDAKernel, 2, "dataset", "image", {"r": 4, "value": 1}),
        (kernel.KDAKernel, 1, "dataframe", "image", {"r": 4, "value": 1}),
        (kernel.KDAKernel, 2, "dataframe", "image", {"r": 4, "value": 1}),
    ],
)
def test_kernel_peaks(testdata_cbma, tmp_path_factory, kern, res, param, return_type, kwargs):
    """Peak/COMs of kernel maps should match the foci fed in (assuming focus isn't masked out).

    Notes
    -----
    Remember that dataframe --> dataset won't work.
    Only testing dataset --> dataset with ALEKernel because it takes a while.
    Test on multiple template resolutions.
    """
    tmpdir = tmp_path_factory.mktemp("test_kernel_peaks")
    testdata_cbma.update_path(tmpdir)

    id_ = "pain_03.nidm-1"

    template = get_template(space=f"mni152_{res}mm", mask="brain")
    masker = get_masker(template)

    xyz = testdata_cbma.coordinates.loc[testdata_cbma.coordinates["id"] == id_, ["x", "y", "z"]]
    ijk = mm2vox(xyz, masker.mask_img.affine)
    ijk = np.squeeze(ijk.astype(int))

    if param == "dataframe":
        input_ = testdata_cbma.coordinates.copy()
    elif param == "dataset":
        input_ = testdata_cbma.copy()

    kern_instance = kern(**kwargs)
    output = kern_instance.transform(input_, masker, return_type=return_type)

    if return_type == "image":
        kern_data = output[0].get_fdata()
    elif return_type == "array":
        kern_data = np.squeeze(masker.inverse_transform(output[:1, :]).get_fdata())
    else:
        f = output.images.loc[output.images["id"] == id_, kern_instance.image_type].values[0]
        kern_data = nib.load(f).get_fdata()

    if isinstance(kern_instance, kernel.ALEKernel):
        loc_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    elif isinstance(kern_instance, (kernel.MKDAKernel, kernel.KDAKernel)):
        loc_idx = np.array(center_of_mass(kern_data)).astype(int).T
    else:
        raise Exception(f"A {type(kern_instance)}? Why?")

    loc_ijk = np.squeeze(loc_idx)

    assert np.array_equal(ijk, loc_ijk)


@pytest.mark.parametrize(
    "kern, kwargs",
    [
        (kernel.ALEKernel, {"sample_size": 20}),
        (kernel.MKDAKernel, {"r": 4, "value": 1}),
        (kernel.KDAKernel, {"r": 4, "value": 1}),
    ],
)
def test_kernel_transform_attributes(testdata_cbma, kern, kwargs):
    """Check that attributes are added at transform."""
    kern_instance = kern(**kwargs)
    assert not hasattr(kern_instance, "filename_pattern")
    assert not hasattr(kern_instance, "image_type")
    _ = kern_instance.transform(testdata_cbma, return_type="image")
    assert hasattr(kern_instance, "filename_pattern")
    assert hasattr(kern_instance, "image_type")


@pytest.mark.parametrize(
    "kern, kwargs, set_kwargs",
    [
        (kernel.ALEKernel, {"sample_size": 20}, {"sample_size": None, "fwhm": 10}),
        (kernel.MKDAKernel, {"r": 4, "value": 1}, {"r": 10, "value": 3}),
        (kernel.KDAKernel, {"r": 4, "value": 1}, {"r": 10, "value": 3}),
    ],
)
def test_kernel_smoke(testdata_cbma, kern, kwargs, set_kwargs):
    """Smoke test for different kernel transformers and check that you can reset params."""
    coordinates = testdata_cbma.coordinates.copy()

    kern_instance = kern(**kwargs)
    ma_maps = kern_instance.transform(coordinates, testdata_cbma.masker, return_type="image")
    assert len(ma_maps) == len(testdata_cbma.ids) - 2
    ma_maps = kern_instance.transform(coordinates, testdata_cbma.masker, return_type="array")
    assert ma_maps.shape[0] == len(testdata_cbma.ids) - 2

    # Test set_params
    kern_instance.set_params(**set_kwargs)
    kern_instance2 = kern(**set_kwargs)
    ma_maps1 = kern_instance.transform(coordinates, testdata_cbma.masker, return_type="array")
    ma_maps2 = kern_instance2.transform(coordinates, testdata_cbma.masker, return_type="array")
    assert ma_maps1.shape[0] == ma_maps2.shape[0] == len(testdata_cbma.ids) - 2
    assert np.array_equal(ma_maps1, ma_maps2)


@pytest.mark.parametrize(
    "kern, kwargs",
    [
        (kernel.ALEKernel, {"sample_size": 20}),
        (kernel.KDAKernel, {}),
        (kernel.MKDAKernel, {}),
    ],
)
def test_kernel_low_high_memory(testdata_cbma, tmp_path_factory, kern, kwargs):
    """Compare kernel results when memory_limit is used vs. not."""
    kern_low_mem = kern(memory_limit="1gb", **kwargs)
    kern_spec_mem = kern(memory_limit="2gb", **kwargs)
    kern_high_mem = kern(memory_limit=None, **kwargs)
    trans_kwargs = {"dataset": testdata_cbma, "return_type": "array"}
    assert np.array_equal(
        kern_low_mem.transform(**trans_kwargs),
        kern_high_mem.transform(**trans_kwargs),
    )
    assert np.array_equal(
        kern_low_mem.transform(**trans_kwargs),
        kern_spec_mem.transform(**trans_kwargs),
    )


def test_ALEKernel_fwhm(testdata_cbma):
    """Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't masked out).

    Test with explicit FWHM.
    """
    coordinates = testdata_cbma.coordinates.copy()

    id_ = "pain_03.nidm-1"
    kern = kernel.ALEKernel(fwhm=10)
    ma_maps = kern.transform(coordinates, masker=testdata_cbma.masker, return_type="image")

    xyz = coordinates.loc[coordinates["id"] == id_, ["x", "y", "z"]]
    ijk = mm2vox(xyz, testdata_cbma.masker.mask_img.affine)
    ijk = np.squeeze(ijk.astype(int))

    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_ALEKernel_sample_size(testdata_cbma):
    """Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't masked out).

    Test with explicit sample size.
    """
    coordinates = testdata_cbma.coordinates.copy()

    id_ = "pain_03.nidm-1"
    kern = kernel.ALEKernel(sample_size=20)
    ma_maps = kern.transform(coordinates, masker=testdata_cbma.masker, return_type="image")

    xyz = coordinates.loc[coordinates["id"] == id_, ["x", "y", "z"]]
    ijk = mm2vox(xyz, testdata_cbma.masker.mask_img.affine)
    ijk = np.squeeze(ijk.astype(int))

    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_Peaks2MapsKernel(testdata_cbma, tmp_path_factory):
    """Test Peaks2MapsKernel."""
    tmpdir = tmp_path_factory.mktemp("test_Peaks2MapsKernel")

    model_dir = extract.download_peaks2maps_model()

    testdata_cbma.update_path(tmpdir)
    kern = kernel.Peaks2MapsKernel(model_dir=model_dir)
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
    ids = dset.coordinates["id"].unique()
    ma_maps_dset = testdata_cbma.masker.transform(dset.get_images(ids=ids, imtype=kern.image_type))
    assert isinstance(dset_from_dset, Dataset)
    assert np.array_equal(ma_arr, ma_maps_arr)
    assert np.array_equal(ma_arr, ma_maps_dset)
    assert np.array_equal(ma_arr, ma_maps_from_dset_arr)
    assert np.array_equal(ma_arr, ma_arr_from_dset)
    shutil.rmtree(model_dir)


def test_Peaks2MapsKernel_MKDADensity(testdata_cbma, tmp_path_factory):
    """Test that the Peaks2Maps kernel can work with an estimator."""
    tmpdir = tmp_path_factory.mktemp("test_Peaks2MapsKernel_MKDADensity")

    model_dir = extract.download_peaks2maps_model()

    testdata_cbma.update_path(tmpdir)
    kern = kernel.Peaks2MapsKernel(model_dir=model_dir)

    est = MKDADensity(kernel_transformer=kern, null_method="approximate")
    res = est.fit(testdata_cbma)
    assert isinstance(res, nimare.results.MetaResult)
    assert res.get_map("p", return_type="array").dtype == np.float64
