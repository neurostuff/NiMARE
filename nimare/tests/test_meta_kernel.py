"""Test nimare.meta.kernel (CBMA kernel estimators)."""

import time

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import center_of_mass

from nimare.meta import kernel
from nimare.utils import get_masker, get_template, mm2vox


@pytest.mark.parametrize(
    "kern, res, param, return_type, kwargs",
    [
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
def test_kernel_transform_attributes(kern, kwargs):
    """Check that attributes are added at transform."""
    kern_instance = kern(**kwargs)
    assert not hasattr(kern_instance, "filename_pattern")
    assert not hasattr(kern_instance, "image_type")
    kern_instance._infer_names()
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


def test_ALEKernel_memory(testdata_cbma, tmp_path_factory):
    """Test ALEKernel with memory caching enable."""
    cachedir = tmp_path_factory.mktemp("test_ALE_memory")

    coord = testdata_cbma.coordinates.copy()

    kern_cached = kernel.ALEKernel(sample_size=20, memory=str(cachedir), memory_level=2)
    ma_maps_cached = kern_cached.transform(coord, masker=testdata_cbma.masker, return_type="array")

    kern = kernel.ALEKernel(sample_size=20, memory=None)
    start = time.time()
    ma_maps = kern.transform(coord, masker=testdata_cbma.masker, return_type="array")
    done = time.time()
    elapsed = done - start

    assert np.array_equal(ma_maps_cached, ma_maps)

    # Test that memory is actually used
    kern_cached_fast = kernel.ALEKernel(sample_size=20, memory=str(cachedir), memory_level=2)
    start_chached = time.time()
    ma_maps_cached_fast = kern_cached_fast.transform(
        coord, masker=testdata_cbma.masker, return_type="array"
    )
    done_cached = time.time()
    elapsed_cached = done_cached - start_chached

    assert np.array_equal(ma_maps_cached_fast, ma_maps)
    assert elapsed_cached < elapsed


def test_MKDA_kernel_sum_across(testdata_cbma):
    """Test if creating a summary array is equivalent to summing across the sparse array."""
    kern = kernel.MKDAKernel(r=10, value=1)
    coordinates = testdata_cbma.coordinates.copy()
    sparse_ma_maps = kern.transform(coordinates, masker=testdata_cbma.masker, return_type="sparse")
    summary_map = kern.transform(
        coordinates, masker=testdata_cbma.masker, return_type="summary_array"
    )

    summary_sparse_ma_map = sparse_ma_maps.sum(axis=0)
    mask_data = testdata_cbma.masker.mask_img.get_fdata().astype(bool)

    # Indexing the sparse array is slow, perform masking in the dense array
    summary_sparse_ma_map = summary_sparse_ma_map.todense().reshape(-1)
    summary_sparse_ma_map = summary_sparse_ma_map[mask_data.reshape(-1)]

    assert (
        np.testing.assert_array_equal(summary_map, summary_sparse_ma_map.astype(np.int32)) is None
    )
