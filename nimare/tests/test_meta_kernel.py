"""
Test nimare.meta.kernel (CBMA kernel estimators).
"""
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import center_of_mass

from nimare.meta import kernel
from nimare.utils import get_template, get_masker


def test_alekernel_smoke(testdata_cbma):
    """
    Smoke test for nimare.meta.kernel.ALEKernel
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    coordinates = testdata_cbma.coordinates.copy()
    coordinates['sample_size'] = 20

    kern = kernel.ALEKernel()
    ma_maps = kern.transform(
        coordinates,
        testdata_cbma.masker,
        return_type='image'
    )
    assert len(ma_maps) == len(testdata_cbma.ids)
    ma_maps = kern.transform(
        coordinates,
        testdata_cbma.masker,
        return_type='array'
    )
    assert ma_maps.shape[0] == len(testdata_cbma.ids)


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
    coordinates['sample_size'] = 20

    id_ = 'pain_01.nidm-1'
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(
        coordinates,
        testdata_cbma.masker
    )
    ijk = coordinates.loc[coordinates['id'] == id_, ['i', 'j', 'k']]
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
    coordinates['sample_size'] = 20

    id_ = 'pain_01.nidm-1'
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(
        coordinates,
        masker=testdata_cbma.masker
    )

    ijk = coordinates.loc[coordinates['id'] == id_, ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_alekernel_dataset(testdata_cbma):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on Dataset object.
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    testdata_cbma = testdata_cbma.slice(testdata_cbma.ids)
    coordinates = testdata_cbma.coordinates.copy()
    coordinates['sample_size'] = 20
    testdata_cbma.coordinates = coordinates

    id_ = 'pain_01.nidm-1'
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(
        testdata_cbma
    )

    ijk = coordinates.loc[coordinates['id'] == id_, ['i', 'j', 'k']]
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

    id_ = 'pain_01.nidm-1'
    kern = kernel.ALEKernel(fwhm=10)
    ma_maps = kern.transform(
        coordinates,
        masker=testdata_cbma.masker
    )

    ijk = coordinates.loc[coordinates['id'] == id_, ['i', 'j', 'k']]
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

    id_ = 'pain_01.nidm-1'
    kern = kernel.ALEKernel(sample_size=20)
    ma_maps = kern.transform(
        coordinates,
        masker=testdata_cbma.masker
    )

    ijk = coordinates.loc[coordinates['id'] == id_, ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_mkdakernel_smoke(testdata_cbma):
    """
    Smoke test for nimare.meta.kernel.MKDAKernel, using Dataset object.
    """
    kern = kernel.MKDAKernel()
    ma_maps = kern.transform(testdata_cbma)
    assert len(ma_maps) == len(testdata_cbma.ids)
    ma_maps = kern.transform(
        testdata_cbma.coordinates,
        testdata_cbma.masker,
        return_type='array'
    )
    assert ma_maps.shape[0] == len(testdata_cbma.ids)


def test_mkdakernel_1mm(testdata_cbma):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 'pain_01.nidm-1'
    kern = kernel.MKDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker)

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates['id'] == id_,
                                        ['i', 'j', 'k']]
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
    id_ = 'pain_01.nidm-1'
    kern = kernel.MKDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker)

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates['id'] == id_,
                                        ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_smoke(testdata_cbma):
    """
    Smoke test for nimare.meta.kernel.KDAKernel
    """
    kern = kernel.KDAKernel()
    ma_maps = kern.transform(
        testdata_cbma.coordinates,
        testdata_cbma.masker
    )
    assert len(ma_maps) == len(testdata_cbma.ids)
    ma_maps = kern.transform(
        testdata_cbma.coordinates,
        testdata_cbma.masker,
        return_type='array'
    )
    assert ma_maps.shape[0] == len(testdata_cbma.ids)


def test_kdakernel_1mm(testdata_cbma):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 'pain_01.nidm-1'
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker)

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates['id'] == id_,
                                        ['i', 'j', 'k']]
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
    id_ = 'pain_01.nidm-1'
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma.coordinates, testdata_cbma.masker)

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates['id'] == id_,
                                        ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_dataset(testdata_cbma):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on Dataset object.
    """
    id_ = 'pain_01.nidm-1'
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata_cbma)

    ijk = testdata_cbma.coordinates.loc[testdata_cbma.coordinates['id'] == id_,
                                        ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)
