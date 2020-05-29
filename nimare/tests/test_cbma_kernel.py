"""
Test nimare.meta.cbma.kernel (CBMA kernel estimators).
"""
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import center_of_mass

from nimare.meta.cbma import kernel
from nimare.utils import get_template, get_masker


def test_alekernel_smoke(testdata):
    """
    Smoke test for nimare.meta.cbma.kernel.ALEKernel
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    coordinates = testdata['dset'].coordinates.copy()
    coordinates['sample_size'] = 20

    kern = kernel.ALEKernel()
    ma_maps = kern.transform(
        coordinates,
        testdata['dset'].masker,
        return_type='image'
    )
    assert len(ma_maps) == len(testdata['dset'].ids)
    ma_maps = kern.transform(
        coordinates,
        testdata['dset'].masker,
        return_type='array'
    )
    assert ma_maps.shape[0] == len(testdata['dset'].ids)


def test_alekernel1(testdata):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on 1mm template.
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    coordinates = testdata['dset'].coordinates.copy()
    coordinates['sample_size'] = 20

    id_ = 'pain_01.nidm-1'
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(
        coordinates,
        testdata['dset'].masker
    )
    ijk = coordinates.loc[coordinates['id'] == id_, ['i', 'j', 'k']]
    ijk = ijk.values.astype(int)
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.where(kern_data == np.max(kern_data))
    max_ijk = np.array(max_idx).T
    assert np.array_equal(ijk, max_ijk)


def test_alekernel2(testdata):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on 2mm template.
    """
    # Manually override dataset coordinates file sample sizes
    # This column would be extracted from metadata and added to coordinates
    # automatically by the Estimator
    coordinates = testdata['dset'].coordinates.copy()
    coordinates['sample_size'] = 20

    id_ = 'pain_01.nidm-1'
    kern = kernel.ALEKernel()
    ma_maps = kern.transform(
        coordinates,
        masker=testdata['dset'].masker
    )

    ijk = coordinates.loc[coordinates['id'] == id_, ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_mkdakernel_smoke(testdata):
    """
    Smoke test for nimare.meta.cbma.kernel.MKDAKernel
    """
    kern = kernel.MKDAKernel()
    ma_maps = kern.transform(testdata['dset'])
    assert len(ma_maps) == len(testdata['dset'].ids)
    ma_maps = kern.transform(
        testdata['dset'].coordinates,
        testdata['dset'].masker,
        return_type='array'
    )
    assert ma_maps.shape[0] == len(testdata['dset'].ids)


def test_mkdakernel1(testdata):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 'pain_01.nidm-1'
    kern = kernel.MKDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata['dset'].coordinates, testdata['dset'].masker)

    ijk = testdata['dset'].coordinates.loc[testdata['dset'].coordinates['id'] == id_,
                                           ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_mkdakernel2(testdata):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 2mm template.
    """
    id_ = 'pain_01.nidm-1'
    kern = kernel.MKDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata['dset'].coordinates, testdata['dset'].masker)

    ijk = testdata['dset'].coordinates.loc[testdata['dset'].coordinates['id'] == id_,
                                           ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_smoke(testdata):
    """
    Smoke test for nimare.meta.cbma.kernel.KDAKernel
    """
    kern = kernel.KDAKernel()
    ma_maps = kern.transform(
        testdata['dset'].coordinates,
        testdata['dset'].masker
    )
    assert len(ma_maps) == len(testdata['dset'].ids)
    ma_maps = kern.transform(
        testdata['dset'].coordinates,
        testdata['dset'].masker,
        return_type='array'
    )
    assert ma_maps.shape[0] == len(testdata['dset'].ids)


def test_kdakernel1(testdata):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 'pain_01.nidm-1'
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata['dset'].coordinates, testdata['dset'].masker)

    ijk = testdata['dset'].coordinates.loc[testdata['dset'].coordinates['id'] == id_,
                                           ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel2(testdata):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 2mm template.
    """
    id_ = 'pain_01.nidm-1'
    kern = kernel.KDAKernel(r=4, value=1)
    ma_maps = kern.transform(testdata['dset'].coordinates, testdata['dset'].masker)

    ijk = testdata['dset'].coordinates.loc[testdata['dset'].coordinates['id'] == id_,
                                           ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ma_maps[0].get_fdata()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)
