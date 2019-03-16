"""
Test nimare.meta.cbma.kernel (CBMA kernel estimators).
"""
import pytest
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import center_of_mass

from nimare.meta.cbma import kernel
from nimare.utils import get_template, mm2vox


# Fixtures used in the rest of the tests
class DummyDataset(object):
    def __init__(self, df, img):
        self.coordinates = df
        self.mask = img


@pytest.fixture(scope='module')
def testdata1():
    mask_img = get_template(space='mni152_1mm', mask='brain')
    df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'n', 'space'],
                      data=[[1, -28, -20, -16, 20, 'mni'],
                            [2, -28, -20, -16, 5, 'mni']])
    xyz = df[['x', 'y', 'z']].values
    ijk = pd.DataFrame(mm2vox(xyz, mask_img.affine), columns=['i', 'j', 'k'])
    df = pd.concat([df, ijk], axis=1)

    dset = DummyDataset(df, mask_img)
    return dset


@pytest.fixture(scope='module')
def testdata2():
    mask_img = get_template(space='mni152_2mm', mask='brain')
    df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'n', 'space'],
                      data=[[1, -28, -20, -16, 20, 'mni'],
                            [2, -28, -20, -16, 5, 'mni']])
    xyz = df[['x', 'y', 'z']].values
    ijk = pd.DataFrame(mm2vox(xyz, mask_img.affine), columns=['i', 'j', 'k'])
    df = pd.concat([df, ijk], axis=1)

    dset = DummyDataset(df, mask_img)
    return dset


def test_alekernel_smoke(testdata2):
    """
    Smoke test for nimare.meta.cbma.kernel.ALEKernel
    """
    ids = [1, 2]
    kern = kernel.ALEKernel(testdata2.coordinates, testdata2.mask)
    ale_kernels = kern.transform(ids=ids)
    assert len(ale_kernels) == len(ids)


def test_alekernel1(testdata1):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on 1mm template.
    """
    id_ = 1
    kern = kernel.ALEKernel(testdata1.coordinates, testdata1.mask)
    ale_kernels = kern.transform(ids=[id_])

    ijk = testdata1.coordinates.loc[testdata1.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = ijk.values.astype(int)
    kern_data = ale_kernels[0].get_data()
    max_idx = np.where(kern_data == np.max(kern_data))
    max_ijk = np.array(max_idx).T
    assert np.array_equal(ijk, max_ijk)


def test_alekernel2(testdata2):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on 2mm template.
    """
    id_ = 1
    kern = kernel.ALEKernel(testdata2.coordinates, testdata2.mask)
    ale_kernels = kern.transform(ids=[id_])

    ijk = testdata2.coordinates.loc[testdata2.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = ale_kernels[0].get_data()
    max_idx = np.array(np.where(kern_data == np.max(kern_data))).T
    max_ijk = np.squeeze(max_idx)
    assert np.array_equal(ijk, max_ijk)


def test_mkdakernel_smoke(testdata1):
    """
    Smoke test for nimare.meta.cbma.kernel.MKDAKernel
    """
    ids = [1, 2]
    kern = kernel.MKDAKernel(testdata1.coordinates, testdata1.mask)
    ale_kernels = kern.transform(ids=ids)
    assert len(ale_kernels) == len(ids)


def test_mkdakernel1(testdata1):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 1
    kern = kernel.MKDAKernel(testdata1.coordinates, testdata1.mask)
    mkda_kernels = kern.transform(ids=[id_], r=4, value=1)

    ijk = testdata1.coordinates.loc[testdata1.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = mkda_kernels[0].get_data()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_mkdakernel2(testdata2):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 2mm template.
    """
    id_ = 1
    kern = kernel.MKDAKernel(testdata2.coordinates, testdata2.mask)
    mkda_kernels = kern.transform(ids=[id_], r=4, value=1)

    ijk = testdata2.coordinates.loc[testdata2.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = mkda_kernels[0].get_data()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_smoke(testdata1):
    """
    Smoke test for nimare.meta.cbma.kernel.KDAKernel
    """
    ids = [1, 2]
    kern = kernel.KDAKernel(testdata1.coordinates, testdata1.mask)
    ale_kernels = kern.transform(ids=ids)
    assert len(ale_kernels) == len(ids)


def test_kdakernel1(testdata1):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 1
    kern = kernel.KDAKernel(testdata1.coordinates, testdata1.mask)
    kda_kernels = kern.transform(ids=[id_], r=4, value=1)

    ijk = testdata1.coordinates.loc[testdata1.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = kda_kernels[0].get_data()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel2(testdata2):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 2mm template.
    """
    id_ = 1
    kern = kernel.KDAKernel(testdata2.coordinates, testdata2.mask)
    kda_kernels = kern.transform(ids=[id_], r=4, value=1)

    ijk = testdata2.coordinates.loc[testdata2.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = kda_kernels[0].get_data()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)
