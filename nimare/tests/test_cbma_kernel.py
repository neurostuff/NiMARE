"""
Test nimare.meta.cbma.kernel (CBMA kernel estimators).
"""
import pytest
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import center_of_mass

from nimare.meta.cbma import kernel
from nimare.utils import get_template, mm2vox, get_masker


# Fixtures used in the rest of the tests
class DummyDataset(object):
    def __init__(self, df, img):
        self.coordinates = df
        self.masker = get_masker(img)

    def slice(self):
        pass


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
    kern = kernel.ALEKernel()
    ale_kernels = kern.transform(testdata2.coordinates, testdata2.masker)
    assert len(ale_kernels) == 2


def test_alekernel1(testdata1):
    """
    Peaks of ALE kernel maps should match the foci fed in (assuming focus isn't
    masked out).
    Test on 1mm template.
    """
    id_ = 1
    kern = kernel.ALEKernel()
    ale_kernels = kern.transform(testdata1.coordinates, testdata1.masker)

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
    kern = kernel.ALEKernel()
    ale_kernels = kern.transform(testdata2.coordinates, mask=testdata2.masker)

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
    kern = kernel.MKDAKernel()
    ale_kernels = kern.transform(testdata1)
    assert len(ale_kernels) == 2


def test_mkdakernel1(testdata1):
    """
    COMs of MKDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 1
    kern = kernel.MKDAKernel(r=4, value=1)
    maps = kern.transform(testdata1.coordinates, testdata1.masker)

    ijk = testdata1.coordinates.loc[testdata1.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = maps[0].get_data()
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
    kern = kernel.MKDAKernel(r=4, value=1)
    maps = kern.transform(testdata2.coordinates, testdata2.masker)

    ijk = testdata2.coordinates.loc[testdata2.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = maps[0].get_data()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)


def test_kdakernel_smoke(testdata1):
    """
    Smoke test for nimare.meta.cbma.kernel.KDAKernel
    """
    kern = kernel.KDAKernel()
    maps = kern.transform(testdata1.coordinates, testdata1.masker)
    assert len(maps) == 2


def test_kdakernel1(testdata1):
    """
    COMs of KDA kernel maps should match the foci fed in (assuming focus isn't
    masked out and spheres don't overlap).
    Test on 1mm template.
    """
    id_ = 1
    kern = kernel.KDAKernel(r=4, value=1)
    maps = kern.transform(testdata1.coordinates, testdata1.masker)

    ijk = testdata1.coordinates.loc[testdata1.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = maps[0].get_data()
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
    kern = kernel.KDAKernel(r=4, value=1)
    maps = kern.transform(testdata2.coordinates, testdata2.masker)

    ijk = testdata2.coordinates.loc[testdata2.coordinates['id'] == id_,
                                    ['i', 'j', 'k']]
    ijk = np.squeeze(ijk.values.astype(int))
    kern_data = maps[0].get_data()
    com = np.array(center_of_mass(kern_data)).astype(int).T
    com = np.squeeze(com)
    assert np.array_equal(ijk, com)
