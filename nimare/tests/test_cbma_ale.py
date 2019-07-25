"""
Test nimare.meta.cbma.ale (ALE/SCALE meta-analytic algorithms).
"""
import pytest
import pandas as pd

import nimare
from nimare.meta.cbma import ale
from nimare.utils import get_template, mm2vox


# Fixtures used in the rest of the tests
class DummyDataset(object):
    def __init__(self, df, img):
        self.coordinates = df
        self.mask = img


@pytest.fixture(scope='module')
def testdata1():
    mask_img = get_template(space='mni152_2mm', mask='brain')
    df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'n', 'space'],
                      data=[[1, -28, -20, -16, 100, 'mni'],
                            [2, -28, -20, -16, 100, 'mni'],
                            [3, -28, -20, -16, 100, 'mni'],
                            [4, -28, -20, -16, 100, 'mni'],
                            [5, -28, -20, -16, 100, 'mni'],
                            [6, -28, -20, -16, 100, 'mni'],
                            [7, -28, -20, -16, 100, 'mni'],
                            [8, -28, -20, -16, 100, 'mni'],
                            [9, -28, -20, -16, 100, 'mni'],
                            [10, -28, -20, -16, 100, 'mni'],
                            [11, -28, -20, -16, 100, 'mni']])
    xyz = df[['x', 'y', 'z']].values
    ijk = pd.DataFrame(mm2vox(xyz, mask_img.affine), columns=['i', 'j', 'k'])
    df = pd.concat([df, ijk], axis=1)

    dset = DummyDataset(df, mask_img)
    return dset


@pytest.fixture(scope='module')
def testdata2():
    mask_img = get_template(space='mni152_2mm', mask='brain')
    df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'n', 'space'],
                      data=[[1, -24, -20, -16, 100, 'mni'],
                            [2, -24, -20, -16, 100, 'mni'],
                            [3, -24, -20, -16, 100, 'mni'],
                            [4, -24, -20, -16, 100, 'mni'],
                            [5, -24, -20, -16, 100, 'mni'],
                            [6, -24, -20, -16, 100, 'mni'],
                            [7, -24, -20, -16, 100, 'mni'],
                            [8, -24, -20, -16, 100, 'mni'],
                            [9, -24, -20, -16, 100, 'mni'],
                            [10, -24, -20, -16, 100, 'mni'],
                            [11, -24, -20, -16, 100, 'mni']])
    xyz = df[['x', 'y', 'z']].values
    ijk = pd.DataFrame(mm2vox(xyz, mask_img.affine), columns=['i', 'j', 'k'])
    df = pd.concat([df, ijk], axis=1)

    dset = DummyDataset(df, mask_img)
    return dset


def test_ale(testdata1):
    """
    Smoke test for ALE
    """
    ale_meta = ale.ALE(n_iters=5, n_cores=1)
    ale_meta.fit(testdata1)
    assert isinstance(ale_meta.results, nimare.base.MetaResult)


def test_ale_subtraction(testdata1, testdata2):
    """
    Smoke test for ALE
    """
    ale_meta = ale.ALE(n_iters=5, n_cores=1)
    ale_meta.fit(testdata1, dataset2=testdata2)
    assert isinstance(ale_meta.results, nimare.base.MetaResult)
