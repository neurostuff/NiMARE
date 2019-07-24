"""
Test nimare.meta.cbma.mkda (KDA-based meta-analytic algorithms).
"""
import pytest
import pandas as pd

import nimare
from nimare.meta.cbma import mkda
from nimare.utils import get_template, mm2vox


# Fixtures used in the rest of the tests
class DummyDataset(object):
    def __init__(self, df, img):
        self.coordinates = df
        self.mask = img
        self.ids = self.coordinates['id'].unique()


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


def test_mkda_density(testdata1):
    """
    Smoke test for MKDADensity
    """
    mkda_meta = mkda.MKDADensity(n_iters=5, n_cores=1)
    mkda_meta.fit(testdata1)
    assert isinstance(mkda_meta.results, nimare.base.MetaResult)


def test_mkda_chi2_fdr(testdata1):
    """
    Smoke test for MKDAChi2
    """
    mkda_meta = mkda.MKDAChi2(corr='fdr', n_cores=1)
    mkda_meta.fit(testdata1, testdata1)
    assert isinstance(mkda_meta.results, nimare.base.MetaResult)


def test_mkda_chi2_fwe(testdata1):
    """
    Smoke test for MKDAChi2
    """
    mkda_meta = mkda.MKDAChi2(n_iters=5, corr='fwe', n_cores=1)
    mkda_meta.fit(testdata1, testdata1)
    assert isinstance(mkda_meta.results, nimare.base.MetaResult)


def test_kda_density(testdata1):
    """
    Smoke test for KDA
    """
    kda_meta = mkda.KDA(n_iters=5, n_cores=1)
    kda_meta.fit(testdata1)
    assert isinstance(kda_meta.results, nimare.base.MetaResult)
