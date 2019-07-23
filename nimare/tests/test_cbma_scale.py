"""
Test nimare.meta.cbma.ale (ALE/SCALE meta-analytic algorithms).
"""
import pytest
import numpy as np
import pandas as pd
import nibabel as nib

import nimare
from nimare.meta.cbma import ale
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
    mask_img = nib.Nifti1Image(np.ones((20, 20, 20), int), mask_img.affine)
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


def test_scale(testdata1):
    """
    Smoke test for SCALE
    """
    ijk = np.vstack(np.where(testdata1.mask.get_data())).T
    ijk = ijk[:, :100]
    scale_meta = ale.SCALE(testdata1, ijk=ijk)
    scale_meta.fit(n_iters=5, ids=testdata1.ids, n_cores=1)
    assert isinstance(scale_meta.results, nimare.base.MetaResult)
