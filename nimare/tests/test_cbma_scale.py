"""
Test nimare.meta.cbma.ale (SCALE meta-analytic algorithm).
"""
import numpy as np

import nimare
from nimare.meta.cbma import ale


def test_scale(testdata):
    """
    Smoke test for SCALE
    """
    dset = testdata['dset'].slice(testdata['dset'].ids[:3])
    ijk = np.vstack(np.where(testdata['dset'].masker.mask_img.get_fdata())).T
    ijk = ijk[:, :20]
    meta = ale.SCALE(n_iters=5, n_cores=1, ijk=ijk)
    res = meta.fit(dset)
    assert isinstance(res, nimare.base.MetaResult)
