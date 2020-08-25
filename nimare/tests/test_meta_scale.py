"""
Test nimare.meta.ale (SCALE meta-analytic algorithm).
"""
import numpy as np

import nimare
from nimare.meta import ale


def test_scale(testdata_cbma):
    """
    Smoke test for SCALE
    """
    dset = testdata_cbma.slice(testdata_cbma.ids[:3])
    ijk = np.vstack(np.where(testdata_cbma.masker.mask_img.get_fdata())).T
    ijk = ijk[:, :20]
    meta = ale.SCALE(n_iters=5, n_cores=1, ijk=ijk)
    res = meta.fit(dset)
    assert isinstance(res, nimare.results.MetaResult)
