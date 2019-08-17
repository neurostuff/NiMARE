"""
Test nimare.meta.cbma.ale (SCALE meta-analytic algorithm).
"""
import pytest
import numpy as np

import nimare
from nimare.meta.cbma import ale


def test_scale():
    """
    Smoke test for SCALE
    """
    ijk = np.vstack(np.where(pytest.cbma_testdata3.masker.mask_img.get_data())).T
    ijk = ijk[:, :20]
    meta = ale.SCALE(n_iters=5, n_cores=1, ijk=ijk)
    res = meta.fit(pytest.cbma_testdata3)
    assert isinstance(res, nimare.base.MetaResult)
