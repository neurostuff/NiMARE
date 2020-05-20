"""
Test nimare.meta.ibma (image-based meta-analytic algorithms).
"""
import pytest

import nimare
from nimare.meta import ibma


def test_FFX_GLM():
    """
    Smoke test for FFX GLM.
    """
    meta = ibma.FFX_GLM()
    res = meta.fit(pytest.dset_conse)
    assert isinstance(res, nimare.base.MetaResult)


def test_MFX_GLM():
    """
    Smoke test for MFX GLM.
    """
    meta = ibma.MFX_GLM()
    res = meta.fit(pytest.dset_conse)
    assert isinstance(res, nimare.base.MetaResult)


def test_ffx_glm():
    """
    Smoke test for FFX GLM.
    """
    result = ibma.ffx_glm(pytest.data_con, pytest.data_se,
                          pytest.sample_sizes_con, pytest.dset_conse.mask)
    assert isinstance(result, dict)


def test_mfx_glm():
    """
    Smoke test for MFX GLM.
    """
    result = ibma.mfx_glm(pytest.data_con, pytest.data_se,
                          pytest.sample_sizes_con, pytest.dset_conse.mask)
    assert isinstance(result, dict)
