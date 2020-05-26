"""
Test nimare.meta.ibma (image-based meta-analytic algorithms).
"""
import nimare
from nimare.meta import ibma


def test_FFX_GLM(testdata):
    """
    Smoke test for FFX GLM.
    """
    meta = ibma.FFX_GLM()
    res = meta.fit(testdata['data_conse'])
    assert isinstance(res, nimare.base.MetaResult)


def test_MFX_GLM(testdata):
    """
    Smoke test for MFX GLM.
    """
    meta = ibma.MFX_GLM()
    res = meta.fit(testdata['data_conse'])
    assert isinstance(res, nimare.base.MetaResult)


def test_ffx_glm(testdata):
    """
    Smoke test for FFX GLM.
    """
    result = ibma.ffx_glm(testdata['data_con'], testdata['data_se'],
                          testdata['sample_sizes_con'], testdata['data_conse'].masker)
    assert isinstance(result, dict)


def test_mfx_glm(testdata):
    """
    Smoke test for MFX GLM.
    """
    result = ibma.mfx_glm(testdata['data_con'], testdata['data_se'],
                          testdata['sample_sizes_con'], testdata['data_conse'].masker)
    assert isinstance(result, dict)
