"""
Test nimare.meta.esma (effect-size meta-analytic algorithms).
"""
from nimare.meta import esma


def test_z_perm(testdata):
    """
    Smoke test for z permutation.
    """
    result = esma.stouffers(testdata['data_z'], inference='rfx', null='empirical',
                            n_iters=10)
    assert isinstance(result, dict)


def test_stouffers_ffx(testdata):
    """
    Smoke test for Stouffer's FFX.
    """
    result = esma.stouffers(testdata['data_z'], inference='ffx', null='theoretical',
                            n_iters=None)
    assert isinstance(result, dict)


def test_stouffers_rfx(testdata):
    """
    Smoke test for Weighted Stouffer's.
    """
    result = esma.stouffers(testdata['data_z'], inference='rfx', null='theoretical',
                            n_iters=None)
    assert isinstance(result, dict)


def test_weighted_stouffers(testdata):
    """
    Smoke test for Stouffer's RFX.
    """
    result = esma.weighted_stouffers(testdata['data_z'], testdata['sample_sizes_z'])
    assert isinstance(result, dict)


def test_fishers(testdata):
    """
    Smoke test for Fisher's.
    """
    result = esma.fishers(testdata['data_z'])
    assert isinstance(result, dict)


def test_con_perm(testdata):
    """
    Smoke test for contrast permutation.
    """
    result = esma.rfx_glm(testdata['data_beta'], null='empirical', n_iters=10)
    assert isinstance(result, dict)


def test_rfx_glm(testdata):
    """
    Smoke test for RFX GLM.
    """
    result = esma.rfx_glm(testdata['data_beta'], null='theoretical', n_iters=None)
    assert isinstance(result, dict)
