"""
Test nimare.meta.esma (effect-size meta-analytic algorithms).
"""
import pytest

from nimare.meta import esma


def test_z_perm():
    """
    Smoke test for z permutation.
    """
    result = esma.stouffers(pytest.data_z, inference='rfx', null='empirical',
                            n_iters=10)
    assert isinstance(result, dict)


def test_stouffers_ffx():
    """
    Smoke test for Stouffer's FFX.
    """
    result = esma.stouffers(pytest.data_z, inference='ffx', null='theoretical',
                            n_iters=None)
    assert isinstance(result, dict)


def test_stouffers_rfx():
    """
    Smoke test for Weighted Stouffer's.
    """
    result = esma.stouffers(pytest.data_z, inference='rfx', null='theoretical',
                            n_iters=None)
    assert isinstance(result, dict)


def test_weighted_stouffers():
    """
    Smoke test for Stouffer's RFX.
    """
    result = esma.weighted_stouffers(pytest.data_z, pytest.sample_sizes_z)
    assert isinstance(result, dict)


def test_fishers():
    """
    Smoke test for Fisher's.
    """
    result = esma.fishers(pytest.data_z)
    assert isinstance(result, dict)


def test_con_perm():
    """
    Smoke test for contrast permutation.
    """
    result = esma.rfx_glm(pytest.data_con, null='empirical', n_iters=10)
    assert isinstance(result, dict)


def test_rfx_glm():
    """
    Smoke test for RFX GLM.
    """
    result = esma.rfx_glm(pytest.data_con, null='theoretical', n_iters=None)
    assert isinstance(result, dict)
