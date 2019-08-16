"""
Test nimare.meta.esma (effect-size meta-analytic algorithms).
"""
import os.path as op

import pytest
import pandas as pd

from nimare.meta import esma
from .utils import get_test_data_path


@pytest.fixture(scope='module')
def stroop_data():
    path = op.join(get_test_data_path(), 'vonBastian_stroop.csv')
    return pd.read_csv(path)


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


def test_stan_mfx_only_group_level_estimates(stroop_data):
    estimates = stroop_data.groupby('ID')['rt'].mean()
    results = esma.stan_mfx(estimates)
    names = {'estimate', 'estimate_sd', 'tau', 'tau_sd'}
    assert set(list(results.keys())) == names
    assert results['mu'].shape == (5,)
