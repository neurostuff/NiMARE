"""
Test nimare.meta.ibma (image-based meta-analytic estimators).
"""
import pytest

import nimare
from nimare.meta import ibma


def test_fishers():
    """
    Smoke test for Fisher's.
    """
    meta = ibma.Fishers()
    meta.fit(pytest.dset_z)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_z_perm():
    """
    Smoke test for z permutation.
    """
    meta = ibma.Stouffers(inference='rfx', null='empirical', n_iters=10)
    meta.fit(pytest.dset_z)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_stouffers_ffx():
    """
    Smoke test for Stouffer's FFX.
    """
    meta = ibma.Stouffers(inference='ffx', null='theoretical', n_iters=None)
    meta.fit(pytest.dset_z)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_stouffers_rfx():
    """
    Smoke test for Stouffer's RFX.
    """
    meta = ibma.Stouffers(inference='rfx', null='theoretical', n_iters=None)
    meta.fit(pytest.dset_z)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_weighted_stouffers():
    """
    Smoke test for Weighted Stouffer's.
    """
    meta = ibma.WeightedStouffers()
    meta.fit(pytest.dset_z)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_con_perm():
    """
    Smoke test for contrast permutation.
    """
    meta = ibma.RFX_GLM(null='empirical', n_iters=10)
    meta.fit(pytest.dset_conse)
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_rfx_glm():
    """
    Smoke test for RFX GLM.
    """
    meta = ibma.RFX_GLM(null='theoretical', n_iters=None)
    meta.fit(pytest.dset_conse)
    assert isinstance(meta.results, nimare.base.MetaResult)
