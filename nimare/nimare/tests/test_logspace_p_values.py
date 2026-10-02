"""The log-space p-value path must report deeper tails without moving any threshold.

Every conversion this covers is a monotone reparameterization of one tail probability, so
thresholding on ``p``, on ``logp`` or on ``z`` has to keep selecting the same voxels. These
tests hold the three maps of an estimator against each other; the per-conversion accuracy
tests live in ``test_transforms.py`` and ``test_stats.py``.
"""

import numpy as np
import pytest

from nimare.meta import ale, ibma, mkda
from nimare.utils import _minimum_positive_float

_P_FLOOR = _minimum_positive_float()
_LOGP_AT_FLOOR = -np.log10(_P_FLOOR)


@pytest.fixture(scope="module")
def cbma_maps(testdata_cbma):
    """Uncorrected p, z and logp maps from a coordinate-based estimator."""
    return ale.ALE(null_method="approximate").fit(testdata_cbma).maps


@pytest.fixture(scope="module")
def mkda_maps(testdata_cbma):
    """Uncorrected p, z and logp maps from MKDA density."""
    return mkda.MKDADensity(null_method="approximate").fit(testdata_cbma).maps


@pytest.fixture(scope="module")
def fishers_maps(testdata_ibma):
    """Uncorrected p, z and logp maps from a combination test."""
    return ibma.Fishers().fit(testdata_ibma).maps


@pytest.fixture(scope="module")
def dsl_maps(testdata_ibma):
    """Uncorrected p, z and logp maps from a PyMARE-backed regression estimator."""
    return ibma.DerSimonianLaird().fit(testdata_ibma).maps


_MAP_FIXTURES = ["cbma_maps", "mkda_maps", "fishers_maps", "dsl_maps"]


@pytest.mark.parametrize("maps_fixture", _MAP_FIXTURES)
@pytest.mark.parametrize("alpha", [0.05, 0.001])
def test_thresholding_on_p_and_on_logp_select_the_same_voxels(maps_fixture, alpha, request):
    """The two routes workflows and reports threshold by must not disagree."""
    maps = request.getfixturevalue(maps_fixture)
    p, logp = maps["p"], maps["logp"]

    covered = ~np.isnan(p)
    by_p = p[covered] <= alpha
    by_logp = logp[covered] >= -np.log10(alpha)

    assert np.array_equal(by_p, by_logp)


@pytest.mark.parametrize("maps_fixture", _MAP_FIXTURES)
def test_logp_agrees_with_p_until_p_runs_out(maps_fixture, request):
    """Above the float32 floor the two maps are the same number, below it only logp is."""
    maps = request.getfixturevalue(maps_fixture)
    p, logp = maps["p"], maps["logp"]

    above_floor = ~np.isnan(p) & (p > _P_FLOOR)
    assert above_floor.any()
    assert np.allclose(logp[above_floor], -np.log10(p[above_floor]), atol=1e-4)

    at_floor = ~np.isnan(p) & (p == _P_FLOOR)
    assert np.all(logp[at_floor] >= _LOGP_AT_FLOOR), "p bottomed out, so logp must carry on"


@pytest.mark.parametrize("maps_fixture", _MAP_FIXTURES)
def test_no_map_carries_a_non_finite_statistic(maps_fixture, request):
    """An infinite z breaks thresholding, plotting and any maximum over the map."""
    maps = request.getfixturevalue(maps_fixture)

    for name in ("p", "z", "logp"):
        values = maps[name]
        assert not np.isinf(values).any(), name
