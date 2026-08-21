"""Tests for the CBMR observation distributions.

The point of these is fidelity. Each distribution here is a port of a likelihood that already
existed as a method on :mod:`nimare.meta.models`, rewritten to consume the predictor's spatial
patterns instead of a hard-coded notion of groups. A port is only worth having if it computes
the same number, so most of what follows compares the two directly.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta import models
    from nimare.meta.cbmr.distributions import (
        ClusteredNegativeBinomial,
        DistributionError,
        NegativeBinomial,
        Poisson,
        resolve_distribution,
    )
    from nimare.meta.cbmr.predictor import CBMRPredictor
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_GROUP = 5
N_VOXELS = 24
N_BASES = 4
GROUPS = ("schiz", "dep")


@pytest.fixture
def setup():
    """Build a grouped design, a basis, coefficients, and Poisson-drawn foci."""
    rng = np.random.default_rng(17)
    n_experiments = N_PER_GROUP * len(GROUPS)
    annotations = pd.DataFrame(
        {
            "diagnosis": [g for g in GROUPS for _ in range(N_PER_GROUP)],
            "n": rng.normal(size=n_experiments),
        }
    )
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    bases = raw / raw.sum(axis=1, keepdims=True)

    predictor = CBMRPredictor(bind(Design.from_formula("~ s(diagnosis) + n"), annotations), bases)
    spatial = torch.tensor(
        rng.normal(1.0, 0.2, (predictor.n_spatial_columns, N_BASES)), dtype=torch.float64
    )
    global_coef = torch.tensor(rng.normal(0.0, 0.1, 1), dtype=torch.float64)

    with torch.no_grad():
        eta = predictor.linear_predictor(spatial, global_coef).numpy()
    foci = rng.poisson(np.exp(eta)).astype(float)
    return predictor, spatial, global_coef, foci, annotations


def _legacy_pieces(predictor, foci):
    """Split foci into the per-group lists the legacy models expect.

    For a grouped design the predictor's patterns *are* the groups, which is what makes the
    port a like-for-like comparison at all.
    """
    per_voxel, per_experiment, members = [], [], []
    for pattern in range(predictor.patterns.n_patterns):
        rows = np.flatnonzero(predictor.patterns.assignment == pattern)
        members.append(rows)
        per_voxel.append(torch.as_tensor(foci[rows].sum(axis=0), dtype=torch.float64))
        per_experiment.append(torch.as_tensor(foci[rows].sum(axis=1), dtype=torch.float64))
    return per_voxel, per_experiment, members


def _legacy_spatial_coef(predictor, spatial):
    """Return per-pattern spatial coefficients in the legacy models' layout.

    A pattern's effective coefficient is its loading times the term coefficients. Necessary
    because np.unique sorts the pattern rows, so pattern order is not patsy's column order --
    comparing them positionally would silently pair the wrong group with the wrong foci.
    """
    loadings = torch.as_tensor(predictor.patterns.loadings, dtype=spatial.dtype)
    per_pattern = loadings @ spatial
    return per_pattern.reshape(per_pattern.shape[0], per_pattern.shape[1], 1)


def test_patterns_recover_the_groups(setup):
    """A grouped design must give exactly one spatial pattern per group."""
    predictor = setup[0]
    assert predictor.patterns.n_patterns == len(GROUPS)
    assert not predictor.patterns.is_degenerate


def test_poisson_matches_the_legacy_multigroup_likelihood(setup):
    """The ported Poisson likelihood must equal models.PoissonEstimator's."""
    predictor, spatial, global_coef, foci, _ = setup
    per_voxel, per_experiment, members = _legacy_pieces(predictor, foci)

    actual = Poisson().log_likelihood(predictor, spatial, global_coef, None, foci)

    legacy = models.PoissonEstimator(device="cpu")
    expected = legacy._log_likelihood_mult_group(
        spatial_coef=_legacy_spatial_coef(predictor, spatial),
        moderator_coef=global_coef.reshape(1, 1),
        coef_spline_bases=torch.as_tensor(predictor.bases, dtype=torch.float64),
        foci_per_voxel=per_voxel,
        foci_per_experiment=per_experiment,
        moderators=[
            torch.as_tensor(predictor.global_block[rows], dtype=torch.float64) for rows in members
        ],
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize(
    "distribution,legacy_class",
    [
        (NegativeBinomial, models.NegativeBinomialEstimator if TORCH_INSTALLED else None),
        (
            ClusteredNegativeBinomial,
            models.ClusteredNegativeBinomialEstimator if TORCH_INSTALLED else None,
        ),
    ],
)
def test_overdispersed_likelihoods_match_their_legacy_versions(setup, distribution, legacy_class):
    """Both overdispersion ports must equal the marginal likelihoods they came from.

    These are the models that cannot be written for a non-separable predictor at all, so the
    port only had to carry them over from "group" to "pattern". Any drift here would be a
    silently different model.
    """
    predictor, spatial, global_coef, foci, _ = setup
    per_voxel, per_experiment, members = _legacy_pieces(predictor, foci)
    overdispersion = torch.tensor([0.03, 0.07], dtype=torch.float64)

    actual = distribution().log_likelihood(predictor, spatial, global_coef, overdispersion, foci)

    legacy = legacy_class(device="cpu")
    expected = legacy._log_likelihood_mult_group(
        overdispersion_coef=[overdispersion[i] for i in range(len(GROUPS))],
        spatial_coef=_legacy_spatial_coef(predictor, spatial),
        coef_spline_bases=torch.as_tensor(predictor.bases, dtype=torch.float64),
        foci_per_voxel=per_voxel,
        foci_per_experiment=per_experiment,
        moderator_coef=global_coef.reshape(1, 1),
        moderators=[
            torch.as_tensor(predictor.global_block[rows], dtype=torch.float64) for rows in members
        ],
    )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-7)


def test_overdispersion_needs_experiments_sharing_a_pattern(setup):
    """A spatial covariate leaves one experiment per pattern, which cannot fit overdispersion.

    The precise form of the constraint the old API stated as "voxelwise CBMR requires
    model=PoissonEstimator". What matters is not the moderator's kind but whether any
    experiments share a spatial map.
    """
    _, _, _, _, annotations = setup
    rng = np.random.default_rng(18)
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(n)"), annotations), raw / raw.sum(axis=1, keepdims=True)
    )

    with pytest.raises(DistributionError, match="single\n?.*experiment|single experiment"):
        NegativeBinomial().check_design(predictor)


def test_poisson_accepts_any_design(setup):
    """Poisson is defined per cell, so no design is out of reach for it."""
    _, _, _, _, annotations = setup
    rng = np.random.default_rng(19)
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(n)"), annotations), raw / raw.sum(axis=1, keepdims=True)
    )
    Poisson().check_design(predictor)  # must not raise


def test_grouped_design_passes_the_overdispersion_check(setup):
    """The check must not fire on the designs overdispersion is meant for."""
    NegativeBinomial().check_design(setup[0])
    ClusteredNegativeBinomial().check_design(setup[0])


def test_nuisance_parameter_shapes(setup):
    """Poisson owns none; the overdispersed models own one per pattern."""
    predictor = setup[0]
    n_patterns = predictor.patterns.n_patterns

    assert Poisson().n_nuisance_parameters(n_patterns) == 0
    assert Poisson().initial_nuisance(n_patterns) is None
    for distribution in (NegativeBinomial(), ClusteredNegativeBinomial()):
        assert distribution.n_nuisance_parameters(n_patterns) == n_patterns
        assert distribution.initial_nuisance(n_patterns).shape == (n_patterns,)


@pytest.mark.parametrize(
    "given,expected",
    [
        ("poisson", "Poisson"),
        ("Poisson", "Poisson"),
        ("negative_binomial", "NegativeBinomial"),
        ("clustered-negative-binomial", "ClusteredNegativeBinomial"),
    ],
)
def test_distributions_resolve_from_names(given, expected):
    """Names, classes, and instances should all be accepted."""
    assert resolve_distribution(given).name == expected
    assert resolve_distribution(type(resolve_distribution(given))).name == expected


def test_unknown_distribution_names_the_options():
    """A typo should list what was available."""
    with pytest.raises(DistributionError, match="Unknown distribution"):
        resolve_distribution("poison")
    with pytest.raises(DistributionError, match="Cannot interpret"):
        resolve_distribution(42)


def test_overdispersion_gradients_flow(setup):
    """Fitting needs gradients for the overdispersion parameters too."""
    predictor, spatial, global_coef, foci, _ = setup
    spatial = spatial.clone().requires_grad_(True)
    overdispersion = torch.tensor([0.05, 0.05], dtype=torch.float64, requires_grad=True)

    NegativeBinomial().log_likelihood(
        predictor, spatial, global_coef, overdispersion, foci
    ).backward()

    assert overdispersion.grad is not None
    assert torch.all(torch.isfinite(overdispersion.grad))
    assert torch.any(overdispersion.grad != 0)
