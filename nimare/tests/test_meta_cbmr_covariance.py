"""Tests for CBMR's robust covariance estimators.

statsmodels implements exactly these sandwich estimators, so each one here is checked against
it on the materialized design. That is the whole reason to trust an implementation that never
forms the design: agreement with one that does.
"""

import numpy as np
import pandas as pd
import pytest

statsmodels_api = pytest.importorskip("statsmodels.api")

try:
    import torch
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr.covariance import CovarianceError, sandwich_covariance
    from nimare.meta.cbmr.distributions import Poisson
    from nimare.meta.cbmr.model import CBMRModel
    from nimare.meta.cbmr.predictor import CBMRPredictor
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_GROUP = 12
N_VOXELS = 18
N_BASES = 4


@pytest.fixture(scope="module")
def fitted():
    """Fit a grouped design with a scalar moderator on interior-MLE counts."""
    rng = np.random.default_rng(83)
    n_experiments = 2 * N_PER_GROUP
    annotations = pd.DataFrame(
        {
            "dx": ["a"] * N_PER_GROUP + ["b"] * N_PER_GROUP,
            "n": rng.normal(size=n_experiments),
        }
    )
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    bases = raw / raw.sum(axis=1, keepdims=True)

    predictor = CBMRPredictor(bind(Design.from_formula("~ s(dx) + n"), annotations), bases)
    spatial = torch.tensor(
        rng.normal(1.4, 0.25, (predictor.n_spatial_columns, N_BASES)), dtype=torch.float64
    )
    global_coef = torch.tensor(rng.normal(0.0, 0.15, 1), dtype=torch.float64)
    with torch.no_grad():
        eta = predictor.linear_predictor(spatial, global_coef).numpy()
    foci = rng.poisson(np.exp(eta)).astype(float)

    model = CBMRModel(predictor, Poisson()).fit(foci, n_iter=3000, tol=1e-13)
    return model, foci, annotations


def _materialize(predictor):
    n_experiments = predictor.patterns.n_experiments
    spatial = np.einsum("ic,vb->ivcb", predictor.spatial_block, predictor.bases).reshape(
        n_experiments * predictor.n_voxels, -1
    )
    return np.hstack([spatial, np.repeat(predictor.global_block, predictor.n_voxels, axis=0)])


def _statsmodels(model, foci, **fit_kwargs):
    return statsmodels_api.GLM(
        foci.reshape(-1),
        _materialize(model.predictor),
        family=statsmodels_api.families.Poisson(),
    ).fit(**fit_kwargs)


def _reference_sandwich(model, foci, correction):
    """Compute the sandwich directly from the materialized design, per its definition.

    An independent reference for the corrections statsmodels does not implement. With ``X`` the
    explicit design, ``W = diag(mu)``, and ``r = y - mu``::

        A = X' W X                          bread
        hc0:  M = X' diag(r^2) X
        hc1:  M = hc0 meat * n / (n - p)
        hc3:  M = X' diag((r / (1 - h))^2) X,   h_i = mu_i x_i' A^-1 x_i
        V = A^-1 M A^-1

    The point of the comparison is that the shipped implementation never forms ``X``, assembling
    every quantity from the design's Kronecker structure instead.
    """
    design = _materialize(model.predictor)
    coefficients = model.coefficients.detach().numpy()
    mean = np.exp(design @ coefficients)
    residuals = foci.reshape(-1) - mean

    bread = design.T @ (design * mean[:, None])
    bread_inverse = np.linalg.inv(bread)

    if correction == "hc3":
        leverage = mean * np.einsum("ij,jk,ik->i", design, bread_inverse, design, optimize=True)
        residuals = residuals / (1.0 - leverage)

    meat = design.T @ (design * (residuals**2)[:, None])
    if correction == "hc1":
        n_observations, n_parameters = design.shape
        meat = meat * (n_observations / (n_observations - n_parameters))
    return bread_inverse @ meat @ bread_inverse


def test_iid_hc0_matches_statsmodels(fitted):
    """Cell-wise HC0 must equal statsmodels' HC0.

    Only HC0 is checked against statsmodels here, because its GLM implementation silently
    collapses HC1 and HC3 onto HC0 -- passing ``cov_type="HC3"`` to ``sm.GLM`` returns a matrix
    identical to HC0, verified on an independent Poisson fit. The other two corrections are
    checked against :func:`_reference_sandwich` instead.
    """
    model, foci, _ = fitted
    actual = sandwich_covariance(model, foci, meat="iid", correction="hc0")
    expected = _statsmodels(model, foci, cov_type="HC0").cov_params()

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-9)


def test_statsmodels_glm_does_not_distinguish_the_hc_variants(fitted):
    """Pin the reason HC1 and HC3 are not compared against statsmodels.

    If a future statsmodels implements them properly for GLM, this fails and the comparison
    above can be widened -- which is better than leaving a silent gap in the oracle.
    """
    model, foci, _ = fitted
    matrices = {
        variant: np.asarray(_statsmodels(model, foci, cov_type=variant).cov_params())
        for variant in ("HC0", "HC1", "HC3")
    }
    np.testing.assert_allclose(matrices["HC1"], matrices["HC0"], rtol=1e-12)
    np.testing.assert_allclose(matrices["HC3"], matrices["HC0"], rtol=1e-12)


@pytest.mark.parametrize("correction", ["hc0", "hc1", "hc3"])
def test_iid_sandwich_matches_the_direct_computation(fitted, correction):
    """The structured assembly must equal the definition applied to explicit design rows.

    Covers the leverage machinery: hc3 divides each residual by ``1 - h_iv``, and the leverages
    are built from the design's Kronecker structure rather than from its rows.
    """
    model, foci, _ = fitted
    actual = sandwich_covariance(model, foci, meat="iid", correction=correction)
    expected = _reference_sandwich(model, foci, correction)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)


def test_cluster_sandwich_matches_statsmodels(fitted):
    """Clustering by experiment must equal statsmodels' cluster covariance.

    The realistic assumption for coordinate data: an experiment's foci are correlated with each
    other and independent of other experiments'.
    """
    model, foci, _ = fitted
    n_experiments = model.predictor.patterns.n_experiments
    groups = np.repeat(np.arange(n_experiments), N_VOXELS)

    actual = sandwich_covariance(model, foci, meat="cluster", correction="hc0")
    expected = _statsmodels(
        model, foci, cov_type="cluster", cov_kwds={"groups": groups, "use_correction": False}
    ).cov_params()

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-6, atol=1e-10)


def test_cluster_standard_errors_exceed_model_based_ones(fitted):
    """Clustering should widen intervals, which is the point of using it.

    Foci within an experiment are correlated, so treating cells as independent understates
    uncertainty. If this ever reversed, the estimator would be giving false comfort.
    """
    model, foci, _ = fitted
    model_based = np.sqrt(np.diag(model.covariance(foci)))
    clustered = np.sqrt(np.diag(sandwich_covariance(model, foci, meat="cluster")))

    assert np.median(clustered / model_based) > 1.0


def test_sandwich_is_symmetric_and_positive_semidefinite(fitted):
    """A covariance matrix has to look like one."""
    model, foci, _ = fitted
    for meat, correction in (("cluster", "hc1"), ("iid", "hc3")):
        covariance = sandwich_covariance(model, foci, meat=meat, correction=correction)
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-10)
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert eigenvalues.min() > -1e-8 * max(1.0, eigenvalues.max())


def test_ridge_regularizes_the_bread(fitted):
    """A ridge should shrink the covariance, for designs whose information is near-singular."""
    model, foci, _ = fitted
    plain = sandwich_covariance(model, foci, meat="cluster", correction="hc0")
    ridged = sandwich_covariance(model, foci, meat="cluster", correction="hc0", ridge=1.0)

    assert np.trace(ridged) < np.trace(plain)


def test_cluster_with_hc3_is_refused(fitted):
    """Per-observation leverage has no meaning once observations are summed into clusters."""
    model, foci, _ = fitted
    with pytest.raises(CovarianceError, match="no counterpart"):
        sandwich_covariance(model, foci, meat="cluster", correction="hc3")


def test_invalid_options_are_reported(fitted):
    """Typos in the options should name the alternatives."""
    model, foci, _ = fitted
    with pytest.raises(CovarianceError, match="meat must be one of"):
        sandwich_covariance(model, foci, meat="sandwich")
    with pytest.raises(CovarianceError, match="correction must be one of"):
        sandwich_covariance(model, foci, correction="hc9")


def test_hc1_refuses_an_overparameterized_design(fitted):
    """``n / (n - p)`` is meaningless when p exceeds the number of clusters, so say so."""
    model, foci, annotations = fitted
    rng = np.random.default_rng(89)
    wide = rng.uniform(0.05, 1.0, (N_VOXELS, 40))
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(dx) + n"), annotations),
        wide / wide.sum(axis=1, keepdims=True),
    )
    wide_model = CBMRModel(predictor, Poisson())
    wide_foci = np.asarray(foci)

    with pytest.raises(CovarianceError, match="hc1 scales by"):
        sandwich_covariance(wide_model, wide_foci, meat="cluster", correction="hc1")
