"""Check CBMR against an independent GLM implementation.

CBMR is a Poisson GLM with a fixed spline design. Its likelihood
(:meth:`~nimare.meta.models.PoissonEstimator._log_likelihood_single_group`) factors the
sum over experiments and voxels into marginals, which is a computational optimization
rather than a different model::

    Sum_v y_.v s_v + Sum_i y_i. m_i - (Sum_v exp(s_v))(Sum_i exp(m_i))
      == Sum_{i,v} [y_iv eta_iv - exp(eta_iv)]     for eta_iv = s_v + m_i

So statsmodels can fit exactly the same model from the materialized design, with an
independent optimizer (IRLS rather than L-BFGS) and independent covariance code. These
tests are an external oracle: the CBMR-specific fixtures elsewhere prove that behavior is
*unchanged*, while these prove it is *correct*.

Every test here simulates counts from a known model whose maximum likelihood estimate is
interior. That precondition is not incidental -- see
:func:`test_simulated_counts_have_an_interior_maximum` for why it is asserted rather than
assumed.
"""

import warnings

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except ImportError:
    warnings.warn("Torch not installed. CBMR GLM equivalence tests will be skipped.", stacklevel=2)
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta import models

statsmodels_api = pytest.importorskip("statsmodels.api")

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_VOXELS = 40
N_BASES = 6
N_MODERATORS = 2
GROUP_SIZES = {"A": 25, "B": 20}
MODERATOR_NAMES = ["moderator_0", "moderator_1"]


def _simulate_interior_poisson(seed=11):
    """Simulate CBMR-shaped Poisson counts whose MLE is interior.

    Rates are chosen so that counts average around three per experiment-voxel cell. Sparse
    0/1 data -- which is what :func:`nimare.generate.create_coordinate_studyset` produces,
    since no experiment reports two foci in one voxel -- drives the Poisson MLE to the
    boundary instead, making coefficients diverge rather than converge.
    """
    rng = np.random.default_rng(seed)
    groups = list(GROUP_SIZES)

    # Rows sum to one, as a cubic B-spline basis does (it is a partition of unity).
    bases = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    bases /= bases.sum(axis=1, keepdims=True)

    spatial_coef = {
        "A": rng.normal(1.5, 0.4, N_BASES),
        "B": rng.normal(1.2, 0.4, N_BASES),
    }
    moderator_coef = np.array([0.25, -0.15])
    moderators = {g: rng.normal(size=(GROUP_SIZES[g], N_MODERATORS)) for g in groups}

    foci = {}
    for group in groups:
        eta = (bases @ spatial_coef[group])[None, :] + (moderators[group] @ moderator_coef)[
            :, None
        ]
        foci[group] = rng.poisson(np.exp(eta))

    return {
        "groups": groups,
        "bases": bases,
        "moderators": moderators,
        "foci": foci,
        "foci_per_voxel": {g: foci[g].sum(axis=0).astype(float) for g in groups},
        "foci_per_experiment": {g: foci[g].sum(axis=1).astype(float) for g in groups},
        "true_spatial_coef": spatial_coef,
        "true_moderator_coef": moderator_coef,
    }


def _materialize_design(sim):
    """Return the (experiment x voxel) design matrix and response CBMR implies.

    One column block of spline bases per group, plus shared moderator columns -- the
    pooling of moderator coefficients across groups that ``moderators_linear`` being a
    single ``Linear`` (rather than a ``ModuleDict``) encodes.
    """
    groups = sim["groups"]
    bases = sim["bases"]
    n_columns = len(groups) * N_BASES + N_MODERATORS

    blocks, responses = [], []
    for index, group in enumerate(groups):
        n_experiments = GROUP_SIZES[group]
        block = np.zeros((n_experiments * N_VOXELS, n_columns))
        block[:, index * N_BASES : (index + 1) * N_BASES] = np.tile(bases, (n_experiments, 1))
        block[:, len(groups) * N_BASES :] = np.repeat(sim["moderators"][group], N_VOXELS, axis=0)
        blocks.append(block)
        responses.append(sim["foci"][group].ravel())

    return np.vstack(blocks), np.concatenate(responses)


def _fit_statsmodels(sim):
    """Fit the equivalent Poisson GLM with statsmodels."""
    design, response = _materialize_design(sim)
    return statsmodels_api.GLM(
        response,
        design,
        family=statsmodels_api.families.Poisson(),
    ).fit()


def _fit_cbmr(sim, n_iter=3000, tol=1e-11):
    """Fit the same data with CBMR's Poisson estimator, to convergence."""
    estimator = models.PoissonEstimator(
        penalty=False, lr=1.0, lr_decay=0.999, n_iter=n_iter, tol=tol, device="cpu"
    )
    estimator.init_weights(
        groups=sim["groups"],
        moderators=MODERATOR_NAMES,
        spatial_coef_dim=N_BASES,
        moderators_coef_dim=N_MODERATORS,
    )
    estimator.fit(
        sim["bases"],
        sim["moderators"],
        sim["foci_per_voxel"],
        sim["foci_per_experiment"],
    )
    return estimator


def _cbmr_coefficients(estimator, sim):
    """Extract CBMR's fitted coefficients in the layout ``_materialize_design`` uses."""
    spatial = {
        group: estimator.spatial_coef_linears[group].weight.detach().cpu().numpy().ravel()
        for group in sim["groups"]
    }
    moderator = estimator.moderators_linear.weight.detach().cpu().numpy().ravel()
    return spatial, moderator


def test_simulated_counts_have_an_interior_maximum():
    """The simulation must stay in the regime where the Poisson MLE actually exists.

    Guards the premise of every other test in this module. With the sparse 0/1 counts that
    coordinate-based data naturally produces, the Poisson likelihood is maximized only as
    coefficients run to infinity, and both CBMR and statsmodels then wander off together
    -- agreeing with each other while telling you nothing. If someone retunes the
    simulation into that regime, this fails first and says why.
    """
    sim = _simulate_interior_poisson()
    _, response = _materialize_design(sim)

    assert response.mean() > 1.0, "counts too sparse; the MLE will run to the boundary"
    assert (response == 0).mean() < 0.5, "mostly zeros; expect Poisson separation"

    result = _fit_statsmodels(sim)
    assert result.converged
    # exp(20) already overflows a float32 intensity map; a real interior optimum is small.
    assert np.abs(result.params).max() < 20.0


def test_cbmr_poisson_coefficients_match_statsmodels_glm():
    """The L-BFGS fit must reach the same optimum as statsmodels' IRLS."""
    sim = _simulate_interior_poisson()
    expected = _fit_statsmodels(sim)
    spatial, moderator = _cbmr_coefficients(_fit_cbmr(sim), sim)

    for index, group in enumerate(sim["groups"]):
        np.testing.assert_allclose(
            spatial[group],
            expected.params[index * N_BASES : (index + 1) * N_BASES],
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"spatial coefficients disagree for group {group}",
        )
    np.testing.assert_allclose(
        moderator,
        expected.params[len(sim["groups"]) * N_BASES :],
        rtol=1e-4,
        atol=1e-5,
    )


def test_cbmr_poisson_log_likelihood_matches_statsmodels_glm():
    """The factored CBMR likelihood must equal the elementwise Poisson log-likelihood.

    CBMR drops the ``-sum(log(y!))`` term, which is constant in the parameters, so it is
    added back here before comparing.
    """
    from scipy.special import gammaln

    sim = _simulate_interior_poisson()
    expected = _fit_statsmodels(sim)
    estimator = _fit_cbmr(sim)
    spatial, moderator = _cbmr_coefficients(estimator, sim)

    _, response = _materialize_design(sim)
    log_likelihood = 0.0
    for group in sim["groups"]:
        spatial_term = sim["bases"] @ spatial[group]
        moderator_term = sim["moderators"][group] @ moderator
        log_likelihood += (
            sim["foci_per_voxel"][group] @ spatial_term
            + sim["foci_per_experiment"][group] @ moderator_term
            - np.exp(spatial_term).sum() * np.exp(moderator_term).sum()
        )
    log_likelihood -= gammaln(response + 1).sum()

    np.testing.assert_allclose(log_likelihood, expected.llf, rtol=1e-6)


def test_cbmr_poisson_spatial_standard_errors_match_statsmodels_fisher():
    """Spatial inverse-Fisher standard errors must match statsmodels' nonrobust ones."""
    sim = _simulate_interior_poisson()
    expected = _fit_statsmodels(sim)
    estimator = _fit_cbmr(sim)
    _, tables = estimator.summary()

    spatial_se = tables["spatial_regression_coef_se"]
    for index, group in enumerate(sim["groups"]):
        np.testing.assert_allclose(
            np.asarray(spatial_se.loc[group], dtype=float).ravel(),
            expected.bse[index * N_BASES : (index + 1) * N_BASES],
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"spatial standard errors disagree for group {group}",
        )


def test_cbmr_poisson_moderator_standard_errors_match_statsmodels_fisher():
    """Pooled moderator standard errors must use every group's information.

    Regression test. These were previously computed from only the last group, because
    ``ll_single_group_kwargs`` and ``group_spatial_coef`` leaked out of the
    ``for group in self.groups`` loop in ``standard_error_estimation`` -- inflating them by
    roughly ``sqrt(n_total / n_last_group)`` while inverting the moderator block alone
    deflated them ~6%, netting a 46% overestimate on this simulation.
    """
    sim = _simulate_interior_poisson()
    expected = _fit_statsmodels(sim)
    estimator = _fit_cbmr(sim)
    _, tables = estimator.summary()

    np.testing.assert_allclose(
        np.asarray(tables["moderators_regression_se"], dtype=float).ravel(),
        expected.bse[len(sim["groups"]) * N_BASES :],
        rtol=1e-6,
        atol=1e-9,
    )
