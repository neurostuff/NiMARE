"""Optimizer backends for the voxelwise CBMR model.

The scipy L-BFGS-B path behind ``backend="approximate"``, which fits the additive
log-linear model directly rather than through torch. Moved out of ``nimare.meta.utils``,
where it sat interleaved with the unrelated ALE and KDA kernel machinery.
"""

import logging

import numpy as np
from scipy import optimize
from scipy import sparse as sp_sparse

LGR = logging.getLogger(__name__)


def _safe_exp(values):
    """Exponentiate after clipping to avoid numerical overflow."""
    return np.exp(np.clip(values, -100, 100))


def _spatial_cbmr_kron_vector_product(moderators, bases, coefficient):
    """Compute ``kron(moderators, bases) @ coefficient`` without forming ``kron``.

    Parameters
    ----------
    moderators : :obj:`numpy.ndarray` of shape ``(n_experiments, n_moderators)``
        Experiment-level design matrix.
    bases : :obj:`numpy.ndarray` of shape ``(n_voxels, n_bases)``
        Spatial B-spline basis matrix.
    coefficient : :obj:`numpy.ndarray` of shape ``(n_moderators * n_bases, 1)``
        Flattened spatially varying coefficient matrix.

    Returns
    -------
    :obj:`numpy.ndarray` of shape ``(n_experiments * n_voxels, 1)``
        Flattened linear predictor.
    """
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    coefficient = np.asarray(coefficient).reshape((n_moderators, n_bases))
    return (moderators @ coefficient @ bases.T).reshape((n_experiments * n_voxels, 1))


def _spatial_cbmr_log_poisson_nll(moderators, bases, coefficient, foci):
    """Return the Poisson negative log-likelihood for the approximate solver."""
    n_experiments = moderators.shape[0]
    n_voxels = bases.shape[0]
    eta = _spatial_cbmr_kron_vector_product(moderators, bases, coefficient).reshape(
        (n_experiments, n_voxels)
    )
    mean = _safe_exp(eta)
    if sp_sparse.issparse(foci):
        foci = foci.toarray()
    return -float(np.mean(foci * eta - mean))


def _spatial_cbmr_gradient(moderators, bases, coefficient, foci):
    """Compute the negative Poisson score for a spatially varying coefficient."""
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    if sp_sparse.issparse(foci):
        foci_csr = foci.tocsr()
        observed_term = (moderators.T @ foci_csr @ bases).reshape((n_moderators * n_bases, 1))
    else:
        observed_term = (moderators.T @ foci @ bases).reshape((n_moderators * n_bases, 1))

    eta = _spatial_cbmr_kron_vector_product(moderators, bases, coefficient).reshape(
        (n_experiments, n_voxels)
    )
    expected = _safe_exp(eta)
    expected_term = (moderators.T @ expected @ bases).reshape((n_moderators * n_bases, 1))
    return -(observed_term - expected_term)


def _fit_spatial_cbmr_additive_log_glm(moderators, bases, foci):
    """Fit an additive log-Poisson approximation used for preconditioning."""
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    if sp_sparse.issparse(foci):
        foci_csr = foci.tocsr()
        foci_by_experiment = np.asarray(foci_csr.mean(axis=1)).ravel()
        foci_by_voxel = np.asarray(foci_csr.mean(axis=0)).ravel()
    else:
        foci_by_experiment = foci.mean(axis=1)
        foci_by_voxel = foci.mean(axis=0)

    def objective(params):
        basis_coef = params[:n_bases]
        moderator_coef = params[n_bases:]
        basis_linear = bases @ basis_coef
        moderator_linear = moderators @ moderator_coef
        log_like = (
            (foci_by_voxel * basis_linear).mean()
            + (foci_by_experiment * moderator_linear).mean()
            - _safe_exp(basis_linear).mean() * _safe_exp(moderator_linear).mean()
        )
        return -log_like

    def gradient(params):
        basis_coef = params[:n_bases]
        moderator_coef = params[n_bases:]
        exp_basis = _safe_exp(bases @ basis_coef)
        exp_moderator = _safe_exp(moderators @ moderator_coef)
        basis_grad = (bases.T @ foci_by_voxel) / n_voxels - (
            bases.T @ exp_basis
        ) / n_voxels * exp_moderator.mean()
        moderator_grad = (moderators.T @ foci_by_experiment) / n_experiments - (
            moderators.T @ exp_moderator
        ) / n_experiments * exp_basis.mean()
        return -np.concatenate([basis_grad, moderator_grad])

    result = optimize.minimize(
        fun=objective,
        jac=gradient,
        x0=np.zeros(n_bases + n_moderators),
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 200},
    )
    return result.x[:n_bases], result.x[n_bases:]


def _compute_spatial_cbmr_preconditioner(moderators, bases, mean_moderator, mean_basis, damping):
    """Build an approximate Kronecker preconditioner for the gradient step."""
    moderator_info = moderators.T @ (moderators * mean_moderator)
    basis_info = bases.T @ (bases * mean_basis)
    moderator_info_inv = np.linalg.pinv(moderator_info + damping * np.eye(moderators.shape[1]))
    basis_info_inv = np.linalg.pinv(basis_info + damping * np.eye(bases.shape[1]))
    return np.kron(moderator_info_inv, basis_info_inv)


def fit_voxelwise_cbmr_approximate(
    moderators,
    bases,
    foci,
    tol=1e-10,
    max_iter=100,
    alpha=1.0,
    damping=1e-4,
    compute_nll=False,
):
    """Fit a spatially varying log-Poisson GLM with a preconditioned gradient step."""
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    LGR.info(
        "SpatialCBMR approximate model: %d experiments, %d voxels, %d moderators, %d bases.",
        n_experiments,
        n_voxels,
        n_moderators,
        n_bases,
    )
    basis_coef, moderator_coef = _fit_spatial_cbmr_additive_log_glm(moderators, bases, foci)
    mean_moderator = _safe_exp(moderators @ moderator_coef)[:, None]
    mean_basis = _safe_exp(bases @ basis_coef)[:, None]
    preconditioner = _compute_spatial_cbmr_preconditioner(
        moderators,
        bases,
        mean_moderator,
        mean_basis,
        damping=damping,
    )

    coefficient = np.zeros((n_moderators, n_bases), dtype=np.float64)
    coefficient[-1] = basis_coef
    coefficient = coefficient.reshape((n_moderators * n_bases, 1))
    for iteration in range(max_iter):
        gradient = _spatial_cbmr_gradient(moderators, bases, coefficient, foci)
        new_coefficient = coefficient - alpha * (preconditioner @ gradient)
        if not np.isfinite(new_coefficient).all():
            raise FloatingPointError(
                "SpatialCBMR approximate regression produced non-finite coefficients. "
                "Try reducing alpha or increasing damping."
            )
        delta = float(np.linalg.norm(new_coefficient - coefficient))
        relative_delta = delta / max(float(np.linalg.norm(coefficient)), 1.0)
        coefficient = new_coefficient
        if compute_nll:
            nll = _spatial_cbmr_log_poisson_nll(moderators, bases, coefficient, foci)
            LGR.info(
                "Iteration %d: delta=%g, relative_delta=%g, nll=%g",
                iteration,
                delta,
                relative_delta,
                nll,
            )
        else:
            LGR.debug(
                "Iteration %d: delta=%g, relative_delta=%g",
                iteration,
                delta,
                relative_delta,
            )
        if delta < tol or relative_delta < tol:
            LGR.info("SpatialCBMR approximate model converged in %d iterations.", iteration + 1)
            break
    else:
        LGR.warning(
            "SpatialCBMR approximate model did not converge within %d iterations.",
            max_iter,
        )
    return coefficient


fit_spatial_cbmr_approximate = fit_voxelwise_cbmr_approximate
