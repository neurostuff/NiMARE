"""Closed-form observed information for the CBMR distributions.

Why this exists
---------------
:meth:`~nimare.meta.cbmr.model.CBMRModel.information_matrix` differentiates the log-likelihood
twice with ``torch.func.hessian``, which is ``jacfwd(jacrev(...))``. Forward mode carries one
tangent per parameter through the likelihood simultaneously, so the intermediate log-intensity
tensor has shape ``(n_parameters, n_patterns, n_voxels)``. For a 21-group model over 21,789
voxels that is 28.7 GB -- an out-of-memory kill rather than a slow fit -- and the size is driven
by the group count, so resampling to a coarser grid does not avoid it.

All three distributions admit a closed form, and it is the same shape in each case. Writing the
likelihood over patterns as ``predictor.py`` does, ``-log L = sum_p f_p(S_p, gamma)`` with
``S = (L b) B^T`` the log-intensity, the predictor is *linear* in the coefficients, so the chain
rule contributes no curvature term and::

    H_bb[(c,k),(c',k')] = sum_p L_pc L_pc' (B^T Sigma_p B)[k,k']

with ``Sigma_p`` the second differential of ``f_p`` in the log-intensity. What differs between
the distributions is only ``Sigma_p``:

Poisson
    ``Sigma_p = T_p diag(exp(S_p))``. The data terms are linear in the coefficients, so they
    differentiate away: the Poisson Hessian does not depend on the foci at all. This is the
    canonical-link property -- observed and expected information coincide.
NegativeBinomial
    ``Sigma_p = diag(w_p)`` with ``w_pv = (R + Y_pv) u_pv A / (u_pv + A)^2``, writing
    ``A = S1/(theta S2)`` and ``R = S1 A``. Same shape, different weight, and now foci
    dependent. The Poisson weight is the ``theta -> 0`` limit of this one.
ClusteredNegativeBinomial
    The spatial coefficients reach this likelihood only through the scalar
    ``E_p = sum_v exp(S_pv)``, so ``Sigma_p`` is diagonal *plus rank one* and the spatial block
    picks up a correction ``g'' a_p a_p^T`` with ``a_p = B^T exp(S_p)``.

Each is ``n_patterns`` symmetric rank-updates of width ``n_bases``, costing
``O(n_patterns x n_voxels x n_bases^2)`` and allocating one ``(n_voxels, n_bases)`` temporary.
No ``n_parameters``-sized intermediate is formed anywhere, which is the axis that made the model
impossible to fit.

The nuisance parameters are held fixed at their fitted values throughout, which is what
``information_matrix`` has always done and what CBMR reports regression standard errors from.
That is what makes the overdispersed cases tractable: their special functions then depend on
constants alone.

Every identity above is proved symbolically, term by term, at
https://github.com/jdkent/cbmr-proofs -- 59 claims, each a residual simplified by sympy to
exactly zero -- and checked numerically against ``torch.func.hessian`` in
``nimare/tests/test_meta_cbmr_information.py``.
"""

import numpy as np
from scipy.special import digamma, polygamma

from nimare.meta.cbmr.distributions import (
    ClusteredNegativeBinomial,
    NegativeBinomial,
    Poisson,
)


def _fitted_pieces(model):
    """Return the quantities all three closed forms are written in terms of."""
    predictor = model.predictor
    flat = model.coefficients.detach().cpu().numpy()
    n_spatial, n_global = model.n_spatial, model.n_global

    spatial = flat[:n_spatial].reshape(predictor.n_spatial_columns, predictor.n_bases)
    loadings = predictor.patterns.loadings
    intensity = np.exp((loadings @ spatial) @ predictor.bases.T)  # exp(S), (n_patterns, n_voxels)

    if n_global:
        global_block = predictor.global_block
        weight = np.exp(global_block @ flat[n_spatial:])  # exp(m_i)
    else:
        global_block = None
        weight = np.ones(predictor.patterns.assignment.size)
    return predictor, loadings, intensity, global_block, weight


def _scatter_spatial(information, row, n_bases, block):
    """Add ``L_pc L_pc' * block`` into every spatial cross block the pattern's support touches."""
    for c in np.flatnonzero(row):
        rows = slice(c * n_bases, (c + 1) * n_bases)
        for d in np.flatnonzero(row):
            information[rows, d * n_bases : (d + 1) * n_bases] += (row[c] * row[d]) * block


def _scatter_cross(information, row, n_bases, n_spatial, vector, moderator_row):
    """Add ``L_pc * vector_k * moderator_q`` into the spatial-by-global cross block."""
    outer = np.outer(vector, moderator_row)
    for c in np.flatnonzero(row):
        information[c * n_bases : (c + 1) * n_bases, n_spatial:] += row[c] * outer


def _nuisance(model):
    """Return the fitted nuisance parameters on the statistical scale."""
    return model.distribution.transform_nuisance(model.nuisance).detach().cpu().numpy()


def poisson_information_matrix(model, foci=None):
    """Return the observed Fisher information of a fitted Poisson CBMR, in closed form.

    Takes no foci: the Hessian of this likelihood does not depend on them. Identical to
    ``torch.func.hessian`` of the negative log-likelihood to within floating-point rounding.
    """
    predictor, loadings, intensity, global_block, weight = _fitted_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    bases = predictor.bases
    assignment = predictor.patterns.assignment

    total = np.zeros(predictor.patterns.n_patterns)  # T_p
    np.add.at(total, assignment, weight)

    information = np.zeros((n_spatial + n_global, n_spatial + n_global))
    for p, row in enumerate(loadings):
        if not np.any(row) or total[p] == 0.0:
            continue
        _scatter_spatial(
            information, row, n_bases,
            bases.T @ (bases * (total[p] * intensity[p])[:, None]),
        )

    if n_global:
        marginal_basis = intensity @ bases  # a_pk
        moderator = np.zeros((predictor.patterns.n_patterns, n_global))  # U_pq
        np.add.at(moderator, assignment, weight[:, None] * global_block)
        cross = np.einsum(
            "pc,pk,pq->ckq", loadings, marginal_basis, moderator, optimize=True
        ).reshape(n_spatial, n_global)
        information[:n_spatial, n_spatial:] = cross
        information[n_spatial:, :n_spatial] = cross.T

        energy = intensity.sum(axis=1)  # E_p
        information[n_spatial:, n_spatial:] = global_block.T @ (
            global_block * (weight * energy[assignment])[:, None]
        )
    return information


def negative_binomial_information_matrix(model, foci):
    """Return the observed Fisher information of a fitted NegativeBinomial CBMR, in closed form.

    Written as ``Psi(R, A, S)`` with ``R = S1^2/(theta S2)`` and ``A = S1/(theta S2)``, which is
    what makes the derivation tractable: the special functions depend only on ``R``, the voxels
    only on ``S``, and the moderators only on ``R`` and ``A``.

    Unlike the Poisson case the foci do not drop out -- ``R + Y_v`` is a per-voxel weight -- but
    they enter only through the pattern marginals.
    """
    predictor, loadings, intensity, global_block, weight = _fitted_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    n_voxels, bases = predictor.n_voxels, predictor.bases
    assignment = predictor.patterns.assignment
    overdispersion = _nuisance(model)
    marginal = predictor.patterns.marginal_by_pattern(foci)

    information = np.zeros((n_spatial + n_global, n_spatial + n_global))
    for p, row in enumerate(loadings):
        if not np.any(row):
            continue
        members = np.flatnonzero(assignment == p)
        member_weight = weight[members]
        sum_1 = member_weight.sum()
        sum_2 = np.square(member_weight).sum()
        a = sum_1 / (overdispersion[p] * sum_2)
        r = sum_1 * a

        u = intensity[p]
        counts = marginal[p]
        denominator = u + a

        _scatter_spatial(
            information, row, n_bases,
            bases.T @ (bases * ((r + counts) * u * a / denominator**2)[:, None]),
        )
        if not n_global:
            continue

        # dR/dgamma and dA/dgamma, through their log derivatives.
        member_block = global_block[members]
        u_1 = member_weight @ member_block
        u_2 = np.square(member_weight) @ member_block
        u_1_2 = member_block.T @ (member_block * member_weight[:, None])
        u_2_2 = member_block.T @ (member_block * np.square(member_weight)[:, None])
        d_log_1, d_log_2 = u_1 / sum_1, 2 * u_2 / sum_2
        dd_log_1 = u_1_2 / sum_1 - np.outer(u_1, u_1) / sum_1**2
        dd_log_2 = 4 * u_2_2 / sum_2 - 4 * np.outer(u_2, u_2) / sum_2**2
        d_log_a, d_log_r = d_log_1 - d_log_2, 2 * d_log_1 - d_log_2
        a_1, r_1 = a * d_log_a, r * d_log_r
        a_2 = a * (np.outer(d_log_a, d_log_a) + dd_log_1 - dd_log_2)
        r_2 = r * (np.outer(d_log_r, d_log_r) + 2 * dd_log_1 - dd_log_2)

        # d2Psi/dS dR and d2Psi/dS dA, each carried into gamma by its own chain rule factor.
        _scatter_cross(information, row, n_bases, n_spatial, bases.T @ (u / denominator), r_1)
        _scatter_cross(
            information, row, n_bases, n_spatial,
            -(bases.T @ ((r + counts) * u / denominator**2)), a_1,
        )

        psi_r = (
            -digamma(counts + r).sum() + n_voxels * digamma(r)
            - n_voxels * np.log(a) + np.log(denominator).sum()
        )
        psi_a = -n_voxels * r / a + ((r + counts) / denominator).sum()
        psi_rr = -polygamma(1, counts + r).sum() + n_voxels * polygamma(1, r)
        psi_ra = -n_voxels / a + (1.0 / denominator).sum()
        psi_aa = n_voxels * r / a**2 - ((r + counts) / denominator**2).sum()
        information[n_spatial:, n_spatial:] += (
            psi_rr * np.outer(r_1, r_1)
            + psi_ra * (np.outer(r_1, a_1) + np.outer(a_1, r_1))
            + psi_aa * np.outer(a_1, a_1)
            + psi_r * r_2
            + psi_a * a_2
        )

    if n_global:
        information[n_spatial:, :n_spatial] = information[:n_spatial, n_spatial:].T
    return information


def clustered_negative_binomial_information_matrix(model, foci):
    """Return the observed information of a fitted ClusteredNegativeBinomial, in closed form.

    The spatial coefficients reach this likelihood only through the scalar
    ``E_p = sum_v exp(S_pv)``, so the spatial block is not the plain ``B^T diag(w) B`` of the
    other two but that shape plus a rank-one correction, with ``a_p = B^T exp(S_p)``.

    Only the per-experiment foci totals appear; the per-voxel marginals enter the likelihood
    linearly and so vanish from the Hessian, as in the Poisson case.
    """
    predictor, loadings, intensity, global_block, weight = _fitted_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    bases = predictor.bases
    assignment = predictor.patterns.assignment
    overdispersion = _nuisance(model)
    per_experiment = _experiment_totals(foci)

    information = np.zeros((n_spatial + n_global, n_spatial + n_global))
    for p, row in enumerate(loadings):
        if not np.any(row):
            continue
        members = np.flatnonzero(assignment == p)
        precision = 1.0 / overdispersion[p]
        u = intensity[p]
        energy = u.sum()  # E_p

        member_weight = weight[members]
        excess = per_experiment[members] + precision
        denominator = energy * member_weight + precision

        gradient = (excess * member_weight / denominator).sum()
        curvature = -(excess * np.square(member_weight) / denominator**2).sum()
        marginal_basis = bases.T @ u  # a_k

        _scatter_spatial(
            information, row, n_bases,
            gradient * (bases.T @ (bases * u[:, None]))
            + curvature * np.outer(marginal_basis, marginal_basis),
        )
        if not n_global:
            continue

        member_block = global_block[members]
        scale = excess * precision * member_weight / denominator**2
        _scatter_cross(
            information, row, n_bases, n_spatial, marginal_basis, scale @ member_block
        )
        information[n_spatial:, n_spatial:] += member_block.T @ (
            member_block * (scale * energy)[:, None]
        )

    if n_global:
        information[n_spatial:, :n_spatial] = information[:n_spatial, n_spatial:].T
    return information


def _experiment_totals(foci):
    """Return the foci count per experiment, without densifying a sparse matrix."""
    return np.asarray(foci.sum(axis=1)).reshape(-1).astype(float)


#: The closed form for each distribution, most specific class first so a subclass resolves to
#: its own entry rather than a base's.
CLOSED_FORMS = (
    (ClusteredNegativeBinomial, clustered_negative_binomial_information_matrix),
    (NegativeBinomial, negative_binomial_information_matrix),
    (Poisson, poisson_information_matrix),
)


def closed_form_information(distribution):
    """Return the closed-form information matrix for ``distribution``, or None if there is none.

    None is the signal to fall back to automatic differentiation, so a distribution added
    without a derivation keeps working rather than silently getting the wrong Hessian.
    """
    for klass, function in CLOSED_FORMS:
        if isinstance(distribution, klass):
            return function
    return None
