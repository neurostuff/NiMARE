"""Closed-form observed information for the CBMR distributions.

``torch.func.hessian`` is ``jacfwd(jacrev(...))``. Forward mode carries one tangent per
parameter, so its intermediate has shape ``(n_parameters, n_patterns, n_voxels)`` -- 28.7 GB for
a 21-group model over 21,789 voxels. All three distributions have a closed form that avoids it::

    H_bb[(c,k),(c',k')] = sum_p L_pc L_pc' (B^T Sigma_p B)[k,k']

Only ``Sigma_p``, the second derivative of the pattern's term in the log-intensity, differs
between them. Each is ``n_patterns`` rank-updates of width ``n_bases``, so nothing of size
``n_parameters`` is built.
"""

import numpy as np
from scipy.special import digamma, polygamma

from nimare.meta.cbmr.distributions import (
    ClusteredNegativeBinomial,
    NegativeBinomial,
    Poisson,
)
from nimare.meta.cbmr.predictor import experiment_totals


def _intensity_pieces(model):
    """Return the fitted intensity and per-experiment weights all closed forms share.

    ``weight`` is ``E_i exp(m_i)``, and it comes from
    :meth:`~nimare.meta.cbmr.predictor.CBMRPredictor.experiment_weights` rather than being
    rebuilt here. Rebuilding it was how this function could disagree with the likelihood: the
    exposure would have been present in the fit and absent from every closed form, while the
    autodiff fallback stayed correct and so agreed with neither.
    """
    predictor = model.predictor
    flat = model.coefficients.detach().cpu().numpy()
    n_spatial, n_global = model.n_spatial, model.n_global

    spatial = flat[:n_spatial].reshape(predictor.n_spatial_columns, predictor.n_bases)
    loadings = predictor.patterns.loadings
    intensity = np.exp((loadings @ spatial) @ predictor.bases.T)  # exp(S), (n_patterns, n_voxels)

    global_coef = model.unpack(model.coefficients.detach())[1]
    weight = predictor.experiment_weights(global_coef).detach().cpu().numpy()
    global_block = predictor.global_block if n_global else None
    return predictor, loadings, intensity, global_block, weight


def _symmetrize(matrix):
    """Return the exact symmetric part of a matrix that is symmetric in exact arithmetic.

    ``B^T diag(w) B`` is symmetric on paper, but BLAS sums the ``[i, j]`` and ``[j, i]`` dot
    products in different orders, so they can differ in the last bit. ``eigvalsh`` and the
    Cholesky read one triangle only, so that difference would quietly decide which value is used.
    """
    return 0.5 * (matrix + matrix.T)


def _finalize(information, n_spatial, n_global):
    """Mirror the spatial-by-global block and make the moderator block exactly symmetric."""
    if n_global:
        information[n_spatial:, :n_spatial] = information[:n_spatial, n_spatial:].T
        information[n_spatial:, n_spatial:] = _symmetrize(information[n_spatial:, n_spatial:])
    return information


def _scatter_spatial(information, row, n_bases, block):
    """Add ``L_pc L_pc' * block`` into every spatial cross block the pattern's support touches."""
    block = _symmetrize(block)
    support = np.flatnonzero(row)
    for c in support:
        rows = slice(c * n_bases, (c + 1) * n_bases)
        for d in support:
            information[rows, d * n_bases : (d + 1) * n_bases] += (row[c] * row[d]) * block


def _scatter_cross(information, row, n_bases, n_spatial, vector, moderator_row):
    """Add ``L_pc * vector_k * moderator_q`` into the spatial-by-global cross block."""
    outer = np.outer(vector, moderator_row)
    for c in np.flatnonzero(row):
        information[c * n_bases : (c + 1) * n_bases, n_spatial:] += row[c] * outer


def _nuisance(model):
    """Return the fitted nuisance parameters on the statistical scale."""
    return model.distribution.transform_nuisance(model.nuisance).detach().cpu().numpy()


def poisson_information_matrix(model):
    """Return the observed Fisher information of a fitted Poisson model.

    Takes no foci. Under a log link the Poisson information depends only on the fitted
    coefficients, so the counts drop out.

    Parameters
    ----------
    model : :class:`~nimare.meta.cbmr.model.CBMRModel`
        Fitted model.

    Returns
    -------
    :obj:`numpy.ndarray`
        Shape ``(n_parameters, n_parameters)``.
    """
    predictor, loadings, intensity, global_block, weight = _intensity_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    bases = predictor.bases
    assignment = predictor.patterns.assignment

    total = np.zeros(predictor.patterns.n_patterns)  # T_p
    np.add.at(total, assignment, weight)

    information = np.zeros((n_spatial + n_global, n_spatial + n_global))
    for p, row in enumerate(loadings):
        _scatter_spatial(
            information, row, n_bases, bases.T @ (bases * (total[p] * intensity[p])[:, None])
        )

    if n_global:
        marginal_basis = intensity @ bases  # a_pk
        moderator = np.zeros((predictor.patterns.n_patterns, n_global))  # U_pq
        np.add.at(moderator, assignment, weight[:, None] * global_block)
        information[:n_spatial, n_spatial:] = np.einsum(
            "pc,pk,pq->ckq", loadings, marginal_basis, moderator, optimize=True
        ).reshape(n_spatial, n_global)

        energy = intensity.sum(axis=1)  # E_p
        information[n_spatial:, n_spatial:] = global_block.T @ (
            global_block * (weight * energy[assignment])[:, None]
        )
    return _finalize(information, n_spatial, n_global)


def negative_binomial_information_matrix(model, foci):
    """Return the observed Fisher information of a fitted NegativeBinomial model.

    Parameters
    ----------
    model : :class:`~nimare.meta.cbmr.model.CBMRModel`
        Fitted model.
    foci : array_like or :obj:`scipy.sparse.spmatrix`
        Foci counts. Used, unlike the Poisson case, but only through the pattern marginals.

    Returns
    -------
    :obj:`numpy.ndarray`
        Shape ``(n_parameters, n_parameters)``.

    Notes
    -----
    Uses ``A = S1/(theta S2)`` and ``R = S1 A`` in place of NiMARE's shape and probability. In
    those terms the gamma functions depend only on ``R``, the voxels only on the log-intensity,
    and the moderators only on ``R`` and ``A``.

    Precision is poor when ``theta`` approaches zero. The moderator block then sums large terms
    of opposite sign, and the fit is a Poisson one in all but name.
    """
    predictor, loadings, intensity, global_block, weight = _intensity_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    n_voxels, bases = predictor.n_voxels, predictor.bases
    assignment = predictor.patterns.assignment
    overdispersion = _nuisance(model)
    marginal = predictor.patterns.marginal_by_pattern(foci)

    information = np.zeros((n_spatial + n_global, n_spatial + n_global))
    for p, row in enumerate(loadings):
        members = np.flatnonzero(assignment == p)
        member_weight = weight[members]
        sum_1 = member_weight.sum()
        sum_2 = np.square(member_weight).sum()
        a = sum_1 / (overdispersion[p] * sum_2)
        r = sum_1 * a

        u = intensity[p]
        counts = marginal[p]
        denominator = u + a

        weight_v = (r + counts) * u * a / denominator**2
        _scatter_spatial(information, row, n_bases, bases.T @ (bases * weight_v[:, None]))
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

        # d2Psi/dS dR and d2Psi/dS dA, each carried into gamma by its own chain factor.
        _scatter_cross(information, row, n_bases, n_spatial, bases.T @ (u / denominator), r_1)
        cross_a = -(bases.T @ ((r + counts) * u / denominator**2))
        _scatter_cross(information, row, n_bases, n_spatial, cross_a, a_1)

        psi_r = (
            -digamma(counts + r).sum()
            + n_voxels * digamma(r)
            - n_voxels * np.log(a)
            + np.log(denominator).sum()
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

    return _finalize(information, n_spatial, n_global)


def clustered_negative_binomial_information_matrix(model, foci):
    """Return the observed Fisher information of a fitted ClusteredNegativeBinomial model.

    Parameters
    ----------
    model : :class:`~nimare.meta.cbmr.model.CBMRModel`
        Fitted model.
    foci : array_like or :obj:`scipy.sparse.spmatrix`
        Foci counts. Only the per-experiment totals matter here.

    Returns
    -------
    :obj:`numpy.ndarray`
        Shape ``(n_parameters, n_parameters)``.

    Notes
    -----
    The spatial coefficients reach this likelihood only through the scalar
    ``E_p = sum_v exp(S_pv)``. That makes the spatial block ``B^T diag(w) B`` plus a rank-one
    correction, where the other two distributions need only the first term.
    """
    predictor, loadings, intensity, global_block, weight = _intensity_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    bases = predictor.bases
    assignment = predictor.patterns.assignment
    overdispersion = _nuisance(model)
    per_experiment = experiment_totals(foci)

    information = np.zeros((n_spatial + n_global, n_spatial + n_global))
    for p, row in enumerate(loadings):
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

        block = gradient * (bases.T @ (bases * u[:, None])) + curvature * np.outer(
            marginal_basis, marginal_basis
        )
        _scatter_spatial(information, row, n_bases, block)
        if not n_global:
            continue

        member_block = global_block[members]
        scale = excess * precision * member_weight / denominator**2
        _scatter_cross(information, row, n_bases, n_spatial, marginal_basis, scale @ member_block)
        information[n_spatial:, n_spatial:] += member_block.T @ (
            member_block * (scale * energy)[:, None]
        )

    return _finalize(information, n_spatial, n_global)


def _poisson_entry(model, foci):
    """Adapt the Poisson signature to the ``(model, foci)`` the dispatch uses."""
    return poisson_information_matrix(model)


#: Most specific class first, so a subclass resolves to its own entry rather than a base's.
CLOSED_FORMS = (
    (ClusteredNegativeBinomial, clustered_negative_binomial_information_matrix),
    (NegativeBinomial, negative_binomial_information_matrix),
    (Poisson, _poisson_entry),
)


def closed_form_information(distribution):
    """Return the closed-form information function for ``distribution``, or None.

    Parameters
    ----------
    distribution : :class:`~nimare.meta.cbmr.distributions.Distribution`
        Observation distribution.

    Returns
    -------
    callable or None
        None means no derivation covers this distribution, and the caller should fall back to
        automatic differentiation rather than use another distribution's formula.
    """
    for klass, function in CLOSED_FORMS:
        if isinstance(distribution, klass):
            return function
    return None
