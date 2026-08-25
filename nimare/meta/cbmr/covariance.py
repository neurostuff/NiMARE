"""Robust covariance estimators for a term-based CBMR model.

The Fisher information gives standard errors that are correct only if the Poisson mean-variance
relationship holds. Foci are overdispersed and clustered within experiments, so it usually does
not, and a sandwich estimator replaces the model-based variance with an empirical one::

    V = A^-1 M A^-1

``A`` is the observed information -- the "bread", from
:meth:`~nimare.meta.cbmr.model.CBMRModel.information_matrix` -- and ``M`` is the "meat", built
from the score contributions. Two ways to group them:

``meat="iid"``
    One contribution per experiment-voxel cell. Robust to the wrong variance function but still
    assumes cells are independent, which they are not: an experiment's foci are its own.
``meat="cluster"``
    One contribution per experiment, summing its cells first. Robust to arbitrary correlation
    within an experiment, which is the realistic assumption for coordinate data, and the reason
    it is the default.

Leverage corrections (``hc0``, ``hc1``, ``hc3``) adjust for the fact that residuals from a
fitted model are systematically too small. ``hc3`` is the most conservative and the most
expensive, since it needs the leverage of every cell.

Nothing here materializes the (experiment x voxel) design. A design row is
``[S[i] kron B[v] ; G[i]]``, so every quantity below is assembled from that structure -- see the
einsum subscripts, where ``i`` indexes experiments, ``v`` voxels, ``c``/``d`` spatial columns and
``p``/``q`` basis functions.
"""

import logging

import numpy as np
from scipy.linalg import lapack

from nimare.meta.cbmr.predictor import _as_dense_array

LGR = logging.getLogger(__name__)

VALID_MEATS = ("cluster", "iid")
VALID_CORRECTIONS = (None, "hc0", "hc1", "hc3")


class CovarianceError(ValueError):
    """Raised when a covariance estimator cannot be used as requested."""


def _validate(meat, correction):
    """Normalize and check the sandwich options."""
    meat = str(meat).lower()
    if meat not in VALID_MEATS:
        raise CovarianceError(f"meat must be one of {VALID_MEATS}, got {meat!r}.")
    if correction is not None:
        correction = str(correction).lower()
    if correction not in VALID_CORRECTIONS:
        raise CovarianceError(
            f"correction must be one of {VALID_CORRECTIONS}, got {correction!r}."
        )
    if meat == "cluster" and correction == "hc3":
        raise CovarianceError(
            "hc3 rescales each observation by its own leverage, which has no counterpart once "
            "observations are summed into clusters. Use hc1 with meat='cluster', or hc3 with "
            "meat='iid'."
        )
    return meat, correction


def _fitted_pieces(model, foci):
    """Return residuals, fitted means, and the design's structural blocks."""
    predictor = model.predictor
    counts = _as_dense_array(foci)

    spatial_coef, global_coef = model.unpack(model.coefficients.detach())
    log_intensity = predictor.log_intensity_by_pattern(spatial_coef).detach().cpu().numpy()
    moderator = predictor.moderator_effect(global_coef).detach().cpu().numpy()

    eta = log_intensity[predictor.patterns.assignment] + moderator[:, None]
    mean = np.exp(eta)
    return counts - mean, mean, predictor.spatial_block, predictor.global_block, predictor.bases


def _leverage(model, foci, spatial_block, global_block, bases, mean):
    """Return the leverage of every experiment-voxel cell.

    ``h_iv = x_iv' A^-1 x_iv`` scaled by the cell's mean, as for any GLM hat value. Assembled
    from the design's structure rather than from the rows themselves, which would be an
    (experiment x voxel) x parameters array.
    """
    bread_inverse = np.linalg.inv(model.information_matrix(foci))
    n_spatial = model.n_spatial
    n_columns, n_bases = spatial_block.shape[1], bases.shape[1]

    spatial_covariance = bread_inverse[:n_spatial, :n_spatial].reshape(
        n_columns, n_bases, n_columns, n_bases
    )
    leverage = np.einsum(
        "ic,vp,cpdq,id,vq->iv",
        spatial_block,
        bases,
        spatial_covariance,
        spatial_block,
        bases,
        optimize=True,
    )
    if global_block is not None:
        cross = bread_inverse[:n_spatial, n_spatial:].reshape(n_columns, n_bases, -1)
        leverage = leverage + 2.0 * np.einsum(
            "ic,vp,cpk,ik->iv", spatial_block, bases, cross, global_block, optimize=True
        )
        global_covariance = bread_inverse[n_spatial:, n_spatial:]
        leverage = (
            leverage
            + np.einsum(
                "ik,kl,il->i", global_block, global_covariance, global_block, optimize=True
            )[:, None]
        )
    return leverage * mean


def _meat_iid(residuals, spatial_block, global_block, bases, n_spatial):
    """Assemble the meat from one contribution per experiment-voxel cell."""
    weights = residuals**2
    spatial_spatial = np.einsum(
        "ic,id,vp,vq,iv->cpdq",
        spatial_block,
        spatial_block,
        bases,
        bases,
        weights,
        optimize=True,
    )
    size = n_spatial if global_block is None else n_spatial + global_block.shape[1]
    meat = np.zeros((size, size), dtype=float)
    meat[:n_spatial, :n_spatial] = spatial_spatial.reshape(n_spatial, n_spatial)

    if global_block is not None:
        cross = np.einsum(
            "ic,vp,iv,ik->cpk", spatial_block, bases, weights, global_block, optimize=True
        ).reshape(n_spatial, -1)
        meat[:n_spatial, n_spatial:] = cross
        meat[n_spatial:, :n_spatial] = cross.T
        meat[n_spatial:, n_spatial:] = np.einsum(
            "ik,il,iv->kl", global_block, global_block, weights, optimize=True
        )
    return meat


def _scores_by_cluster(residuals, spatial_block, global_block, bases, n_spatial):
    """Return one score vector per experiment, summing its cells."""
    projected = residuals @ bases  # (experiments, bases)
    spatial_scores = np.einsum("ic,ip->icp", spatial_block, projected, optimize=True).reshape(
        residuals.shape[0], n_spatial
    )
    if global_block is None:
        return spatial_scores
    global_scores = global_block * residuals.sum(axis=1)[:, None]
    return np.hstack([spatial_scores, global_scores])


def sandwich_covariance(model, foci, meat="cluster", correction="hc1", ridge=0.0):
    """Return the sandwich covariance of the regression coefficients.

    Parameters
    ----------
    model : :class:`~nimare.meta.cbmr.model.CBMRModel`
        Fitted model.
    foci : array_like
        Foci counts the model was fitted to.
    meat : {"cluster", "iid"}, optional
        How score contributions are grouped. Default is ``"cluster"``, which allows arbitrary
        correlation within an experiment.
    correction : {None, "hc0", "hc1", "hc3"}, optional
        Leverage correction. Default is ``"hc1"``.
    ridge : :obj:`float`, optional
        Added to the diagonal of the bread before inversion, for designs whose information is
        near-singular. Default is 0.0.

    Returns
    -------
    :obj:`numpy.ndarray`
        Covariance matrix over the flat coefficient vector, in the design's parameter layout.
    """
    meat_kind, correction = _validate(meat, correction)
    residuals, mean, spatial_block, global_block, bases = _fitted_pieces(model, foci)
    n_spatial = model.n_spatial
    n_parameters = model.n_parameters

    if correction == "hc3":
        leverage = _leverage(model, foci, spatial_block, global_block, bases, mean)
        # Clip below one: a leverage at or above one would divide by zero, which happens when a
        # cell is fitted exactly and carries no information about anything else.
        residuals = residuals / np.clip(1.0 - leverage, 1e-6, None)

    if meat_kind == "iid":
        meat_matrix = _meat_iid(residuals, spatial_block, global_block, bases, n_spatial)
        n_observations = residuals.size
    else:
        scores = _scores_by_cluster(residuals, spatial_block, global_block, bases, n_spatial)
        meat_matrix = scores.T @ scores
        n_observations = residuals.shape[0]

    if correction == "hc1":
        if n_observations <= n_parameters:
            raise CovarianceError(
                f"hc1 scales by n / (n - p), but this design has {n_parameters} parameters and "
                f"only {n_observations} "
                f"{'experiments' if meat_kind == 'cluster' else 'observations'} to fit them. Use "
                "correction='hc0', a coarser spline_spacing, or fewer s() terms."
            )
        meat_matrix = meat_matrix * (n_observations / (n_observations - n_parameters))

    bread = model.information_matrix(foci)
    if ridge:
        bread = bread + ridge * np.eye(n_parameters)
    bread_inverse = np.linalg.inv(bread)
    return bread_inverse @ meat_matrix @ bread_inverse


# ---------------------------------------------------------------------------------------------
# Structure of the Fisher information, and the strategies it permits.
#
# The information matrix's beta-beta block is ``sum_p L_pc L_pc' (B^T Sigma_p B)``, so the factor
# ``L_pc L_pc'`` decides which entries can be nonzero at all -- independently of the
# distribution, since it is a statement about the design. That sparsity is what the routines
# below exploit. Each is proved at https://github.com/jdkent/cbmr-proofs.
# ---------------------------------------------------------------------------------------------


def spatial_components(loadings, n_bases):
    """Return the groups of spatial parameter indices that can covary.

    Two spatial columns are coupled when some pattern loads on both, since that is the only way a
    cross term can be nonzero, and coupling is transitive -- so this is connected components over
    the columns, seeded by each pattern's support. A cell-means factor gives one component per
    level; a sum-to-zero term or a spatial covariate couples everything.

    Independent of the global block: this is the structure of ``D`` in the bordered form
    ``[[D, C], [C^T, E]]``, which survives moderators even though full block diagonality does not.
    """
    n_columns = loadings.shape[1]
    parent = np.arange(n_columns)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in loadings:
        support = np.flatnonzero(row)
        for other in support[1:]:
            a, b = find(support[0]), find(other)
            if a != b:
                parent[b] = a

    members = {}
    for column in range(n_columns):
        members.setdefault(find(column), []).append(column)
    return [
        np.concatenate([np.arange(c * n_bases, (c + 1) * n_bases) for c in sorted(columns)])
        for columns in members.values()
    ]


def is_block_diagonal(information, blocks):
    """Whether ``information`` really is block diagonal over ``blocks``.

    Nonzero *counts* rather than magnitudes: the claim being checked is that the off-block
    entries are structurally absent, so a small-but-nonzero entry means the derivation is wrong,
    not that the block is nearly separable. Rounding a real coupling to zero would silently
    understate a standard error.
    """
    if len(blocks) < 2:
        return False
    return all(
        np.count_nonzero(information[index])
        == np.count_nonzero(information[np.ix_(index, index)])
        for index in blocks
    )


def symmetric_condition_number(information, blocks=None):
    """Return the exact 2-norm condition number of a symmetric matrix.

    For symmetric ``H`` the singular values are the absolute eigenvalues, so a symmetric
    eigensolver replaces the SVD that ``np.linalg.cond`` forms -- exact, with no hypothesis on
    the design, and several times faster. When ``H`` is block diagonal its spectrum is the union
    of the blocks', so the eigensolver runs per block instead of once on the whole.
    """
    if blocks is None:
        values = np.linalg.eigvalsh(information)
    else:
        values = np.concatenate(
            [np.linalg.eigvalsh(information[np.ix_(index, index)]) for index in blocks]
        )
    smallest = np.abs(values).min()
    return np.inf if smallest == 0 else np.abs(values).max() / smallest


def cholesky_inverse(information):
    """Invert a positive definite matrix from one Cholesky factor, or return None.

    ``None`` rather than an exception for a non-positive-definite matrix: that is a statement
    about the fit, which the caller reports, not an error in this routine.
    """
    factor, info = lapack.dpotrf(information, lower=1, clean=1)
    if info != 0:
        return None
    inverse, info = lapack.dpotri(factor, lower=1)
    if info != 0:
        return None
    return np.tril(inverse) + np.tril(inverse, -1).T


def blockwise_inverse(information, blocks):
    """Invert a block diagonal matrix one block at a time."""
    inverse = np.zeros_like(information)
    for index in blocks:
        inverse[np.ix_(index, index)] = np.linalg.inv(information[np.ix_(index, index)])
    return inverse


def bordered_inverse(information, blocks, n_spatial):
    """Invert ``[[D, C], [C^T, E]]`` with ``D`` block diagonal, by Schur complement.

    Recovers most of the blockwise saving for a design whose spatial columns separate but which
    carries global moderators: the cross block couples every spatial column through ``gamma``, so
    the matrix is not block diagonal at all, but ``D`` underneath it still is. Returns None if
    ``D`` or the Schur complement is singular, so the caller can fall back rather than propagate
    a nonsense covariance.
    """
    border = information[:n_spatial, n_spatial:]
    try:
        d_inverse = blockwise_inverse(information[:n_spatial, :n_spatial], blocks)
        weighted = d_inverse @ border  # D^-1 C
        schur = information[n_spatial:, n_spatial:] - border.T @ weighted
        schur_inverse = np.linalg.inv(schur)
    except np.linalg.LinAlgError:
        return None
    inverse = np.empty_like(information)
    inverse[:n_spatial, :n_spatial] = d_inverse + weighted @ schur_inverse @ weighted.T
    inverse[:n_spatial, n_spatial:] = -weighted @ schur_inverse
    inverse[n_spatial:, :n_spatial] = inverse[:n_spatial, n_spatial:].T
    inverse[n_spatial:, n_spatial:] = schur_inverse
    return inverse


def fisher_covariance(model, information):
    """Invert the observed information, exploiting whatever structure the design left.

    A ladder, most specific rung first:

    block diagonal
        the spatial columns separate and there are no moderators. The spectrum is the union of
        the blocks' and the inverse is blockwise. Only a lone cell-means factor reaches here.
    bordered block diagonal
        the spatial columns separate but moderators border them. The inverse still goes
        blockwise, through a Schur complement; the condition number cannot follow, because the
        spectrum of a bordered matrix is not the union of its parts.
    dense
        everything else. Still two improvements on a plain SVD-plus-LU: the condition number
        comes from a symmetric eigensolver, and the inverse from a Cholesky factor when the
        matrix is positive definite.

    Every rung returns the same matrix and reports the same condition number to the last digit;
    only the work differs. Structural claims are checked against the matrix in hand rather than
    assumed.
    """
    n_spatial = model.n_spatial
    loadings, n_bases = model.predictor.patterns.loadings, model.predictor.n_bases
    blocks = spatial_components(loadings, n_bases) if not model.n_global else []
    separable = bool(blocks) and is_block_diagonal(information, blocks)

    condition = symmetric_condition_number(information, blocks if separable else None)
    if condition > 1.0 / np.finfo(float).eps:
        LGR.warning(
            f"The Fisher information matrix has condition number {condition:.3g}, past what "
            "double precision can invert meaningfully, so these standard errors should not be "
            f"trusted. {_CONDITION_ADVICE}"
        )

    # Every route funnels through one handler: a singular design must explain itself the same
    # way whichever rung happened to reach it, and numpy's bare "Singular matrix" tells a user
    # nothing about which knob to turn.
    try:
        if separable:
            return blockwise_inverse(information, blocks)

        if model.n_global:
            spatial = spatial_components(loadings, n_bases)
            if len(spatial) > 1 and is_block_diagonal(
                information[:n_spatial, :n_spatial], spatial
            ):
                bordered = bordered_inverse(information, spatial, n_spatial)
                if bordered is not None:
                    return bordered

        inverse = cholesky_inverse(information)
        if inverse is not None:
            return inverse
        return np.linalg.inv(information)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            f"The Fisher information matrix over {model.n_parameters} coefficients is singular, "
            f"so no standard errors exist for this fit. {_CONDITION_ADVICE}"
        ) from error


#: Repeated where a singular or ill-conditioned information matrix is reported, so a user gets
#: the same actionable message from either path.
_CONDITION_ADVICE = (
    "The design asks for more spatial detail than these foci can support. Try a coarser "
    "spline_spacing, a higher incidence_threshold, fewer s() terms, or more experiments; "
    "CBMRResult.describe_terms() reports the parameter budget per term."
)
