"""Linear predictor assembly for CBMR.

Builds ``eta`` from a :class:`~nimare.meta.cbmr.terms.BoundDesign` and evaluates the Poisson
log-likelihood against it.

The interesting part is memory. Written out, the predictor is an (experiment x voxel) array:
at 200 experiments and 228,483 voxels that is 366 MB, and autograd needs several multiples of
it. The existing global CBMR path avoids ever forming it by exploiting the fact that a Poisson
log-likelihood over a separable predictor collapses onto marginals::

    sum_v y_.v s_v + sum_i y_i. m_i - (sum_v exp(s_v))(sum_i exp(m_i))

which costs O(n_voxels + n_experiments) instead of their product. That is why CBMR historically
had two model classes: one fast separable path and one general path for voxelwise moderators.

Both are the same computation. What decides it is not a mode the user picks but how many
*distinct spatial loadings* the design produces. Stack the experiment-level blocks of every
spatial term side by side and take the unique rows: each distinct row is a pattern of spatial
coefficients, and every experiment sharing that row shares a log-intensity map. Then

    log L = sum_p [ Y_p . s_p  +  sum_{i in p} y_i. m_i  -  (sum_v exp(s_p)) T_p ]

with ``Y_p`` the voxel marginal within pattern ``p`` and ``T_p = sum_{i in p} E_i exp(m_i)``,
where ``E_i`` is the experiment's exposure and is ``1`` unless the design says otherwise. Cost is
O(n_patterns x n_voxels).

An exposure enters only through ``T_p``, on the natural scale. Written that way its own
contribution to the likelihood, ``sum_i y_i. log E_i``, is free of every parameter and so is never
formed -- which is what makes ``E_i = 0`` exactly representable rather than a logarithm of zero to
be managed. The cost is that log-likelihoods are comparable only between models sharing an
exposure; see the note in :mod:`nimare.meta.cbmr.terms`.

That single expression covers the whole range. Group-only spatial terms give one pattern per
group, recovering the fast path exactly. A spatial covariate gives every experiment its own
pattern, recovering the general path. Nothing in between needs special handling, and no
combination is rejected for want of an implementation -- expensive designs are merely
expensive, which the parameter budget already warns about.
"""

import numpy as np
import scipy.sparse

from nimare.meta.cbmr._torch import torch


def _as_dense_array(value):
    """Return a dense float array from a possibly sparse matrix."""
    if scipy.sparse.issparse(value):
        return np.asarray(value.todense(), dtype=float)
    return np.asarray(value, dtype=float)


def experiment_totals(foci):
    """Return the foci count per experiment, for a sparse matrix or an array alike.

    ``sum`` works on both, so the experiment-by-voxel grid is never materialised.
    """
    return np.asarray(foci.sum(axis=1)).reshape(-1)


def _as_tensor(array, dtype=torch.float64):
    """Convert an array to a torch tensor, copying anything that is not already contiguous.

    Everything reaching this function today is contiguous by construction -- ``np.hstack`` output,
    patsy design matrices, freshly allocated accumulators -- but that is a property of the current
    call sites rather than a guarantee. A pandas column selection is a view over the frame's
    blocks and can be non-contiguous, read-only or negatively strided depending on how the
    requested columns sit relative to the frame's own order; torch rejects the last outright and
    warns about the second. One guard at the boundary beats auditing the layout at each caller.
    """
    if torch.is_tensor(array):
        return array.to(dtype)
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)


class SpatialPatterns:
    """Experiments grouped by which spatial coefficients they load on.

    Parameters
    ----------
    spatial_block : :obj:`numpy.ndarray`
        The experiment-level blocks of every spatial term, concatenated column-wise.

    Attributes
    ----------
    loadings : :obj:`numpy.ndarray`
        One row per distinct pattern, of shape ``(n_patterns, n_spatial_columns)``.
    assignment : :obj:`numpy.ndarray`
        Index into ``loadings`` for each experiment.
    """

    def __init__(self, spatial_block):
        spatial_block = np.asarray(spatial_block, dtype=float)
        if spatial_block.ndim != 2:
            raise ValueError("spatial_block must be two-dimensional.")
        loadings, assignment = np.unique(spatial_block, axis=0, return_inverse=True)
        self.loadings = loadings
        self.assignment = np.asarray(assignment).reshape(-1)

    @property
    def n_patterns(self):
        """Number of distinct spatial loadings."""
        return self.loadings.shape[0]

    @property
    def n_experiments(self):
        """Number of experiments."""
        return self.assignment.shape[0]

    @property
    def is_degenerate(self):
        """Whether every experiment has its own pattern, so nothing is shared.

        The general (experiment x voxel) case. Not an error -- a spatial covariate legitimately
        gives every experiment a distinct map -- but it is the expensive end of the range and
        worth being able to report.
        """
        return self.n_patterns == self.n_experiments

    def marginal_by_pattern(self, foci):
        """Sum foci counts over the experiments sharing each pattern.

        Written as a product with the ``(n_experiments, n_patterns)`` membership matrix. That
        keeps a sparse ``foci`` sparse, so the cost is one pass over the stored nonzeros rather
        than one over the full experiment-by-voxel grid -- the array this module exists to avoid,
        31 GB at 17,000 experiments over a 2 mm mask. The same expression handles a dense
        ``foci``, and is faster there than the scatter-add it replaces.

        Returns
        -------
        :obj:`numpy.ndarray`
            Shape ``(n_patterns, n_voxels)``.
        """
        if foci.shape[0] != self.n_experiments:
            raise ValueError(
                f"foci has {foci.shape[0]} rows but the design covers {self.n_experiments} "
                "experiments."
            )
        membership = scipy.sparse.csr_matrix(
            (np.ones(self.n_experiments), (np.arange(self.n_experiments), self.assignment)),
            shape=(self.n_experiments, self.n_patterns),
        )
        marginal = membership.T @ foci
        if scipy.sparse.issparse(marginal):
            marginal = marginal.todense()
        return np.asarray(marginal, dtype=float)


class CBMRPredictor:
    """The linear predictor implied by a bound design and a spline basis.

    Parameters
    ----------
    bound_design : :class:`~nimare.meta.cbmr.terms.BoundDesign`
        Design resolved against the experiment annotations.
    bases : :obj:`numpy.ndarray`
        Spline basis, of shape ``(n_voxels, n_bases)``.
    """

    def __init__(self, bound_design, bases):
        self.design = bound_design
        self.bases = np.asarray(bases, dtype=float)
        if self.bases.ndim != 2:
            raise ValueError("bases must be two-dimensional (n_voxels x n_bases).")

        spatial_blocks = [block.block for block in bound_design.blocks if block.term.spatial]
        # An exposure term is non-spatial but owns no coefficient, so it must not reach
        # ``global_block``: everything downstream reads that as the fitted moderator columns and
        # would quietly hand the exposure a coefficient, which is the one thing it must not have.
        global_blocks = [
            block.block
            for block in bound_design.blocks
            if not block.term.spatial and not block.term.exposure
        ]
        if not spatial_blocks:
            raise ValueError(
                "A CBMR design needs at least one spatial term; the spatial intensity is the "
                "quantity being estimated."
            )

        self.spatial_block = np.hstack(spatial_blocks)
        self.global_block = np.hstack(global_blocks) if global_blocks else None
        self.exposure = bound_design.exposure
        self.patterns = SpatialPatterns(self.spatial_block)

    @property
    def n_voxels(self):
        """Number of voxels the basis is evaluated at."""
        return self.bases.shape[0]

    @property
    def n_bases(self):
        """Width of the spline basis."""
        return self.bases.shape[1]

    @property
    def n_spatial_columns(self):
        """Total experiment-level columns across all spatial terms."""
        return self.spatial_block.shape[1]

    @property
    def n_global_columns(self):
        """Total experiment-level columns across all non-spatial terms."""
        return 0 if self.global_block is None else self.global_block.shape[1]

    @property
    def n_parameters(self):
        """Total number of coefficients."""
        return self.n_spatial_columns * self.n_bases + self.n_global_columns

    @property
    def has_exposure(self):
        """Whether the design carries an exposure."""
        return self.exposure is not None

    def experiment_weights(self, global_coef, dtype=None):
        """Return ``E_i exp(m_i)``, the multiplier on each experiment's spatial intensity.

        The single place this product is formed. Every consumer -- the likelihood, the closed-form
        information, the sandwich, the overdispersed marginals -- needs exactly it, and an earlier
        arrangement in which two of them built it separately is how a change to one could be
        silently missing from the other.
        """
        weights = torch.exp(self.moderator_effect(global_coef))
        if dtype is not None:
            weights = weights.to(dtype)
        if self.exposure is None:
            return weights
        return _as_tensor(self.exposure, weights.dtype) * weights

    def log_intensity_by_pattern(self, spatial_coef):
        """Return the spatial log-intensity for each distinct pattern.

        Parameters
        ----------
        spatial_coef : :obj:`torch.Tensor`
            Shape ``(n_spatial_columns, n_bases)``.

        Returns
        -------
        :obj:`torch.Tensor`
            Shape ``(n_patterns, n_voxels)``.
        """
        loadings = _as_tensor(self.patterns.loadings, spatial_coef.dtype)
        bases = _as_tensor(self.bases, spatial_coef.dtype)
        return (loadings @ spatial_coef) @ bases.T

    def moderator_effect(self, global_coef):
        """Return the scalar linear predictor contributed per experiment.

        Parameters
        ----------
        global_coef : :obj:`torch.Tensor` or None
            Shape ``(n_global_columns,)``. None when the design has no non-spatial terms.

        Returns
        -------
        :obj:`torch.Tensor`
            Shape ``(n_experiments,)``. Zeros when there are no non-spatial terms.
        """
        if self.global_block is None or global_coef is None:
            return torch.zeros(self.patterns.n_experiments, dtype=torch.float64)
        block = _as_tensor(self.global_block, global_coef.dtype)
        return block @ global_coef

    def linear_predictor(self, spatial_coef, global_coef=None):
        """Return the full (experiment x voxel) linear predictor.

        Materializes the array this module exists to avoid, so it is for diagnostics and tests
        rather than for fitting. :func:`poisson_log_likelihood` never calls it.

        Carries the coefficients only. The exposure multiplies the *mean* rather than adding to
        the predictor, so it is applied by :meth:`fitted_mean` instead; adding ``log(E_i)`` here
        would reintroduce the logarithm of zero the multiplicative form exists to avoid.
        """
        by_pattern = self.log_intensity_by_pattern(spatial_coef)
        assignment = torch.as_tensor(self.patterns.assignment, dtype=torch.long)
        eta = by_pattern[assignment]
        return eta + self.moderator_effect(global_coef)[:, None]

    def fitted_mean(self, spatial_coef, global_coef=None):
        """Return the fitted (experiment x voxel) mean, exposure included.

        ``mu_iv = E_i exp(m_i) exp(s_{p(i)}(v))``. Materializes the full array, so it is for the
        sandwich and for diagnostics rather than for fitting.
        """
        by_pattern = torch.exp(self.log_intensity_by_pattern(spatial_coef))
        assignment = torch.as_tensor(self.patterns.assignment, dtype=torch.long)
        weights = self.experiment_weights(global_coef, by_pattern.dtype)
        return by_pattern[assignment] * weights[:, None]


def poisson_log_likelihood(predictor, spatial_coef, global_coef, foci):
    """Poisson log-likelihood of ``foci`` under ``predictor``, up to a constant.

    Drops the ``-sum(log(y!))`` term, which does not depend on the parameters, matching what
    CBMR has always reported, and with an exposure also drops ``sum_i y_i. log E_i``, which does
    not either. The second makes log-likelihoods incomparable across designs with different
    exposures; see the module docstring.

    Evaluated on marginals rather than on the (experiment x voxel) array; see the module
    docstring. Exact, not an approximation -- verified against the elementwise form.

    Parameters
    ----------
    predictor : :class:`CBMRPredictor`
        Assembled predictor.
    spatial_coef : :obj:`torch.Tensor`
        Shape ``(n_spatial_columns, n_bases)``.
    global_coef : :obj:`torch.Tensor` or None
        Shape ``(n_global_columns,)``.
    foci : :obj:`scipy.sparse.spmatrix` or :obj:`numpy.ndarray`
        Foci counts, of shape ``(n_experiments, n_voxels)``.

    Returns
    -------
    :obj:`torch.Tensor`
        Scalar log-likelihood.
    """
    marginal_by_pattern = _as_tensor(
        predictor.patterns.marginal_by_pattern(foci), spatial_coef.dtype
    )
    foci_per_experiment = _as_tensor(experiment_totals(foci), spatial_coef.dtype)

    log_intensity = predictor.log_intensity_by_pattern(spatial_coef)
    moderator = predictor.moderator_effect(global_coef).to(spatial_coef.dtype)

    # sum_p Y_p . s_p
    spatial_term = torch.sum(marginal_by_pattern * log_intensity)
    # sum_i y_i. m_i
    moderator_term = torch.dot(foci_per_experiment, moderator)

    # sum_p (sum_v exp(s_p)) T_p, with T_p the summed weight within pattern p
    intensity_sum = torch.exp(log_intensity).sum(dim=1)
    assignment = torch.as_tensor(predictor.patterns.assignment, dtype=torch.long)
    moderator_sum = torch.zeros(
        predictor.patterns.n_patterns, dtype=spatial_coef.dtype
    ).index_add_(0, assignment, predictor.experiment_weights(global_coef, spatial_coef.dtype))

    return spatial_term + moderator_term - torch.dot(intensity_sum, moderator_sum)
