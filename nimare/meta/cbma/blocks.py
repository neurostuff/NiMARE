r"""Block peak-selection likelihood: a reported maximum, or a block that stayed silent.

Where :mod:`nimare.meta.cbma.censored` treats an observation as an interval about one estimate,
a block treats a group of correlated elements together and the observation is *the largest of
them, if it cleared the threshold*. This is the model a coordinate table actually generates: a
paper prints one peak per cluster, not one value per voxel.

For block :math:`i` with :math:`M_i` elements,

.. math::
    Y_{ij} = m + U_i + \epsilon_{ij},\quad U_i \sim N(0, \tau^2),
    \ \epsilon_{ij}\sim N(0,\sigma_i^2),

so the block is equicorrelated at :math:`\tau^2/(\tau^2+\sigma^2)`. Conditionally on
:math:`U_i` the elements are independent, which makes the maximum's conditional law exact:

.. math::
    P(H_i \le h \mid U_i) = \Phi\!\left(\tfrac{h-m-U_i}{\sigma_i}\right)^{M_i},
    \qquad
    f_{H_i\mid U_i}(h) = \frac{M_i}{\sigma_i}
    \Phi\!\left(\tfrac{h-m-U_i}{\sigma_i}\right)^{M_i-1}
    \varphi\!\left(\tfrac{h-m-U_i}{\sigma_i}\right).

The unconditional versions are integrals over :math:`U_i` with no closed form, and are taken by
Gauss-Hermite quadrature. Everything else is derived in ``proofs/block_peak_selection.py``.

Notes
-----
**On capping the number of peaks.** One maximum per *block* is not a cap on a study's peak count:
a whole-brain model has many blocks, each contributing its own maximum-or-silence, and the
study's total is whatever survives. It becomes a cap only when a study is given a single block,
which is a thing to do deliberately and to say, not a default. :func:`block_loglik` takes as many
blocks per study as it is given and imposes no limit.

**When heights are worth reading.** The reported height is a location family in the mean, so its
mean-score is minus its height-score and its contribution is an ordinary location-family Fisher
information. Comparing that with the report indicator's information shows how much the height
adds, and the answer depends strongly on the threshold: at a cut where reporting is near a coin
flip the indicator carries most of the information and the height adds a few percent, while at a
liberal cut the height carries nearly all of it. ``use_heights=False`` computes the
indicator-only likelihood so the two can be measured against each other on the same data rather
than argued about.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, logsumexp

#: Gauss-Hermite nodes for the integral over the study effect. Doubling this must not move a
#: log-likelihood materially; :func:`quadrature_is_converged` is the check.
DEFAULT_NODES = 64


def _quadrature(between_variance, nodes):
    r"""Nodes and log-weights for :math:`\\int f(u)\\,N(u; 0, \\tau^2)\\,du`.

    A zero between-study variance is a point mass, not a degenerate integral, so it returns the
    single node at zero rather than a rule that would divide by the scale.
    """
    between_variance = float(between_variance)
    if between_variance <= 0:
        return np.zeros(1), np.zeros(1)
    positions, weights = np.polynomial.hermite.hermgauss(int(nodes))
    return positions * np.sqrt(2.0 * between_variance), np.log(weights) - 0.5 * np.log(np.pi)


def _standardised(values, mean, offsets, scales):
    """Standardised distance of each block's cut-point from the mean, per quadrature node."""
    return (values[:, None] - mean - offsets[None, :]) / scales[:, None]


def block_loglik(
    mean,
    between_variance,
    element_variances,
    *,
    heights,
    thresholds,
    elements,
    use_heights=True,
    nodes=DEFAULT_NODES,
):
    r"""Total log-likelihood of reported block maxima and silent blocks.

    Parameters
    ----------
    mean, between_variance : :obj:`float`
        The marginal mean :math:`m` and the between-study variance :math:`\tau^2`.
    element_variances : :obj:`numpy.ndarray`
        Within-study variance :math:`\sigma_i^2` of one element of block :math:`i`.
    heights : :obj:`numpy.ndarray`
        Reported maximum of each block, or ``nan`` where the block reported nothing. A block is
        silent if and only if its height is not finite, so silence is a state of the data rather
        than a separate argument that could disagree with it.
    thresholds : :obj:`numpy.ndarray`
        The threshold each block's maximum had to clear. Supplied, never inferred: the smallest
        reported height is an order statistic and moves with how much signal a study had.
    elements : :obj:`numpy.ndarray`
        Number of elements :math:`M_i` in each block.
    use_heights : :obj:`bool`, default=True
        Read the reported height. With ``False`` a report contributes only
        :math:`\log[1 - P(H \le c)]`, which is the indicator-only likelihood -- the comparison
        the design assessment calls modest and which this makes measurable.
    nodes : :obj:`int`
        Gauss-Hermite nodes for the integral over the study effect.

    Returns
    -------
    :obj:`float`
        The log-likelihood, or ``-inf`` where any block has numerically zero probability so that
        an optimiser walks away from the point rather than through it.

    Raises
    ------
    ValueError
        If a reported height does not clear its own threshold. That is not a numerical problem
        to clip; it is a table and a threshold that contradict each other, and continuing would
        fit a model to an impossible observation.
    """
    heights = np.asarray(heights, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    elements = np.asarray(elements, dtype=float)
    element_variances = np.asarray(element_variances, dtype=float)
    if not (heights.shape == thresholds.shape == elements.shape == element_variances.shape):
        raise ValueError(
            "heights, thresholds, elements and element_variances must have the same shape; got "
            f"{heights.shape}, {thresholds.shape}, {elements.shape} and "
            f"{element_variances.shape}."
        )
    if np.any(element_variances <= 0):
        raise ValueError("Every element variance must be positive.")
    if np.any(elements < 1):
        raise ValueError("Every block must have at least one element.")

    reported = np.isfinite(heights)
    if np.any(reported & (heights < thresholds)):
        offender = int(np.flatnonzero(reported & (heights < thresholds))[0])
        raise ValueError(
            f"Block {offender} reports a maximum of {heights[offender]:g} below its own "
            f"threshold of {thresholds[offender]:g}. A table and a threshold that contradict "
            "each other are a data problem, not something to clip."
        )

    offsets, log_weights = _quadrature(between_variance, nodes)
    scales = np.sqrt(element_variances)

    total = 0.0

    if reported.any():
        if use_heights:
            standardised = _standardised(heights[reported], mean, offsets, scales[reported])
            counts = elements[reported][:, None]
            log_density = (
                np.log(elements[reported])[:, None]
                - np.log(scales[reported])[:, None]
                + (counts - 1.0) * log_ndtr(standardised)
                - 0.5 * standardised**2
                - 0.5 * np.log(2.0 * np.pi)
            )
            terms = logsumexp(log_density + log_weights[None, :], axis=1)
        else:
            standardised = _standardised(thresholds[reported], mean, offsets, scales[reported])
            log_silent = elements[reported][:, None] * log_ndtr(standardised)
            silent = logsumexp(log_silent + log_weights[None, :], axis=1)
            # log(1 - exp(x)) for x <= 0, computed the stable way.
            terms = np.log(-np.expm1(np.minimum(silent, -1e-300)))
        if not np.all(np.isfinite(terms)):
            return -np.inf
        total += float(terms.sum())

    silent_blocks = ~reported
    if silent_blocks.any():
        standardised = _standardised(
            thresholds[silent_blocks], mean, offsets, scales[silent_blocks]
        )
        log_silent = elements[silent_blocks][:, None] * log_ndtr(standardised)
        terms = logsumexp(log_silent + log_weights[None, :], axis=1)
        if not np.all(np.isfinite(terms)):
            return -np.inf
        total += float(terms.sum())

    return total


def quadrature_is_converged(
    mean, between_variance, element_variances, *, tolerance=1e-6, **kwargs
):
    """Report whether doubling the quadrature nodes moves the log-likelihood.

    A quadrature rule that has not converged produces a likelihood that is smooth, plausible and
    wrong, which no amount of downstream checking catches. This is cheap and belongs in any run
    that reports a number.
    """
    kwargs.pop("nodes", None)
    coarse = block_loglik(mean, between_variance, element_variances, nodes=DEFAULT_NODES, **kwargs)
    fine = block_loglik(
        mean, between_variance, element_variances, nodes=2 * DEFAULT_NODES, **kwargs
    )
    if not (np.isfinite(coarse) and np.isfinite(fine)):
        return False, np.inf
    gap = abs(fine - coarse) / max(abs(fine), 1.0)
    return bool(gap < tolerance), float(gap)


def block_report_probability(
    mean, between_variance, element_variances, *, thresholds, elements, nodes=DEFAULT_NODES
):
    r"""Probability that each block reports, :math:`1 - \mathbb{E}_U[\Phi(\cdot)^{M}]`.

    Exposed because it is what a reporting model can be checked against on real tables: the
    fraction of blocks that printed a peak is observable, and a model that cannot match it is
    not describing the corpus whatever its likelihood says.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    elements = np.asarray(elements, dtype=float)
    scales = np.sqrt(np.asarray(element_variances, dtype=float))
    offsets, log_weights = _quadrature(between_variance, nodes)
    standardised = _standardised(thresholds, mean, offsets, scales)
    silent = logsumexp(elements[:, None] * log_ndtr(standardised) + log_weights[None, :], axis=1)
    return -np.expm1(np.minimum(silent, 0.0))
