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

**The block is exchangeable, and real spatial correlation is not.** Conditioning on the study
effect leaves the elements independent, so every pair within a block correlates equally. Measured
against exact multivariate-normal orthant probabilities at matched mean correlation, the
exchangeable model **understates a block's silence probability** -- so it overstates how often a
block reports -- by up to 0.030 at nine elements with slowly decaying correlation and a liberal
cut. Restricted to strict cuts, standardised 2.5 and above, the worst error falls to 0.005; at a
mean within-block correlation of 0.15 or less it is 0.010. So this is usable for the strict
thresholds published tables come from and is *not* a general substitute for a spatial model: for
heavily smoothed data at a liberal cut the conditional multivariate-normal route is needed.
Measured in ``experiments/exchangeable_block_error.py``, which states a 0.01 kill condition in
advance and fails it.

**This is a composite likelihood unless you say otherwise.** ``study_index`` groups blocks
by the study they came from and makes the likelihood exact. Without it every block is treated as
its own study, which overstates the information badly once a study contributes several blocks --
see :func:`standard_error_inflation`.

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
    # numpy's hermgauss underflows above roughly 400 nodes: at 600 it returns zero weights and
    # warns inside its own routine, and the resulting rule yields nan rather than a more accurate
    # answer. Refuse it rather than return a number computed from a broken rule.
    if np.any(weights <= 0):
        raise ValueError(
            f"A {int(nodes)}-node Gauss-Hermite rule underflows: "
            f"{int(np.sum(weights <= 0))} of its weights are zero. Use fewer nodes; more is "
            "not more accurate here."
        )
    return positions * np.sqrt(2.0 * between_variance), np.log(weights) - 0.5 * np.log(np.pi)


def _standardised(values, mean, offsets, scales):
    """Standardised distance of each block's cut-point from the mean, per quadrature node."""
    return (values[:, None] - mean - offsets[None, :]) / scales[:, None]


def _conditional_log_terms(
    mean, element_variances, heights, thresholds, elements, offsets, use_heights
):
    """Log-probability of each block's observation given each study-effect node.

    The ``(blocks, nodes)`` matrix everything else is built from. Keeping it explicit is what
    makes the exact per-study likelihood possible: the shared study effect has to be held fixed
    across a study's blocks before it is integrated out, not integrated out block by block.
    """
    scales = np.sqrt(element_variances)
    reported = np.isfinite(heights)
    terms = np.empty((heights.size, offsets.size), dtype=float)

    cut = _standardised(thresholds, mean, offsets, scales)
    log_silent = elements[:, None] * log_ndtr(cut)

    if reported.any():
        if use_heights:
            standardised = _standardised(heights[reported], mean, offsets, scales[reported])
            counts = elements[reported][:, None]
            terms[reported] = (
                np.log(elements[reported])[:, None]
                - np.log(scales[reported])[:, None]
                + (counts - 1.0) * log_ndtr(standardised)
                - 0.5 * standardised**2
                - 0.5 * np.log(2.0 * np.pi)
            )
        else:
            # log(1 - exp(x)) for x <= 0, computed the stable way, conditional on the node
            # rather than on the marginal silence probability.
            terms[reported] = np.log(-np.expm1(np.minimum(log_silent[reported], -1e-300)))
    silent = ~reported
    if silent.any():
        terms[silent] = log_silent[silent]
    return terms


def block_loglik(
    mean,
    between_variance,
    element_variances,
    *,
    heights,
    thresholds,
    elements,
    study_index=None,
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
    study_index : :obj:`numpy.ndarray`, optional
        Which study each block belongs to. **This changes the likelihood, not just its
        bookkeeping.** Blocks of one study share that study's effect, so the exact likelihood
        integrates it out once per study,
        :math:`L_i = \int \prod_b t_b(u)\,p(u)\,du`. Left unset, every block is treated as its
        own study, which gives the *composite* likelihood
        :math:`\tilde L_i = \prod_b \int t_b(u)\,p(u)\,du` -- correct only when that is
        actually true.

        The two are not close. For an all-silent study the composite version understates the
        likelihood and **overstates the information**, because it counts one draw of the study
        effect as many. Measured at nine elements per block and a strict threshold, the ratio of
        exact to composite standard error runs 1.000, 0.798, 0.615, 0.421, 0.204 at 1, 2, 4, 10
        and 50 blocks per study: a whole-brain composite fit reports an interval five times too
        narrow. Derived in ``proofs/composite_block_likelihood.py``;
        :func:`standard_error_inflation` measures it for a given configuration.
    use_heights : :obj:`bool`, default=True
        Read the reported height. With ``False`` a report contributes only the probability that
        the block reported at all, which is the indicator-only likelihood -- the comparison the
        design assessment calls modest and which this makes measurable.
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
    terms = _conditional_log_terms(
        mean, element_variances, heights, thresholds, elements, offsets, use_heights
    )

    if study_index is None:
        totals = logsumexp(terms + log_weights[None, :], axis=1)
    else:
        labels = np.asarray(study_index).reshape(-1)
        if labels.size != heights.size:
            raise ValueError(
                f"study_index must have one entry per block; got {labels.size} for "
                f"{heights.size} blocks."
            )
        codes, positions = np.unique(labels, return_inverse=True)
        stacked = np.zeros((codes.size, offsets.size), dtype=float)
        np.add.at(stacked, positions, terms)
        totals = logsumexp(stacked + log_weights[None, :], axis=1)

    if not np.all(np.isfinite(totals)):
        return -np.inf
    return float(totals.sum())


def standard_error_inflation(
    mean, between_variance, element_variances, *, step=1e-3, study_index=None, **kwargs
):
    r"""Ratio of the exact standard error to the composite one, at this configuration.

    Returns ``exact_se / composite_se``, which is at most one: the composite likelihood treats
    one draw of a study's effect as many independent ones, so it overstates the information and
    understates the interval. A value of 0.5 means a composite fit's interval is half the width
    it should be.

    This is the calibration the design assessment asks for in its step 4 -- "assess approximation
    error against the small exact models" -- and it is a diagnostic, not a correction. Applying
    it would need a sandwich or Godambe adjustment, which this does not do.
    """
    if study_index is None:
        raise ValueError(
            "standard_error_inflation compares a grouped fit against an ungrouped one, so it "
            "needs study_index; without it the two are the same likelihood."
        )

    def curvature(grouping):
        values = [
            block_loglik(
                candidate,
                between_variance,
                element_variances,
                study_index=grouping,
                **kwargs,
            )
            for candidate in (mean - step, mean, mean + step)
        ]
        if not all(np.isfinite(values)):
            return np.nan
        return -(values[0] - 2 * values[1] + values[2]) / step**2

    exact = curvature(study_index)
    composite = curvature(None)
    if not (np.isfinite(exact) and np.isfinite(composite)) or composite <= 0 or exact <= 0:
        return np.nan
    return float(np.sqrt(exact / composite))


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
