r"""A predefined block with general covariance: silence, a recorded maximum, and its location.

Section 3 of the revised plan asks for the exact small-block formulation with arbitrary
covariance, in place of the exchangeable block in :mod:`nimare.meta.cbma.blocks`. For a fixed,
predefined block :math:`Y \sim N(\mu, \Sigma)` of :math:`M` elements:

.. math::
    P\left(\max_j Y_j \le c\right) = \Phi_M(c\mathbf 1;\ \mu,\ \Sigma),
    \qquad
    p(J=j,\ H=h) = f_{Y_j}(h)\ P\!\left(Y_{-j} \le h\mathbf 1 \mid Y_j = h\right),

with the Gaussian conditional mean :math:`\mu_{-j} + \Sigma_{-j,j}\Sigma_{jj}^{-1}(h-\mu_j)` and
conditional covariance the Schur complement
:math:`\Sigma_{-j,-j} - \Sigma_{-j,j}\Sigma_{jj}^{-1}\Sigma_{j,-j}`. Sum over :math:`j` only when
the table did not record which location the maximum was at. Derived in
``proofs/general_covariance_blocks.py``.

**Why this exists rather than the exchangeable block.** The exchangeable model generates
within-block dependence from a shared *study* effect, which ties the within-block correlation to
the between-study variance: :math:`r = \tau^2/(\tau^2+\sigma_\epsilon^2)`. So it cannot represent
a smooth field with no heterogeneity -- correlated voxels, identical studies -- because
:math:`\tau^2 \to 0` forces :math:`r \to 0`. Representing a within-block correlation of 0.5 at a
sampling sd of 0.25 *requires* a between-study sd of 0.25 in that family. Sampling covariance and
between-study structure have to be separate parameters, which is what
:mod:`nimare.meta.cbma.spatial` writes as :math:`S_i` and :math:`D`, and what this module takes
as an arbitrary :math:`\Sigma`.

Notes
-----
**A predefined block is not a published cluster.** Cluster extent criteria, sub-peak suppression
and table-length selection are events outside this set. The plan is explicit that this is a
tractable *reference* for small blocks against which approximations are measured, not a claim
that every published cluster is a predetermined region.

**The orthant probability is numerical.** SciPy's multivariate normal CDF is quasi-Monte Carlo,
so every quantity here carries its error -- around :math:`10^{-5}` at these dimensions.
:func:`partition_residual` is the check that the pieces still exhaust the probability, and it
reports the residual rather than asserting a tolerance.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

#: Above this many elements the orthant probability stops being a reference and starts being a
#: Monte Carlo estimate with an error larger than the quantities it is compared against.
MAX_REFERENCE_ELEMENTS = 12


def _validate(mean, covariance):
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    size = mean.size
    if covariance.shape != (size, size):
        raise ValueError(f"covariance must be square of side {size}; got {covariance.shape}.")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("covariance must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() <= 0:
        raise ValueError(
            "covariance must be positive definite; its smallest eigenvalue is "
            f"{eigenvalues.min():.3e}. A singular block is a smaller block with a duplicated "
            "element, not a degenerate one to be regularised silently."
        )
    if size > MAX_REFERENCE_ELEMENTS:
        raise ValueError(
            f"This is a small-block reference: {size} elements exceeds "
            f"{MAX_REFERENCE_ELEMENTS}, beyond which the orthant probability's Monte Carlo "
            "error swamps what it is used to measure. Use the block or composite likelihoods "
            "for larger regions, with their approximation error measured."
        )
    return mean, covariance


def block_silence_log_probability(mean, covariance, threshold):
    r"""Log :math:`\Phi_M(c\mathbf 1;\ \mu,\ \Sigma)`: no element cleared the threshold."""
    mean, covariance = _validate(mean, covariance)
    probability = float(
        multivariate_normal(mean=mean, cov=covariance).cdf(np.full(mean.size, float(threshold)))
    )
    if not 0.0 < probability <= 1.0:
        return -np.inf
    return float(np.log(probability))


def _conditional(mean, covariance, index, value):
    """Gaussian conditional mean and covariance of the other elements given one of them."""
    others = [position for position in range(mean.size) if position != index]
    cross = covariance[np.ix_(others, [index])] / covariance[index, index]
    conditional_mean = mean[others] + (cross * (value - mean[index])).ravel()
    conditional_covariance = (
        covariance[np.ix_(others, others)] - cross @ covariance[np.ix_([index], others)]
    )
    return conditional_mean, conditional_covariance


def recorded_maximum_log_density(mean, covariance, index, height):
    r"""Log density of a maximum recorded **at a named location**.

    .. math:: p(J=j, H=h) = f_{Y_j}(h)\ P(Y_{-j} \le h\mathbf 1 \mid Y_j = h)

    Use this when the table says *where* the peak was, which published tables do. Summing over
    locations -- :func:`maximum_log_density` -- discards that information and is only correct
    when the location genuinely was not recorded.
    """
    mean, covariance = _validate(mean, covariance)
    index = int(index)
    if not 0 <= index < mean.size:
        raise ValueError(f"index must be a block position in [0, {mean.size}); got {index}.")
    height = float(height)

    marginal = norm.logpdf(height, loc=mean[index], scale=np.sqrt(covariance[index, index]))
    if mean.size == 1:
        return float(marginal)

    conditional_mean, conditional_covariance = _conditional(mean, covariance, index, height)
    if conditional_mean.size == 1:
        tail = norm.cdf(
            height, loc=conditional_mean[0], scale=np.sqrt(conditional_covariance[0, 0])
        )
    else:
        tail = multivariate_normal(mean=conditional_mean, cov=conditional_covariance).cdf(
            np.full(conditional_mean.size, height)
        )
    tail = float(tail)
    if tail <= 0.0:
        return -np.inf
    return float(marginal + np.log(tail))


def maximum_log_density(mean, covariance, height):
    """Log density of the block maximum with its location **not** recorded, summed over j."""
    mean, covariance = _validate(mean, covariance)
    terms = [
        recorded_maximum_log_density(mean, covariance, index, height) for index in range(mean.size)
    ]
    finite = [term for term in terms if np.isfinite(term)]
    if not finite:
        return -np.inf
    largest = max(finite)
    return float(largest + np.log(sum(np.exp(term - largest) for term in finite)))


def partition_residual(mean, covariance, threshold, upper=None, by_location=True):
    r"""How far the reporting events are from exhausting the probability.

    Integrates every recorded-location density above the threshold and adds the silence
    probability. Returns a dict with ``silence``, ``reported`` (per location when
    ``by_location``), ``total`` and ``residual``.

    The residual is **reported, not asserted small**: the orthant probability is quasi-Monte
    Carlo and carries its own error, so a residual near :math:`10^{-6}` is the integrator's and
    a residual near :math:`10^{-2}` is a formulation error. Distinguishing them is the caller's
    job and cannot be done by a tolerance chosen here.
    """
    mean, covariance = _validate(mean, covariance)
    threshold = float(threshold)
    if upper is None:
        upper = float(np.max(mean) + 8.0 * np.sqrt(np.max(np.diag(covariance))))

    silence = float(np.exp(block_silence_log_probability(mean, covariance, threshold)))
    masses = []
    for index in range(mean.size):
        mass = quad(
            lambda value, position=index: float(
                np.exp(recorded_maximum_log_density(mean, covariance, position, value))
            ),
            threshold,
            upper,
            limit=200,
        )[0]
        masses.append(float(mass))

    total = silence + float(np.sum(masses))
    return {
        "silence": silence,
        "reported": masses if by_location else float(np.sum(masses)),
        "total": total,
        "residual": abs(total - 1.0),
    }


def draw_constrained_block(
    mean, covariance, threshold, index=None, height=None, rng=None, sweeps=25
):
    r"""Draw block values consistent with the observed reporting event, for augmentation.

    Section 3.1's computational alternative: when orthant probabilities are the bottleneck,
    impute the block's unobserved values from the Gaussian **constrained by the genuine
    observation event** and integrate by Monte Carlo instead.

    * silence (``index is None``) constrains every element below ``threshold``;
    * a recorded maximum at ``index`` with ``height`` fixes that element and constrains the rest
      below it.

    Componentwise truncated-normal Gibbs sweeps, using the conditional Gaussian.

    .. warning::
        These are **integration variables inside one model**, not completed images to pass
        onward as observed evidence. Their uncertainty is the thing being integrated; taking a
        single draw and treating it as data is the imputation error this whole construction
        exists to avoid. General cluster-extent, suppression and table-selection events involve
        inequalities outside this truncation set and need their actual operator.
    """
    mean, covariance = _validate(mean, covariance)
    generator = np.random.default_rng() if rng is None else rng
    threshold = float(threshold)
    size = mean.size

    upper_bounds = np.full(size, threshold)
    fixed = np.zeros(size, dtype=bool)
    state = np.minimum(mean.copy(), threshold - 0.1 * np.sqrt(np.diag(covariance)))

    if index is not None:
        if height is None:
            raise ValueError("A recorded maximum needs both its index and its height.")
        index = int(index)
        if float(height) < threshold:
            raise ValueError(
                f"A recorded maximum of {float(height):g} does not clear its threshold of "
                f"{threshold:g}; a table and a threshold that contradict each other are a data "
                "problem, not something to sample around."
            )
        fixed[index] = True
        state[index] = float(height)
        upper_bounds[:] = float(height)
        upper_bounds[index] = float(height)

    precision = np.linalg.inv(covariance)
    for _ in range(int(sweeps)):
        for position in range(size):
            if fixed[position]:
                continue
            others = [other for other in range(size) if other != position]
            conditional_variance = 1.0 / precision[position, position]
            conditional_mean = mean[position] - conditional_variance * float(
                precision[position, others] @ (state[others] - mean[others])
            )
            scale = np.sqrt(conditional_variance)
            limit = norm.cdf((upper_bounds[position] - conditional_mean) / scale)
            limit = min(max(limit, 1e-12), 1.0 - 1e-12)
            state[position] = conditional_mean + scale * norm.ppf(generator.uniform(0.0, limit))
    return state
