r"""Prospective planning from a meta-analysis: power, assurance, and the ceiling assurance obeys.

A meta-analytic effect is often wanted in order to size a future study. The quantity that
answers that question is not the power at the estimated effect but the **assurance**, the power
averaged over what is known about the future study's own effect:

.. math::
    \mathcal{A}(n) = \int \operatorname{Power}(n, \theta)\,
    p(\theta_{\rm new}\mid\text{data})\,d\theta,
    \qquad
    p(\theta_{\rm new}\mid\text{data}) = N\!\left(\hat m,\ \hat\tau^2
    + \operatorname{se}(\hat m)^2\right).

The predictive variance has two terms and both are load-bearing. Dropping the estimation error
treats the meta-analytic mean as known; dropping the heterogeneity treats a future study as a
repeat of the mean, which is the commoner and larger mistake.

**The direction of the error from plugging in.** Power is a sigmoid in the effect whose
inflection sits exactly at the effect giving 50% power. By Jensen, assurance is therefore
*below* plug-in power whenever the estimate exceeds that effect and above it when it falls short:
plug-in power is optimistic precisely in the regime where a study is being designed to be
adequately powered. Derived in ``proofs/assurance_not_plug_in_power.py``, and confirmed there
against the exact noncentral t.

**The limit, and what it is a limit on.** In the normal approximation
:math:`\mathcal{A}(n) = \Phi\!\left((\sqrt n \hat m - z)/\sqrt{1 + n s^2}\right)`, so as
:math:`n \to \infty` assurance tends to :math:`\Phi(\hat m / s)` and no sample size buys more.
That limit is for rejection **in a stated direction**. Two-sided power tends to one at every
non-zero effect, so two-sided assurance tends to one and has no informative ceiling. Assurance
here is therefore directional by default: a rejection in the wrong direction is not a successful
study. :func:`assurance_ceiling` reports the limit, because a power curve that climbs to one is
a power curve that has forgotten heterogeneity. It is an *asymptotic* value, not a
finite-sample bound: with wrong-sign components in the predictive distribution, assurance need
not be monotone and can exceed its own limit at finite :math:`n`.

Notes
-----
**What this does not do.** Everything here is for a single prespecified effect at a single
location, with a simple one- or two-sample t design. Whole-brain voxel, cluster or TFCE power
depends on the spatial covariance, the signal's extent, the search volume and the correction
procedure, none of which appear here; :func:`roi_standardised_effect` exists only to make the
point that an ROI's standardised effect is not the average of its voxels' and needs their
covariance.

**Where the predictive variance comes from matters more than this code does.** An assurance
computed from a heterogeneity estimated on a handful of image studies inherits that estimate's
uncertainty, which this does not propagate. Treat the output as conditional on the supplied
predictive standard deviation, and vary it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nct, norm

DESIGNS = ("one-sample", "two-sample")

#: Grid resolution for the average over the predictive distribution. Adaptive quadrature is not
#: used: it returns ``nan`` by roundoff once the integrand is nearly constant, and a ``nan``
#: silently compared against a threshold is a verdict from a missing number. Nor is Gauss-Hermite,
#: which cannot resolve a power curve's transition -- see :func:`_predictive_grid`.
#:
#: Unlike the block module this needs no underflow guard on the node count, because the rule here
#: is a refined linspace and not a Gauss-Hermite one: there are no quadrature weights to
#: underflow. Raising it is safe and does make the answer more accurate.
DEFAULT_NODES = 96

#: Above this non-centrality scipy's noncentral t is unreliable, while the power is one to
#: machine precision. :func:`two_sided_power` substitutes exactly one there *after* checking the
#: normal approximation agrees, so the substitution is verified rather than assumed.
_FAR_NONCENTRALITY = 30.0


def design_scale(sample_size, design="one-sample", allocation=0.5):
    r"""Standard-error multiplier :math:`\kappa` linking a standardised effect to a t statistic.

    For a one-sample or paired design :math:`\kappa = 1/\sqrt{n}`; for two independent groups
    :math:`\kappa = \sqrt{1/n_1 + 1/n_2}`. The non-centrality is then :math:`\theta/\kappa`.

    Parameters
    ----------
    sample_size : array_like
        Total number of observations.
    design : {"one-sample", "two-sample"}
        Which convention applies. The document is explicit that these are different estimands
        with different power implications and cannot both be served by a total count alone.
    allocation : :obj:`float`
        Fraction of the total in the first group, for a two-sample design.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}; got {design!r}.")
    total = np.asarray(sample_size, dtype=float)
    if np.any(total < 3):
        raise ValueError("A t design needs at least three observations.")
    if design == "one-sample":
        return 1.0 / np.sqrt(total)
    if not 0.0 < allocation < 1.0:
        raise ValueError("allocation must lie strictly between zero and one.")
    first = total * allocation
    second = total - first
    if np.any(first < 2) or np.any(second < 2):
        raise ValueError("Each group of a two-sample design needs at least two observations.")
    return np.sqrt(1.0 / first + 1.0 / second)


def degrees_of_freedom(sample_size, design="one-sample"):
    """Degrees of freedom of the t statistic for the design."""
    total = np.asarray(sample_size, dtype=float)
    return total - 1.0 if design == "one-sample" else total - 2.0


def two_sided_power(effect, sample_size, *, alpha=0.05, design="one-sample", allocation=0.5):
    r"""Exact two-sided power of a t test at a fixed standardised effect.

    Uses the noncentral t rather than a normal approximation. On a grid of sample sizes from 10
    to 80 and effects from 0.2 to 0.8 the normal approximation is out by up to 0.099, all in the
    optimistic direction, so the approximation is fine for algebra and not for a recommendation.

    Power is even in the effect, and tends to one; beyond a non-centrality of 30 scipy's
    noncentral t stops being reliable while the power is one to machine precision, so exactly one
    is substituted there and the substitution is checked against the normal approximation first.
    """
    scale = design_scale(sample_size, design, allocation)
    dof = degrees_of_freedom(sample_size, design)
    noncentrality = np.abs(np.asarray(effect, dtype=float)) / scale
    critical = nct.ppf(1.0 - alpha / 2.0, dof, 0.0)

    noncentrality, critical, dof = np.broadcast_arrays(
        noncentrality, np.asarray(critical, dtype=float), np.asarray(dof, dtype=float)
    )
    out = np.empty(noncentrality.shape, dtype=float)
    far = noncentrality > _FAR_NONCENTRALITY
    near = ~far
    if near.any():
        out[near] = nct.sf(critical[near], dof[near], noncentrality[near]) + nct.sf(
            critical[near], dof[near], -noncentrality[near]
        )
    if far.any():
        if np.any(norm.cdf(noncentrality[far] - critical[far]) < 1.0 - 1e-12):
            raise ValueError(
                "The far-field power substitution was reached where the power is not "
                "numerically one; this is a bug rather than a data problem."
            )
        out[far] = 1.0
    if not np.all(np.isfinite(out)):
        raise ValueError("The noncentral-t power returned a non-finite value.")
    return out if out.ndim else float(out)


def directional_power(
    effect, sample_size, *, direction=1.0, alpha=0.05, design="one-sample", allocation=0.5
):
    r"""Probability of rejecting **in a stated direction**, which is what planning wants.

    A two-sided rejection in the wrong direction is not a successful study, and the distinction
    is not cosmetic: two-sided power tends to one for any non-zero effect, so two-sided assurance
    tends to one too and has no informative ceiling. Correctly-signed rejection tends to the
    probability that a future study's own effect has the stated sign, which is
    :func:`assurance_ceiling`.

    The critical value is still the two-sided one, so this is the power of the usual test
    decomposed by which tail it lands in, not a one-sided test at ``alpha``.
    """
    scale = design_scale(sample_size, design, allocation)
    dof = degrees_of_freedom(sample_size, design)
    signed = np.sign(float(direction)) or 1.0
    noncentrality = signed * np.asarray(effect, dtype=float) / scale
    critical = nct.ppf(1.0 - alpha / 2.0, dof, 0.0)

    noncentrality, critical, dof = np.broadcast_arrays(
        noncentrality, np.asarray(critical, dtype=float), np.asarray(dof, dtype=float)
    )
    out = np.empty(noncentrality.shape, dtype=float)
    far = np.abs(noncentrality) > _FAR_NONCENTRALITY
    near = ~far
    if near.any():
        out[near] = nct.sf(critical[near], dof[near], noncentrality[near])
    if far.any():
        # Far above the critical value the answer is one; far below it is zero. Both are checked
        # against the normal approximation before being substituted.
        high = far & (noncentrality > 0)
        low = far & (noncentrality <= 0)
        if np.any(high):
            if np.any(norm.cdf(noncentrality[high] - critical[high]) < 1.0 - 1e-12):
                raise ValueError(
                    "The far-field substitution was reached where the power is not "
                    "numerically one; this is a bug rather than a data problem."
                )
            out[high] = 1.0
        if np.any(low):
            if np.any(norm.cdf(noncentrality[low] - critical[low]) > 1e-12):
                raise ValueError(
                    "The far-field substitution was reached where the power is not "
                    "numerically zero; this is a bug rather than a data problem."
                )
            out[low] = 0.0
    if not np.all(np.isfinite(out)):
        raise ValueError("The noncentral-t power returned a non-finite value.")
    return out if out.ndim else float(out)


def predictive_standard_deviation(between_variance, standard_error):
    r"""Combine heterogeneity and estimation error into a future study's predictive spread.

    Both terms are required. Passing zero heterogeneity says a future study *is* the
    meta-analytic mean, which is a different and much stronger claim than having estimated that
    mean precisely.
    """
    between_variance = np.asarray(between_variance, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)
    if np.any(between_variance < 0) or np.any(standard_error < 0):
        raise ValueError("Neither the heterogeneity nor the standard error can be negative.")
    return np.sqrt(between_variance + standard_error**2)


def assurance(
    sample_size,
    mean,
    predictive_sd,
    *,
    alpha=0.05,
    design="one-sample",
    allocation=0.5,
    directional=True,
    nodes=DEFAULT_NODES,
):
    r"""Power averaged over a future study's predictive effect distribution.

    Parameters
    ----------
    sample_size : array_like
        Sample sizes to evaluate.
    mean : :obj:`float`
        The meta-analytic estimate :math:`\hat m`.
    predictive_sd : :obj:`float`
        Standard deviation of the predictive distribution, from
        :func:`predictive_standard_deviation`. Zero reduces this to plug-in power, which is
        offered only so the difference can be measured.
    directional : :obj:`bool`, default=True
        Count only rejections in the direction of ``mean``. This is the default because a
        two-sided rejection in the wrong direction is not a successful study, and because
        two-sided assurance tends to one for any non-zero effect and so has no informative
        ceiling -- the ceiling in :func:`assurance_ceiling` is the directional one.

    Returns
    -------
    :obj:`numpy.ndarray`
        Assurance at each sample size.
    """
    predictive_sd = float(predictive_sd)
    if predictive_sd < 0:
        raise ValueError("The predictive standard deviation cannot be negative.")
    sizes = np.atleast_1d(np.asarray(sample_size, dtype=float))
    sign = np.sign(float(mean)) or 1.0

    def power(effects, size_array):
        if directional:
            return directional_power(
                effects,
                size_array,
                direction=sign,
                alpha=alpha,
                design=design,
                allocation=allocation,
            )
        return two_sided_power(
            effects, size_array, alpha=alpha, design=design, allocation=allocation
        )

    if predictive_sd == 0.0:
        return np.asarray(power(np.full(sizes.shape, float(mean)), sizes), dtype=float)

    grid = _predictive_grid(
        float(mean),
        predictive_sd,
        sizes,
        alpha=alpha,
        design=design,
        allocation=allocation,
        nodes=int(nodes),
    )
    weight = norm.pdf(grid, loc=float(mean), scale=predictive_sd)
    out = np.empty(sizes.shape, dtype=float)
    for index, size in enumerate(sizes):
        powers = power(grid, np.full(grid.shape, size))
        out[index] = float(np.trapezoid(powers * weight, grid))
    return out


def _predictive_grid(mean, predictive_sd, sizes, *, alpha, design, allocation, nodes):
    """Integration grid covering the predictive distribution *and* every power transition.

    Gauss-Hermite is wrong for this integral and was used here at first. The power curve is a
    step of width about ``12 * kappa`` in the effect, and at large samples that is far narrower
    than the spacing of any fixed-node rule across a predictive distribution several tenths
    wide: at 50,000 observations a 96-node rule returned 0.900 for a quantity whose analytic
    limit is 0.885, overshooting a ceiling it cannot exceed. The grid below is refined around
    each sample size's own transition, which is where all the variation is.
    """
    span = np.linspace(mean - 12.0 * predictive_sd, mean + 12.0 * predictive_sd, 4 * nodes + 1)
    pieces = [span]
    for size in np.atleast_1d(sizes):
        scale = float(np.atleast_1d(design_scale(size, design, allocation))[0])
        dof = float(np.atleast_1d(degrees_of_freedom(size, design))[0])
        centre = float(nct.ppf(1.0 - alpha / 2.0, dof, 0.0)) * scale
        margin = 12.0 * scale
        for sign in (-1.0, 1.0):
            pieces.append(
                np.linspace(sign * centre - margin, sign * centre + margin, 4 * nodes + 1)
            )
    grid = np.unique(np.concatenate(pieces))
    return grid[(grid >= mean - 12.0 * predictive_sd) & (grid <= mean + 12.0 * predictive_sd)]


def assurance_ceiling(
    mean, predictive_sd, *, directional=True, alpha=0.05, null_mass=None, direction=None
):
    r"""Report the limit assurance approaches as the sample size grows.

    For a fixed predictive distribution and the correctly signed tail of a two-sided
    :math:`\alpha`-level test,

    .. math:: \lim_{n\to\infty}\mathcal{A}_+(n)
        = P(\text{sign}\cdot\theta > 0) + \frac{\alpha}{2}\,P(\theta = 0),

    and the two-sided limit is :math:`P(\theta \ne 0) + \alpha\,P(\theta = 0)`. Under a
    continuous predictive distribution the point mass is zero and these reduce to
    :math:`\Phi(|\hat m|/s)` and one respectively.

    .. warning::
        **This is an asymptotic value, not a universal finite-sample ceiling.** With a predictive
        distribution carrying wrong-sign components, assurance need not be monotone in the
        sample size and can exceed its own limit at finite :math:`n`: the external audit
        supplies a mixture whose finite-:math:`n` assurance is .012823 against a limit of .0125.
        Treat this as the value the curve tends to, and check monotonicity rather than assuming
        it.

        A previous version returned one whenever ``predictive_sd`` was zero. That is wrong for a
        point mass **at zero effect**, whose directional rejection probability is
        :math:`\alpha/2` and whose two-sided probability is :math:`\alpha`, in the limit as at
        any :math:`n`. Caught by the audit.

    Parameters
    ----------
    null_mass : :obj:`float`, optional
        :math:`P(\theta = 0)` under the predictive distribution, for a mixture carrying an
        atom there. Defaults to one when ``predictive_sd`` is zero and ``mean`` is zero -- a
        degenerate distribution at no effect -- and to zero otherwise.
    direction : :obj:`float`, optional
        The sign being tested for. Defaults to the sign of ``mean``. Specify it explicitly
        rather than reading it off a fitted mean after the fact.
    """
    mean = float(mean)
    predictive_sd = float(predictive_sd)
    sign = float(np.sign(direction if direction is not None else mean)) or 1.0

    if null_mass is None:
        null_mass = 1.0 if (predictive_sd == 0.0 and mean == 0.0) else 0.0
    null_mass = float(null_mass)
    if not 0.0 <= null_mass <= 1.0:
        raise ValueError("null_mass must be a probability.")

    if predictive_sd == 0.0:
        correct_sign = 1.0 - null_mass if sign * mean > 0 else 0.0
    else:
        correct_sign = (1.0 - null_mass) * float(norm.cdf(sign * mean / predictive_sd))

    if directional:
        return correct_sign + alpha / 2.0 * null_mass
    non_null = 1.0 - null_mass
    return non_null + alpha * null_mass


def required_sample_size(
    target,
    mean,
    predictive_sd,
    *,
    alpha=0.05,
    design="one-sample",
    allocation=0.5,
    directional=True,
    maximum=100000,
):
    """Smallest sample size reaching a target assurance, with an explicit status.

    Returns ``(size, status)``. ``status`` is one of:

    ``"reached"``
        ``size`` is the smallest sample size on the searched integers meeting the target.
    ``"not_reached_within_budget"``
        The target is not met at ``maximum``; ``size`` is ``None``. This is **not** the same as
        unattainable -- a larger budget may reach it.
    ``"above_asymptotic_limit"``
        The target exceeds :func:`assurance_ceiling`, so no sample size reaches it under this
        predictive distribution. ``size`` is ``None``.

    These three are not interchangeable and were previously collapsed into a bare ``None``.

    The search brackets by doubling and then bisects, but **clamps the bracket to**
    ``maximum`` so a feasible size below the budget cannot be stepped over. The old version
    doubled past the budget and then reported nothing: at mean .3, zero predictive spread,
    target .8 and maximum 100 it returned ``None`` although assurance at 100 is .8439 and 90
    suffices. Caught by the external audit.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("The target assurance must lie strictly between zero and one.")
    if int(maximum) < 3:
        raise ValueError("The maximum sample size must be at least three.")

    limit = assurance_ceiling(mean, predictive_sd, directional=directional, alpha=alpha)
    if target >= limit:
        return None, "above_asymptotic_limit"

    def reached(size):
        return (
            float(
                assurance(
                    size,
                    mean,
                    predictive_sd,
                    alpha=alpha,
                    design=design,
                    allocation=allocation,
                    directional=directional,
                )[0]
            )
            >= target
        )

    ceiling_size = int(maximum)
    if not reached(ceiling_size):
        return None, "not_reached_within_budget"

    low, high = 3, min(8, ceiling_size)
    while high < ceiling_size and not reached(high):
        low, high = high, min(high * 2, ceiling_size)
    while low + 1 < high:
        middle = (low + high) // 2
        if reached(middle):
            high = middle
        else:
            low = middle
    return int(high), "reached"


def assurance_is_converged(sample_size, mean, predictive_sd, *, tolerance=1e-4, **kwargs):
    """Report whether refining the integration grid moves the assurance.

    Exposed for the same reason the block module exposes its quadrature check: an integration
    rule that has not resolved the integrand returns a smooth, plausible, wrong number, and no
    downstream comparison catches it. This one was not hypothetical -- the first version of
    :func:`assurance` failed it.
    """
    kwargs.pop("nodes", None)
    coarse = assurance(sample_size, mean, predictive_sd, nodes=DEFAULT_NODES, **kwargs)
    fine = assurance(sample_size, mean, predictive_sd, nodes=4 * DEFAULT_NODES, **kwargs)
    gap = float(np.max(np.abs(np.asarray(fine) - np.asarray(coarse))))
    return bool(gap < tolerance), gap


def roi_standardised_effect(means, covariance):
    r"""Standardised effect of an ROI average, which is not the average of voxelwise effects.

    .. math::
        g_{\rm ROI} = \frac{\mathbf{1}^\top\mu / V}
        {\sqrt{\mathbf{1}^\top\Sigma\mathbf{1} / V^2}}.

    The denominator is the square root of the *mean of the covariance matrix's entries*, so it
    depends on every pairwise covariance in the region. Averaging per-voxel standardised effects
    reconstructs it only when every voxel has the same variance and the correlations are one --
    that is, never.
    """
    means = np.asarray(means, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (means.size, means.size):
        raise ValueError(
            f"covariance must be square of side {means.size}; got {covariance.shape}."
        )
    total = float(np.sum(covariance))
    if total <= 0:
        raise ValueError("The ROI average has non-positive variance; check the covariance.")
    return float(means.mean() / np.sqrt(total / means.size**2))
