"""Selection-corrected likelihood machinery for effect-size CBMA.

Coordinate-based meta-analyses see a reported peak only when the originating
study's statistic cleared that study's reporting threshold. A study that
reports nothing in a region has therefore not given us *no* information: it has
told us its statistic there fell *below* threshold. That is a **left-censored**
observation, not a missing one and not a zero.

This module implements the likelihood that takes that seriously, so that no
imputation of study images is required -- the unobserved values are integrated
out in closed form.

Two likelihoods are available, and the distinction matters a great deal:

``"censored"`` (L1)
    The full-data likelihood. A study that reported contributes the plain
    normal density :math:`f(g_i)`; a study that did not contributes
    ``P(|G_i| <= c_i)``. These integrate to one over the sample space.
    Correct whenever the full roster of studies is known -- which it always is
    in NiMARE, since the collection enumerates them.

``"conditional"`` (L2)
    The likelihood *conditional on having been reported*: reporting studies
    contribute a truncated density :math:`f(g_i) / P(|G_i| > c_i)` and
    non-reporting studies are dropped entirely. Appropriate only when the
    denominator of studies is unknown.

.. warning::
    Combining the two -- truncating the reporters *and* censoring the
    non-reporters -- applies the selection correction twice and is badly biased
    toward zero. It is not a valid likelihood (it does not integrate to one).

The same conditional-likelihood algebra appears under different names in
several literatures: the Tobit model in econometrics, "non-detect" estimation
in environmental statistics, and winner's-curse correction in statistical
genetics. Neuroimaging peaks are a winner's-curse problem with a spatial index.

References
----------
* Tench et al. (2017), *Coordinate based random effect size meta-analysis of
  neuroimaging studies*, NeuroImage 153:293-306.
* Costafreda (2012), *Parametric coordinate-based meta-analysis*,
  J Neurosci Methods 210:291-300.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp

__all__ = [
    "log1mexp",
    "log_interval_prob",
    "region_moments",
    "selection_moments",
    "loglikelihood",
    "score",
    "information",
    "solve_delta",
]

#: -0.5 * log(2 * pi), the normalizing constant of the standard normal log-pdf.
_LOG_SQRT_2PI = 0.9189385332046727

#: Largest |delta| (in Hedges' g units) the solver will consider. Effect sizes
#: beyond this are not physically meaningful and signal a degenerate fit.
G_MAX = 50.0

_TINY = np.finfo(float).tiny


def _log_phi(x):
    """Return the standard normal log-pdf, with ``-inf`` at infinite arguments."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, -np.inf)
    finite = np.isfinite(x)
    out[finite] = -0.5 * x[finite] ** 2 - _LOG_SQRT_2PI
    return out


def log1mexp(x):
    """Compute ``log(1 - exp(x))`` for ``x < 0`` without cancellation.

    Parameters
    ----------
    x : array_like
        Values, all strictly negative.

    Returns
    -------
    :class:`numpy.ndarray`
        ``log(1 - exp(x))``, evaluated stably on both sides of ``-log(2)``.

    Notes
    -----
    Naive evaluation loses all precision for ``x`` near zero (where
    ``1 - exp(x)`` cancels) and for very negative ``x`` (where ``exp(x)``
    underflows). Switching formula at ``-log(2)`` keeps both regimes accurate.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > -np.log(2.0), np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def log_interval_prob(a, b):
    """Compute ``log(Phi(b) - Phi(a))`` stably for ``a <= b``.

    Parameters
    ----------
    a, b : array_like
        Standardized interval limits, ``a <= b``. Infinities are allowed.

    Returns
    -------
    :class:`numpy.ndarray`
        The log-probability that a standard normal falls in ``[a, b]``.

    Notes
    -----
    Two failure modes are handled explicitly.

    1.  When both limits are large and positive, ``log_ndtr`` returns values
        near zero for both and their difference cancels catastrophically. The
        interval is first reflected about the origin (which leaves the
        probability unchanged) so that it always sits in the left tail.
    2.  When the interval is very narrow, the ``log1mexp`` argument approaches
        zero and precision is lost. A midpoint rule,
        ``Phi(b) - Phi(a) ~ (b - a) * phi((a + b) / 2)``, is used instead.

    Together these keep the result accurate even where ``ndtr(b) - ndtr(a)``
    underflows to exactly zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = np.broadcast_arrays(a, b)

    # Reflect so the interval lies in the left tail (Phi(b)-Phi(a) is invariant).
    flip = (a + b) > 0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)

    log_lo = log_ndtr(lo)
    log_hi = log_ndtr(hi)

    width = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        wide = log_hi + log1mexp(np.minimum(log_lo - log_hi, -_TINY))
        narrow = _log_phi(0.5 * (lo + hi)) + np.log(np.maximum(width, _TINY))

    out = np.where(width > 1e-5, wide, narrow)
    return np.where(width <= 0, -np.inf, out)


def region_moments(lower, upper):
    """Return log-probability, mean and variance of a truncated standard normal.

    Parameters
    ----------
    lower, upper : array_like
        Standardized limits of the region, ``lower <= upper``. Infinities are
        allowed, so one-sided regions are expressed by passing ``-inf``.

    Returns
    -------
    log_prob : :class:`numpy.ndarray`
        ``log P(lower < Z < upper)``.
    mean : :class:`numpy.ndarray`
        ``E[Z | lower < Z < upper]``.
    var : :class:`numpy.ndarray`
        ``Var[Z | lower < Z < upper]``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower, upper = np.broadcast_arrays(lower, upper)

    log_prob = log_interval_prob(lower, upper)
    r_lo = np.exp(_log_phi(lower) - log_prob)
    r_hi = np.exp(_log_phi(upper) - log_prob)

    mean = r_lo - r_hi
    # x * phi(x) -> 0 as |x| -> inf; substitute before multiplying so that
    # `inf * 0` is never formed (np.where evaluates both branches).
    lo_term = np.where(np.isfinite(lower), lower, 0.0) * r_lo
    hi_term = np.where(np.isfinite(upper), upper, 0.0) * r_hi
    var = 1.0 + (lo_term - hi_term) - mean**2
    return log_prob, mean, np.clip(var, 0.0, None)


def selection_moments(lower, upper):
    """Return moments over the *complement* of ``[lower, upper]``.

    This is the region a study's statistic must land in for a peak to be
    reported.

    Parameters
    ----------
    lower, upper : array_like
        Standardized limits of the *censoring* region.

    Returns
    -------
    log_prob : :class:`numpy.ndarray`
        ``log P(Z < lower or Z > upper)``.
    mean : :class:`numpy.ndarray`
        ``E[Z | Z < lower or Z > upper]``.
    var : :class:`numpy.ndarray`
        ``Var[Z | Z < lower or Z > upper]``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower, upper = np.broadcast_arrays(lower, upper)

    # Both tails are positive quantities, so logsumexp is exact here -- much
    # better conditioned than taking the complement of the interval.
    log_prob = logsumexp(np.stack([log_ndtr(lower), log_ndtr(-upper)]), axis=0)
    r_lo = np.exp(_log_phi(lower) - log_prob)
    r_hi = np.exp(_log_phi(upper) - log_prob)

    mean = r_hi - r_lo
    lo_term = np.where(np.isfinite(lower), lower, 0.0) * r_lo
    hi_term = np.where(np.isfinite(upper), upper, 0.0) * r_hi
    var = 1.0 + (hi_term - lo_term) - mean**2
    return log_prob, mean, np.clip(var, 0.0, None)


def _limits(delta, s, c, two_sided):
    """Standardize the censoring region ``|g| <= c`` (or ``g <= c``)."""
    upper = (c - delta) / s
    lower = (-c - delta) / s if two_sided else np.full_like(upper, -np.inf)
    return lower, upper


def loglikelihood(delta, g, s, c, reported, two_sided=True, likelihood="censored"):
    """Evaluate the selection-corrected log-likelihood at ``delta``.

    Parameters
    ----------
    delta : float
        Candidate population effect size, in Hedges' g units.
    g : :class:`numpy.ndarray`
        Per-study observed effect sizes. Entries where ``reported`` is False are
        ignored.
    s : :class:`numpy.ndarray`
        Per-study standard deviations, ``sqrt(within variance + tau^2)``.
    c : :class:`numpy.ndarray`
        Per-study reporting thresholds, in Hedges' g units, non-negative.
    reported : :class:`numpy.ndarray` of bool
        Whether each study reported a focus in this region.
    two_sided : :obj:`bool`, default=True
        Whether studies could have reported effects of either sign.
    likelihood : {"censored", "conditional"}, default="censored"
        See the module docstring.

    Returns
    -------
    :obj:`float`
        The log-likelihood.
    """
    lower, upper = _limits(delta, s, c, two_sided)
    total = 0.0

    if likelihood == "censored":
        z = (g[reported] - delta) / s[reported]
        total += float(np.sum(-np.log(s[reported]) - 0.5 * z**2 - _LOG_SQRT_2PI))
    elif likelihood == "conditional":
        z = (g[reported] - delta) / s[reported]
        log_sel, _, _ = selection_moments(lower[reported], upper[reported])
        total += float(np.sum(-np.log(s[reported]) - 0.5 * z**2 - _LOG_SQRT_2PI - log_sel))
    else:
        raise ValueError(f"likelihood must be 'censored' or 'conditional'; got {likelihood!r}.")

    if likelihood == "censored" and np.any(~reported):
        log_cens, _, _ = region_moments(lower[~reported], upper[~reported])
        total += float(np.sum(log_cens))

    return total


def score(delta, g, s, c, reported, two_sided=True, likelihood="censored"):
    """Evaluate the derivative of :func:`loglikelihood` with respect to ``delta``.

    Returns
    -------
    :obj:`float`
        ``d/d delta`` of the log-likelihood.

    Notes
    -----
    Both contributions have the same form, ``(E[G | region] - delta) / s**2``,
    because the log-derivative of a normal orthant probability is the truncated
    mean. Ratios ``phi / P`` are formed as ``exp(log phi - log P)`` so that they
    stay finite even when ``P`` underflows: the censored score then approaches
    its correct linear asymptote ``(c - delta) / s**2`` rather than ``0/0``.
    """
    lower, upper = _limits(delta, s, c, two_sided)
    total = 0.0

    if likelihood == "censored":
        total += float(np.sum((g[reported] - delta) / s[reported] ** 2))
    elif likelihood == "conditional":
        _, mean_sel, _ = selection_moments(lower[reported], upper[reported])
        total += float(np.sum((g[reported] - delta) / s[reported] ** 2 - mean_sel / s[reported]))
    else:
        raise ValueError(f"likelihood must be 'censored' or 'conditional'; got {likelihood!r}.")

    if likelihood == "censored" and np.any(~reported):
        _, mean_cens, _ = region_moments(lower[~reported], upper[~reported])
        total += float(np.sum(mean_cens / s[~reported]))

    return total


def information(delta, g, s, c, reported, two_sided=True, likelihood="censored"):
    """Evaluate the observed Fisher information for ``delta``.

    Returns
    -------
    :obj:`float`
        ``-d^2/d delta^2`` of the log-likelihood, in closed form.

    Notes
    -----
    For a region that does not depend on ``delta``,
    ``d^2/d delta^2 log P = (Var[G | region] - s^2) / s^4``. A censored study
    therefore contributes ``(1 - v) / s^2`` where ``v`` is its truncated
    variance ratio -- always positive but smaller than a reporting study's
    ``1 / s^2``, since truncating to an interval reduces variance. Under the
    conditional likelihood a reporting study contributes ``v_sel / s^2``, which
    *exceeds* ``1 / s^2`` because tail truncation increases variance.
    """
    lower, upper = _limits(delta, s, c, two_sided)
    total = 0.0

    if likelihood == "censored":
        total += float(np.sum(1.0 / s[reported] ** 2))
    elif likelihood == "conditional":
        _, _, var_sel = selection_moments(lower[reported], upper[reported])
        total += float(np.sum(var_sel / s[reported] ** 2))
    else:
        raise ValueError(f"likelihood must be 'censored' or 'conditional'; got {likelihood!r}.")

    if likelihood == "censored" and np.any(~reported):
        _, _, var_cens = region_moments(lower[~reported], upper[~reported])
        total += float(np.sum((1.0 - var_cens) / s[~reported] ** 2))

    return total


def _bracket(g, s, c, reported, two_sided, likelihood):
    """Return ``(lo, hi)`` guaranteed to bracket the score's root, when one exists."""
    weights = 1.0 / s**2
    total_weight = weights.sum()
    reported_term = float(np.sum(g[reported] * weights[reported]))

    if likelihood == "censored" and two_sided:
        # E[G | |G| <= c] lies strictly inside (-c, c), so these are exact
        # bounds: the score is negative above `hi` and positive below `lo`.
        censored_term = float(np.sum(c[~reported] * weights[~reported]))
        lo = (reported_term - censored_term) / total_weight
        hi = (reported_term + censored_term) / total_weight
        return lo - 1e-6, hi + 1e-6

    # One-sided censoring is unbounded below, and the conditional likelihood is
    # unbounded above, so fall back to an expanding bracket.
    centre = reported_term / total_weight if np.any(reported) else 0.0
    width = max(1.0, 2.0 * (np.max(np.abs(g[reported])) + np.max(c)) + 5.0 * np.max(s))
    return centre - width, centre + width


def solve_delta(g, s, c, reported, two_sided=True, likelihood="censored", xtol=1e-10):
    """Maximize the selection-corrected likelihood over ``delta``.

    Parameters
    ----------
    g, s, c, reported : :class:`numpy.ndarray`
        As in :func:`loglikelihood`.
    two_sided : :obj:`bool`, default=True
        Whether effects of either sign were reportable.
    likelihood : {"censored", "conditional"}, default="censored"
        See the module docstring.
    xtol : :obj:`float`, default=1e-10
        Absolute tolerance passed to the root finder.

    Returns
    -------
    delta : :obj:`float`
        The maximum-likelihood effect size, or NaN if it does not exist.
    se : :obj:`float`
        Standard error from the observed information, or ``inf`` at a boundary.
    converged : :obj:`bool`
        Whether a finite interior maximum was found.

    Notes
    -----
    The log-likelihood is strictly concave in ``delta``: the reporting term is
    a Gaussian (or a natural exponential family, under the conditional
    likelihood), and truncating a log-concave density to a region strictly
    reduces its variance, which makes the censoring term concave too. The score
    is therefore strictly decreasing and Brent's method on a valid bracket
    converges to the unique root.
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    c = np.asarray(c, dtype=float)
    reported = np.asarray(reported, dtype=bool)

    if not np.any(reported):
        # A cluster is defined by at least one reported focus, so this should be
        # unreachable; returning NaN beats a spuriously precise zero.
        return np.nan, np.inf, False

    s = np.maximum(s, 1e-6)

    def _score(d):
        return score(d, g, s, c, reported, two_sided=two_sided, likelihood=likelihood)

    lo, hi = _bracket(g, s, c, reported, two_sided, likelihood)

    # Expand until the root is bracketed, or give up at a physically absurd
    # effect size (the conditional likelihood genuinely diverges when a lone
    # reported value sits exactly at the threshold).
    for _ in range(64):
        if _score(lo) > 0 > _score(hi):
            break
        width = hi - lo
        lo, hi = lo - width, hi + width
        if hi - lo > 4 * G_MAX:
            sign = 1.0 if _score(0.0) > 0 else -1.0
            return sign * G_MAX, np.inf, False
    else:  # pragma: no cover - the width guard above fires first
        return np.nan, np.inf, False

    delta = float(brentq(_score, lo, hi, xtol=xtol, rtol=1e-12))
    info = information(delta, g, s, c, reported, two_sided=two_sided, likelihood=likelihood)
    se = float(1.0 / np.sqrt(info)) if info > 0 else np.inf
    return delta, se, True
