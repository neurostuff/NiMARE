"""High-precision references for the CBMR negative binomial, where float64 cannot serve.

Near zero overdispersion the likelihood holds ``lgamma(y + r) - lgamma(r)`` with ``r`` of order
one over the overdispersion, and float64 derivatives of it cancel. These evaluate the same
likelihood, transcribed from its ``(r, p)`` form, in mpmath at 50 digits.
"""

import numpy as np
import pytest


def negative_binomial_log_likelihood(predictor, overdispersion, foci, n_spatial):
    """Return the moment-matched negative-binomial log-likelihood as an mpmath function.

    The function takes the flat coefficient vector, spatial coefficients first, as separate
    mpmath arguments. The ``-lgamma(y + 1)`` term is parameter-free and left out.
    """
    mpmath = pytest.importorskip("mpmath")
    n_bases, n_columns = predictor.n_bases, predictor.n_spatial_columns
    marginal = predictor.patterns.marginal_by_pattern(foci)
    exposure = (
        np.ones(predictor.patterns.n_experiments)
        if predictor.exposure is None
        else np.asarray(predictor.exposure, dtype=float)
    )
    block = (
        np.zeros((predictor.patterns.n_experiments, 0))
        if predictor.global_block is None
        else predictor.global_block
    )

    def log_likelihood(*vector):
        spatial = [vector[c * n_bases : (c + 1) * n_bases] for c in range(n_columns)]
        gamma = vector[n_spatial:]
        weight = [
            mpmath.mpf(float(e)) * mpmath.exp(mpmath.fsum(x * g for x, g in zip(row, gamma)))
            for e, row in zip(exposure, block)
        ]
        total = mpmath.mpf(0)
        for p in range(predictor.patterns.n_patterns):
            rows = np.flatnonzero(predictor.patterns.assignment == p)
            sum_1 = mpmath.fsum(weight[i] for i in rows)
            sum_2 = mpmath.fsum(weight[i] ** 2 for i in rows)
            size = sum_1**2 / (mpmath.mpf(float(overdispersion[p])) * sum_2)
            for v in range(predictor.n_voxels):
                log_intensity = mpmath.fsum(
                    predictor.patterns.loadings[p, c] * predictor.bases[v, k] * spatial[c][k]
                    for c in range(n_columns)
                    for k in range(n_bases)
                )
                mean = sum_1 * mpmath.exp(log_intensity)
                count = marginal[p, v]
                total += (
                    mpmath.loggamma(count + size)
                    - mpmath.loggamma(size)
                    + size * (mpmath.log(size) - mpmath.log(size + mean))
                    + count * (mpmath.log(mean) - mpmath.log(size + mean))
                )
        return total

    return log_likelihood


def negative_hessian(function, point, digits=50):
    """Return minus the Hessian of an mpmath ``function`` at ``point``, as float64."""
    mpmath = pytest.importorskip("mpmath")
    n = len(point)
    information = np.zeros((n, n))
    with mpmath.workdps(digits):
        at = [mpmath.mpf(float(value)) for value in point]
        for i in range(n):
            for j in range(i, n):
                orders = [0] * n
                orders[i] += 1
                orders[j] += 1
                information[i, j] = information[j, i] = -float(mpmath.diff(function, at, orders))
    return information
