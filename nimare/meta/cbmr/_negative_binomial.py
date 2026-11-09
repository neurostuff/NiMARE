"""Pieces of the moment-matched negative binomial shared by its likelihood and its information.

Both are written in the dispersion ``k = theta S2 / S1^2`` of the single negative-binomial
variable matching a pattern's voxel total, rather than in its size ``1/k``, which grows without
bound as the overdispersion approaches zero.
"""

import numpy as np


def matched_dispersion(overdispersion, member_weight):
    """Return ``k = theta S2 / S1^2`` for one pattern, from numpy arrays or torch tensors alike.

    ``S1`` and ``S2`` are the sums of the members' weights and squared weights.
    """
    return overdispersion * (member_weight**2).sum() / member_weight.sum() ** 2


def count_histogram(counts):
    """Return ``n_j``, the number of voxels with more than ``j`` foci, for ``j < max(y)``.

    ``None`` when the counts are not all non-negative integers, which have no such sum. The
    voxel marginals of integer foci are sums of integers and so are exactly integral.
    """
    counts = np.asarray(counts)
    if np.any(counts != np.rint(counts)) or np.any(counts < 0):
        return None
    return counts.size - np.cumsum(np.bincount(counts.astype(np.int64)))[:-1]


def log1p_ratio_excess(x):
    """Return ``log(1 + x) / x - 1 / (1 + x)`` without cancellation as ``x`` approaches zero.

    Both terms tend to one while their difference is about ``x / 2``, so the direct form loses
    all precision at ``x = 1e-12``. Below ``1e-2`` the power series
    ``sum_n (-1)^(n+1) n x^n / (n + 1)`` is summed instead; ten terms reach double precision.
    """
    x = np.asarray(x, dtype=float)
    small = x < 1e-2
    result = np.empty_like(x)
    xs = x[small]
    result[small] = sum((-1) ** (n + 1) * n * xs**n / (n + 1) for n in range(1, 11))
    xl = x[~small]
    result[~small] = np.log1p(xl) / xl - 1.0 / (1.0 + xl)
    return result
