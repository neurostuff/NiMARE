"""One-sample OLS over group contributions, with blockwise sign-flip inference.

Derived from Nilearn's :func:`nilearn.mass_univariate.permuted_ols` (BSD 3-Clause): the
argument names, the ``h0_max_t`` / ``logp_max_t`` outputs and the max-statistic scheme all
follow it, so the two can be swapped. Only the one-sample scheme is carried over -- Nilearn's
TFCE, cluster-level and confound-orthogonalization paths are not, since
:class:`~nimare.meta.ibma.PermutedOLS` does not use them. It lives here because Nilearn's
``exchangeability_blocks`` argument is missing from some supported versions, and because the
group-weighted statistic has no equivalent there. The sign-flip scheme is from Winkler et al.
(2014); every statistic is evaluated through :mod:`pymare.stats`.
"""

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from pymare.stats import (
    encode_groups,
    group_mean,
    one_sample_t_from_sufficient_statistics,
    satterthwaite_dof,
    weighted_intercept_cr2,
    weighted_intercept_cr2_sufficient_statistics,
)
from sklearn.utils import check_random_state


def _permutation_maxima(
    sign_flips,
    group_contributions,
    group_sum_squares,
    two_sided_test,
    max_bytes,
    cr2_sufficient_statistics=None,
):
    """Calculate null maxima in memory-bounded vectorized batches."""
    n_voxels = group_contributions.shape[1]
    bytes_per_permutation = max(1, 2 * n_voxels * np.dtype(float).itemsize)
    batch_size = max(1, max_bytes // bytes_per_permutation)
    maxima = np.empty(sign_flips.shape[0], dtype=float)

    for start in range(0, sign_flips.shape[0], batch_size):
        stop = min(start + batch_size, sign_flips.shape[0])
        if cr2_sufficient_statistics is None:
            group_sums = sign_flips[start:stop] @ group_contributions
            permuted_t = one_sample_t_from_sufficient_statistics(
                group_sums,
                group_sum_squares,
                group_contributions.shape[0],
            )
        else:
            permuted_t = weighted_intercept_cr2(
                sign_flips[start:stop],
                cr2_sufficient_statistics,
            )
        if two_sided_test:
            np.abs(permuted_t, out=permuted_t)
        maxima[start:stop] = np.nanmax(permuted_t, axis=1)

    return maxima


def _empirical_max_p(observed_t, h0_max_t, two_sided_test):
    """Convert observed statistics and max-t null values to adjusted p-values."""
    observed_for_test = np.abs(observed_t) if two_sided_test else observed_t
    sorted_null = np.sort(h0_max_t)
    exceedances = h0_max_t.size - np.searchsorted(
        sorted_null,
        observed_for_test,
        side="left",
    )
    return (exceedances + 1) / (h0_max_t.size + 1)


def _permuted_ols(
    target_vars,
    *,
    exchangeability_blocks=None,
    n_perm=0,
    two_sided_test=True,
    random_state=None,
    n_jobs=1,
    sign_flips=None,
    group_weights=None,
):
    """Fit a one-sample OLS over group contributions with blockwise sign-flip inference.

    Each block contributes the mean of its available maps, so a block carries the same total
    weight however many maps it supplied. Permutations apply one random sign per block, which
    permits correlated maps within a block while assuming independent, jointly sign-symmetric
    blocks.

    Collapsing to one row per block is a correctness requirement, not a convenience:
    :func:`pymare.stats.weighted_intercept_cr2` computes leverage as ``q_g / W`` per row, which
    is the CR2 leverage only when each row *is* a cluster.

    Parameters
    ----------
    target_vars : :obj:`numpy.ndarray` of shape (n_maps, n_voxels)
        The maps to combine.
    exchangeability_blocks : None or array-like of shape (n_maps,), optional
        One block label per map. Maps sharing a label are treated as dependent and are
        sign-flipped together. None gives every map its own block, which recovers the ordinary
        one-sample test.
    group_weights : None or :obj:`numpy.ndarray` of shape (n_blocks,), optional
        One fixed positive weight per block, applied after maps have been averaged within the
        block. When omitted, every block receives equal weight and the statistic reduces
        algebraically to the ordinary one-sample t.
    sign_flips : None or :obj:`numpy.ndarray` of shape (n_perm, n_blocks), optional
        Sign flips to use instead of drawing fresh ones. Lets a caller share one null across
        several calls, which is how a single max-statistic null is built over liberal-mask bags.

    Returns
    -------
    :obj:`dict`
        With keys ``"t"``, ``"logp_max_t"``, ``"h0_max_t"`` and ``"dof"``.
    """
    target_vars = np.asarray(target_vars, dtype=float)
    if target_vars.ndim != 2:
        raise ValueError("target_vars must have shape (n_maps, n_voxels).")

    if not isinstance(n_perm, (int, np.integer)) or n_perm < 0:
        raise ValueError("n_perm must be a non-negative integer.")

    block_codes, block_labels = encode_groups(
        exchangeability_blocks,
        n_observations=target_vars.shape[0],
    )
    n_blocks = block_labels.size
    if n_blocks < 2:
        raise ValueError("At least two independent blocks are required.")

    # One sign flip per block already lines up with one row per block.
    group_contributions = group_mean(target_vars, block_codes)

    n_samples = group_contributions.shape[0]
    group_sums = group_contributions.sum(axis=0)
    group_sum_squares = np.einsum(
        "ij,ij->j",
        group_contributions,
        group_contributions,
    )
    cr2_sufficient_statistics = None
    if group_weights is None:
        # Equal weights collapse the CR2 sandwich to s^2 / m, so the cheaper
        # sufficient-statistic form gives the same statistic and the degrees of freedom are
        # exactly m - 1. Asserting that beats calling satterthwaite_dof, which warns below 4
        # about an approximation that is not being made here.
        observed_t = one_sample_t_from_sufficient_statistics(
            group_sums,
            group_sum_squares,
            n_samples,
        )
        dof = float(n_samples - 1)
    else:
        weights = np.asarray(group_weights, dtype=float).ravel()
        if weights.size != n_samples:
            raise ValueError("group_weights must contain one weight per block.")
        cr2_sufficient_statistics = weighted_intercept_cr2_sufficient_statistics(
            group_contributions,
            weights,
        )
        observed_t = weighted_intercept_cr2(
            np.ones((1, n_samples)),
            cr2_sufficient_statistics,
        ).squeeze(axis=0)
        # Unequal weights shrink the reference below the block count, by an amount the
        # naive m - 1 cannot see. One dominant study is enough to matter.
        dof = float(
            np.ravel(
                satterthwaite_dof(
                    np.ones((n_samples, 1)),
                    weights[:, None],
                    np.arange(n_samples),
                )
            )[0]
        )

    result = {
        "t": observed_t[None, :],
        "logp_max_t": np.array([], dtype=float),
        "h0_max_t": np.array([], dtype=float),
        "dof": dof,
    }
    if n_perm == 0:
        return result

    if sign_flips is None:
        rng = check_random_state(random_state)
        sign_flips = rng.choice((-1.0, 1.0), size=(n_perm, n_blocks))
    else:
        sign_flips = np.asarray(sign_flips, dtype=float)
        if sign_flips.shape != (n_perm, n_blocks):
            raise ValueError("sign_flips must have shape (n_perm, n_blocks).")
        if not np.all(np.isin(sign_flips, (-1, 1))):
            raise ValueError("sign_flips may only contain -1 and 1.")

    n_jobs = effective_n_jobs(n_jobs)
    max_bytes = 64 * 1024**2
    if n_jobs == 1 or n_perm == 1:
        h0_max_t = _permutation_maxima(
            sign_flips,
            group_contributions,
            group_sum_squares,
            two_sided_test,
            max_bytes,
            cr2_sufficient_statistics,
        )
    else:
        sign_chunks = np.array_split(sign_flips, min(n_jobs, n_perm))
        null_chunks = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_permutation_maxima)(
                chunk,
                group_contributions,
                group_sum_squares,
                two_sided_test,
                max_bytes // n_jobs,
                cr2_sufficient_statistics,
            )
            for chunk in sign_chunks
        )
        h0_max_t = np.concatenate(null_chunks)

    p_values = _empirical_max_p(observed_t, h0_max_t, two_sided_test)

    result["logp_max_t"] = -np.log10(p_values)[None, :]
    result["h0_max_t"] = h0_max_t
    return result
