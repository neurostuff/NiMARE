"""Various statistical helper functions."""

import logging

import numpy as np

from nimare import utils

LGR = logging.getLogger(__name__)


def one_way(data, n):
    """One-way chi-square test of independence.

    Takes a 1D array as input and compares activation at each voxel to
    proportion expected under a uniform distribution throughout the array.
    Note that if you're testing activation with this, make sure that only
    valid voxels (e.g., in-mask gray matter voxels) are included in the
    array, or results won't make any sense!

    Parameters
    ----------
    data : 1D array_like
        Counts across voxels.
    n : :obj:`int`
        Maximum possible count (aka total number of units) for all cells in
        ``data``. If data is n_voxels long, then ``n`` is the number of studies
        in the analysis.

    Returns
    -------
    chi2 : :class:`numpy.ndarray`
        Chi-square values

    Notes
    -----
    Taken from Neurosynth.
    """
    term = np.asarray(data, dtype=np.float64)
    expected_term = np.mean(term, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (term - expected_term) ** 2 * n / (expected_term * (n - expected_term))
    return chi2


def two_way_counts(selected, unselected, n_selected, n_unselected):
    """Two-way chi-square test from paired study-count vectors.

    Parameters
    ----------
    selected, unselected : (N,) array_like
        Per-voxel active-study counts for the two groups.
    n_selected, n_unselected : :obj:`int`
        Number of studies in each group.

    Returns
    -------
    chi_sq : :class:`numpy.ndarray`
        Chi-square values, one per voxel.
    """
    a = np.asarray(selected, dtype=np.float64)
    b = np.asarray(unselected, dtype=np.float64)
    total = n_selected + n_unselected
    row0 = a + b
    row1 = total - row0

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = total * (a * n_unselected - b * n_selected) ** 2
        denominator = row0 * row1 * n_selected * n_unselected
        chi_sq = numerator / denominator

    chi_sq[denominator == 0] = 2.0
    return chi_sq


def two_way(cells):
    """Two-way chi-square test of independence.

    Takes a 3D array as input: N(voxels) x 2 x 2, where the last two
    dimensions are the contingency table for each of N voxels.

    Parameters
    ----------
    cells : (N, 2, 2) array_like
        Concatenated set of contingency tables. There are N contingency tables,
        with the last two dimensions being the tables for each input.

    Returns
    -------
    chi_sq : :class:`numpy.ndarray`
        Chi-square values.

    Notes
    -----
    Taken from Neurosynth.
    """
    cells = np.asarray(cells, dtype=np.float64)
    if cells.ndim != 3 or cells.shape[1:] != (2, 2):
        raise ValueError(
            "two_way expects an array of shape (n_tables, 2, 2); " f"got {cells.shape!r}."
        )

    return two_way_counts(
        selected=cells[:, 0, 0],
        unselected=cells[:, 0, 1],
        n_selected=np.asarray(cells[:, :, 0].sum(axis=1)),
        n_unselected=np.asarray(cells[:, :, 1].sum(axis=1)),
    )


def pearson(x, y):
    """Correlate row vector x with each row vector in 2D array y, quickly.

    Parameters
    ----------
    x : (1, N) array_like
        Row vector to correlate with each row in ``y``.
    y : (M, N) array_like
        Array, for which each row is correlated with ``x``.

    Returns
    -------
    rs : (M,) :class:`numpy.ndarray`
        Pearson correlation coefficients for ``x`` against each row of ``y``.
    """
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(np.sum(datam**2, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    rs = temp / (datass[1:] * datass[0])
    return rs


def null_to_p(test_value, null_array, tail="two", symmetric=False):
    """Return p-value for test value(s) against null array.

    .. versionchanged:: 0.0.7

        * [FIX] Add parameter *symmetric*.

    Parameters
    ----------
    test_value : 1D array_like
        Values for which to determine p-value.
    null_array : 1D array_like
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    symmetric : bool
        When tail="two", indicates how to compute p-values. When False (default),
        both one-tailed p-values are computed, and the two-tailed p is double
        the minimum one-tailed p. When True, it is assumed that the null
        distribution is zero-centered and symmetric, and the two-tailed p-value
        is computed as P(abs(test_value) >= abs(null_array)).

    Returns
    -------
    p_value : :obj:`float`
        P-value(s) associated with the test value when compared against the null
        distribution. Return type matches input type (i.e., a float if
        test_value is a single float, and an array if test_value is an array).

    Notes
    -----
    P-values are clipped based on the number of elements in the null array.
    Therefore no p-values of 0 or 1 should be produced.

    When the null distribution is known to be symmetric and centered on zero,
    and two-tailed p-values are desired, use symmetric=True, as it is
    approximately twice as efficient computationally, and has lower variance.
    """
    if tail not in {"two", "upper", "lower"}:
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')

    return_first = isinstance(test_value, (float, int))
    test_value = np.atleast_1d(test_value)
    null_array = np.array(null_array)

    # For efficiency's sake, if there are more than 1000 values, pass only the unique
    # values through percentileofscore(), and then reconstruct.
    if len(test_value) > 1000:
        reconstruct = True
        test_value, uniq_idx = np.unique(test_value, return_inverse=True)
    else:
        reconstruct = False

    def compute_p(t, null):
        null = np.sort(null)
        idx = np.searchsorted(null, t, side="left").astype(float)
        return 1 - idx / len(null)

    if tail == "two":
        if symmetric:
            p = compute_p(np.abs(test_value), np.abs(null_array))
        else:
            p_l = compute_p(test_value, null_array)
            p_r = compute_p(test_value * -1, null_array * -1)
            p = 2 * np.minimum(p_l, p_r)
    elif tail == "lower":
        p = compute_p(test_value * -1, null_array * -1)
    else:
        p = compute_p(test_value, null_array)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value)
    smallest_value = 1.0 / len(null_array)
    result = np.maximum(smallest_value, np.minimum(p, 1.0 - smallest_value))

    if reconstruct:
        result = result[uniq_idx]

    return result[0] if return_first else result


def nullhist_to_p(test_values, histogram_weights, histogram_bins):
    """Return one-sided p-value for test value against null histogram.

    .. versionadded:: 0.0.4

    Parameters
    ----------
    test_values : float or 1D array_like
        Values for which to determine p-value. Can be a single value or a one-dimensional array.
        If a one-dimensional array, it should have the same length as the histogram_weights' last
        dimension.
    histogram_weights : (B [x V]) array
        Histogram weights representing the null distribution against which test_value is compared.
        These should be raw weights or counts, not a cumulatively-summed null distribution.
    histogram_bins : (B) array
        Histogram bin centers. Note that this differs from numpy.histogram's behavior, which uses
        bin *edges*. Histogram bins created with numpy will need to be adjusted accordingly.

    Returns
    -------
    p_value : :obj:`float`
        P-value associated with the test value when compared against the null distribution.
        P-values reflect the probability of a test value at or above the observed value if the
        test value was drawn from the null distribution.
        This is a one-sided p-value.

    Notes
    -----
    P-values are clipped based on the largest observed non-zero weight in the null histogram.
    Therefore no p-values of 0 should be produced.
    """
    test_values = np.asarray(test_values)
    return_value = False
    if test_values.ndim == 0:
        return_value = True
        test_values = np.atleast_1d(test_values)
    assert test_values.ndim == 1
    assert histogram_bins.ndim == 1
    assert histogram_weights.shape[0] == histogram_bins.shape[0]
    assert histogram_weights.ndim in (1, 2)
    if histogram_weights.ndim == 2:
        assert histogram_weights.shape[1] == test_values.shape[0]
        voxelwise_null = True
    else:
        histogram_weights = histogram_weights[:, None]
        voxelwise_null = False

    n_bins = len(histogram_bins)
    inv_step = 1 / (histogram_bins[1] - histogram_bins[0])  # assume equal spacing

    # Convert histograms to null distributions
    # The value in each bin represents the probability of finding a test value
    # (stored in histogram_bins) of that value or lower.
    null_distribution = histogram_weights / np.sum(histogram_weights, axis=0)
    null_distribution = np.cumsum(null_distribution[::-1, :], axis=0)[::-1, :]
    null_distribution /= np.max(null_distribution, axis=0)
    null_distribution = np.squeeze(null_distribution)

    smallest_value = np.min(null_distribution[null_distribution != 0])

    p_values = np.ones(test_values.shape)
    idx = np.where(test_values > 0)[0]
    value_bins = utils._round2(test_values[idx] * inv_step)
    value_bins[value_bins >= n_bins] = n_bins - 1  # limit to within null distribution

    # Get p-values by getting the value_bins-th value in null_distribution
    if voxelwise_null:
        p_values[idx] = null_distribution[value_bins, idx]
    else:
        p_values[idx] = null_distribution[value_bins]

    # ensure p_value in the following range:
    # smallest_value <= p_value <= 1.0
    p_values = np.maximum(smallest_value, np.minimum(p_values, 1.0))
    if return_value:
        p_values = p_values[0]
    return p_values


def _check_nlogp(nlogp):
    """Return ``nlogp`` as a float array, rejecting anything that is not an ``nlogp``.

    An ``nlogp`` is at most zero. Catching a positive one here turns the easy
    mistake of passing p-values to a log-space correction into an error rather than into a
    silently wrong result: every p-value would come back adjusted to one.
    """
    nlogp = np.asarray(nlogp, dtype=float)
    if np.any(nlogp > 0):
        raise ValueError(
            "nlogp must hold natural logarithms of p-values, which are at most 0; got a "
            f"maximum of {np.nanmax(nlogp)}."
        )
    return nlogp


def nlogp_bonferroni(nlogp):
    """Perform Bonferroni correction on ``nlogp`` values.

    .. versionadded:: 0.21.0

    The log-space counterpart of :func:`pymare.stats.bonferroni`. Multiplying by the number
    of tests is an addition in logs, so nothing underflows on the way through and a corrected
    p-value below the smallest representable double survives as its logarithm.

    Parameters
    ----------
    nlogp : :obj:`numpy.ndarray`
        Natural logarithms of the uncorrected p-values.

    Returns
    -------
    :obj:`numpy.ndarray`
        Natural logarithms of the corrected p-values.
    """
    nlogp = _check_nlogp(nlogp)
    # The cap is against log(1): a corrected p-value cannot exceed one.
    return np.minimum(nlogp + np.log(nlogp.size), 0.0)


def nlogp_fdr(nlogp, method="bh"):
    """Perform FDR correction on ``nlogp`` values.

    .. versionadded:: 0.21.0

    The log-space counterpart of :func:`pymare.stats.fdr`, step for step: the step-up
    procedure only ever divides a p-value by a positive factor and takes running minima, both
    of which carry over to logs unchanged. Sorting by ``nlogp`` is sorting by ``p``, so the
    set of tests declared significant at any alpha is identical.

    Parameters
    ----------
    nlogp : :obj:`numpy.ndarray`
        Natural logarithms of the uncorrected p-values.
    method : {"bh", "by"}, optional
        Either "bh" (Benjamini-Hochberg :footcite:p:`benjamini1995controlling`) or "by"
        (Benjamini-Yekutieli :footcite:p:`benjamini2001control`). Default is "bh".

    Returns
    -------
    :obj:`numpy.ndarray`
        Natural logarithms of the corrected p-values.

    References
    ----------
    .. footbibliography::
    """
    nlogp = _check_nlogp(nlogp)
    n_tests = nlogp.size

    sort_idx = np.argsort(nlogp)
    revert_idx = np.argsort(sort_idx)

    log_ecdffactor = np.log(np.arange(1, n_tests + 1) / n_tests)
    if method == "by":
        log_ecdffactor = log_ecdffactor - np.log(np.sum(1 / np.arange(1, n_tests + 1)))

    log_adjusted = nlogp[sort_idx] - log_ecdffactor
    log_adjusted = np.minimum.accumulate(log_adjusted[::-1])[::-1]
    return np.minimum(log_adjusted, 0.0)[revert_idx]
