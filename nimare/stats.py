"""Various statistical helper functions."""
import logging
import warnings

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
    term = data.astype("float64")
    no_term = n - term
    t_exp = np.mean(term, 0)
    t_exp = np.array([t_exp] * data.shape[0])
    nt_exp = n - t_exp
    t_mss = (term - t_exp) ** 2 / t_exp
    nt_mss = (no_term - nt_exp) ** 2 / nt_exp
    chi2 = t_mss + nt_mss
    return chi2


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
    # Mute divide-by-zero warning for bad voxels since we account for that
    # later
    warnings.simplefilter("ignore", RuntimeWarning)

    cells = cells.astype("float64")  # Make sure we don't overflow
    total = np.apply_over_axes(np.sum, cells, [1, 2]).ravel()
    chi_sq = np.zeros(cells.shape, dtype="float64")
    for i in range(2):
        for j in range(2):
            exp = np.sum(cells[:, i, :], 1).ravel() * np.sum(cells[:, :, j], 1).ravel() / total
            bad_vox = np.where(exp == 0)[0]
            chi_sq[:, i, j] = (cells[:, i, j] - exp) ** 2 / exp
            chi_sq[bad_vox, i, j] = 1.0  # Set p-value for invalid voxels to 1
    chi_sq = np.apply_over_axes(np.sum, chi_sq, [1, 2]).ravel()
    return chi_sq


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
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array))
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
        # Pair each test value with its associated null distribution
        for i_voxel, voxel_idx in enumerate(idx):
            p_values[voxel_idx] = null_distribution[value_bins[i_voxel], voxel_idx]
    else:
        p_values[idx] = null_distribution[value_bins]

    # ensure p_value in the following range:
    # smallest_value <= p_value <= 1.0
    p_values = np.maximum(smallest_value, np.minimum(p_values, 1.0))
    if return_value:
        p_values = p_values[0]
    return p_values


def fdr(p, q=0.05):
    """Determine FDR threshold given a p value array and desired false discovery rate q.

    Parameters
    ----------
    p : 1D :class:`numpy.ndarray`
        Array of p-values.
    q : :obj:`float`, optional
        False discovery rate in fraction form. Default is 0.05 (5%).

    Returns
    -------
    :obj:`float`
        P-value threshold for desired false discovery rate.

    Notes
    -----
    Taken from Neurosynth.
    """
    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype="float") * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if any(below) else -1


def hedges_g(y, n_subjects1, n_subjects2=None):
    """Calculate Hedges' G.

    .. todo::

        Support calculation across voxels as well.

    Parameters
    ----------
    y : 2D array of shape (n_studies, max_study_size)
        Subject-level values for which to calculate Hedges G.
        Multiple studies may be provided.
        The array contains as many rows as there are studies, and as many columns as the maximum
        sample size in the studyset. Extra columns in each row should be filled with NaNs.
    n_subjects1 : :obj:`numpy.ndarray` of shape (n_studies,)
        Number of subjects in the first group of each study.
    n_subjects2 : None or int
        Number of subjects in the second group of each study.
        If None, the dataset is assumed to be have one sample.
        Technically, this parameter is probably unnecessary, since the second group's sample size
        can be inferred from ``n_subjects1`` and ``y``.

    Returns
    -------
    g_arr : 1D array of shape (n_studies,)

    Notes
    -----
    Clues for Python version from https://en.wikipedia.org/wiki/Effect_size#Hedges'_g.

    I also updated the original code to support varying sample sizes across studies.

    Notes
    -----
    R Code:

    .. highlight:: r
    .. code-block:: r

        g <- function (y) { # Calculate Hedges' g
            J * apply(y, 2, function (y_i) {
                (mean(y_i[1:n.subj]) - mean(y_i[-(1:n.subj)])) /
                sqrt((var(y_i[1:n.subj]) + var(y_i[-(1:n.subj)])) / 2)
            })
        }
    """
    from scipy.special import gamma

    if n_subjects2 is not None:
        # Must be an array
        assert n_subjects2.shape == n_subjects1.shape

        # Constants
        df = (n_subjects1 + n_subjects2) - 2  # degrees of freedom
        J_arr = gamma(df / 2) / (gamma((df - 1) / 2) * np.sqrt(df / 2))  # Hedges' correction

        g_arr = np.zeros(y.shape[0])
        for i_study in range(y.shape[0]):
            n1, n2 = n_subjects1[i_study], n_subjects2[i_study]
            study_data = y[i_study, : n1 + n2]
            var1 = np.var(study_data[:n1], axis=0)
            var2 = np.var(study_data[n1:], axis=0)

            mean_diff = np.mean(study_data[:n1]) - np.mean(study_data[n1:])
            pooled_std = np.sqrt((((n1 - 1) * var1) + ((n2 - 1) * var2)) / ((n1 + n2) - 2))
            g = mean_diff / pooled_std
            g_arr[i_study] = g

        g_arr = g_arr * J_arr * df

    else:
        df = n_subjects1 - 1
        J_arr = gamma(df / 2) / (gamma((df - 1) / 2) * np.sqrt(df / 2))  # Hedges' correction
        g_arr = J_arr * df * np.nanmean(y, axis=1) / np.nanstd(y, axis=1)

    return g_arr


def hedges_g_var(g, n_subjects1, n_subjects2=None):
    """Calculate variance of Hedges' G.

    .. todo::

        Support one-sample tests.

    Parameters
    ----------
    g : :obj:`numpy.ndarray` of shape (n_studies,)
        Hedges' G values.
    n_subjects1 : :obj:`numpy.ndarray` of shape (n_studies,)
    n_subjects2 : None or :obj:`numpy.ndarray` of shape (n_studies,)
        Can be None if the G values come from one-sample tests.

    Returns
    -------
    g_var : :obj:`numpy.ndarray` of shape (n_studies,)
        Hedges' G variance values.

    Notes
    -----
    Clues for Python version from https://constantinyvesplessen.com/post/g/#hedges-g,
    except using proper J instead of approximation.

    I also updated the original code to support varying sample sizes across studies.

    Notes
    -----
    R Code:

    .. highlight:: r
    .. code-block:: r

        df <- 2 * n.subj - 2 # Degrees of freedom
        J <- gamma(df / 2) / gamma((df - 1) / 2) * sqrt(2 / df) # Hedges' correction
        g_var <- function (g) { # Variance of Hedges' g
            1 / n.subj + (1 - (df - 2) / (df * J^2)) * g^2
        }
    """
    if n_subjects2 is not None:
        assert g.shape == n_subjects1.shape == n_subjects2.shape
        g_var = ((n_subjects1 + n_subjects2) / (n_subjects1 * n_subjects2)) + (
            (g**2) / (2 * (n_subjects1 + n_subjects2))
        )
    else:
        raise ValueError("One-sample tests are not yet supported.")

    return g_var
