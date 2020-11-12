"""Various statistical helper functions
"""
import logging
import warnings

import numpy as np
from scipy import stats

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
    datass = np.sqrt(np.sum(datam ** 2, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    rs = temp / (datass[1:] * datass[0])
    return rs


def null_to_p(test_value, null_array, tail="two"):
    """Return p-value for test value against null array.

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

    Returns
    -------
    p_value : :obj:`float`
        P-value associated with the test value when compared against the null
        distribution.

    Notes
    -----
    P-values are clipped based on the number of elements in the null array.
    Therefore no p-values of 0 or 1 should be produced.
    """
    test_value = np.atleast_1d(test_value)

    # For efficiency's sake, if there are more than 1000 values, pass only the unique
    # values through percentileofscore(), and then reconstruct.
    if len(test_value) > 1000:
        reconstruct = True
        test_value, uniq_idx = np.unique(test_value, return_inverse=True)
    else:
        reconstruct = False

    # TODO: this runs in N^2 time; is there a more efficient alternative?
    p = np.array([stats.percentileofscore(null_array, v, "strict") for v in test_value])
    p /= 100.0
    if tail == "two":
        p = (0.5 - np.abs(p - 0.5)) * 2
    elif tail == "upper":
        p = 1 - p
    elif tail != "lower":
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')

    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array))

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value)
    result = np.maximum(smallest_value, np.minimum(p, 1.0 - smallest_value))

    if reconstruct:
        result = result[uniq_idx]

    return result


def fdr(p, q=0.05):
    """Determine FDR threshold given a p value array and desired false
    discovery rate q.

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
