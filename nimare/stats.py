"""Various statistical helper functions"""
import warnings

import numpy as np
from scipy import stats
from scipy.special import ndtri

from .due import due
from . import references


def one_way(data, n):
    """ One-way chi-square test of independence.
    Takes a 1D array as input and compares activation at each voxel to
    proportion expected under a uniform distribution throughout the array.
    Note that if you're testing activation with this, make sure that only
    valid voxels (e.g., in-mask gray matter voxels) are included in the
    array, or results won't make any sense!

    Returns
    -------
    Chi2 values
    """
    term = data.astype('float64')
    no_term = n - term
    t_exp = np.mean(term, 0)
    t_exp = np.array([t_exp, ] * data.shape[0])
    nt_exp = n - t_exp
    t_mss = (term - t_exp) ** 2 / t_exp
    nt_mss = (no_term - nt_exp) ** 2 / nt_exp
    chi2 = t_mss + nt_mss
    return chi2


def two_way(cells):
    """ Two-way chi-square test of independence.
    Takes a 3D array as input: N(voxels) x 2 x 2, where the last two
    dimensions are the contingency table for each of N voxels. Returns an
    array of chi2 values.
    """
    # Mute divide-by-zero warning for bad voxels since we account for that
    # later
    warnings.simplefilter("ignore", RuntimeWarning)

    cells = cells.astype('float64')  # Make sure we don't overflow
    total = np.apply_over_axes(np.sum, cells, [1, 2]).ravel()
    chi_sq = np.zeros(cells.shape, dtype='float64')
    for i in range(2):
        for j in range(2):
            exp = np.sum(cells[:, i, :], 1).ravel() * \
                np.sum(cells[:, :, j], 1).ravel() / total
            bad_vox = np.where(exp == 0)[0]
            chi_sq[:, i, j] = (cells[:, i, j] - exp) ** 2 / exp
            chi_sq[bad_vox, i, j] = 1.0  # Set p-value for invalid voxels to 1
    chi_sq = np.apply_over_axes(np.sum, chi_sq, [1, 2]).ravel()
    return chi_sq


def pearson(x, y):
    """
    Correlates row vector x with each row vector in 2D array y.

    Parameters
    ----------
    x : (1, N) array_like
    y : (M, N) array_like
    """
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(np.sum(datam**2, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    rs = temp / (datass[1:] * datass[0])
    return rs


def null_to_p(test_value, null_array, tail='two'):
    """Return two-sided p-value for test value against null array.
    """
    if tail == 'two':
        p_value = (50 - np.abs(stats.percentileofscore(
            null_array, test_value) - 50.)) * 2. / 100.
    elif tail == 'upper':
        p_value = 1 - (stats.percentileofscore(null_array, test_value) / 100.)
    elif tail == 'lower':
        p_value = stats.percentileofscore(null_array, test_value) / 100.
    else:
        raise ValueError('Argument "tail" must be one of ["two", "upper", '
                         '"lower"]')
    return p_value


def p_to_z(p, tail='two'):
    """Convert p-values to z-values.
    """
    eps = np.spacing(1)
    p = np.array(p)
    p[p < eps] = eps
    if tail == 'two':
        z = ndtri(1 - (p / 2))
        z = np.array(z)
    elif tail == 'one':
        z = ndtri(1 - p)
        z = np.array(z)
        z[z < 0] = 0
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    if z.shape == ():
        z = z[()]
    return z


@due.dcite(references.T2Z_TRANSFORM,
           description='Introduces T-to-Z transform.')
@due.dcite(references.T2Z_IMPLEMENTATION,
           description='Python implementation of T-to-Z transform.')
def t_to_z(t_values, dof):
    """
    From Vanessa Sochat's TtoZ package.
    """
    # Select just the nonzero voxels
    nonzero = t_values[t_values != 0]

    # We will store our results here
    z_values = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = (nonzero <= c)
    k2 = (nonzero > c)

    # Subset the data into two sets
    t1 = nonzero[k1]
    t2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_t1 = stats.t.cdf(t1, df=dof)
    z_values_t1 = stats.norm.ppf(p_values_t1)

    # Calculate p values for > 0
    p_values_t2 = stats.t.cdf(-t2, df=dof)
    z_values_t2 = -stats.norm.ppf(p_values_t2)
    z_values[k1] = z_values_t1
    z_values[k2] = z_values_t2

    # Write new image to file
    out = np.zeros(t_values.shape)
    out[t_values != 0] = z_values
    return out


def fdr(p, q=.05):
    """ Determine FDR threshold given a p value array and desired false
    discovery rate q. """
    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype='float') * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if any(below) else -1
