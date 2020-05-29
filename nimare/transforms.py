"""Miscellaneous spatial and statistical transforms
"""
import numpy as np
from scipy import stats
from scipy.special import ndtri

from .due import due
from . import references


def sd_to_var(sd, n):
    """Convert standard deviation to sampling variance.

    Parameters
    ----------
    sd : array_like
        Standard deviation of the sample
    n : int
        Sample size

    Returns
    -------
    var : array_like
        Sampling variance of the parameter
    """
    se = sd / np.sqrt(n)
    var = se_to_var(se)
    return var


def se_to_var(se):
    """Convert standard error values to sampling variance.

    Parameters
    ----------
    se : array_like
        Standard error of the parameter

    Returns
    -------
    var : array_like
        Sampling variance of the parameter
    """
    var = se ** 2
    return var


def svar_to_var(svar, n):
    """Convert "sample variance" (variance of the individual observations in a
    single sample) to "sampling variance" (variance of sampling distribution
    for the parameter).

    Parameters
    ----------
    svar : array_like
        Sample variance
    n : int
        Sample size

    Returns
    -------
    var : array_like
        Sampling variance of the parameter
    """
    var = svar / n
    return var


def t_to_beta(t, var):
    """Convert t-statistic to parameter estimate using sampling variance.

    Parameters
    ----------
    t : array_like
        T-statistics of the parameter
    var : array_like
        Sampling variance of the parameter

    Returns
    -------
    pe : array_like
        Parameter estimates
    """
    pe = t * np.sqrt(var)
    return pe


def p_to_z(p, tail='two'):
    """Convert p-values to (unsigned) z-values.

    Parameters
    ----------
    p : array_like
        P-values
    tail : {'one', 'two'}, optional
        Whether p-values come from one-tailed or two-tailed test. Default is
        'two'.

    Returns
    -------
    z : array_like
        Z-statistics (unsigned)
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
    Convert t-statistics to z-statistics.

    An implementation of [1]_ from Vanessa Sochat's TtoZ package [2]_.

    Parameters
    ----------
    t_values : array_like
        T-statistics
    dof : int
        Degrees of freedom

    Returns
    -------
    z_values : array_like
        Z-statistics

    References
    ----------
    .. [1] Hughett, P. (2007). Accurate Computation of the F-to-z and t-to-z
           Transforms for Large Arguments. Journal of Statistical Software,
           23(1), 1-5.
    .. [2] Sochat, V. (2015, October 21). TtoZ Original Release. Zenodo.
           http://doi.org/10.5281/zenodo.32508
    """
    # Select just the nonzero voxels
    nonzero = t_values[t_values != 0]

    # We will store our results here
    z_values_nonzero = np.zeros(len(nonzero))

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
    z_values_nonzero[k1] = z_values_t1
    z_values_nonzero[k2] = z_values_t2

    z_values = np.zeros(t_values.shape)
    z_values[t_values != 0] = z_values_nonzero
    return z_values
