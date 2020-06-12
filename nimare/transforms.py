"""Miscellaneous spatial and statistical transforms
"""
import logging

import numpy as np
import nibabel as nib
from scipy import stats
from scipy.special import ndtri

from .due import due
from . import references

LGR = logging.getLogger(__name__)


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
        Standard error of the sample parameter

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


def vox2mm(ijk, affine):
    """
    Convert matrix subscripts to coordinates.
    From here:
    http://blog.chrisgorgolewski.org/2014/12/how-to-convert-between-voxel-and-mm.html
    """
    xyz = nib.affines.apply_affine(affine, ijk)
    return xyz


def mm2vox(xyz, affine):
    """
    Convert coordinates to matrix subscripts.
    From here:
    http://blog.chrisgorgolewski.org/2014/12/how-to-convert-between-voxel-and-mm.html
    """
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz).astype(int)
    return ijk


@due.dcite(references.LANCASTER_TRANSFORM,
           description='Introduces the Lancaster MNI-to-Talairach transform, '
                       'as well as its inverse, the Talairach-to-MNI '
                       'transform.')
@due.dcite(references.LANCASTER_TRANSFORM_VALIDATION,
           description='Validates the Lancaster MNI-to-Talairach and '
                       'Talairach-to-MNI transforms.')
def tal2mni(coords):
    """
    Python version of BrainMap's tal2icbm_other.m.
    This function converts coordinates from Talairach space to MNI
    space (normalized using templates other than those contained
    in SPM and FSL) using the tal2icbm transform developed and
    validated by Jack Lancaster at the Research Imaging Center in
    San Antonio, Texas.
    http://www3.interscience.wiley.com/cgi-bin/abstract/114104479/ABSTRACT
    FORMAT outpoints = tal2icbm_other(inpoints)
    Where inpoints is N by 3 or 3 by N matrix of coordinates
    (N being the number of points)
    ric.uthscsa.edu 3/14/07
    """
    # Find which dimensions are of size 3
    shape = np.array(coords.shape)
    if all(shape == 3):
        LGR.info('Input is an ambiguous 3x3 matrix.\nAssuming coords are row '
                 'vectors (Nx3).')
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError('Input must be an Nx3 or 3xN matrix.')
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array([[0.9357, 0.0029, -0.0072, -1.0423],
                           [-0.0065, 0.9396, -0.0726, -1.3940],
                           [0.0103, 0.0752, 0.8967, 3.6475],
                           [0.0000, 0.0000, 0.0000, 1.0000]])

    # Invert the transformation matrix
    icbm_other = np.linalg.inv(icbm_other)

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]
    if use_dim == 1:
        out_coords = out_coords.transpose()
    return out_coords


@due.dcite(references.LANCASTER_TRANSFORM,
           description='Introduces the Lancaster MNI-to-Talairach transform, '
                       'as well as its inverse, the Talairach-to-MNI '
                       'transform.')
@due.dcite(references.LANCASTER_TRANSFORM_VALIDATION,
           description='Validates the Lancaster MNI-to-Talairach and '
                       'Talairach-to-MNI transforms.')
def mni2tal(coords):
    """
    Python version of BrainMap's icbm_other2tal.m.
    This function converts coordinates from MNI space (normalized using
    templates other than those contained in SPM and FSL) to Talairach space
    using the icbm2tal transform developed and validated by Jack Lancaster at
    the Research Imaging Center in San Antonio, Texas.
    http://www3.interscience.wiley.com/cgi-bin/abstract/114104479/ABSTRACT
    FORMAT outpoints = icbm_other2tal(inpoints)
    Where inpoints is N by 3 or 3 by N matrix of coordinates
    (N being the number of points)
    ric.uthscsa.edu 3/14/07
    """
    # Find which dimensions are of size 3
    shape = np.array(coords.shape)
    if all(shape == 3):
        LGR.info('Input is an ambiguous 3x3 matrix.\nAssuming coords are row '
                 'vectors (Nx3).')
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError('Input must be an Nx3 or 3xN matrix.')
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array([[0.9357, 0.0029, -0.0072, -1.0423],
                           [-0.0065, 0.9396, -0.0726, -1.3940],
                           [0.0103, 0.0752, 0.8967, 3.6475],
                           [0.0000, 0.0000, 0.0000, 1.0000]])

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]
    if use_dim == 1:
        out_coords = out_coords.transpose()
    return out_coords
