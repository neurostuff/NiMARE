"""Miscellaneous spatial and statistical transforms
"""
import logging
import os.path as op

import numpy as np
import nibabel as nib
from scipy import stats
from scipy.special import ndtri

from .due import due
from . import references, utils

LGR = logging.getLogger(__name__)


def resolve_transforms(target, available_data, masker):
    available_types = list(available_data.keys())
    if target == 'z':
        if ('t' in available_types) and ('sample_sizes' in available_data):
            dof = np.sum(available_data['sample_sizes']) - len(available_data['sample_sizes'])
            t = masker.transform(available_data['t'])
            z = t_to_z(t, dof)
        elif ('p' in available_types):
            p = masker.transform(available_data['p'])
            z = p_to_z(p)
        else:
            return None
        z = masker.inverse_transform(z)
        return z
    else:
        return None


def transform_images(df, target, masker, metadata_df=None, out_dir=None):
    mask_img = masker.mask_img
    new_mask = np.ones(mask_img.shape, int)
    new_mask = nib.Nifti1Image(new_mask, mask_img.affine, header=mask_img.header)
    new_masker = utils.get_masker(new_mask)
    res = masker.mask_img.header.get_zooms()
    res = 'x'.join([str(r) for r in res])
    target_ids = df.loc[df[target].isnull(), 'id']
    for id_ in target_ids:
        row = df.loc[df['id'] == id_].iloc[0]

        # Determine output filename, if file can be generated
        if out_dir is None:
            options = [r for r in row.values if isinstance(r, str) and op.isfile(r)]
            id_out_dir = op.dirname(options[0])
        else:
            id_out_dir = out_dir
        new_file = op.join(id_out_dir, f'{id_}_{res}_{target}.nii.gz')

        # Grab columns with actual values
        available_data = row[~row.isnull()].to_dict()
        if metadata_df is not None:
            metadata_row = metadata_df.loc[metadata_df['id'] == id_].iloc[0]
            metadata = metadata_row[~metadata_row.isnull()].to_dict()
            for k, v in metadata.items():
                if k not in available_data.keys():
                    available_data[k] = v

        # Get converted data
        img = resolve_transforms(target, available_data, new_masker)
        if img is not None:
            img.to_filename(new_file)
            df.loc[df['id'] == id_, target] = new_file
        else:
            df.loc[df['id'] == id_, target] = None
    return df


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
