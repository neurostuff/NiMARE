"""
Utilities
"""
from __future__ import division

from os.path import abspath, join, dirname, sep

import numpy as np
import nibabel as nib

from .due import due, Doi


def get_template(space='Mni305_1mm'):
    if space == 'Mni305_1mm':
        template_file = join(get_resource_path(), 'templates/MNI305_1mm.nii.gz')
    else:
        raise ValueError('Space {0} not supported'.format(space))
    return template_file


def get_mask(space='Mni305_1mm'):
    if space == 'Mni305_1mm':
        mask_file = join(get_resource_path(), 'templates/MNI305_1mm_mask.nii.gz')
    else:
        raise ValueError('Space {0} not supported'.format(space))
    return mask_file


def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def intersection(alpha, beta):
    """Return subset of alpha and beta that are present in both.
    """
    raise Exception('Not implemented')


def diff(alpha, beta):
    """Return subset of alpha that is not present in beta.
    """
    raise Exception('Not implemented')


def round2(ndarray):
    """
    Numpy rounds X.5 values to nearest even integer. We want to round to the
    nearest integer away from zero.
    """
    onedarray = ndarray.flatten()
    signs = np.sign(onedarray)  # pylint: disable=no-member
    idx = np.where(np.abs(onedarray-np.round(onedarray)) == 0.5)[0]
    x = np.abs(onedarray)
    y = np.round(x)
    y[idx] = np.ceil(x[idx])
    y *= signs
    rounded = y.reshape(ndarray.shape)
    return rounded.astype(int)


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
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz)
    return ijk


def thresh_str(num):
    """
    Create string of decimal values from number.
    """
    str_ = str(num - int(num))[2:]
    return str_


@due.dcite(Doi('10.1002/hbm.20345'),
           description='Introduces the Lancaster MNI-to-Talairach transform, '
                       'as well as its inverse, the Talairach-to-MNI '
                       'transform.')
@due.dcite(Doi('10.1016/j.neuroimage.2010.02.048'),
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
        print('Input is an ambiguous 3x3 matrix.\nAssuming coords are row vectors (Nx3).')
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError('Input must be an Nx3 or 3xN matrix.')
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array([[ 0.9357,     0.0029,    -0.0072,    -1.0423],
                           [-0.0065,     0.9396,    -0.0726,    -1.3940],
                           [ 0.0103,     0.0752,     0.8967,     3.6475],
                           [ 0.0000,     0.0000,     0.0000,     1.0000]])

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


@due.dcite(Doi('10.1002/hbm.20345'),
           description='Introduces the Lancaster MNI-to-Talairach transform, '
                       'as well as its inverse, the Talairach-to-MNI '
                       'transform.')
@due.dcite(Doi('10.1016/j.neuroimage.2010.02.048'),
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
        print('Input is an ambiguous 3x3 matrix.\nAssuming coords are row vectors (Nx3).')
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError('Input must be an Nx3 or 3xN matrix.')
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array([[ 0.9357,     0.0029,    -0.0072,    -1.0423],
                           [-0.0065,     0.9396,    -0.0726,    -1.3940],
                           [ 0.0103,     0.0752,     0.8967,     3.6475],
                           [ 0.0000,     0.0000,     0.0000,     1.0000]])

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]
    if use_dim == 1:
        out_coords = out_coords.transpose()
    return out_coords


def read_nifti(file_):
    """
    Read nifti file and return both matrix and affine.
    """
    info = nib.load(file_)
    affine = info.get_affine()
    data = np.asarray(np.squeeze(info.get_data()))
    return data, affine


def save_nifti(matrix, filename, affine):
    """
    Save matrix to nifti file using affine matrix.
    """
    img = nib.Nifti1Image(matrix, affine)
    img.to_filename(filename)


@due.dcite(Doi('10.1016/j.neuroimage.2010.07.033'),
           description='Introduces the MNI152 template.')
def cite_mni152():
    """
    Dummy function to cite MNI152 paper with duecredit.
    """
    pass


def get_resource_path():
    """
    Returns the path to general resources, terminated with separator. Resources
    are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return abspath(join(dirname(__file__), 'resources') + sep)
