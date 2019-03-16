"""
Utilities
"""
from __future__ import division

import os.path as op
import logging

import numpy as np
import nibabel as nib
from nilearn import datasets

from .due import due, Doi, BibTeX

LGR = logging.getLogger(__name__)


def get_template(space='mni152_1mm', mask=None):
    if space == 'mni152_1mm':
        if mask is None:
            img = nib.load(datasets.fetch_icbm152_2009()['t1'])
        elif mask == 'brain':
            img = nib.load(datasets.fetch_icbm152_2009()['mask'])
        elif mask == 'gm':
            img = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)
        else:
            raise ValueError('Mask {0} not supported'.format(mask))
    elif space == 'mni152_2mm':
        if mask is None:
            img = datasets.load_mni152_template()
        elif mask == 'brain':
            img = datasets.load_mni152_brain_mask()
        elif mask == 'gm':
            # this approach seems to approximate the 0.2 thresholded
            # GM mask pretty well
            temp_img = datasets.load_mni152_template()
            data = temp_img.get_data()
            data = data * -1
            data[data != 0] += np.abs(np.min(data))
            data = (data > 1200).astype(int)
            img = nib.Nifti1Image(data, temp_img.affine)
        else:
            raise ValueError('Mask {0} not supported'.format(mask))
    elif space == 'ale_2mm':
        if mask is None:
            img = datasets.load_mni152_template()
        else:
            # Not the same as the nilearn brain mask, but should correspond to
            # the default "more conservative" MNI152 mask in GingerALE.
            img = nib.load(op.join(get_resource_path(), 'templates/MNI152_2x2x2_brainmask.nii.gz'))
    else:
        raise ValueError('Space {0} not supported'.format(space))
    return img


def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


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
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz).astype(int)
    return ijk


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


def get_resource_path():
    """
    Returns the path to general resources, terminated with separator. Resources
    are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), 'resources') + op.sep)
