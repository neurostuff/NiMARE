"""
Utilities for coordinate-based meta-analysis estimators
"""
import os
import math
import logging
import requests
from io import BytesIO
from tarfile import TarFile

import numpy as np
import numpy.linalg as npl
import nibabel as nb
from scipy import ndimage
from lzma import LZMAFile
from tqdm.auto import tqdm

from .peaks2maps import model_fn
from ...due import due
from ... import references

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LGR = logging.getLogger(__name__)


def _get_resize_arg(target_shape):
    mni_shape_mm = np.array([148.0, 184.0, 156.0])
    target_resolution_mm = np.ceil(
        mni_shape_mm / np.array(target_shape)).astype(
        np.int32)
    target_affine = np.array([[4., 0., 0., -75.],
                              [0., 4., 0., -105.],
                              [0., 0., 4., -70.],
                              [0., 0., 0., 1.]])
    target_affine[0, 0] = target_resolution_mm[0]
    target_affine[1, 1] = target_resolution_mm[1]
    target_affine[2, 2] = target_resolution_mm[2]
    return target_affine, list(target_shape)


def _get_generator(contrasts_coordinates, target_shape, affine,
                   skip_out_of_bounds=False):
    def generator():
        for contrast in contrasts_coordinates:
            encoded_coords = np.zeros(list(target_shape))
            for real_pt in contrast:
                vox_pt = np.rint(nb.affines.apply_affine(
                    npl.inv(affine), real_pt)).astype(int)
                if skip_out_of_bounds and (vox_pt[0] >= 32 or
                                           vox_pt[1] >= 32 or vox_pt[2] >= 32):
                    continue
                encoded_coords[vox_pt[0], vox_pt[1], vox_pt[2]] = 1
            yield (encoded_coords, encoded_coords)

    return generator


def _get_checkpoint_dir():
    from appdirs import AppDirs
    dirs = AppDirs(appname="nimare", appauthor="neurostuff", version="1.0")
    checkpoint_dir = os.path.join(dirs.user_data_dir, "ohbm2018_model")
    if not os.path.exists(checkpoint_dir):
        LGR.info('Downloading the model (this is a one-off operation)...')
        url = "https://zenodo.org/record/1257721/files/ohbm2018_model.tar.xz?download=1"
        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)
        f = BytesIO()

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 * 1024
        wrote = 0
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size),
                         unit='MB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
        if total_size != 0 and wrote != total_size:
            raise Exception("Download interrupted")

        f.seek(0)
        LGR.info('Uncompressing the model to %s...'.format(checkpoint_dir))
        tarfile = TarFile(fileobj=LZMAFile(f), mode="r")
        tarfile.extractall(dirs.user_data_dir)
    return checkpoint_dir


@due.dcite(references.PEAKS2MAPS,
           description='Transforms coordinates of peaks to unthresholded maps using a deep '
                       'convolutional neural net.')
def peaks2maps(contrasts_coordinates, skip_out_of_bounds=True,
               tf_verbosity_level=None):
    """
    Generate modeled activation (MA) maps using depp ConvNet model peaks2maps

    Parameters
    ----------
    contrasts_coordinates : list of lists that are len == 3
        List of contrasts and their coordinates
    skip_out_of_bounds : aboolean, optional
        Remove coordinates outside of the bounding box of the peaks2maps model
    tf_verbosity_level : int
        Tensorflow verbosity logging level

    Returns
    -------
    ma_values : array-like
        1d array of modeled activation values.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        if "No module named 'tensorflow'" in str(e):
            raise Exception("tensorflow not installed - see https://www.tensorflow.org/install/ "
                            "for instructions")
        else:
            raise

    if tf_verbosity_level is None:
        tf_verbosity_level = tf.logging.FATAL
    target_shape = (32, 32, 32)
    affine, _ = _get_resize_arg(target_shape)
    tf.logging.set_verbosity(tf_verbosity_level)

    def generate_input_fn():
        dataset = tf.data.Dataset.from_generator(_get_generator(contrasts_coordinates,
                                                                target_shape, affine,
                                                                skip_out_of_bounds=skip_out_of_bounds),
                                                 (tf.float32, tf.float32),
                                                 (tf.TensorShape(target_shape), tf.TensorShape(target_shape)))
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    model_dir = _get_checkpoint_dir()
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    results = model.predict(generate_input_fn)
    results = [result for result in results]
    assert len(results) == len(contrasts_coordinates), "returned %d" % len(results)

    niis = [nb.Nifti1Image(np.squeeze(result), affine) for result in results]
    return niis


def compute_ma(shape, ijk, kernel):
    """
    Generate modeled activation (MA) maps.
    Replaces the values around each focus in ijk with the contrast-specific
    kernel. Takes the element-wise maximum when looping through foci, which
    accounts for foci which are near to one another and may have overlapping
    kernels.

    Parameters
    ----------
    shape : tuple
        Shape of brain image + buffer. Typically (91, 109, 91) + (30, 30, 30).
    ijk : array-like
        Indices of foci. Each row is a coordinate, with the three columns
        corresponding to index in each of three dimensions.
    kernel : array-like
        3D array of smoothing kernel. Typically of shape (30, 30, 30).

    Returns
    -------
    ma_values : array-like
        1d array of modeled activation values.
    """
    ma_values = np.zeros(shape)
    mid = int(np.floor(kernel.shape[0] / 2.))
    mid1 = mid + 1
    for j_peak in range(ijk.shape[0]):
        i, j, k = ijk[j_peak, :]
        xl = max(i - mid, 0)
        xh = min(i + mid1, ma_values.shape[0])
        yl = max(j - mid, 0)
        yh = min(j + mid1, ma_values.shape[1])
        zl = max(k - mid, 0)
        zh = min(k + mid1, ma_values.shape[2])
        xlk = mid - (i - xl)
        xhk = mid - (i - xh)
        ylk = mid - (j - yl)
        yhk = mid - (j - yh)
        zlk = mid - (k - zl)
        zhk = mid - (k - zh)

        if ((xl >= 0) & (xh >= 0) & (yl >= 0) & (yh >= 0) & (zl >= 0) &
                (zh >= 0) & (xlk >= 0) & (xhk >= 0) & (ylk >= 0) & (yhk >= 0) &
                (zlk >= 0) & (zhk >= 0)):
            ma_values[xl:xh, yl:yh, zl:zh] = np.maximum(
                ma_values[xl:xh, yl:yh, zl:zh],
                kernel[xlk:xhk, ylk:yhk, zlk:zhk])
    return ma_values


@due.dcite(references.ALE_KERNEL,
           description='Introduces sample size-dependent kernels to ALE.')
def get_ale_kernel(img, n=None, fwhm=None):
    """
    Estimate 3D Gaussian and sigma (in voxels) for ALE kernel given
    sample size (n) or fwhm (in mm).
    """
    if n is not None and fwhm is not None:
        raise ValueError('Only one of n and fwhm may be specified')
    elif n is None and fwhm is None:
        raise ValueError('Either n or fwhm must be provided')
    elif n is not None:
        uncertain_templates = (5.7 / (2. * np.sqrt(2. / np.pi)) *
                               np.sqrt(8. * np.log(2.)))  # pylint: disable=no-member
        # Assuming 11.6 mm ED between matching points
        uncertain_subjects = (11.6 / (2 * np.sqrt(2 / np.pi)) *
                              np.sqrt(8 * np.log(2))) / np.sqrt(n)  # pylint: disable=no-member
        fwhm = np.sqrt(uncertain_subjects ** 2 + uncertain_templates ** 2)

    fwhm_vox = fwhm / np.sqrt(np.prod(img.header.get_zooms()))
    sigma_vox = fwhm_vox * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)  # pylint: disable=no-member

    data = np.zeros((31, 31, 31))
    mid = int(np.floor(data.shape[0] / 2.))
    data[mid, mid, mid] = 1.
    kernel = ndimage.filters.gaussian_filter(data, sigma_vox, mode='constant')

    # Crop kernel to drop surrounding zeros
    mn = np.min(np.where(kernel > np.spacing(1))[0])
    mx = np.max(np.where(kernel > np.spacing(1))[0])
    kernel = kernel[mn:mx + 1, mn:mx + 1, mn:mx + 1]
    mid = int(np.floor(data.shape[0] / 2.))
    return sigma_vox, kernel
