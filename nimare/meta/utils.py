"""Utilities for coordinate-based meta-analysis estimators."""
import logging

import numpy as np
from scipy import ndimage

from .. import references
from ..due import due
from ..utils import _determine_chunk_size

LGR = logging.getLogger(__name__)


def compute_kda_ma(
    shape,
    vox_dims,
    ijks,
    r,
    value=1.0,
    exp_idx=None,
    sum_overlap=False,
    memory_limit=None,
    memmap_filename=None,
):
    """Compute (M)KDA modeled activation (MA) map.

    .. versionchanged:: 0.0.8

        * [ENH] Add *memmap_filename* parameter for memory mapping arrays.

    .. versionadded:: 0.0.4

    Replaces the values around each focus in ijk with binary sphere.

    Parameters
    ----------
    shape : :obj:`tuple`
        Shape of brain image + buffer. Typically (91, 109, 91).
    vox_dims : array_like
        Size (in mm) of each dimension of a voxel.
    ijks : array-like
        Indices of foci. Each row is a coordinate, with the three columns
        corresponding to index in each of three dimensions.
    r : :obj:`int`
        Sphere radius, in mm.
    value : :obj:`int`
        Value for sphere.
    exp_idx : array_like
        Optional indices of experiments. If passed, must be of same length as
        ijks. Each unique value identifies all coordinates in ijk that come from
        the same experiment. If None passed, it is assumed that all coordinates
        come from the same experiment.
    sum_overlap : :obj:`bool`
        Whether to sum voxel values in overlapping spheres.
    memory_limit : :obj:`str` or None, optional
        Memory limit to apply to data. If None, no memory management will be applied.
        Otherwise, the memory limit will be used to (1) assign memory-mapped files and
        (2) restrict memory during array creation to the limit.
        Default is None.
    memmap_filename : :obj:`str`, optional
        If passed, use this file for memory mapping arrays

    Returns
    -------
    kernel_data : :obj:`numpy.array`
        3d or 4d array. If `exp_idx` is none, a 3d array in the same shape as
        the `shape` argument is returned. If `exp_idx` is passed, a 4d array
        is returned, where the first dimension has size equal to the number of
        unique experiments, and the remaining 3 dimensions are equal to `shape`.
    """
    squeeze = exp_idx is None
    if exp_idx is None:
        exp_idx = np.ones(len(ijks))

    uniq, exp_idx = np.unique(exp_idx, return_inverse=True)
    n_studies = len(uniq)

    kernel_shape = (n_studies,) + shape
    if memmap_filename:
        # Use a memmapped 4D array
        kernel_data = np.memmap(memmap_filename, dtype=type(value), mode="w+", shape=kernel_shape)
    else:
        kernel_data = np.zeros(kernel_shape, dtype=type(value))

    n_dim = ijks.shape[1]
    xx, yy, zz = [slice(-r // vox_dims[i], r // vox_dims[i] + 0.01, 1) for i in range(n_dim)]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    kernel = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** 0.5 <= r]

    if memory_limit:
        chunk_size = _determine_chunk_size(limit=memory_limit, arr=ijks[0])

    for i, peak in enumerate(ijks):
        sphere = np.round(kernel.T + peak)
        idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, shape), 1) <= -1)
        sphere = sphere[idx, :].astype(int)
        exp = exp_idx[i]
        if sum_overlap:
            kernel_data[exp][tuple(sphere.T)] += value
        else:
            kernel_data[exp][tuple(sphere.T)] = value

        if memmap_filename and i % chunk_size == 0:
            # Write changes to disk
            kernel_data.flush()

    if squeeze:
        kernel_data = np.squeeze(kernel_data, axis=0)

    return kernel_data


def compute_ale_ma(shape, ijk, kernel):
    """Generate ALE modeled activation (MA) maps.

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
    mid = int(np.floor(kernel.shape[0] / 2.0))
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

        if (
            (xl >= 0)
            & (xh >= 0)
            & (yl >= 0)
            & (yh >= 0)
            & (zl >= 0)
            & (zh >= 0)
            & (xlk >= 0)
            & (xhk >= 0)
            & (ylk >= 0)
            & (yhk >= 0)
            & (zlk >= 0)
            & (zhk >= 0)
        ):
            ma_values[xl:xh, yl:yh, zl:zh] = np.maximum(
                ma_values[xl:xh, yl:yh, zl:zh], kernel[xlk:xhk, ylk:yhk, zlk:zhk]
            )
    return ma_values


@due.dcite(references.ALE_KERNEL, description="Introduces sample size-dependent kernels to ALE.")
def get_ale_kernel(img, sample_size=None, fwhm=None):
    """Estimate 3D Gaussian and sigma (in voxels) for ALE kernel given sample size or fwhm."""
    if sample_size is not None and fwhm is not None:
        raise ValueError('Only one of "sample_size" and "fwhm" may be specified')
    elif sample_size is None and fwhm is None:
        raise ValueError('Either "sample_size" or "fwhm" must be provided')
    elif sample_size is not None:
        uncertain_templates = (
            5.7 / (2.0 * np.sqrt(2.0 / np.pi)) * np.sqrt(8.0 * np.log(2.0))
        )  # pylint: disable=no-member
        # Assuming 11.6 mm ED between matching points
        uncertain_subjects = (11.6 / (2 * np.sqrt(2 / np.pi)) * np.sqrt(8 * np.log(2))) / np.sqrt(
            sample_size
        )  # pylint: disable=no-member
        fwhm = np.sqrt(uncertain_subjects**2 + uncertain_templates**2)

    fwhm_vox = fwhm / np.sqrt(np.prod(img.header.get_zooms()))
    sigma_vox = (
        fwhm_vox * np.sqrt(2.0) / (np.sqrt(2.0 * np.log(2.0)) * 2.0)
    )  # pylint: disable=no-member

    data = np.zeros((31, 31, 31))
    mid = int(np.floor(data.shape[0] / 2.0))
    data[mid, mid, mid] = 1.0
    kernel = ndimage.filters.gaussian_filter(data, sigma_vox, mode="constant")

    # Crop kernel to drop surrounding zeros
    mn = np.min(np.where(kernel > np.spacing(1))[0])
    mx = np.max(np.where(kernel > np.spacing(1))[0])
    kernel = kernel[mn : mx + 1, mn : mx + 1, mn : mx + 1]
    mid = int(np.floor(data.shape[0] / 2.0))
    return sigma_vox, kernel
