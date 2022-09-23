"""Utilities for coordinate-based meta-analysis estimators."""
import logging
import warnings

import numpy as np
import sparse
from scipy import ndimage

from nimare.utils import unique_rows

LGR = logging.getLogger(__name__)


def compute_kda_ma(
    mask,
    ijks,
    r,
    value=1.0,
    exp_idx=None,
    sum_overlap=False,
):
    """Compute (M)KDA modeled activation (MA) map.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays.
        * Return 4D sparse array.
        * `shape` and `vox_dims` parameters have been removed. That information is now extracted
          from the new parameter `mask`.

    .. versionadded:: 0.0.4

    Replaces the values around each focus in ijk with binary sphere.

    Parameters
    ----------
    mask : img_like
        Mask to extract the MA maps shape (typically (91, 109, 91)) and voxel dimension.
        The mask is applied the data coordinated before creating the kernel_data.
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

    Returns
    -------
    kernel_data : :obj:`sparse._coo.core.COO`
        4D sparse array. If `exp_idx` is none, a 3d array in the same
        shape as the `shape` argument is returned. If `exp_idx` is passed, a 4d array
        is returned, where the first dimension has size equal to the number of
        unique experiments, and the remaining 3 dimensions are equal to `shape`.
    """
    shape = mask.shape
    vox_dims = mask.header.get_zooms()

    mask_data = mask.get_fdata().astype(bool)

    if exp_idx is None:
        exp_idx = np.ones(len(ijks))

    exp_idx_uniq, exp_idx = np.unique(exp_idx, return_inverse=True)
    n_studies = len(exp_idx_uniq)

    kernel_shape = (n_studies,) + shape

    n_dim = ijks.shape[1]
    xx, yy, zz = [slice(-r // vox_dims[i], r // vox_dims[i] + 0.01, 1) for i in range(n_dim)]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    kernel = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** 0.5 <= r]

    def _convolve_sphere(kernel, peaks):
        """Convolve peaks with a spherical kernel.

        Parameters
        ----------
        kernel : 2D numpy.ndarray
            IJK coordinates of a sphere, relative to a central point
            (not the brain template).
        peaks : 2D numpy.ndarray
            The IJK coordinates of peaks to convolve with the kernel.

        Returns
        -------
        sphere_coords : 2D numpy.ndarray
            All coordinates that fall within any sphere.
            Coordinates from overlapping spheres will appear twice.
        """
        # Convolve spheres
        sphere_coords = np.zeros((kernel.shape[1] * len(peaks), 3), dtype=int)
        chunk_idx = np.arange(0, (kernel.shape[1]), dtype=int)
        for peak in peaks:
            sphere_coords[chunk_idx, :] = kernel.T + peak
            chunk_idx = chunk_idx + kernel.shape[1]

        return sphere_coords

    all_coords = []
    all_exp = []
    all_data = []
    # Loop over experiments
    for i_exp, _ in enumerate(exp_idx_uniq):
        # Index peaks by experiment
        curr_exp_idx = exp_idx == i_exp
        peaks = ijks[curr_exp_idx]

        all_spheres = _convolve_sphere(kernel, peaks)

        if sum_overlap:
            all_spheres, counts = unique_rows(all_spheres, return_counts=True)
            counts = counts * value
        else:
            all_spheres = unique_rows(all_spheres)

        # Mask coordinates beyond space
        idx = np.all(
            np.concatenate([all_spheres >= 0, np.less(all_spheres, shape)], axis=1), axis=1
        )

        all_spheres = all_spheres[idx, :]

        sphere_idx_inside_mask = np.where(mask_data[tuple(all_spheres.T)])[0]
        sphere_idx_filtered = all_spheres[sphere_idx_inside_mask, :].T
        nonzero_idx = tuple(sphere_idx_filtered)

        if sum_overlap:
            nonzero_to_append = counts[idx][sphere_idx_inside_mask]
        else:
            nonzero_to_append = np.ones((len(sphere_idx_inside_mask),)) * value

        all_exp.append(np.full(nonzero_idx[0].shape[0], i_exp))
        all_coords.append(np.vstack(nonzero_idx))
        all_data.append(nonzero_to_append)

    exp = np.hstack(all_exp)
    coords = np.vstack((exp.flatten(), np.hstack(all_coords)))

    data = np.hstack(all_data).flatten()
    kernel_data = sparse.COO(coords, data, shape=kernel_shape)

    return kernel_data


def compute_ale_ma(mask, ijks, kernel=None, exp_idx=None, sample_sizes=None, use_dict=False):
    """Generate ALE modeled activation (MA) maps.

    Replaces the values around each focus in ijk with the contrast-specific
    kernel. Takes the element-wise maximum when looping through foci, which
    accounts for foci which are near to one another and may have overlapping
    kernels.

    .. versionchanged:: 0.0.12

        * This function now returns a 4D sparse array.
        * `shape` parameter has been removed. That information is now extracted
          from the new parameter `mask`.
        * Replace `ijk` with `ijks`.
        * New parameters: `exp_idx`, `sample_sizes`, and `use_dict`.

    Parameters
    ----------
    mask : img_like
        Mask to extract the MA maps shape (typically (91, 109, 91)) and voxel dimension.
        The mask is applied to the coordinates before creating the kernel_data.
    ijks : array-like
        Indices of foci. Each row is a coordinate, with the three columns
        corresponding to index in each of three dimensions.
    kernel : array-like, or None, optional
        3D array of smoothing kernel. Typically of shape (30, 30, 30).
    exp_idx : array_like
        Optional indices of experiments. If passed, must be of same length as
        ijks. Each unique value identifies all coordinates in ijk that come from
        the same experiment. If None passed, it is assumed that all coordinates
        come from the same experiment.
    sample_sizes : array_like, :obj:`int` or None, optional
        Array of smaple sizes or sample size, used to derive FWHM for Gaussian kernel.
    use_dict : :obj:`bool`, optional
        If True, empty kernels dictionary is used to retain the kernel for each element of
        sample_sizes. If False and sample_sizes is int, the ale kernel is calculated for
        sample_sizes. If False and sample_sizes is None, the unique kernels is used.

    Returns
    -------
    kernel_data : :obj:`sparse._coo.core.COO`
        4D sparse array. If `exp_idx` is none, a 3d array in the same
        shape as the `shape` argument is returned. If `exp_idx` is passed, a 4d array
        is returned, where the first dimension has size equal to the number of
        unique experiments, and the remaining 3 dimensions are equal to `shape`.
    """
    if use_dict:
        if kernel is not None:
            warnings.warn("The kernel provided will be replace by an empty dictionary.")
        kernels = {}  # retain kernels in dictionary to speed things up
        if not isinstance(sample_sizes, np.ndarray):
            raise ValueError("To use a kernel dictionary sample_sizes must be a list.")
    elif sample_sizes is not None:
        if not isinstance(sample_sizes, int):
            raise ValueError("If use_dict is False, sample_sizes provided must be integer.")
    else:
        if kernel is None:
            raise ValueError("3D array of smoothing kernel must be provided.")

    if exp_idx is None:
        exp_idx = np.ones(len(ijks))

    shape = mask.shape
    mask_data = mask.get_fdata().astype(bool)

    exp_idx_uniq, exp_idx = np.unique(exp_idx, return_inverse=True)
    n_studies = len(exp_idx_uniq)

    kernel_shape = (n_studies,) + shape
    all_exp = []
    all_coords = []
    all_data = []
    for i_exp, _ in enumerate(exp_idx_uniq):

        # Index peaks by experiment
        curr_exp_idx = exp_idx == i_exp
        ijk = ijks[curr_exp_idx]

        if use_dict:
            # Get sample_size from input
            sample_size = sample_sizes[curr_exp_idx][0]
            if sample_size not in kernels.keys():
                _, kernel = get_ale_kernel(mask, sample_size=sample_size)
                kernels[sample_size] = kernel
            else:
                kernel = kernels[sample_size]
        elif sample_sizes is not None:
            _, kernel = get_ale_kernel(mask, sample_size=sample_sizes)

        mid = int(np.floor(kernel.shape[0] / 2.0))
        mid1 = mid + 1
        ma_values = np.zeros(shape)
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
        # Set voxel outside the mask to zero.
        ma_values[~mask_data] = 0
        nonzero_idx = np.where(ma_values > 0)

        all_exp.append(np.full(nonzero_idx[0].shape[0], i_exp))
        all_coords.append(np.vstack(nonzero_idx))
        all_data.append(ma_values[nonzero_idx])

    exp = np.hstack(all_exp)
    coords = np.vstack((exp.flatten(), np.hstack(all_coords)))
    data = np.hstack(all_data).flatten()

    kernel_data = sparse.COO(coords, data, shape=kernel_shape)

    return kernel_data


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
    kernel = ndimage.gaussian_filter(data, sigma_vox, mode="constant")

    # Crop kernel to drop surrounding zeros
    mn = np.min(np.where(kernel > np.spacing(1))[0])
    mx = np.max(np.where(kernel > np.spacing(1))[0])
    kernel = kernel[mn : mx + 1, mn : mx + 1, mn : mx + 1]
    mid = int(np.floor(data.shape[0] / 2.0))
    return sigma_vox, kernel


def _get_last_bin(arr1d):
    """Index the last location in a 1D array with a non-zero value."""
    if np.any(arr1d):
        last_bin = np.where(arr1d)[0][-1]

    else:
        last_bin = 0

    return last_bin


def _calculate_cluster_measures(arr3d, threshold, conn, tail="upper"):
    """Calculate maximum cluster mass and size for an array.

    This method assesses both positive and negative clusters.

    Parameters
    ----------
    arr3d : :obj:`numpy.ndarray`
        Unthresholded 3D summary-statistic matrix. This matrix will end up changed in place.
    threshold : :obj:`float`
        Uncorrected summary-statistic thresholded for defining clusters.
    conn : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.

    Returns
    -------
    max_size, max_mass : :obj:`float`
        Maximum cluster size and mass from the matrix.
    """
    if tail == "upper":
        arr3d[arr3d <= threshold] = 0
    else:
        arr3d[np.abs(arr3d) <= threshold] = 0

    labeled_arr3d = np.empty(arr3d.shape, int)
    labeled_arr3d, _ = ndimage.label(arr3d > 0, conn)

    if tail == "two":
        # Label positive and negative clusters separately
        n_positive_clusters = np.max(labeled_arr3d)
        temp_labeled_arr3d, _ = ndimage.label(arr3d < 0, conn)
        temp_labeled_arr3d[temp_labeled_arr3d > 0] += n_positive_clusters
        labeled_arr3d = labeled_arr3d + temp_labeled_arr3d
        del temp_labeled_arr3d

    clust_sizes = np.bincount(labeled_arr3d.flatten())
    clust_vals = np.arange(0, clust_sizes.shape[0])

    # Cluster mass-based inference
    max_mass = 0
    for unique_val in clust_vals[1:]:
        ss_vals = np.abs(arr3d[labeled_arr3d == unique_val]) - threshold
        max_mass = np.maximum(max_mass, np.sum(ss_vals))

    # Cluster size-based inference
    clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
    if clust_sizes.size:
        max_size = np.max(clust_sizes)
    else:
        max_size = 0

    return max_size, max_mass
