"""Utilities for coordinate-based meta-analysis estimators."""

import warnings

import numpy as np
import sparse
from numba import jit
from scipy import ndimage
from scipy import sparse as sp_sparse

from nimare.utils import DEFAULT_FLOAT_DTYPE, _mask_img_to_bool, unique_rows

# based on local benchmarks, tested 20, 30, 40, 50, 100, 200 studies
# sorting provides speed benefits starting betwee 30 and 40 studies
KDA_SORT_MIN_STUDIES = 40
# occupancy-mask vs. unique-rows crossover observed around ~50 foci/study
KDA_OCCUPANCY_MIN_FOCI = 50


@jit(nopython=True, cache=True)
def _convolve_sphere(kernel, ijks, index, max_shape):
    """Convolve peaks with a spherical kernel.

    Parameters
    ----------
    kernel : 2D numpy.ndarray
        IJK coordinates of a sphere, relative to a central point
        (not the brain template).
    peaks : 2D numpy.ndarray
        The IJK coordinates of peaks to convolve with the kernel.
    max_shape: 1D numpy.ndarray
        The maximum shape of the image volume.

    Returns
    -------
    sphere_coords : 2D numpy.ndarray
        All coordinates that fall within any sphere.ß∑
        Coordinates from overlapping spheres will appear twice.
    """

    def np_all_axis1(x):
        """Numba compatible version of np.all(x, axis=1)."""
        out = np.ones(x.shape[0], dtype=np.bool_)
        for i in range(x.shape[1]):
            out = np.logical_and(out, x[:, i])
        return out

    peaks = ijks[index]
    sphere_coords = np.zeros((kernel.shape[1] * len(peaks), 3), dtype=np.int32)
    chunk_idx = np.arange(0, (kernel.shape[1]), dtype=np.int64)
    for peak in peaks:
        sphere_coords[chunk_idx, :] = kernel.T + peak
        chunk_idx = chunk_idx + kernel.shape[1]

    # Mask coordinates beyond space
    idx = np_all_axis1(np.logical_and(sphere_coords >= 0, np.less(sphere_coords, max_shape)))

    return sphere_coords[idx, :]


@jit(nopython=True, cache=True)
def _convolve_sphere_to_mask(kernel, ijks, index, max_shape):
    """Convolve peaks with a spherical kernel into a boolean occupancy mask."""
    peaks = ijks[index]
    occ = np.zeros((max_shape[0], max_shape[1], max_shape[2]), dtype=np.bool_)
    for peak in peaks:
        for i in range(kernel.shape[1]):
            x = kernel[0, i] + peak[0]
            y = kernel[1, i] + peak[1]
            z = kernel[2, i] + peak[2]
            if (
                (x >= 0)
                and (y >= 0)
                and (z >= 0)
                and (x < max_shape[0])
                and (y < max_shape[1])
                and (z < max_shape[2])
            ):
                occ[x, y, z] = True
    return occ


@jit(nopython=True, cache=True)
def _sum_across_studies_last_seen(kernel, ijks, exp_idx, n_studies, max_shape, value):
    """Accumulate study counts directly while deduplicating voxels within each study.

    This matches the previous Python implementation for ``sum_across_studies=True``:
    each voxel contributes at most once per study before being added into the across-study
    summary map, even if multiple peaks from the same study overlap there.
    """
    all_values = np.zeros((max_shape[0], max_shape[1], max_shape[2]), dtype=np.int32)
    last_seen = np.full((max_shape[0], max_shape[1], max_shape[2]), -1, dtype=np.int32)

    for study_idx in range(n_studies):
        for peak_idx in range(ijks.shape[0]):
            if exp_idx[peak_idx] != study_idx:
                continue

            peak = ijks[peak_idx]
            for kernel_idx in range(kernel.shape[1]):
                x = kernel[0, kernel_idx] + peak[0]
                y = kernel[1, kernel_idx] + peak[1]
                z = kernel[2, kernel_idx] + peak[2]
                if (
                    (x >= 0)
                    and (y >= 0)
                    and (z >= 0)
                    and (x < max_shape[0])
                    and (y < max_shape[1])
                    and (z < max_shape[2])
                    and (last_seen[x, y, z] != study_idx)
                ):
                    last_seen[x, y, z] = study_idx
                    all_values[x, y, z] += value

    return all_values


def compute_kda_ma(
    mask,
    ijks,
    r,
    value=1.0,
    exp_idx=None,
    sum_overlap=False,
    sum_across_studies=False,
):
    """Compute (M)KDA modeled activation (MA) map.

    .. versionchanged:: 0.2.2

        * Return masked study-by-voxel CSR matrices for sparse outputs.
        * `shape` and `vox_dims` parameters have been removed. That information is now extracted
          from the new parameter `mask`.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays.

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
    sum_across_studies : :obj:`bool`
        Whether to sum voxel values across studies.

    Returns
    -------
    kernel_data : :obj:`numpy.ndarray` or tuple
        If ``sum_across_studies`` is True, returns a masked 1D summary array.
        Otherwise returns a tuple of:

        1. A masked study-by-voxel CSR matrix of shape ``(n_studies, n_mask_voxels)``
        2. An array mapping flattened full-volume voxel indices to masked voxel indices.
    """
    if sum_overlap and sum_across_studies:
        raise NotImplementedError("sum_overlap and sum_across_studies cannot both be True.")

    if exp_idx is None:
        exp_idx = np.ones(len(ijks), dtype=np.int32)

    ijks = ijks.astype(np.int32, copy=False)
    shape = mask.shape
    vox_dims = mask.header.get_zooms()
    max_shape = np.array(shape, dtype=np.int32)
    mask_data = _mask_img_to_bool(mask)
    mask_flat_to_masked = _get_mask_flat_to_masked(mask)
    n_voxels = int(mask_data.sum())

    exp_idx_uniq, exp_idx = np.unique(exp_idx, return_inverse=True)
    n_studies = len(exp_idx_uniq)

    n_dim = ijks.shape[1]
    xx, yy, zz = [slice(-r // vox_dims[i], r // vox_dims[i] + 0.01, 1) for i in range(n_dim)]
    cube = np.vstack([row.ravel() for row in (np.mgrid[xx, yy, zz]).astype(np.int32)])
    kernel = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** 0.5 <= r]

    if sum_across_studies:
        # The JIT helper preserves the previous semantics while avoiding per-study temporary
        # arrays: deduplicate voxels within each study, then accumulate once across studies.
        all_values = _sum_across_studies_last_seen(
            kernel,
            ijks,
            exp_idx.astype(np.int32),
            n_studies,
            max_shape,
            np.int32(value),
        )

        # Only return values within the mask
        all_values = all_values.reshape(-1)
        kernel_data = all_values[mask_data.reshape(-1)]

    else:
        exp_counts = np.bincount(exp_idx, minlength=n_studies)
        use_occ_by_exp = (not sum_overlap) & (exp_counts >= KDA_OCCUPANCY_MIN_FOCI)
        flat_stride_y = shape[2]
        flat_stride_x = shape[1] * shape[2]
        indptr = [0]
        indices_parts = []
        data_parts = []
        value = DEFAULT_FLOAT_DTYPE(value)

        for i_exp, _ in enumerate(exp_idx_uniq):
            curr_exp_idx = exp_idx == i_exp
            use_occ = use_occ_by_exp[i_exp]

            if sum_overlap:
                all_spheres = _convolve_sphere(kernel, ijks, curr_exp_idx, max_shape)
                if all_spheres.size:
                    flat_coords = (
                        all_spheres[:, 0] * flat_stride_x
                        + all_spheres[:, 1] * flat_stride_y
                        + all_spheres[:, 2]
                    )
                    cols = mask_flat_to_masked[flat_coords]
                    cols = cols[cols >= 0]
                    if cols.size:
                        cols, counts = np.unique(cols, return_counts=True)
                        vals = counts.astype(DEFAULT_FLOAT_DTYPE, copy=False) * value
                    else:
                        cols = np.array([], dtype=np.int32)
                        vals = np.array([], dtype=DEFAULT_FLOAT_DTYPE)
                else:
                    cols = np.array([], dtype=np.int32)
                    vals = np.array([], dtype=DEFAULT_FLOAT_DTYPE)
            elif use_occ:
                occ = _convolve_sphere_to_mask(kernel, ijks, curr_exp_idx, max_shape)
                occ &= mask_data
                flat_occ = np.flatnonzero(occ.reshape(-1))
                cols = mask_flat_to_masked[flat_occ]
                vals = np.full(cols.shape[0], value, dtype=DEFAULT_FLOAT_DTYPE)
            else:
                all_spheres = _convolve_sphere(kernel, ijks, curr_exp_idx, max_shape)
                if all_spheres.size:
                    all_spheres = unique_rows(all_spheres)
                    flat_coords = (
                        all_spheres[:, 0] * flat_stride_x
                        + all_spheres[:, 1] * flat_stride_y
                        + all_spheres[:, 2]
                    )
                    cols = mask_flat_to_masked[flat_coords]
                    cols = cols[cols >= 0]
                    cols.sort()
                    vals = np.full(cols.shape[0], value, dtype=DEFAULT_FLOAT_DTYPE)
                else:
                    cols = np.array([], dtype=np.int32)
                    vals = np.array([], dtype=DEFAULT_FLOAT_DTYPE)

            cols = cols.astype(np.int32, copy=False)
            vals = vals.astype(DEFAULT_FLOAT_DTYPE, copy=False)
            indices_parts.append(cols)
            data_parts.append(vals)
            indptr.append(indptr[-1] + cols.shape[0])

        indices = (
            np.concatenate(indices_parts).astype(np.int32, copy=False)
            if indices_parts
            else np.array([], dtype=np.int32)
        )
        data = (
            np.concatenate(data_parts).astype(DEFAULT_FLOAT_DTYPE, copy=False)
            if data_parts
            else np.array([], dtype=DEFAULT_FLOAT_DTYPE)
        )
        indptr = np.array(indptr, dtype=np.int64)

        kernel_data = sp_sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_studies, n_voxels),
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        kernel_data.sort_indices()
        kernel_data = kernel_data, mask_flat_to_masked

    return kernel_data


def _get_mask_flat_to_masked(mask_img):
    """Map flattened full-volume voxel indices to masked voxel indices."""
    mask_data = _mask_img_to_bool(mask_img).reshape(-1)
    mask_flat_to_masked = np.full(mask_data.shape[0], -1, dtype=np.int32)
    mask_flat_to_masked[mask_data] = np.arange(mask_data.sum(), dtype=np.int32)
    return mask_flat_to_masked


def _coo_to_masked_csr(ma_values, mask_img, mask_flat_to_masked=None):
    """Convert 4D ALE MA maps to a study-by-voxel CSR matrix within the analysis mask."""
    if sp_sparse.isspmatrix_csr(ma_values):
        return ma_values, mask_flat_to_masked

    if not isinstance(ma_values, sparse._coo.core.COO):
        return ma_values, mask_flat_to_masked

    if mask_flat_to_masked is None:
        mask_flat_to_masked = _get_mask_flat_to_masked(mask_img)

    flat_voxels = np.ravel_multi_index(ma_values.coords[1:], dims=mask_img.shape)
    rows = ma_values.coords[0].astype(np.int32, copy=False)
    cols = mask_flat_to_masked[flat_voxels]
    valid_mask = cols >= 0
    data = ma_values.data.astype(DEFAULT_FLOAT_DTYPE, copy=False)
    n_voxels = int(mask_flat_to_masked.max()) + 1 if mask_flat_to_masked.size else 0
    csr = sp_sparse.csr_matrix(
        (data[valid_mask], (rows[valid_mask], cols[valid_mask])),
        shape=(ma_values.shape[0], n_voxels),
        dtype=DEFAULT_FLOAT_DTYPE,
    )
    csr.sort_indices()
    return csr, mask_flat_to_masked


def _kernel_to_sparse_support(kernel):
    """Convert a dense ALE kernel to sparse offsets and values."""
    nonzero_idx = np.array(np.where(kernel > 0), dtype=np.int32)
    center = np.floor(np.array(kernel.shape) / 2.0).astype(np.int32)[:, None]
    offsets = (nonzero_idx - center).T.astype(np.int32, copy=False)
    values = kernel[tuple(nonzero_idx)].astype(DEFAULT_FLOAT_DTYPE, copy=False)
    return offsets, values


@jit(nopython=True, cache=True)
def _convolve_ale_kernel_to_masked_cols(
    offsets,
    kernel_values,
    peaks,
    shape,
    mask_flat_to_masked,
    flat_stride_x,
    flat_stride_y,
):
    """Expand sparse ALE kernel support around study peaks and keep in-mask voxels only."""
    max_entries = peaks.shape[0] * offsets.shape[0]
    cols = np.empty(max_entries, dtype=np.int32)
    vals = np.empty(max_entries, dtype=kernel_values.dtype)
    n_entries = 0

    for peak_idx in range(peaks.shape[0]):
        peak = peaks[peak_idx]
        for kernel_idx in range(offsets.shape[0]):
            x = offsets[kernel_idx, 0] + peak[0]
            y = offsets[kernel_idx, 1] + peak[1]
            z = offsets[kernel_idx, 2] + peak[2]
            if (
                (x >= 0)
                and (y >= 0)
                and (z >= 0)
                and (x < shape[0])
                and (y < shape[1])
                and (z < shape[2])
            ):
                flat_idx = x * flat_stride_x + y * flat_stride_y + z
                masked_col = mask_flat_to_masked[flat_idx]
                if masked_col >= 0:
                    cols[n_entries] = masked_col
                    vals[n_entries] = kernel_values[kernel_idx]
                    n_entries += 1

    return cols[:n_entries], vals[:n_entries]


def compute_ale_ma(
    mask,
    ijks,
    kernel=None,
    exp_idx=None,
    sample_sizes=None,
    use_dict=False,
):
    """Generate masked ALE MA maps directly as a study-by-voxel CSR matrix.

    Returns
    -------
    kernel_data : :class:`scipy.sparse.csr_matrix`
        Study-by-masked-voxel CSR matrix of ALE MA values.
    max_ma_values : :class:`numpy.ndarray`
        Row-wise maxima for each study MA map.
    mask_flat_to_masked : :class:`numpy.ndarray`
        Lookup array mapping flattened full-volume voxel indices to masked voxel indices.
    """
    if use_dict:
        if kernel is not None:
            warnings.warn("The kernel provided will be replace by an empty dictionary.")
        kernel_supports = {}
        if not isinstance(sample_sizes, np.ndarray):
            raise ValueError("To use a kernel dictionary sample_sizes must be a list.")
    elif sample_sizes is not None:
        if not isinstance(sample_sizes, int):
            raise ValueError("If use_dict is False, sample_sizes provided must be integer.")
        _, kernel = get_ale_kernel(mask, sample_size=sample_sizes)
        kernel_support = _kernel_to_sparse_support(kernel)
    else:
        if kernel is None:
            raise ValueError("3D array of smoothing kernel must be provided.")
        kernel_support = _kernel_to_sparse_support(kernel)

    if exp_idx is None:
        exp_idx = np.ones(len(ijks), dtype=np.int32)

    ijks = ijks.astype(np.int32, copy=False)
    shape = np.array(mask.shape, dtype=np.int32)
    flat_stride_y = shape[2]
    flat_stride_x = shape[1] * shape[2]
    mask_flat_to_masked = _get_mask_flat_to_masked(mask)

    exp_idx_uniq, exp_idx = np.unique(exp_idx, return_inverse=True)
    n_studies = len(exp_idx_uniq)
    n_voxels = int(mask_flat_to_masked.max()) + 1 if mask_flat_to_masked.size else 0

    indptr = [0]
    indices_parts = []
    data_parts = []
    max_ma_values = np.zeros(n_studies, dtype=DEFAULT_FLOAT_DTYPE)

    for i_exp, _ in enumerate(exp_idx_uniq):
        curr_exp_idx = exp_idx == i_exp
        study_ijks = ijks[curr_exp_idx]

        if use_dict:
            sample_size = sample_sizes[curr_exp_idx][0]
            if sample_size not in kernel_supports:
                _, kernel = get_ale_kernel(mask, sample_size=sample_size)
                kernel_supports[sample_size] = _kernel_to_sparse_support(kernel)
            offsets, kernel_values = kernel_supports[sample_size]
        else:
            offsets, kernel_values = kernel_support

        cols, vals = _convolve_ale_kernel_to_masked_cols(
            offsets,
            kernel_values,
            study_ijks,
            shape,
            mask_flat_to_masked,
            flat_stride_x,
            flat_stride_y,
        )

        if cols.size:
            order = np.argsort(cols, kind="mergesort")
            cols = cols[order]
            vals = vals[order]
            starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
            cols = cols[starts]
            vals = np.maximum.reduceat(vals, starts).astype(DEFAULT_FLOAT_DTYPE, copy=False)
            indices_parts.append(cols)
            data_parts.append(vals)
            indptr.append(indptr[-1] + cols.shape[0])
            max_ma_values[i_exp] = vals.max()
        else:
            indptr.append(indptr[-1])

    indices = (
        np.concatenate(indices_parts).astype(np.int32, copy=False)
        if indices_parts
        else np.array([], dtype=np.int32)
    )
    data = (
        np.concatenate(data_parts).astype(DEFAULT_FLOAT_DTYPE, copy=False)
        if data_parts
        else np.array([], dtype=DEFAULT_FLOAT_DTYPE)
    )
    indptr = np.array(indptr, dtype=np.int64)

    kernel_data = sp_sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_studies, n_voxels),
        dtype=DEFAULT_FLOAT_DTYPE,
    )
    kernel_data.sort_indices()

    return kernel_data, max_ma_values, mask_flat_to_masked


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


@jit(nopython=True, cache=True)
def _apply_liberal_mask(data):
    """Separate input image data in bags of voxels that have a valid value across the same studies.

    Parameters
    ----------
    data : (S x V) :class:`numpy.ndarray`
        2D numpy array (S x V) of images, where S is study and V is voxel.

    Returns
    -------
    values_lst : :obj:`list` of :obj:`numpy.ndarray`
        List of 2D numpy arrays (s x v) of images, where the voxel v have a valid
        value in study s.
    voxel_mask_lst : :obj:`list` of :obj:`numpy.ndarray`
        List of 1D numpy arrays (v) of voxel indices for the corresponding bag.
    study_mask_lst : :obj:`list` of :obj:`numpy.ndarray`
        List of 1D numpy arrays (s) of study indices for the corresponding bag.

    Notes
    -----
    Parts of the function are implemented with nested for loops to
    improve the speed with the numba compiler.

    """
    MIN_STUDY_THRESH = 2

    n_voxels = data.shape[1]
    # Get indices of non-nan and zero value of studies for each voxel
    mask = ~np.isnan(data) & (data != 0)
    study_by_voxels_idxs = [np.where(mask[:, i])[0] for i in range(n_voxels)]

    # Group studies by the same number of non-nan voxels
    matches = []
    all_indices = []
    for col_i in range(n_voxels):
        if col_i in all_indices:
            continue

        vox_match = [col_i]
        all_indices.append(col_i)
        for col_j in range(col_i + 1, n_voxels):
            if (
                len(study_by_voxels_idxs[col_i]) == len(study_by_voxels_idxs[col_j])
                and np.array_equal(study_by_voxels_idxs[col_i], study_by_voxels_idxs[col_j])
                and col_j not in all_indices
            ):
                vox_match.append(col_j)
                all_indices.append(col_j)

        matches.append(np.array(vox_match))

    values_lst, voxel_mask_lst, study_mask_lst = [], [], []
    for voxel_mask in matches:
        n_masked_voxels = len(voxel_mask)
        # This is the same for all voxels in the match
        study_mask = study_by_voxels_idxs[voxel_mask[0]]

        if len(study_mask) < MIN_STUDY_THRESH:
            # TODO: Figure out how raise a warning in numba
            # warnings.warn(
            #     f"Removing voxels: {voxel_mask} from the analysis. Not present in 2+ studies."
            # )
            continue

        values = np.zeros((len(study_mask), n_masked_voxels))
        for vox_i, vox in enumerate(voxel_mask):
            for std_i, study in enumerate(study_mask):
                values[std_i, vox_i] = data[study, vox]

        values_lst.append(values)
        voxel_mask_lst.append(voxel_mask)
        study_mask_lst.append(study_mask)

    return values_lst, voxel_mask_lst, study_mask_lst
