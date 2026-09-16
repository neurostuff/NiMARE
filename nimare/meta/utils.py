"""Utilities for coordinate-based meta-analysis estimators."""

import logging
import warnings

import numpy as np
import sparse
from numba import jit
from scipy import ndimage
from scipy import sparse as sp_sparse
from scipy import stats

from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _mask_img_to_bool,
    _nlogp_to_logp_values,
    unique_rows,
)

# based on local benchmarks, tested 20, 30, 40, 50, 100, 200 studies
# sorting provides speed benefits starting betwee 30 and 40 studies
KDA_SORT_MIN_STUDIES = 40
# occupancy-mask vs. unique-rows crossover observed around ~50 foci/study
KDA_OCCUPANCY_MIN_FOCI = 50
LGR = logging.getLogger(__name__)

#: Share of the maximum-statistic null taken as exceedances for the generalized Pareto fit,
#: and the fraction of them dropped on each retry when the fit is rejected.
_GPD_TAIL_FRACTION = 0.10
_GPD_SHRINK_DIVISOR = 20
#: The fit is trusted only this many times above the empirical p floor. Below that it was
#: measured to run about twice anticonservative, so the empirical tail is kept instead.
_GPD_FLOOR_MULTIPLE = 5.0
#: Clamps before a logarithm and before inverting the normal survival function. Only guard
#: against a p of exactly zero; both sit far below any p a permutation count can produce.
_LOGP_FLOOR = 1e-300
_Z_FROM_P_FLOOR = 1e-16

_EPS = float(np.finfo(np.float64).tiny)


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


def sphere_kernel_offsets(r, vox_dims, n_dim=3):
    """Voxel offsets of a sphere of radius ``r`` mm, one row per voxel.

    The observed-statistic path and the fused permutation path must agree
    exactly on which voxels a focus reaches, so both take the sphere from here
    rather than each rebuilding it. Returns ``(n_voxels, n_dim)`` int32 offsets.
    """
    vox_dims = np.asarray(vox_dims, dtype=np.float64)
    slices = [slice(-r // vox_dims[i], r // vox_dims[i] + 0.01, 1) for i in range(n_dim)]
    cube = np.vstack([row.ravel() for row in np.mgrid[tuple(slices)].astype(np.int32)])
    inside = np.sum(np.dot(np.diag(vox_dims), cube) ** 2, axis=0) ** 0.5 <= r
    return np.ascontiguousarray(cube[:, inside].T, dtype=np.int32)


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

    kernel = sphere_kernel_offsets(r, vox_dims, n_dim=ijks.shape[1]).T

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


def _padded_flat_to_masked(mask_img, offsets):
    """:func:`_get_mask_flat_to_masked` on a grid padded wide enough for ``offsets``.

    Dilating a set of foci by a sphere means adding each sphere offset to each focus's voxel
    index. On an unpadded grid that arithmetic has to be bounds-checked per axis, which means
    materializing the candidate *coordinates* -- an ``(n_foci, n_sphere_voxels, 3)`` array that
    exists only to be compared against the shape -- before any of them can be flattened. Pad
    the grid instead and every candidate flat index is inside it by construction, with the
    padding reading back as -1, so the bounds check collapses into the ``>= 0`` test the mask
    lookup already needs and the coordinates never have to exist.

    Padding is twice the largest offset on each axis, not once: a focus is kept when it is
    within one offset of the image, so its own padded index can sit one offset outside the
    unpadded block and a sphere offset can then push it one further. A focus beyond that cannot
    reach an in-mask voxel at all, which is why dropping it is exact rather than a tolerance.

    Returns the lookup, the padded shape, and the per-axis padding.
    """
    mask = _mask_img_to_bool(mask_img)
    shape = np.asarray(mask_img.shape[:3], dtype=np.int64)
    pad = 2 * np.abs(np.asarray(offsets, dtype=np.int64)).max(axis=0)
    padded_shape = shape + 2 * pad
    lookup = np.full(int(np.prod(padded_shape)), -1, dtype=np.int32)
    inner = lookup.reshape(padded_shape)[
        pad[0] : pad[0] + shape[0], pad[1] : pad[1] + shape[1], pad[2] : pad[2] + shape[2]
    ]
    inner[mask.reshape(shape)] = np.arange(int(mask.sum()), dtype=np.int32)
    return lookup, padded_shape, pad


def _coo_to_masked_csr(ma_values, mask_img, mask_flat_to_masked=None):
    """Convert legacy COO ALE MA maps to a study-by-voxel CSR matrix within the mask."""
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

    mass_values = np.abs(arr3d) - threshold

    def _max_cluster_stats(mask):
        labeled_arr3d, n_clusters = ndimage.label(mask, conn)
        if not n_clusters:
            return 0, 0.0

        cluster_ids = np.arange(1, n_clusters + 1)
        cluster_sizes = np.bincount(labeled_arr3d.ravel())[1:]
        cluster_masses = np.asarray(
            ndimage.sum(mass_values, labels=labeled_arr3d, index=cluster_ids)
        )
        return np.max(cluster_sizes), np.max(cluster_masses)

    max_size, max_mass = _max_cluster_stats(arr3d > 0)

    if tail == "two":
        neg_max_size, neg_max_mass = _max_cluster_stats(arr3d < 0)
        max_size = max(max_size, neg_max_size)
        max_mass = max(max_mass, neg_max_mass)

    return max_size, max_mass


def _usable(data):
    """Return which entries of an image array are usable statistics.

    isfinite rather than ~isnan: an infinite value is not a usable statistic either, and it
    survives an isnan check. Input maps do carry them -- a t map divided by a zero standard
    error, say -- and one would otherwise reach PyMARE and turn a whole model's output into
    NaN. Zero is the placeholder a NeuroVault map carries where it has no coverage.
    """
    return np.isfinite(data) & (data != 0)


def _liberal_mask_bags(mask):
    """Group voxels by which studies cover them.

    Parameters
    ----------
    mask : (S x V) :class:`numpy.ndarray` of :obj:`bool`
        Which entries of the image data are usable.

    Returns
    -------
    :obj:`list` of :obj:`tuple`
        One ``(voxel_mask, study_mask)`` pair per bag, in order of first appearance. Bags
        covered by fewer than two studies are dropped, since they cannot be fitted.

    Notes
    -----
    Split out from :func:`_apply_liberal_mask` because an estimator with several image
    inputs cuts them all along one shared coverage pattern, so the grouping is worked out
    once and every input is then sliced with it.

    Each voxel's pattern is packed into bytes and handed to :func:`numpy.unique`, so the
    grouping costs one sort rather than a quadratic pairwise comparison.
    """
    # Necessary condition for :attr:`~nimare.meta._dependence.MIN_INDEPENDENT_UNITS`, not the
    # same count: this is studies, and a bag meeting it can still hold one group.
    MIN_STUDY_THRESH = 2

    # Pack each voxel's column of S booleans into ceil(S / 8) bytes, so that a whole pattern
    # is a single row np.unique can sort on. Padding bits are zero for every voxel alike, so
    # they cannot merge two distinct patterns.
    keys = np.ascontiguousarray(np.packbits(mask, axis=0).T)
    _, first_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    # Older numpy returned a column vector here; newer versions return 1D.
    inverse = np.reshape(inverse, -1)

    # np.unique orders groups lexicographically by packed pattern. Reorder to first
    # appearance so the bags come back in the same order the voxels do.
    by_appearance = np.argsort(first_idx, kind="stable")
    appearance_rank = np.empty(first_idx.size, dtype=np.intp)
    appearance_rank[by_appearance] = np.arange(first_idx.size)

    # Sorting voxels by group puts each bag's voxel indices in one contiguous, ascending run.
    voxels_by_group = np.argsort(appearance_rank[inverse], kind="stable")
    group_sizes = np.bincount(inverse, minlength=first_idx.size)[by_appearance]
    group_bounds = np.concatenate(([0], np.cumsum(group_sizes)))

    bags = []
    for group in range(first_idx.size):
        voxel_mask = voxels_by_group[group_bounds[group] : group_bounds[group + 1]]
        # Identical by construction for every voxel in the group.
        study_mask = np.flatnonzero(mask[:, voxel_mask[0]])

        if study_mask.size >= MIN_STUDY_THRESH:
            bags.append((voxel_mask, study_mask))

    return bags


def _liberal_mask_values(data, bags):
    """Slice one image input into the bags of :func:`_liberal_mask_bags`."""
    return [
        data[np.ix_(study_mask, voxel_mask)].astype(np.float64, copy=False)
        for voxel_mask, study_mask in bags
    ]


def _apply_liberal_mask(data, validity=None):
    """Separate input image data in bags of voxels that have a valid value across the same studies.

    Parameters
    ----------
    data : (S x V) :class:`numpy.ndarray`
        2D numpy array (S x V) of images, where S is study and V is voxel.
    validity : None or (S x V) :class:`numpy.ndarray` of :obj:`bool`, optional
        Which entries of ``data`` are usable. Default is :func:`_usable` of ``data``.

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
    Bags are returned in order of first appearance, i.e. sorted by their lowest voxel index.

    An estimator with several image inputs should call :func:`_liberal_mask_bags` and
    :func:`_liberal_mask_values` instead, so that the grouping is worked out once for all of
    them rather than repeated per input.

    """
    mask = _usable(data) if validity is None else np.asarray(validity)
    bags = _liberal_mask_bags(mask)

    return (
        _liberal_mask_values(data, bags),
        [voxel_mask for voxel_mask, _ in bags],
        [study_mask for _, study_mask in bags],
    )


def _gpd_goodness_of_fit(excess, shape, scale, n_boot=200, seed=0):
    """p-value for "these exceedances are generalized Pareto", by parametric bootstrap.

    The parameters were estimated from the same data, so the textbook Cramer-von Mises null
    distribution does not apply -- using it accepts fits it should reject, which is how a tail
    approximation ends up anticonservative. Simulating from the fitted distribution and
    refitting each replicate gives the right reference.
    """
    rng = np.random.default_rng(seed)
    observed_stat = stats.cramervonmises(
        excess, stats.genpareto(shape, loc=0.0, scale=scale).cdf
    ).statistic
    n = excess.size

    worse = 0
    for _ in range(n_boot):
        sample = stats.genpareto.rvs(shape, loc=0.0, scale=scale, size=n, random_state=rng)
        try:
            boot_shape, _, boot_scale = stats.genpareto.fit(sample, floc=0.0)
            if not np.isfinite(boot_shape) or boot_scale <= 0:
                continue
            statistic = stats.cramervonmises(
                sample, stats.genpareto(boot_shape, loc=0.0, scale=boot_scale).cdf
            ).statistic
        except Exception:  # noqa: BLE001
            continue
        worse += statistic >= observed_stat
    return (1 + worse) / (1 + n_boot)


def _gpd_tail_p(observed, null_maxima, min_exceedances=30, alpha=0.05):
    """Corrected p-values from a generalized Pareto fit to the tail of the null maxima.

    A permutation p-value cannot go below ``1 / (1 + n_iters)``, so resolving a corrected p of
    1e-4 needs ten thousand permutations however uninteresting the other 9999 are. Extreme value
    theory says the exceedances of a high threshold converge to a generalized Pareto
    distribution whatever the parent, so the tail can be *modelled* rather than counted. This
    is the tail approximation of Winkler et al. (2016), which they recommend specifically for
    familywise error.

    Validated here rather than taken on faith, and the result bounds what it may do. Above five
    times the empirical floor the fitted p is 0.85-1.00 of a 40000-permutation truth; at and
    below the floor it runs about twice anticonservative in 36-60% of runs, on every parent
    distribution tried. So it refines p-values only in the range where it was shown to work and
    defers to the empirical tail below -- which gives up the extrapolation past the floor that
    the published method is prized for. With a few hundred exceedances this implementation did
    not earn it.

    The threshold is chosen the way they choose it: start with the largest tenth of the null
    maxima, test the fit, and if it is rejected drop the smallest exceedance and refit, until
    the fit is acceptable or too few points remain. Falling back to the empirical tail when no
    fit is accepted is what keeps this safe -- it can only ever refine a p-value it would
    otherwise have quantized, never invent one on a tail that is not Pareto.

    Returns ``None`` when no acceptable fit exists, leaving the caller on the empirical tail.
    """
    maxima = np.sort(np.asarray(null_maxima, dtype=float))
    n_total = maxima.size
    if n_total < 100:
        return None  # too few to say anything about a tail

    n_exceed = max(int(round(_GPD_TAIL_FRACTION * n_total)), min_exceedances)
    while n_exceed >= min_exceedances:
        threshold = maxima[n_total - n_exceed - 1]
        excess = maxima[n_total - n_exceed :] - threshold
        if not np.all(np.isfinite(excess)) or excess.max() <= 0:
            n_exceed -= max(1, n_exceed // _GPD_SHRINK_DIVISOR)
            continue
        try:
            shape, _, scale = stats.genpareto.fit(excess, floc=0.0)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                raise ValueError
            fitted = stats.genpareto(shape, loc=0.0, scale=scale)
            goodness = _gpd_goodness_of_fit(excess, shape, scale, seed=n_exceed)
        except Exception:  # noqa: BLE001 -- any failure just means try a shorter tail
            n_exceed -= max(1, n_exceed // _GPD_SHRINK_DIVISOR)
            continue

        if goodness > alpha:
            rate = n_exceed / n_total
            observed = np.asarray(observed, dtype=float)
            in_tail = observed > threshold
            p_corrected = np.empty(observed.shape, dtype=float)
            # Below the threshold the empirical tail is well resolved, so keep it there.
            #
            # Deliberately not :func:`nimare.stats.null_to_p`: that returns ``1 - idx / n``
            # clamped into ``[1/n, 1 - 1/n]``, where this is the ``(1 + exceedances) / (1 + n)``
            # randomization estimator. The difference matters for a maximum-statistic null,
            # where the smallest attainable p is the whole point -- the clamped form can report
            # ``1/n`` for a statistic no permutation reached, which is anticonservative.
            empirical = (1 + np.sum(maxima[None, :] >= observed[:, None], axis=1)) / (1 + n_total)
            p_corrected[~in_tail] = empirical[~in_tail]
            p_corrected[in_tail] = rate * fitted.sf(observed[in_tail] - threshold)
            # How far this can be trusted was measured, not assumed: against a
            # 40000-permutation reference, across three parent distributions, 25 repetitions
            # each. Comfortably above the empirical floor it is accurate -- at p = .05 and .01
            # the fitted value is 0.85-1.00 of the truth and essentially never more than twice
            # too small. At and below the floor (1/501 for a 500-permutation run) it runs about
            # twice anticonservative in 36-60% of runs, on every parent tried.
            #
            # So the fit is used only where it was shown to work, five times the floor and
            # above, and the empirical tail is kept below that. That forgoes the extrapolation
            # past the floor which is the published method's main selling point; with a few
            # hundred exceedances this implementation did not earn it, and an anticonservative
            # familywise p is worse than a quantized one.
            # Fall back to the empirical p there, rather than clamping the fitted one up to the
            # floor. Clamping was the bug: for a statistic beyond every null maximum the fit
            # returns something tiny, gets raised to 5/(1 + n), and the caller is handed
            # 0.009980 where the empirical value is 0.001996 -- five times too conservative, and
            # worse than not fitting a tail at all. The intent was always to prefer the
            # empirical tail where the fit is not trusted; this does that.
            floor = _GPD_FLOOR_MULTIPLE / (1.0 + n_total)
            untrusted = in_tail & (p_corrected < floor)
            p_corrected[untrusted] = empirical[untrusted]
            return np.clip(p_corrected, 0.0, 1.0)
        n_exceed -= max(1, n_exceed // _GPD_SHRINK_DIVISOR)

    return None


def _max_statistic_maps(observed, null_maxima, sign, tail_approximation=False):
    """Corrected ``-log10(p)`` and signed z for a statistic against its maximum-statistic null."""
    p_corrected = None
    if tail_approximation:
        p_corrected = _gpd_tail_p(observed, null_maxima)
    if p_corrected is None:
        p_corrected = (1 + np.sum(null_maxima[None, :] >= observed[:, None], axis=1)) / (
            1 + len(null_maxima)
        )
    logp = _nlogp_to_logp_values(np.log(np.clip(p_corrected, _LOGP_FLOOR, None)))
    z_corrected = stats.norm.isf(np.clip(p_corrected, _Z_FROM_P_FLOOR, 1.0) / 2.0) * sign
    return (
        logp.astype(DEFAULT_FLOAT_DTYPE),
        z_corrected.astype(DEFAULT_FLOAT_DTYPE),
    )
