"""Utilities for coordinate-based meta-analysis estimators."""

import logging
import warnings

import numpy as np
import sparse
from numba import jit
from scipy import ndimage, optimize
from scipy import sparse as sp_sparse

from nimare.utils import DEFAULT_FLOAT_DTYPE, _mask_img_to_bool, unique_rows

# based on local benchmarks, tested 20, 30, 40, 50, 100, 200 studies
# sorting provides speed benefits starting betwee 30 and 40 studies
KDA_SORT_MIN_STUDIES = 40
# occupancy-mask vs. unique-rows crossover observed around ~50 foci/study
KDA_OCCUPANCY_MIN_FOCI = 50
LGR = logging.getLogger(__name__)
_EPS = float(np.finfo(np.float64).tiny)


def _safe_exp(values):
    """Exponentiate after clipping to avoid numerical overflow."""
    return np.exp(np.clip(values, -100, 100))


def _spatial_cbmr_kron_vector_product(moderators, bases, coefficient):
    """Compute ``kron(moderators, bases) @ coefficient`` without forming ``kron``.

    Parameters
    ----------
    moderators : :obj:`numpy.ndarray` of shape ``(n_experiments, n_moderators)``
        Experiment-level design matrix.
    bases : :obj:`numpy.ndarray` of shape ``(n_voxels, n_bases)``
        Spatial B-spline basis matrix.
    coefficient : :obj:`numpy.ndarray` of shape ``(n_moderators * n_bases, 1)``
        Flattened spatially varying coefficient matrix.

    Returns
    -------
    :obj:`numpy.ndarray` of shape ``(n_experiments * n_voxels, 1)``
        Flattened linear predictor.
    """
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    coefficient = np.asarray(coefficient).reshape((n_moderators, n_bases))
    return (moderators @ coefficient @ bases.T).reshape((n_experiments * n_voxels, 1))


def _spatial_cbmr_log_poisson_nll(moderators, bases, coefficient, foci):
    """Return the Poisson negative log-likelihood for the approximate solver."""
    n_experiments = moderators.shape[0]
    n_voxels = bases.shape[0]
    eta = _spatial_cbmr_kron_vector_product(moderators, bases, coefficient).reshape(
        (n_experiments, n_voxels)
    )
    mean = _safe_exp(eta)
    if sp_sparse.issparse(foci):
        foci = foci.toarray()
    return -float(np.mean(foci * eta - mean))


def _spatial_cbmr_gradient(moderators, bases, coefficient, foci):
    """Compute the negative Poisson score for a spatially varying coefficient."""
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    if sp_sparse.issparse(foci):
        foci_csr = foci.tocsr()
        observed_term = (moderators.T @ foci_csr @ bases).reshape((n_moderators * n_bases, 1))
    else:
        observed_term = (moderators.T @ foci @ bases).reshape((n_moderators * n_bases, 1))

    eta = _spatial_cbmr_kron_vector_product(moderators, bases, coefficient).reshape(
        (n_experiments, n_voxels)
    )
    expected = _safe_exp(eta)
    expected_term = (moderators.T @ expected @ bases).reshape((n_moderators * n_bases, 1))
    return -(observed_term - expected_term)


def _fit_spatial_cbmr_additive_log_glm(moderators, bases, foci):
    """Fit an additive log-Poisson approximation used for preconditioning."""
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    if sp_sparse.issparse(foci):
        foci_csr = foci.tocsr()
        foci_by_experiment = np.asarray(foci_csr.mean(axis=1)).ravel()
        foci_by_voxel = np.asarray(foci_csr.mean(axis=0)).ravel()
    else:
        foci_by_experiment = foci.mean(axis=1)
        foci_by_voxel = foci.mean(axis=0)

    def objective(params):
        basis_coef = params[:n_bases]
        moderator_coef = params[n_bases:]
        basis_linear = bases @ basis_coef
        moderator_linear = moderators @ moderator_coef
        log_like = (
            (foci_by_voxel * basis_linear).mean()
            + (foci_by_experiment * moderator_linear).mean()
            - _safe_exp(basis_linear).mean() * _safe_exp(moderator_linear).mean()
        )
        return -log_like

    def gradient(params):
        basis_coef = params[:n_bases]
        moderator_coef = params[n_bases:]
        exp_basis = _safe_exp(bases @ basis_coef)
        exp_moderator = _safe_exp(moderators @ moderator_coef)
        basis_grad = (bases.T @ foci_by_voxel) / n_voxels - (
            bases.T @ exp_basis
        ) / n_voxels * exp_moderator.mean()
        moderator_grad = (moderators.T @ foci_by_experiment) / n_experiments - (
            moderators.T @ exp_moderator
        ) / n_experiments * exp_basis.mean()
        return -np.concatenate([basis_grad, moderator_grad])

    result = optimize.minimize(
        fun=objective,
        jac=gradient,
        x0=np.zeros(n_bases + n_moderators),
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 200},
    )
    return result.x[:n_bases], result.x[n_bases:]


def _compute_spatial_cbmr_preconditioner(moderators, bases, mean_moderator, mean_basis, damping):
    """Build an approximate Kronecker preconditioner for the gradient step."""
    moderator_info = moderators.T @ (moderators * mean_moderator)
    basis_info = bases.T @ (bases * mean_basis)
    moderator_info_inv = np.linalg.pinv(moderator_info + damping * np.eye(moderators.shape[1]))
    basis_info_inv = np.linalg.pinv(basis_info + damping * np.eye(bases.shape[1]))
    return np.kron(moderator_info_inv, basis_info_inv)


def fit_voxelwise_cbmr_approximate(
    moderators,
    bases,
    foci,
    tol=1e-10,
    max_iter=100,
    alpha=1.0,
    damping=1e-4,
    compute_nll=False,
):
    """Fit a spatially varying log-Poisson GLM with a preconditioned gradient step."""
    n_experiments, n_moderators = moderators.shape
    n_voxels, n_bases = bases.shape
    LGR.info(
        "SpatialCBMR approximate model: %d experiments, %d voxels, %d moderators, %d bases.",
        n_experiments,
        n_voxels,
        n_moderators,
        n_bases,
    )
    basis_coef, moderator_coef = _fit_spatial_cbmr_additive_log_glm(moderators, bases, foci)
    mean_moderator = _safe_exp(moderators @ moderator_coef)[:, None]
    mean_basis = _safe_exp(bases @ basis_coef)[:, None]
    preconditioner = _compute_spatial_cbmr_preconditioner(
        moderators,
        bases,
        mean_moderator,
        mean_basis,
        damping=damping,
    )

    coefficient = np.zeros((n_moderators, n_bases), dtype=np.float64)
    coefficient[-1] = basis_coef
    coefficient = coefficient.reshape((n_moderators * n_bases, 1))
    for iteration in range(max_iter):
        gradient = _spatial_cbmr_gradient(moderators, bases, coefficient, foci)
        new_coefficient = coefficient - alpha * (preconditioner @ gradient)
        if not np.isfinite(new_coefficient).all():
            raise FloatingPointError(
                "SpatialCBMR approximate regression produced non-finite coefficients. "
                "Try reducing alpha or increasing damping."
            )
        delta = float(np.linalg.norm(new_coefficient - coefficient))
        relative_delta = delta / max(float(np.linalg.norm(coefficient)), 1.0)
        coefficient = new_coefficient
        if compute_nll:
            nll = _spatial_cbmr_log_poisson_nll(moderators, bases, coefficient, foci)
            LGR.info(
                "Iteration %d: delta=%g, relative_delta=%g, nll=%g",
                iteration,
                delta,
                relative_delta,
                nll,
            )
        else:
            LGR.debug(
                "Iteration %d: delta=%g, relative_delta=%g",
                iteration,
                delta,
                relative_delta,
            )
        if delta < tol or relative_delta < tol:
            LGR.info("SpatialCBMR approximate model converged in %d iterations.", iteration + 1)
            break
    else:
        LGR.warning(
            "SpatialCBMR approximate model did not converge within %d iterations.",
            max_iter,
        )
    return coefficient


fit_spatial_cbmr_approximate = fit_voxelwise_cbmr_approximate


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
        Which entries of ``data`` are usable. Default is those that are neither NaN nor zero.

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
    # isfinite, not ~isnan: an infinite value is not a usable statistic either.
    mask = np.isfinite(data) & (data != 0) if validity is None else np.asarray(validity)
    bags = _liberal_mask_bags(mask)

    return (
        _liberal_mask_values(data, bags),
        [voxel_mask for voxel_mask, _ in bags],
        [study_mask for _, study_mask in bags],
    )
