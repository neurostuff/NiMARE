"""Spatial B-spline basis construction for CBMR.

The spatial half of the model: a tensor product of cubic B-spline bases over the x, y and z
voxel indices, evaluated at every in-mask voxel. Moved here from ``nimare.utils``, which had
no other CBMR-specific code and only one caller for these.
"""

import numpy as np

SPLINE_SUPPORT_THRESHOLD = 0.1
"""Minimum peak value for a tensor-product B-spline basis to count as supported."""


def coef_spline_bases(axis_coords, spacing, margin):
    """
    Coefficient of cubic B-spline bases in any x/y/z direction.

    Parameters
    ----------
    axis_coords : value range in x/y/z direction
    spacing: (equally spaced) knots spacing in x/y/z direction,
    margin: extend the region where B-splines are constructed (min-margin, max_margin)
            to avoid weakly-supported B-spline on the edge
    Returns
    -------
    coef_spline : 2-D ndarray (n_points x n_spline_bases)
    """
    import patsy

    # create B-spline basis for x/y/z coordinate
    wider_axis_coords = np.arange(np.min(axis_coords) - margin, np.max(axis_coords) + margin)
    knots = np.arange(  # noqa: F841
        np.min(axis_coords) - margin, np.max(axis_coords) + margin, step=spacing
    )
    design_matrix = patsy.dmatrix(
        "bs(x, knots=knots, degree=3,include_intercept=False)",
        data={"x": wider_axis_coords},
        return_type="matrix",
    )
    design_array = np.array(design_matrix)[:, 1:]  # remove the first column (every element is 1)
    coef_spline = design_array[margin : -margin + 1, :]
    # remove the basis with no/weakly support from the square
    supported_basis = np.sum(coef_spline, axis=0) != 0
    coef_spline = coef_spline[:, supported_basis]

    return coef_spline


def b_spline_bases(masker_voxels, spacing, margin=10):
    """Cubic B-spline bases for spatial intensity.

    The whole coefficient matrix is constructed by taking tensor product of
    all B-spline bases coefficient matrix in three direction.

    Parameters
    ----------
    masker_voxels : :obj:`numpy.ndarray`
        matrix with element either 0 or 1, indicating if it's within brain mask,
    spacing : :obj:`int`
        (equally spaced) knots spacing in x/y/z direction,
    margin : :obj:`int`
        extend the region where B-splines are constructed (min-margin, max_margin)
        to avoid weakly-supported B-spline on the edge
    Returns
    -------
    X : :obj:`numpy.ndarray`
        2-D ndarray (n_voxel x n_spline_bases) only keeps with within-brain voxels
    """
    mask = np.asanyarray(masker_voxels).astype(bool, copy=False)

    # remove the blank space around the brain mask
    xx = np.where(mask.sum(axis=(1, 2)) > 0)[0]
    yy = np.where(mask.sum(axis=(0, 2)) > 0)[0]
    zz = np.where(mask.sum(axis=(0, 1)) > 0)[0]

    x_spline = coef_spline_bases(xx, spacing, margin)
    y_spline = coef_spline_bases(yy, spacing, margin)
    z_spline = coef_spline_bases(zz, spacing, margin)

    cropped_mask = mask[
        np.min(xx) : np.max(xx) + 1,
        np.min(yy) : np.max(yy) + 1,
        np.min(zz) : np.max(zz) + 1,
    ]
    brain_coords = np.argwhere(cropped_mask)

    # Build tensor-product spline rows only for in-mask voxels.
    x_rows = x_spline[brain_coords[:, 0]]
    y_rows = y_spline[brain_coords[:, 1]]
    z_rows = z_spline[brain_coords[:, 2]]

    # Most tensor-product bases are centered outside the brain and carry no support in it,
    # so building the whole product before discarding them wastes about two thirds of the
    # peak memory -- 13.1 GB against the 4.8 GB kept, at spacing=5 on the 2 mm mask. Instead
    # walk the product one (i, j) plane at a time.
    #
    # B-spline bases are non-negative, so max_v(x_i y_j z_k) <= max_v(x_i y_j) * max(z_k):
    # once an (i, j) plane falls under the support threshold no z can lift it back over, and
    # the plane is skipped without ever being formed. Iterating i then j then k reproduces
    # the C-order columns of the full product, so the output is identical to filtering it.
    z_max = z_rows.max() if z_rows.size else 0.0
    n_voxels = brain_coords.shape[0]

    def _supported_columns(i, j):
        """Return the plane for basis pair (i, j) and its support mask, or None if empty."""
        xy_row = x_rows[:, i] * y_rows[:, j]
        if xy_row.max() * z_max < SPLINE_SUPPORT_THRESHOLD:
            return None, None
        plane = xy_row[:, None] * z_rows
        return plane, plane.max(axis=0) >= SPLINE_SUPPORT_THRESHOLD

    # First pass counts the survivors so the output can be allocated exactly once; holding
    # the planes instead would put the discarded columns back into peak memory.
    masks = {}
    n_kept = 0
    for i in range(x_rows.shape[1]):
        for j in range(y_rows.shape[1]):
            _, mask = _supported_columns(i, j)
            if mask is not None and mask.any():
                masks[(i, j)] = mask
                n_kept += int(mask.sum())

    X = np.empty((n_voxels, n_kept), dtype=x_rows.dtype)
    offset = 0
    for (i, j), mask in masks.items():
        plane, _ = _supported_columns(i, j)
        width = int(mask.sum())
        X[:, offset : offset + width] = plane[:, mask]
        offset += width
    return X
