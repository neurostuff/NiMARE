"""The peak representation a kernel transformer consumes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from scipy import sparse


def grid_shape(mask_img):
    """Return the image grid the peak columns span."""
    return tuple(int(size) for size in mask_img.shape)


def n_grid_columns(mask_img):
    """Return how many columns a peak block over ``mask_img``'s grid has."""
    return int(np.prod(grid_shape(mask_img)))


def peak_matrix(studyset, mask_img):
    """Return the analysis-by-grid-voxel matrix of peak counts.

    Rows follow ``studyset.ids``; an analysis with no usable foci is an all-zero
    row. A focus falling outside the image grid is dropped, which is what a
    kernel transformer does with it too.
    """
    block = studyset.coordinate_block()
    shape = grid_shape(mask_img)
    ijk = block.ijk(mask_img.affine)
    group = block.group_of_point()

    within = (ijk >= 0).all(axis=1) & (ijk < np.asarray(shape)).all(axis=1)
    columns = np.ravel_multi_index(ijk[within].T, shape) if within.any() else np.empty(0, int)

    return sparse.coo_matrix(
        (np.ones(int(within.sum()), dtype=np.float32), (group[within], columns)),
        shape=(block.n_groups, n_grid_columns(mask_img)),
    ).tocsr()


def peak_frame(X, shape):
    """Return the ``i``/``j``/``k``/``id`` table a kernel transformer reads.

    ``id`` is the row position in ``X``, so a kernel's maps come back in order.
    """
    X = sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()
    rows, columns = X.nonzero()
    counts = np.asarray(X[rows, columns]).ravel()
    counts = np.rint(counts).astype(int)
    keep = counts > 0
    rows, columns, counts = rows[keep], columns[keep], counts[keep]

    ijk = np.column_stack(np.unravel_index(np.repeat(columns, counts), shape))
    frame = pd.DataFrame(ijk, columns=["i", "j", "k"])
    frame["id"] = np.repeat(rows, counts)
    return frame


def grid_images(batch, mask_img):
    """Return one image per row of a dense grid-space ``batch``."""
    shape = grid_shape(mask_img)
    volume = np.asarray(batch, dtype=float).reshape((-1, *shape))
    return Nifti1Image(np.moveaxis(volume, 0, -1), mask_img.affine)
