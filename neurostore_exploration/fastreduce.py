"""A fast equivalent of MaskerTransformer for a fixed atlas.

An atlas reduction is a fixed linear map from voxels to regions, so it can be
applied as one sparse matrix product instead of unmasking every row into an
image. Validated against :class:`~nimare.ml.MaskerTransformer` in
``12_validate.py``.
"""
from __future__ import annotations

import numpy as np
from nilearn.masking import apply_mask
from scipy import sparse

from nimare.ml import MaskerTransformer


def atlas_operator(atlas, source_masker, reference, masker_kwargs=None):
    """Return ``(operator, names)`` mapping MA-map voxels to atlas regions.

    ``operator`` is ``(n_voxels, n_regions)``, so ``maps @ operator`` is what
    :meth:`~nimare.ml.MaskerTransformer.transform` returns.
    """
    fitted = MaskerTransformer(
        atlas, source_masker=source_masker, masker_kwargs=masker_kwargs
    ).fit(reference)
    masker = fitted.masker_

    if hasattr(masker, "maps_img_") or hasattr(masker, "maps_img"):
        # NiftiMapsMasker regresses the maps out of each image, so the forward
        # operator is the pseudo-inverse of the map matrix.
        maps_img = getattr(masker, "maps_img_", None) or masker.maps_img
        M = apply_mask(maps_img, fitted.mask_img_)  # (n_maps, n_voxels)
        operator = np.linalg.pinv(M)  # (n_voxels, n_maps)
    else:
        labels_img = getattr(masker, "labels_img_", None) or masker.labels_img
        labels = apply_mask(labels_img, fitted.mask_img_).astype(int)
        values = np.unique(labels)
        values = values[values != 0]
        rows = np.flatnonzero(labels > 0)
        cols = np.searchsorted(values, labels[rows])
        counts = np.bincount(cols, minlength=values.size).astype(float)
        # NiftiLabelsMasker averages over each region's voxels.
        operator = sparse.csr_matrix(
            (1.0 / counts[cols], (rows, cols)), shape=(labels.size, values.size)
        )

    names = fitted.region_names_
    return operator, names


def reduce_maps(maps, operator):
    """Apply an atlas operator to a sparse stack of MA maps."""
    out = maps @ operator
    return np.asarray(out.todense()) if sparse.issparse(out) else np.asarray(out)
