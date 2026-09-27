"""Running a NiMARE kernel transformer as a scikit-learn step."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from nimare.ml._peaks import grid_shape, n_grid_columns, peak_frame


class MAKernel(TransformerMixin, BaseEstimator):
    """Convolve peak columns into modeled activation maps.

    Columns come in over the whole image grid, as
    :meth:`~nimare.studyset.Studyset.to_bunch` reports them, and go out over the
    source masker's voxels. Being a transformer, the kernel and its bandwidth
    are fitted, cloned and tuned like any other step. See :doc:`the machine
    learning documentation </machine_learning>`.

    Parameters
    ----------
    kernel : :class:`~nimare.meta.kernel.KernelTransformer`, optional
        The kernel to apply, by default None. An instance or a class. There is
        no default: the choice is scientific.
    source_masker : :class:`~nilearn.maskers.NiftiMasker` or img_like, optional
        The masker whose image grid the peak columns span, normally the
        ``masker`` a bundle carries, by default None.

    Attributes
    ----------
    kernel_ : :class:`~nimare.meta.kernel.KernelTransformer`
        The kernel instance doing the work.
    mask_img_ : :class:`~nibabel.nifti1.Nifti1Image`
        The mask defining the incoming grid and the outgoing voxels.
    n_voxels_ : :obj:`int`
        How many voxel columns the maps have.

    See Also
    --------
    nimare.studyset.Studyset.to_bunch : Where the peak columns come from.

    Examples
    --------
    >>> step = MAKernel(MKDAKernel(r=10), source_masker=bunch.masker)  # doctest: +SKIP
    """

    def __init__(self, kernel=None, source_masker=None):
        self.kernel = kernel
        self.source_masker = source_masker

    def fit(self, X, y=None):
        """Resolve the kernel and the space its peaks are in.

        Parameters
        ----------
        X : array_like or sparse matrix
            Analysis-by-grid-voxel peak counts, used only for their width.
        y : ignored

        Returns
        -------
        :class:`MAKernel`
            The fitted transformer.

        Raises
        ------
        :obj:`ValueError`
            If no kernel or no source masker was given, if the columns do not
            span the masker's grid, or if the kernel derives its width from
            per-analysis sample sizes, which a feature matrix cannot carry.
        """
        if self.kernel is None:
            raise ValueError(
                "MAKernel requires a kernel transformer, such as MKDAKernel(r=10). "
                "There is no default: the choice is scientific."
            )
        if self.source_masker is None:
            raise ValueError(
                "MAKernel requires the source_masker whose grid the peak columns span, "
                "normally the masker a bundle carries."
            )

        from nimare.utils import get_masker

        kernel = self.kernel() if isinstance(self.kernel, type) else self.kernel
        _check_kernel_width(kernel)

        self.kernel_ = kernel
        self.masker_ = get_masker(self.source_masker)
        self.mask_img_ = self.masker_.mask_img
        self.n_voxels_ = int(np.sum(np.asarray(self.mask_img_.dataobj) > 0))
        expected = n_grid_columns(self.mask_img_)
        if X.shape[1] != expected:
            raise ValueError(
                f"MAKernel was given {X.shape[1]} columns, but the grid of its "
                f"source_masker has {expected}. Peak columns span the whole image grid, "
                "so a kernel reaches voxels that a coordinate outside the mask would "
                "otherwise lose. Pass the bundle's own masker as source_masker, and the "
                "peak columns as they came."
            )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Convolve each row's peaks into its MA map.

        Parameters
        ----------
        X : array_like or sparse matrix
            Analysis-by-grid-voxel peak counts.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Analysis-by-voxel MA features, in the source masker's voxel order.
        """
        check_is_fitted(self, ["kernel_"])

        frame = peak_frame(X, grid_shape(self.mask_img_))
        if frame.empty:
            return sparse.csr_matrix((X.shape[0], self.n_voxels_), dtype=float)

        maps = self.kernel_.transform(frame, masker=self.masker_, return_type="sparse")
        maps = maps if sparse.issparse(maps) else sparse.csr_matrix(maps)
        return _scatter_rows(maps.tocsr(), np.unique(frame["id"].to_numpy()), X.shape[0])

    def get_feature_names_out(self, input_features=None):
        """Return one name per voxel of the maps.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        :obj:`numpy.ndarray` of :obj:`str`
            One name per output voxel column.
        """
        check_is_fitted(self, ["kernel_"])
        return np.asarray([f"voxel_{index}" for index in range(self.n_voxels_)], dtype=str)


def _check_kernel_width(kernel):
    """Refuse a kernel whose width comes from per-analysis sample sizes."""
    derives_width = (
        hasattr(kernel, "sample_size")
        and getattr(kernel, "sample_size", None) is None
        and getattr(kernel, "fwhm", None) is None
    )
    if derives_width:
        raise ValueError(
            f"{type(kernel).__name__} with neither 'fwhm' nor 'sample_size' derives a "
            "separate kernel width for every analysis from its sample size, which a "
            "feature matrix cannot carry: a transformer receives a slice of rows "
            "without being told which rows they are. Pass a width that holds across "
            "analyses, such as ALEKernel(fwhm=10) or ALEKernel(sample_size=20)."
        )


def _scatter_rows(maps, rows, n_rows):
    """Return ``maps`` at ``rows`` of an otherwise all-zero ``n_rows`` matrix."""
    if maps.shape[0] != len(rows):
        raise ValueError(
            f"The kernel returned {maps.shape[0]} maps for {len(rows)} analyses with "
            "peaks. MA features cannot be aligned to analyses."
        )
    if maps.shape[0] == n_rows:
        return maps

    scattered = sparse.csr_matrix((n_rows, maps.shape[1]), dtype=maps.dtype)
    scattered = scattered.tolil()
    scattered[rows] = maps
    return scattered.tocsr()
