"""Apply a nilearn masker or atlas to the voxel columns of a bunch."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.spatialimages import SpatialImage
from nilearn.image import load_img
from nilearn.maskers import BaseMasker, NiftiLabelsMasker, NiftiMapsMasker
from nilearn.masking import unmask
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from nimare.ml._peaks import grid_images, mask_source, n_grid_columns


class MaskerTransformer(TransformerMixin, BaseEstimator):
    """Apply a nilearn masker to voxel features.

    A nilearn masker takes images, where a
    :class:`~sklearn.compose.ColumnTransformer` hands out columns of an array.
    This is the bridge: rows are turned back into images in the source mask's
    space, in batches, and handed to the masker, so region definitions,
    smoothing, resampling and aggregation strategy all stay nilearn's.

    An atlas reduces the voxels to regions; a
    :class:`~nilearn.maskers.NiftiMasker` gives voxels back, which is how
    nilearn's smoothing, standardizing and detrending reach these features. How
    many regions an atlas yields depends on the nilearn version as well as the
    atlas; see :doc:`the machine learning documentation </machine_learning>`.

    Parameters
    ----------
    masker : object, optional
        What to apply, by default None. Any nilearn masker, which is cloned
        rather than modified, or anything nilearn loads as an atlas: a
        :class:`~sklearn.utils.Bunch` from a ``nilearn.datasets.fetch_atlas_*``
        function, a 3D or 4D atlas image or a path to one, or the name of a
        fetcher with its arguments in ``masker_kwargs``. A 4D atlas is
        summarised with a :class:`~nilearn.maskers.NiftiMapsMasker` and a 3D
        one with a :class:`~nilearn.maskers.NiftiLabelsMasker`.
    source_masker : :class:`~nilearn.maskers.NiftiMasker` or img_like, optional
        The masker defining the voxel order of the incoming features, normally
        the ``masker`` a bunch carries, by default None. This is where the
        columns came from, not what is applied to them. Columns may be the
        masker's own voxels, as :class:`~nimare.ml.MAKernel` returns them, or
        the whole image grid, as a bunch's peak columns arrive; which of the
        two is read off their width.
    masker_kwargs : :obj:`dict`, optional
        Arguments for the nilearn fetcher when ``masker`` names one, by default
        None.
    batch_size : :obj:`int`, default=32
        How many rows are held in dense image form at once. The default is the
        measured optimum for a 2 mm whole-brain mask, at roughly 60 MB; see
        :doc:`the machine learning documentation </machine_learning>`.

    Attributes
    ----------
    masker_ : :class:`~nilearn.maskers.BaseMasker`
        The fitted nilearn masker doing the work.
    region_names_ : :obj:`list` of :obj:`str` or None
        Region names read from the atlas, when it carries any.
    n_features_out_ : :obj:`int`
        How many columns the fitted masker reports, known once anything has
        been transformed or named.

    Examples
    --------
    >>> transformer = MaskerTransformer(  # doctest: +SKIP
    ...     fetch_atlas_difumo(dimension=64),
    ...     source_masker=bunch.masker,
    ... )
    >>> smoother = MaskerTransformer(  # doctest: +SKIP
    ...     NiftiMasker(smoothing_fwhm=6),
    ...     source_masker=bunch.masker,
    ... )
    """

    def __init__(self, masker=None, source_masker=None, masker_kwargs=None, batch_size=32):
        self.masker = masker
        self.source_masker = source_masker
        self.masker_kwargs = masker_kwargs
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Resolve the atlas and fit its masker in the source mask's space.

        Parameters
        ----------
        X : array_like or sparse matrix
            Analysis-by-voxel features, used only for their width.
        y : ignored

        Returns
        -------
        :class:`MaskerTransformer`
            The fitted transformer.

        Raises
        ------
        :obj:`ValueError`
            If nothing was given to apply, if no source masker was given, or
            if the columns match neither the source masker's voxels nor its
            image grid.
        """
        if self.masker is None:
            raise ValueError(
                "MaskerTransformer requires something to apply: a nilearn masker, a "
                "fetched nilearn atlas, an atlas image or file, or the name of a nilearn "
                "fetcher."
            )
        if self.source_masker is None:
            raise ValueError(
                "MaskerTransformer requires the source_masker that defines the voxel "
                "order of the features, normally the masker a bunch carries."
            )

        self.mask_img_ = mask_source(self.source_masker).mask_img
        self.on_grid_ = _incoming_space(X.shape[1], self.mask_img_)
        masker, region_names = _resolve_atlas(self.masker, self.masker_kwargs)
        masker.set_params(mask_img=self.mask_img_)
        # an atlas masker resamples onto the images it is fitted with, so it
        # gets the mask now rather than resampling once per batch; a voxel
        # masker has nothing to resample and warns if given images
        reduces = hasattr(masker, "labels_img") or hasattr(masker, "maps_img")
        self.masker_ = masker.fit(self.mask_img_) if reduces else masker.fit()
        self.region_names_ = region_names
        self.n_features_in_ = X.shape[1]
        if hasattr(self, "n_features_out_"):
            del self.n_features_out_
        return self

    def transform(self, X):
        """Apply the masker to the features.

        Parameters
        ----------
        X : array_like or sparse matrix
            Analysis-by-voxel features, over the source masker's voxels or over
            its image grid, matching what the transformer was fitted on.

        Returns
        -------
        :obj:`numpy.ndarray`
            Analysis-by-region features for an atlas masker, and
            analysis-by-voxel for a voxel masker.
        """
        check_is_fitted(self, ["masker_"])

        batches = []
        for start in range(0, X.shape[0], self.batch_size):
            batch = X[start : start + self.batch_size]
            if sparse.issparse(batch):
                batch = batch.toarray()
            images = (
                grid_images(batch, self.mask_img_)
                if self.on_grid_
                else unmask(batch, self.mask_img_)
            )
            batches.append(self.masker_.transform(images))

        aggregated = np.vstack(batches)
        self.n_features_out_ = aggregated.shape[1]
        return aggregated

    def inverse_transform(self, X):
        """Return region values spread back over the voxels they summarise.

        Parameters
        ----------
        X : array_like
            Analysis-by-region features, as :meth:`transform` returns them.

        Returns
        -------
        :obj:`numpy.ndarray`
            Analysis-by-voxel features, in the space the transformer reads.
        """
        check_is_fitted(self, ["masker_"])
        images = self.masker_.inverse_transform(np.atleast_2d(X))
        volume = np.asarray(images.dataobj)
        if volume.ndim == 3:
            volume = volume[..., None]
        flat = volume.reshape(-1, volume.shape[-1]).T
        if self.on_grid_:
            return flat
        return flat[:, np.asarray(self.mask_img_.dataobj).ravel() > 0]

    def __sklearn_tags__(self):
        """Declare that this transformer reads sparse input."""
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def get_feature_names_out(self, input_features=None):
        """Return the region names, from the atlas or from the masker.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        :obj:`numpy.ndarray` of :obj:`str`
            One name per region column.
        """
        check_is_fitted(self, ["masker_"])
        n_regions = self._n_features_out()

        for candidate in _name_candidates(self.region_names_):
            if len(candidate) == n_regions:
                return np.asarray(candidate, dtype=str)

        names = _masker_region_names(self.masker_)
        if names is not None and len(names) == n_regions:
            return np.asarray(names, dtype=str)

        return np.asarray([f"region_{idx}" for idx in range(n_regions)], dtype=str)

    def _n_features_out(self):
        """Return how many regions the fitted masker reports, asking it if need be."""
        if not hasattr(self, "n_features_out_"):
            self.transform(np.zeros((1, self.n_features_in_), dtype=float))
        return self.n_features_out_


def _incoming_space(n_columns, mask_img):
    """Report whether columns span the whole image grid rather than the mask.

    The mask is preferred where the two widths coincide.
    """
    n_voxels = int(np.sum(np.asarray(mask_img.dataobj) > 0))
    if n_columns == n_voxels:
        return False
    if n_columns == n_grid_columns(mask_img):
        return True
    raise ValueError(
        f"MaskerTransformer was given {n_columns} columns, but its source_masker has "
        f"{n_voxels} voxels and a grid of {n_grid_columns(mask_img)}. Features must "
        "arrive either over the masker's voxels, as MAKernel returns them, or over its "
        "grid, as a bunch's peak columns do."
    )


def _resolve_atlas(atlas, atlas_kwargs=None):
    """Return ``(unfitted nilearn masker, region names or None)`` for an atlas."""
    region_names = None

    if isinstance(atlas, (str, Path)):
        atlas = _load_atlas(atlas, atlas_kwargs)

    if isinstance(atlas, BaseMasker):
        return clone(atlas), None

    if hasattr(atlas, "maps"):
        # A Bunch from nilearn.datasets.fetch_atlas_*.
        region_names = _atlas_region_names(atlas)
        atlas = atlas.maps

    if not isinstance(atlas, (SpatialImage, str, Path)):
        raise TypeError(
            f"{atlas!r} is not an atlas. Pass a fetched nilearn atlas, an atlas image "
            "or file, the name of a nilearn fetcher, or a nilearn labels or maps masker."
        )

    image = load_img(atlas)
    if image.ndim == 4:
        masker = NiftiMapsMasker(maps_img=image, resampling_target="data", reports=False)
    elif image.ndim == 3:
        masker = NiftiLabelsMasker(labels_img=image, resampling_target="data", reports=False)
    else:
        raise ValueError(
            "An atlas image must be 3D, holding one integer per region, or 4D, holding "
            f"one map per region, not {image.ndim}D."
        )

    return masker, region_names


def _load_atlas(atlas, atlas_kwargs=None):
    """Return the atlas a path or a nilearn fetcher name refers to."""
    name = str(atlas)
    if os.path.exists(name):
        return name

    from nilearn import datasets

    fetcher = getattr(
        datasets, name if name.startswith("fetch_atlas_") else f"fetch_atlas_{name}", None
    )
    if fetcher is None:
        available = sorted(
            attr[len("fetch_atlas_") :]
            for attr in dir(datasets)
            if attr.startswith("fetch_atlas_")
        )
        raise ValueError(
            f"{name!r} is neither a file nor a nilearn atlas fetcher. nilearn fetches: "
            f"{', '.join(available)}."
        )

    return fetcher(**(atlas_kwargs or {}))


def _atlas_region_names(atlas):
    """Return the region names a fetched nilearn atlas carries, or None."""
    labels = getattr(atlas, "labels", None)
    if labels is None:
        return None

    column = _name_column(labels)
    if column is not None:
        labels = labels[column].tolist()

    return [str(label) for label in labels] or None


def _name_column(labels):
    """Return the column of a table of region attributes that holds the names."""
    columns = getattr(labels, "columns", None)
    if columns is None:
        columns = getattr(getattr(labels, "dtype", None), "names", None)
    if columns is None:
        return None

    named = [column for column in columns if "name" in str(column).lower()]
    if named:
        return named[0]

    text = [column for column in columns if _holds_text(labels[column])]
    return text[0] if text else columns[0]


def _holds_text(values):
    """Report whether a column of region attributes holds names rather than numbers."""
    return pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)


def _name_candidates(region_names):
    """Yield the readings of an atlas's names, with and without a background entry."""
    if not region_names:
        return
    yield region_names
    if str(region_names[0]).strip().lower() == "background":
        yield region_names[1:]


def _masker_region_names(atlas_masker):
    """Return the region names a fitted nilearn masker reports, or None."""
    try:
        reported = atlas_masker.get_feature_names_out()
    except (AttributeError, NotFittedError, ValueError, TypeError):
        reported = None

    # nilearn hands back a zero-dimensional array wrapping ``dict_values``.
    if isinstance(reported, np.ndarray) and reported.ndim == 0:
        reported = reported.item()

    for names in (reported, getattr(atlas_masker, "region_names_", None)):
        if isinstance(names, Mapping):
            names = list(names.values())
        if names is not None and len(names):
            return [str(name) for name in names]

    return None


def _is_atlas_like(obj):
    """Report whether an object is a nilearn masker or an atlas."""
    return isinstance(obj, (BaseMasker, SpatialImage)) or hasattr(obj, "maps")


def _is_transformer(obj):
    """Report whether an object is a scikit-learn transformer."""
    return hasattr(obj, "fit") and hasattr(obj, "transform")


def _resolve_voxel_transformer(transformer, masker=None, **kwargs):
    """Return an unfitted transformer for whatever was given for the voxel columns."""
    if isinstance(transformer, type):
        transformer, kwargs = transformer(**kwargs), {}

    if isinstance(transformer, MaskerTransformer) and transformer.source_masker is None:
        # Built without one, which the bunch can supply.
        transformer = clone(transformer)
        transformer.set_params(source_masker=_required_masker(masker))
        return transformer

    if _is_atlas_like(transformer):
        return MaskerTransformer(
            masker=transformer, source_masker=_required_masker(masker), **kwargs
        )

    if _is_transformer(transformer):
        if kwargs:
            raise ValueError(
                "Keyword parameters are only used when the transformer is given as a "
                "class or as an atlas; set them on the transformer instead."
            )
        return transformer

    raise TypeError(
        f"{transformer!r} cannot transform the voxel columns. Pass a scikit-learn "
        "transformer such as TruncatedSVD(n_components=50) or VarianceThreshold(), a "
        "transformer class, or a nilearn masker or atlas for MaskerTransformer to apply."
    )


def _required_masker(masker):
    """Return the source masker a nilearn masker needs, or say it is missing."""
    if masker is None:
        raise ValueError(
            "A nilearn masker needs the masker that defines the voxel order of the map "
            "features, normally the masker a bunch carries."
        )
    return masker
