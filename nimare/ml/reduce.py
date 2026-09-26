"""Reducing modeled activation features, and the atlases that can do it."""

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


class AtlasAggregator(TransformerMixin, BaseEstimator):
    """Aggregate masked voxel features into the regions of a nilearn atlas.

    Rows are converted back into images in the source mask's space, in batches,
    and summarised by a nilearn masker, so region definitions, resampling and
    the aggregation strategy stay nilearn's. How many regions an atlas yields
    therefore depends on the nilearn version as well as on the atlas; see
    :doc:`the machine learning documentation </machine_learning>`.

    Parameters
    ----------
    atlas : object, optional
        The atlas, in any form nilearn loads, by default None: a
        :class:`~sklearn.utils.Bunch` from a ``nilearn.datasets.fetch_atlas_*``
        function, a 3D or 4D atlas image or a path to one, the name of a
        fetcher with its arguments in ``atlas_kwargs``, or a
        :class:`~nilearn.maskers.NiftiLabelsMasker` or
        :class:`~nilearn.maskers.NiftiMapsMasker`, which is cloned rather than
        modified. A 4D atlas is summarised with a maps masker and a 3D one with
        a labels masker.
    masker : :class:`~nilearn.maskers.NiftiMasker` or img_like, optional
        The masker defining the voxel order of the incoming features, normally
        :attr:`FeatureSet.masker`, by default None.
    atlas_kwargs : :obj:`dict`, optional
        Arguments for the nilearn fetcher when ``atlas`` names one, by default
        None.
    batch_size : :obj:`int`, default=32
        How many rows are held in dense image form at once. Larger batches pay
        the masker's per-call setup over more rows, at proportionally more
        memory: 32 rows of a 2 mm whole-brain mask is roughly 60 MB.

    Attributes
    ----------
    atlas_masker_ : :class:`~nilearn.maskers.BaseMasker`
        The fitted nilearn masker doing the aggregation.
    region_names_ : :obj:`list` of :obj:`str` or None
        Region names read from the atlas, when it carries any.
    n_features_out_ : :obj:`int`
        How many regions the fitted masker reports, known once anything has
        been transformed or named.

    Examples
    --------
    >>> reducer = AtlasAggregator(  # doctest: +SKIP
    ...     atlas=fetch_atlas_difumo(dimension=64),
    ...     masker=features.masker,
    ... )
    """

    def __init__(self, atlas=None, masker=None, atlas_kwargs=None, batch_size=32):
        self.atlas = atlas
        self.masker = masker
        self.atlas_kwargs = atlas_kwargs
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
        :class:`AtlasAggregator`
            The fitted aggregator.
        """
        if self.atlas is None:
            raise ValueError(
                "AtlasAggregator requires an atlas: a fetched nilearn atlas, an atlas "
                "image or file, the name of a nilearn fetcher, or a nilearn labels or "
                "maps masker."
            )
        if self.masker is None:
            raise ValueError(
                "AtlasAggregator requires the masker that defines the voxel order of the "
                "features, normally FeatureSet.masker."
            )

        from nimare.utils import get_masker

        self.mask_img_ = get_masker(self.masker).mask_img
        atlas_masker, region_names = _resolve_atlas(self.atlas, self.atlas_kwargs)
        atlas_masker.set_params(mask_img=self.mask_img_)
        self.atlas_masker_ = atlas_masker.fit(self.mask_img_)
        self.region_names_ = region_names
        self.n_features_in_ = X.shape[1]
        if hasattr(self, "n_features_out_"):
            del self.n_features_out_
        return self

    def transform(self, X):
        """Aggregate features into regions.

        Parameters
        ----------
        X : array_like or sparse matrix
            Analysis-by-voxel features in the source masker's voxel order.

        Returns
        -------
        :obj:`numpy.ndarray`
            Analysis-by-region features.
        """
        check_is_fitted(self, ["atlas_masker_"])

        batches = []
        for start in range(0, X.shape[0], self.batch_size):
            batch = X[start : start + self.batch_size]
            if sparse.issparse(batch):
                batch = batch.toarray()
            batches.append(self.atlas_masker_.transform(unmask(batch, self.mask_img_)))

        aggregated = np.vstack(batches)
        self.n_features_out_ = aggregated.shape[1]
        return aggregated

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
        check_is_fitted(self, ["atlas_masker_"])
        n_regions = self._n_features_out()

        for candidate in _name_candidates(self.region_names_):
            if len(candidate) == n_regions:
                return np.asarray(candidate, dtype=str)

        names = _masker_region_names(self.atlas_masker_)
        if names is not None and len(names) == n_regions:
            return np.asarray(names, dtype=str)

        return np.asarray([f"region_{idx}" for idx in range(n_regions)], dtype=str)

    def _n_features_out(self):
        """Return how many regions the fitted masker reports, asking it if need be.

        Neither ``n_elements_`` nor the atlas image answers this on nilearn
        0.13, where a region outside the mask is dropped from the output but
        not from either of them.
        """
        if not hasattr(self, "n_features_out_"):
            self.transform(np.zeros((1, self.n_features_in_), dtype=float))
        return self.n_features_out_


def _resolve_atlas(atlas, atlas_kwargs=None):
    """Return ``(unfitted nilearn masker, region names or None)`` for an atlas.

    Accepts whatever nilearn hands back: a fetched atlas, an image, a path, a
    fetcher name, or a masker built by the caller.
    """
    region_names = None

    if isinstance(atlas, (str, Path)):
        atlas = _load_atlas(atlas, atlas_kwargs)

    if isinstance(atlas, BaseMasker):
        if not (hasattr(atlas, "labels_img") or hasattr(atlas, "maps_img")):
            raise ValueError(
                f"{type(atlas).__name__} extracts voxels rather than regions. Pass a "
                "labels or maps atlas, or a NiftiLabelsMasker or NiftiMapsMasker."
            )
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
    """Return the region names a fetched nilearn atlas carries, or None.

    Fetchers return them as a list, or as a table of region attributes the way
    DiFuMo does.
    """
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


def _is_atlas_like(reducer):
    """Report whether an object describes an atlas rather than a reducer."""
    return isinstance(reducer, (BaseMasker, SpatialImage)) or hasattr(reducer, "maps")


def _is_transformer(reducer):
    """Report whether an object is a scikit-learn transformer."""
    return hasattr(reducer, "fit") and hasattr(reducer, "transform")


def _resolve_map_reducer(reducer, masker=None, **kwargs):
    """Return an unfitted transformer for whatever describes a map reduction.

    A scikit-learn transformer is used as given, a transformer class is built
    from ``kwargs``, and an atlas is wrapped in an :class:`AtlasAggregator`
    bound to ``masker``.
    """
    if isinstance(reducer, type):
        reducer, kwargs = reducer(**kwargs), {}

    if isinstance(reducer, AtlasAggregator) and reducer.masker is None:
        # Built without a masker, which the feature set can supply.
        reducer = clone(reducer)
        reducer.set_params(masker=_required_masker(masker))
        return reducer

    if _is_atlas_like(reducer):
        return AtlasAggregator(atlas=reducer, masker=_required_masker(masker), **kwargs)

    if _is_transformer(reducer):
        if kwargs:
            raise ValueError(
                "Reducer parameters are only used when the reducer is given as a class "
                "or as an atlas; set them on the transformer instead."
            )
        return reducer

    raise TypeError(
        f"{reducer!r} is not a map reducer. Pass a scikit-learn transformer such as "
        "TruncatedSVD(n_components=50) or VarianceThreshold(), a transformer class, or "
        "an atlas for AtlasAggregator to summarise."
    )


def _required_masker(masker):
    """Return the masker an atlas reduction needs, or explain that it is missing."""
    if masker is None:
        raise ValueError(
            "Atlas aggregation needs the masker that defines the voxel order of the map "
            "features, normally FeatureSet.masker."
        )
    return masker
