"""Machine-learning helpers for modeled activation (MA) features.

This module converts a :class:`~nimare.nimads.Studyset` into the aligned arrays
scikit-learn workflows expect: a sparse analysis-by-voxel feature matrix, an
optional target, and the study labels that keep analyses from one study out of
two different partitions.

The division of labour is deliberate. NiMARE owns *extraction*: reading the
Studyset, generating MA maps through a kernel transformer, aligning every row to
its analysis, and reporting what is missing. scikit-learn owns *evaluation*:
splitting, fitting, reducing and scoring. Kernel transformation is row-wise
independent -- an analysis's MA map is a function of that analysis's own foci --
so building the whole matrix before splitting leaks nothing. Everything that
learns *across* rows (decomposition, feature selection, imputation, scaling)
must be fit on training rows only, which is what a
:class:`~sklearn.pipeline.Pipeline` is for;
:meth:`MAFeatureDataset.make_preprocessor` builds the piece that goes in it.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import Memory
from joblib import hash as joblib_hash
from nilearn.masking import unmask
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from nimare.base import NiMAREBase
from nimare.studyset import normalize_collection
from nimare.studyset.columns import ID_COLS
from nimare.studyset.requirements import PerAnalysis

LGR = logging.getLogger(__name__)

__all__ = [
    "AtlasAggregator",
    "MAFeatureDataset",
    "MAFeatureExtractor",
    "make_map_reducer",
]

#: Studyset tables a descriptor or target field may be selected from.
FIELD_SOURCES = ("metadata", "annotations", "texts")

#: Reduction workflows :func:`make_map_reducer` knows by name.
MAP_REDUCERS = ("variance_threshold", "truncated_svd", "atlas_aggregation")

_SOURCE_ALIASES = {
    "annotations_df": "annotations",
    "annotation": "annotations",
    "text": "texts",
    "metadata": "metadata",
    "annotations": "annotations",
    "texts": "texts",
}


# ---------------------------------------------------------------- selectors


def _as_selector(selector):
    """Return ``(source, field)`` for a field selector, with ``source`` optional.

    A selector is a bare field name, a ``(source, field)`` pair matching the
    ``(kind, field)`` vocabulary NiMARE estimators already use in
    ``_required_inputs``, or the equivalent mapping.
    """
    source, field = None, None
    if isinstance(selector, str):
        field = selector
    elif isinstance(selector, Mapping):
        source, field = selector.get("source"), selector.get("field")
        if not field:
            raise ValueError(f"Field selector {selector!r} must define 'field'.")
    elif isinstance(selector, Sequence) and len(selector) == 2:
        source, field = selector
    else:
        raise TypeError(
            f"Field selector {selector!r} must be a field name, a (source, field) pair, "
            "or a mapping with 'source' and 'field'."
        )

    if source is not None:
        if source not in _SOURCE_ALIASES:
            raise ValueError(
                f"Unsupported field selector source {source!r}. Supported sources are "
                f"{', '.join(FIELD_SOURCES)}. A tuple is one (source, field) selector; "
                "put several selectors in a list."
            )
        source = _SOURCE_ALIASES[source]

    return source, str(field)


def _source_frame(studyset, source):
    """Return the Studyset table a source names."""
    return {
        "metadata": studyset.metadata,
        "annotations": studyset.annotations_df,
        "texts": studyset.texts,
    }[source]


def _selectable_columns(studyset, source):
    """Return the columns of one source that a selector may name."""
    return [col for col in _source_frame(studyset, source).columns if col not in ID_COLS]


def _resolve_selector(studyset, selector, what="descriptor"):
    """Return ``(source, field)``, inferring the source from a bare field name."""
    source, field = _as_selector(selector)

    if source is not None:
        if field not in _selectable_columns(studyset, source):
            raise ValueError(
                f"{what.capitalize()} field {field!r} was not found in the Studyset "
                f"{source}. Available {source} fields: "
                f"{_preview(_selectable_columns(studyset, source))}."
            )
        return source, field

    matches = [src for src in FIELD_SOURCES if field in _selectable_columns(studyset, src)]
    if not matches:
        raise ValueError(
            f"{what.capitalize()} field {field!r} was not found in the Studyset metadata, "
            "annotations or texts. Name the source explicitly with a "
            f"(source, field) selector if it should be there."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{what.capitalize()} field {field!r} is ambiguous: it appears in "
            f"{', '.join(matches)}. Name the source explicitly, for example "
            f"('{matches[0]}', '{field}')."
        )
    return matches[0], field


def _preview(values, limit=8):
    """Render a short, bounded preview of a collection for an error message."""
    values = list(values)
    shown = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return shown or "none"


def _read_field(studyset, source, field):
    """Return ``(values, kind)`` for one field, aligned to ``studyset.ids``.

    ``kind`` is ``"numeric"``, ``"categorical"`` or ``"text"``. Numeric metadata
    is read through :class:`~nimare.studyset.requirements.PerAnalysis`, so
    study-level values are inherited by their analyses and list-valued fields
    such as ``sample_sizes`` are reduced the way the rest of NiMARE reduces them.
    """
    raw = _source_frame(studyset, source)[field]

    if source == "texts":
        return raw.to_numpy(dtype=object), "text"

    if source == "metadata":
        values = PerAnalysis(field).values(studyset.store)[studyset.view.index]
        if np.isfinite(values).any() or pd.api.types.is_numeric_dtype(raw):
            return np.asarray(values, dtype=float), "numeric"

    if pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw):
        return raw.to_numpy(dtype=float), "numeric"

    return raw.to_numpy(dtype=object), "categorical"


def _missing_mask(values, kind):
    """Return a boolean mask of values that are absent rather than merely zero."""
    if kind == "numeric":
        return ~np.isfinite(np.asarray(values, dtype=float))

    def absent(value):
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    return np.array([absent(value) for value in values], dtype=bool)


def _jsonable(value):
    """Return a JSON-serialisable stand-in for a provenance value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    return repr(value)


# ------------------------------------------------------------- the container


def _hstack(left, right):
    """Stack two feature blocks, keeping the result sparse if either part is."""
    if right is None:
        return left
    if sparse.issparse(left) or sparse.issparse(right):
        return sparse.hstack([_as_sparse(left), _as_sparse(right)], format="csr")
    return np.hstack([left, right])


def _as_sparse(block):
    """Return a CSR view of a feature block."""
    return block if sparse.issparse(block) else sparse.csr_matrix(block)


def _take_rows(value, rows):
    """Take rows from matrix-like, frame-like or sequence-like data."""
    if value is None:
        return None
    if sparse.issparse(value):
        return value[rows]
    if hasattr(value, "iloc"):
        return value.iloc[rows].copy()
    return np.asarray(value)[rows]


class MAFeatureDataset(NiMAREBase):
    """Aligned modeled activation features, descriptors, target and provenance.

    Every array here shares one row order: row ``i`` is analysis ``ids[i]`` from
    study ``study_ids[i]``. That order is fixed when the dataset is built and is
    preserved by :meth:`split`, :meth:`select_analyses` and the map-reduction
    methods.

    Parameters
    ----------
    map_features : :obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`
        Analysis-by-voxel modeled activation features. Unreduced voxelwise
        features are sparse; a reduced representation may be dense.
    ids : array_like of :obj:`str`
        Full ``"<study_id>-<analysis_id>"`` identifier for each row.
    study_ids : array_like of :obj:`str`
        Study identifier for each row, used as scikit-learn ``groups``.
    descriptor_features : :obj:`numpy.ndarray`, optional
        Numeric descriptor columns appended to :attr:`features`, by default None.
    descriptor_names : :obj:`list` of :obj:`str`, optional
        Names of the descriptor columns, by default None.
    descriptors : :obj:`pandas.DataFrame`, optional
        The selected descriptor values as they were read from the Studyset,
        indexed by :attr:`ids`, by default None. Useful for encoding
        non-numeric fields inside a scikit-learn pipeline.
    target : array_like, optional
        Row-aligned prediction target, by default None.
    provenance : :obj:`dict`, optional
        Conversion settings and source Studyset details, by default None.
    masker : :obj:`nilearn.maskers.NiftiMasker`, optional
        Masker defining the voxel order of unreduced map features, by default
        None.
    map_feature_names : :obj:`list` of :obj:`str`, optional
        Names of the map columns, by default None, which names them
        ``"voxel_<i>"`` on demand.

    Attributes
    ----------
    features : :obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`
        Map features and descriptor features side by side. Built on first
        access from the two blocks, which are the source of truth.
    feature_names : :obj:`list` of :obj:`str`
        Names for :attr:`features` in column order, built on first access.

    See Also
    --------
    MAFeatureExtractor : Builds this container from a Studyset.
    """

    def __init__(
        self,
        map_features: Any,
        ids: Sequence[str],
        study_ids: Sequence[str],
        *,
        descriptor_features: Any | None = None,
        descriptor_names: Sequence[str] | None = None,
        descriptors: pd.DataFrame | None = None,
        target: Any | None = None,
        provenance: dict[str, Any] | None = None,
        masker: Any | None = None,
        map_feature_names: Sequence[str] | None = None,
    ) -> None:
        n_rows = map_features.shape[0]

        ids = np.asarray(ids, dtype=str)
        study_ids = np.asarray(study_ids, dtype=str)

        if len(ids) != n_rows:
            raise ValueError(f"ids has {len(ids)} entries but map_features has {n_rows} rows.")
        if len(study_ids) != n_rows:
            raise ValueError(
                f"study_ids has {len(study_ids)} entries but map_features has {n_rows} rows."
            )
        if target is not None and len(target) != n_rows:
            raise ValueError(
                f"target has {len(target)} entries but map_features has {n_rows} rows."
            )
        if descriptor_features is not None:
            descriptor_features = np.asarray(descriptor_features)
            if descriptor_features.ndim != 2:
                raise ValueError("descriptor_features must be two-dimensional.")
            if descriptor_features.shape[0] != n_rows:
                raise ValueError(
                    f"descriptor_features has {descriptor_features.shape[0]} rows but "
                    f"map_features has {n_rows} rows."
                )
            if descriptor_names is not None and len(descriptor_names) != (
                descriptor_features.shape[1]
            ):
                raise ValueError("descriptor_names must name every descriptor column.")
        if map_feature_names is not None and len(map_feature_names) != map_features.shape[1]:
            raise ValueError("map_feature_names must name every map column.")

        self._map_features = map_features
        self._descriptor_features = descriptor_features
        self._descriptor_names = None if descriptor_names is None else list(descriptor_names)
        self._map_feature_names = None if map_feature_names is None else list(map_feature_names)
        self._features = None
        self._feature_names = None

        self.ids = ids
        self.study_ids = study_ids
        self.descriptors = descriptors
        self.target = None if target is None else np.asarray(target)
        self.provenance = {} if provenance is None else provenance
        self.masker = masker

    def __repr__(self):
        """Show the dataset's shape."""
        n_rows, n_features = self.shape
        return (
            f"{self.__class__.__name__}(n_rows={n_rows}, n_features={n_features}, "
            f"n_studies={len(np.unique(self.study_ids))})"
        )

    def __len__(self):
        """Return the number of analysis rows."""
        return len(self.ids)

    # ----------------------------------------------------------- feature views

    @property
    def features(self):
        """:obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`: map and descriptor columns."""
        if self._features is None:
            self._features = _hstack(self._map_features, self._descriptor_features)
        return self._features

    @property
    def map_features(self):
        """:obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`: the map block alone."""
        return self._map_features

    @property
    def descriptor_features(self):
        """:obj:`numpy.ndarray` or None: the descriptor block alone."""
        return self._descriptor_features

    @property
    def shape(self):
        """:obj:`tuple`: ``(n_rows, n_features)``."""
        n_descriptors = (
            0 if self._descriptor_features is None else (self._descriptor_features.shape[1])
        )
        return (self._map_features.shape[0], self._map_features.shape[1] + n_descriptors)

    @property
    def map_columns(self):
        """:obj:`slice`: the columns of :attr:`features` holding map features."""
        return slice(0, self._map_features.shape[1])

    @property
    def descriptor_columns(self):
        """:obj:`slice`: the columns of :attr:`features` holding descriptor features."""
        return slice(self._map_features.shape[1], self.shape[1])

    @property
    def feature_names(self):
        """:obj:`list` of :obj:`str`: names for :attr:`features`, in column order.

        Built on first access. Naming 200,000 voxels eagerly costs more memory
        than the sparse matrix being named.
        """
        if self._feature_names is None:
            names = self._map_feature_names
            if names is None:
                names = [f"voxel_{idx}" for idx in range(self._map_features.shape[1])]
            self._feature_names = list(names) + list(self._descriptor_names or [])
        return self._feature_names

    # ----------------------------------------------------------------- export

    def to_sklearn(self, return_X_y=False):
        """Export the dataset for scikit-learn.

        Parameters
        ----------
        return_X_y : :obj:`bool`, default=False
            If True, return ``(data, target)`` instead of a Bunch, following
            the :mod:`sklearn.datasets` convention.

        Returns
        -------
        :class:`sklearn.utils.Bunch` or :obj:`tuple`
            A Bunch with ``data``, ``target``, ``groups``, ``feature_names``,
            ``ids``, ``provenance``, ``map_columns`` and ``descriptor_columns``;
            or ``(data, target)`` when ``return_X_y`` is True.

        Notes
        -----
        ``data`` stays sparse while the map features are unreduced. Pass
        ``groups`` straight to a scikit-learn group splitter.
        """
        if return_X_y:
            return self.features, self.target

        return Bunch(
            data=self.features,
            target=self.target,
            groups=self.study_ids,
            feature_names=self.feature_names,
            ids=self.ids,
            descriptors=self.descriptors,
            provenance=self.provenance,
            map_columns=self.map_columns,
            descriptor_columns=self.descriptor_columns,
        )

    # ------------------------------------------------------------- evaluation

    def split(self, test_size=0.25, random_state=None):
        """Split into train and test partitions without splitting a study.

        Parameters
        ----------
        test_size : :obj:`float` or :obj:`int`, default=0.25
            Proportion of *studies* held out, or the number of studies to hold
            out. Analyses are not split evenly when studies contribute
            different numbers of analyses.
        random_state : :obj:`int` or None, optional
            Seed for a reproducible split, by default None.

        Returns
        -------
        (:class:`MAFeatureDataset`, :class:`MAFeatureDataset`)
            Train and test datasets. No study appears in both.

        Raises
        ------
        :obj:`ValueError`
            If the available studies cannot serve the requested split. Nothing
            partial is returned.

        See Also
        --------
        sklearn.model_selection.GroupKFold : Cross-validation over the same groups.
        """
        n_studies = len(np.unique(self.study_ids))
        n_test = self._check_split(test_size, n_studies)

        splitter = GroupShuffleSplit(n_splits=1, test_size=n_test, random_state=random_state)
        train_rows, test_rows = next(splitter.split(np.empty(len(self)), groups=self.study_ids))

        return self.select_analyses(train_rows), self.select_analyses(test_rows)

    @staticmethod
    def _check_split(test_size, n_studies):
        """Return the number of held-out studies, or explain why there is none."""
        if n_studies < 2:
            raise ValueError(
                f"A grouped split needs at least 2 studies, but the dataset has {n_studies}. "
                "Analyses from one study are never split across partitions."
            )

        if isinstance(test_size, (int, np.integer)) and not isinstance(test_size, bool):
            n_test = int(test_size)
        elif isinstance(test_size, (float, np.floating)):
            if not 0.0 < test_size < 1.0:
                raise ValueError(
                    f"test_size={test_size!r} must be between 0 and 1 when given as a fraction."
                )
            n_test = int(np.ceil(test_size * n_studies))
        else:
            raise ValueError(f"test_size={test_size!r} must be a float or an int.")

        if not 1 <= n_test <= n_studies - 1:
            raise ValueError(
                f"test_size={test_size!r} holds out {n_test} of {n_studies} studies, which "
                "leaves one partition empty. Lower test_size or use more studies."
            )
        return n_test

    def make_preprocessor(
        self,
        map_reducer="truncated_svd",
        descriptor_transformer="passthrough",
        **reducer_params,
    ):
        """Build the unfitted preprocessing step for a scikit-learn pipeline.

        The returned transformer reduces the map columns and handles the
        descriptor columns separately, so a pipeline fits both on training rows
        only and no information reaches held-out rows.

        Parameters
        ----------
        map_reducer : :obj:`str` or estimator or None, default="truncated_svd"
            A name :func:`make_map_reducer` knows, an already-built
            scikit-learn transformer, or None/``"passthrough"`` to leave the map
            columns alone.
        descriptor_transformer : estimator or :obj:`str`, default="passthrough"
            What to apply to the descriptor columns, for example
            :class:`~sklearn.impute.SimpleImputer` when descriptors were kept
            with missing values.
        **reducer_params
            Passed to :func:`make_map_reducer` when ``map_reducer`` is a name.

        Returns
        -------
        :class:`sklearn.compose.ColumnTransformer`
            Unfitted, with ``sparse_threshold=1.0`` so that unreduced voxelwise
            features are never densified on the way through.

        Examples
        --------
        >>> pipeline = make_pipeline(  # doctest: +SKIP
        ...     dataset.make_preprocessor("truncated_svd", n_components=50),
        ...     LogisticRegression(),
        ... )
        """
        if map_reducer is None or (isinstance(map_reducer, str) and map_reducer == "passthrough"):
            reducer = "passthrough"
        elif isinstance(map_reducer, str):
            reducer = make_map_reducer(map_reducer, masker=self.masker, **reducer_params)
        else:
            if reducer_params:
                raise ValueError(
                    "Reducer parameters are only used when map_reducer names a workflow; "
                    "set them on the transformer instead."
                )
            reducer = map_reducer

        transformers = [("maps", reducer, self.map_columns)]
        if self._descriptor_features is not None:
            transformers.append(("descriptors", descriptor_transformer, self.descriptor_columns))

        return ColumnTransformer(transformers, sparse_threshold=1.0)

    # -------------------------------------------------------- map reduction

    def fit_transform_maps(self, reducer):
        """Fit a reducer on this dataset's map features and apply it.

        Fit it on the training dataset only, then pass the same, now fitted,
        reducer to :meth:`transform_maps` for held-out data.

        Parameters
        ----------
        reducer : estimator
            A scikit-learn transformer, for example one from
            :func:`make_map_reducer`.

        Returns
        -------
        :class:`MAFeatureDataset`
            A dataset with reduced map features and everything else unchanged.
        """
        return self._with_map_features(reducer.fit_transform(self._map_features), reducer)

    def transform_maps(self, reducer):
        """Apply an already fitted reducer to this dataset's map features.

        Parameters
        ----------
        reducer : estimator
            A transformer already fitted by :meth:`fit_transform_maps` on the
            training dataset.

        Returns
        -------
        :class:`MAFeatureDataset`
            A dataset with reduced map features and everything else unchanged.

        Raises
        ------
        :class:`sklearn.exceptions.NotFittedError`
            If the reducer has not been fitted, which would mean fitting it on
            held-out data.
        """
        try:
            check_is_fitted(reducer)
        except NotFittedError:
            raise NotFittedError(
                f"{type(reducer).__name__} is not fitted. Fit it on the training dataset "
                "with fit_transform_maps() first; fitting it here would use held-out data."
            ) from None
        except TypeError:
            # Not a scikit-learn estimator; let its own transform complain.
            pass

        return self._with_map_features(reducer.transform(self._map_features), reducer)

    def _with_map_features(self, reduced, reducer):
        """Return a copy carrying reduced map features."""
        if reduced.shape[0] != len(self):
            raise ValueError(
                f"{type(reducer).__name__} returned {reduced.shape[0]} rows for "
                f"{len(self)} analyses. A map reducer must preserve the analysis rows."
            )

        try:
            names = list(reducer.get_feature_names_out())
        except (AttributeError, NotFittedError, ValueError):
            names = [f"component_{idx}" for idx in range(reduced.shape[1])]
        if len(names) != reduced.shape[1]:
            names = [f"component_{idx}" for idx in range(reduced.shape[1])]

        provenance = copy.deepcopy(self.provenance)
        provenance.setdefault("map_reductions", []).append(
            {
                "reducer": type(reducer).__name__,
                "params": _jsonable(getattr(reducer, "get_params", dict)()),
                "n_features_out": int(reduced.shape[1]),
            }
        )

        return self._rebuild(map_features=reduced, map_feature_names=names, provenance=provenance)

    # -------------------------------------------------------------- plumbing

    def select_analyses(self, rows):
        """Return the dataset restricted to the analyses a mask or positions select.

        Parameters
        ----------
        rows : array_like
            A boolean mask over the rows, or an array of row positions.

        Returns
        -------
        :class:`MAFeatureDataset`
            A dataset holding the selected rows, in the order they were given.
        """
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if len(rows) != len(self):
                raise ValueError(
                    f"A boolean mask must have one entry per analysis row ({len(self)})."
                )
            rows = np.flatnonzero(rows)
        rows = rows.astype(int, copy=False)

        descriptors = None
        if self.descriptors is not None:
            descriptors = self.descriptors.iloc[rows].copy()

        return self._rebuild(
            map_features=_take_rows(self._map_features, rows),
            descriptor_features=_take_rows(self._descriptor_features, rows),
            descriptors=descriptors,
            ids=self.ids[rows],
            study_ids=self.study_ids[rows],
            target=None if self.target is None else self.target[rows],
        )

    def copy(self):
        """Return an independent copy of the dataset.

        Returns
        -------
        :class:`MAFeatureDataset`
            A copy that shares no mutable state with this dataset.
        """
        return self._rebuild(
            map_features=self._map_features.copy(),
            descriptor_features=(
                None if self._descriptor_features is None else self._descriptor_features.copy()
            ),
            descriptors=None if self.descriptors is None else self.descriptors.copy(),
            ids=self.ids.copy(),
            study_ids=self.study_ids.copy(),
            target=None if self.target is None else self.target.copy(),
            provenance=copy.deepcopy(self.provenance),
        )

    def _rebuild(self, **changes):
        """Return a new dataset, keeping whatever was not named in ``changes``."""
        kwargs = {
            "map_features": self._map_features,
            "ids": self.ids,
            "study_ids": self.study_ids,
            "descriptor_features": self._descriptor_features,
            "descriptor_names": self._descriptor_names,
            "descriptors": self.descriptors,
            "target": self.target,
            "provenance": self.provenance,
            "masker": self.masker,
            "map_feature_names": self._map_feature_names,
        }
        kwargs.update(changes)
        positional = (kwargs.pop("map_features"), kwargs.pop("ids"), kwargs.pop("study_ids"))
        return MAFeatureDataset(*positional, **kwargs)


# ------------------------------------------------------------- the extractor


class MAFeatureExtractor(NiMAREBase):
    """Convert a Studyset into modeled activation feature data.

    This is a NiMARE conversion helper, not a scikit-learn estimator: it has no
    ``fit``. Call :meth:`transform` for the NiMARE container or
    :meth:`to_sklearn` for the scikit-learn bundle, then let a
    :class:`~sklearn.pipeline.Pipeline` own everything that learns from the data.

    Parameters
    ----------
    kernel_transformer : :class:`~nimare.meta.kernel.KernelTransformer`
        Kernel transformer instance or class used to generate the MA maps.
        There is no default: the choice is scientific.
    descriptor_fields : :obj:`list`, optional
        Fields appended to the feature matrix as extra numeric columns, by
        default None. Each is a field name, a ``(source, field)`` tuple, or a
        mapping with ``source`` and ``field``; sources are ``"metadata"``,
        ``"annotations"`` and ``"texts"``. A bare field name is looked up in
        each source in turn, and an ambiguous name asks for the tuple form.
        Non-numeric fields are rejected -- see the Notes.
    target_field : :obj:`str` or :obj:`tuple` or :obj:`dict`, optional
        Field exported as ``y``, by default None. Scalar numeric and scalar
        categorical fields are supported directly.
    target_transformer : :obj:`callable` or transformer, optional
        Applied to the raw target values before they become ``y``, by default
        None. Required for text fields, which have no scalar reading. Use a
        row-wise transform such as a label extractor; a transform that learns
        from the distribution of ``y`` belongs in
        :class:`~sklearn.compose.TransformedTargetRegressor`.
    missing_coordinates : {"drop", "include"}, default="drop"
        What to do with analyses that report no coordinates. ``"drop"`` removes
        them before rows are built and records their ids in provenance;
        ``"include"`` keeps them as all-zero sparse map rows.
    missing_values : {"raise", "drop", "keep"}, default="raise"
        What to do when a selected descriptor or target value is missing.
        ``"raise"`` reports the analyses and fields involved; ``"drop"`` removes
        those analyses and records them in provenance; ``"keep"`` leaves NaN in
        place for a pipeline to impute.
    cache_maps : :obj:`bool`, default=True
        Whether to hold the most recently generated map matrix in memory, so
        that comparing several reducers over one Studyset generates maps once.
    memory : :class:`joblib.Memory`, :obj:`str` or :class:`pathlib.Path`, optional
        Cache location for MA map generation across processes. Used only when
        the kernel transformer does not define its own; the kernel's own
        ``memory`` always wins.
    memory_level : :obj:`int`, default=1
        How eagerly ``memory`` caches, following the NiMARE convention.

    Notes
    -----
    Descriptor fields must be numeric, because the exported feature matrix is
    numeric. A categorical or text field raises, and names the two ways to use
    it: encode it yourself and select the numeric result, or leave it out of
    the matrix and encode it inside your pipeline, reading the raw values from
    :attr:`MAFeatureDataset.descriptors`. Encoding at extraction time would fit
    the encoder on every row, including the rows you are about to hold out.

    Examples
    --------
    >>> extractor = MAFeatureExtractor(  # doctest: +SKIP
    ...     kernel_transformer=MKDAKernel(r=10),
    ...     target_field=("metadata", "comparison_task"),
    ... )
    >>> data = extractor.transform(studyset)  # doctest: +SKIP
    >>> train, test = data.split(test_size=0.25, random_state=13)  # doctest: +SKIP

    See Also
    --------
    MAFeatureDataset : The container this returns.
    make_map_reducer : Reduction workflows for the voxelwise map features.
    """

    def __init__(
        self,
        kernel_transformer: Any,
        descriptor_fields: Sequence[Any] | None = None,
        target_field: Any | None = None,
        target_transformer: Any | None = None,
        missing_coordinates: str = "drop",
        missing_values: str = "raise",
        cache_maps: bool = True,
        memory: Any = None,
        memory_level: int = 1,
    ):
        self.kernel_transformer = kernel_transformer
        self.descriptor_fields = descriptor_fields
        self.target_field = target_field
        self.target_transformer = target_transformer
        self.missing_coordinates = missing_coordinates
        self.missing_values = missing_values
        self.cache_maps = cache_maps
        self.memory = memory
        self.memory_level = memory_level
        self._map_memo = None

    # ------------------------------------------------------------- public API

    def transform(self, studyset):
        """Convert a Studyset into a feature dataset.

        Parameters
        ----------
        studyset : :class:`~nimare.nimads.Studyset`
            The Studyset to convert.

        Returns
        -------
        :class:`MAFeatureDataset`
            One row per retained analysis, with map features, any descriptor
            features, any target, study groups and provenance.
        """
        studyset = normalize_collection(studyset)
        self._validate_options()

        ids = np.asarray(studyset.ids, dtype=str)
        if len(ids) == 0:
            raise ValueError("The Studyset has no analyses to convert.")

        unique_ids, counts = np.unique(ids, return_counts=True)
        if len(unique_ids) != len(ids):
            raise ValueError(
                "Analysis identifiers must be unique, but these repeat: "
                f"{_preview(unique_ids[counts > 1])}."
            )

        study_ids = studyset.metadata["study_id"].to_numpy(dtype=str)
        # The coordinate block is what the kernel transformer reads, so it is also what
        # decides which analyses can have a map at all.
        has_coordinates = studyset.coordinate_block().group_sizes() > 0

        descriptors, descriptor_names = self._read_descriptors(studyset)
        target, target_missing = self._read_target(studyset)

        retained, dropped = self._retained_rows(ids, has_coordinates, descriptors, target_missing)

        studyset_rows = studyset.select_analyses(retained)
        map_features = self._map_matrix(studyset_rows, ids[retained], has_coordinates[retained])

        descriptor_frame = None
        descriptor_matrix = None
        if descriptor_names:
            descriptor_frame = pd.DataFrame(
                {name: values[retained] for name, (values, _) in descriptors.items()},
                index=pd.Index(ids[retained], name="id"),
            )
            descriptor_matrix = np.column_stack(
                [descriptors[name][0][retained].astype(float) for name in descriptor_names]
            )

        return MAFeatureDataset(
            map_features,
            ids=ids[retained],
            study_ids=study_ids[retained],
            descriptor_features=descriptor_matrix,
            descriptor_names=descriptor_names,
            descriptors=descriptor_frame,
            target=None if target is None else target[retained],
            provenance=self._provenance(studyset, ids, retained, dropped, descriptor_names),
            masker=studyset.masker,
        )

    def to_sklearn(self, studyset, return_X_y=False):
        """Convert a Studyset straight into the scikit-learn bundle.

        Parameters
        ----------
        studyset : :class:`~nimare.nimads.Studyset`
            The Studyset to convert.
        return_X_y : :obj:`bool`, default=False
            If True, return ``(data, target)`` instead of a Bunch.

        Returns
        -------
        :class:`sklearn.utils.Bunch` or :obj:`tuple`
            See :meth:`MAFeatureDataset.to_sklearn`.
        """
        return self.transform(studyset).to_sklearn(return_X_y=return_X_y)

    # ------------------------------------------------------------- validation

    def _validate_options(self):
        """Check the option vocabulary before any work is done."""
        if self.missing_coordinates not in ("drop", "include"):
            raise ValueError(
                "missing_coordinates must be 'drop' or 'include', not "
                f"{self.missing_coordinates!r}."
            )
        if self.missing_values not in ("raise", "drop", "keep"):
            raise ValueError(
                "missing_values must be 'raise', 'drop' or 'keep', not "
                f"{self.missing_values!r}."
            )

    # -------------------------------------------------------------- selection

    def _read_descriptors(self, studyset):
        """Return ``({name: (values, kind)}, names)`` for the selected descriptors."""
        selectors = self.descriptor_fields
        if selectors is None:
            return {}, []
        # A tuple is one ``(source, field)`` selector; a list holds several.
        if isinstance(selectors, (str, Mapping, tuple)):
            selectors = [selectors]

        descriptors, names = {}, []
        for selector in selectors:
            source, field = _resolve_selector(studyset, selector, what="descriptor")
            values, kind = _read_field(studyset, source, field)
            if kind != "numeric":
                raise ValueError(
                    f"Descriptor field {field!r} from {source} is {kind}, and the feature "
                    "matrix is numeric. Either encode it yourself and select the numeric "
                    "result, or leave it out of the matrix and encode it inside your "
                    "pipeline from MAFeatureDataset.descriptors, which keeps the encoder "
                    "from being fitted on held-out analyses."
                )
            if field in descriptors:
                raise ValueError(f"Descriptor field {field!r} was selected more than once.")
            descriptors[field] = (values, kind)
            names.append(field)

        return descriptors, names

    def _read_target(self, studyset):
        """Return ``(target values, missing mask)`` for the selected target."""
        if self.target_field is None:
            return None, None

        source, field = _resolve_selector(studyset, self.target_field, what="target")
        values, kind = _read_field(studyset, source, field)
        missing = _missing_mask(values, kind)

        if self.target_transformer is not None:
            values = self._apply_target_transformer(values)
            values = np.asarray(values)
            if values.ndim != 1:
                raise ValueError(
                    f"target_transformer returned {values.ndim}-dimensional values; "
                    "a target must be one value per analysis."
                )
            kind = "numeric" if values.dtype.kind in "fiu" else "categorical"
            missing = _missing_mask(values, kind)
        elif kind == "text":
            raise ValueError(
                f"Target field {field!r} from texts is free text, which has no scalar "
                "reading. Pass target_transformer with a label extractor that turns it "
                "into one value per analysis."
            )
        elif kind == "categorical" and _has_multiple_labels(values):
            raise ValueError(
                f"Target field {field!r} holds several labels per analysis. Pass "
                "target_transformer with a label extractor that chooses one."
            )

        present = values[~missing]
        if len(present) and len(np.unique(present)) == 1:
            raise ValueError(
                f"Target field {field!r} has the single value {present[0]!r} for every "
                "analysis, so there is nothing to predict."
            )

        return values, missing

    def _apply_target_transformer(self, values):
        """Apply the target transformer, whichever shape it has."""
        transformer = self.target_transformer
        if hasattr(transformer, "fit_transform"):
            return transformer.fit_transform(values)
        if callable(transformer):
            return transformer(values)
        raise TypeError(
            "target_transformer must be callable or a transformer with fit_transform, "
            f"not {type(transformer).__name__}."
        )

    def _retained_rows(self, ids, has_coordinates, descriptors, target_missing):
        """Return the retained-row mask and a record of what was dropped."""
        retained = np.ones(len(ids), dtype=bool)
        dropped = {"no_coordinates": [], "missing_values": {}}

        if self.missing_coordinates == "drop":
            retained &= has_coordinates
            dropped["no_coordinates"] = ids[~has_coordinates].tolist()

        missing_by_field = {}
        for name, (values, kind) in descriptors.items():
            missing = _missing_mask(values, kind)
            if missing.any():
                missing_by_field[name] = ids[missing].tolist()
        if target_missing is not None and target_missing.any():
            missing_by_field["<target>"] = ids[target_missing].tolist()

        if missing_by_field:
            if self.missing_values == "raise":
                raise ValueError(
                    "Missing values in "
                    + "; ".join(
                        f"{field} ({len(affected)} analyses: {_preview(affected, 3)})"
                        for field, affected in missing_by_field.items()
                    )
                    + ". Fix the Studyset, or choose missing_values='drop' to remove those "
                    "analyses or missing_values='keep' to impute them in your pipeline."
                )
            if self.missing_values == "drop":
                for affected in missing_by_field.values():
                    retained &= ~np.isin(ids, affected)
            dropped["missing_values"] = missing_by_field

        if not retained.any():
            raise ValueError(
                "No analyses are left after applying missing_coordinates="
                f"{self.missing_coordinates!r} and missing_values={self.missing_values!r}."
            )

        return retained, dropped

    # ------------------------------------------------------------ map features

    def _map_matrix(self, studyset, ids, has_coordinates):
        """Return the analysis-by-voxel matrix, aligned row for row to ``ids``."""
        kernel_transformer = self._resolve_kernel()
        key = self._memo_key(studyset, kernel_transformer) if self.cache_maps else None

        if key is not None and self._map_memo is not None and self._map_memo[0] == key:
            maps = self._map_memo[1]
        else:
            maps = kernel_transformer.transform(studyset, return_type="sparse")
            maps = _as_sparse(maps).tocsr()
            if key is not None:
                self._map_memo = (key, maps)

        return _align_map_rows(maps, ids, has_coordinates)

    def _resolve_kernel(self):
        """Return a kernel transformer instance, with caching wired up if asked."""
        kernel_transformer = self.kernel_transformer
        if isinstance(kernel_transformer, type):
            kernel_transformer = kernel_transformer()

        if _memory_location(self.memory) is None:
            return kernel_transformer
        if _memory_location(getattr(kernel_transformer, "memory", None)) is not None:
            # The kernel already caches somewhere; do not second-guess it.
            return kernel_transformer

        kernel_transformer = copy.deepcopy(kernel_transformer)
        kernel_transformer.memory = (
            self.memory if isinstance(self.memory, Memory) else Memory(location=self.memory)
        )
        kernel_transformer.memory_level = max(int(self.memory_level), 1)
        return kernel_transformer

    @staticmethod
    def _memo_key(studyset, kernel_transformer):
        """Fingerprint everything the generated maps depend on."""
        mask_img = getattr(studyset.masker, "mask_img", None)
        return joblib_hash(
            (
                np.asarray(studyset.ids, dtype=str),
                studyset.coordinates[["id", "x", "y", "z"]].to_numpy(),
                studyset.sample_sizes(),
                None if mask_img is None else (mask_img.shape, mask_img.affine),
                type(kernel_transformer).__module__,
                type(kernel_transformer).__qualname__,
                _kernel_params(kernel_transformer),
            )
        )

    # -------------------------------------------------------------- provenance

    def _provenance(self, studyset, ids, retained, dropped, descriptor_names):
        """Record what this conversion did, for reproducibility."""
        from nimare import __version__

        kernel_transformer = self.kernel_transformer
        kernel_name = (
            kernel_transformer.__name__
            if isinstance(kernel_transformer, type)
            else type(kernel_transformer).__name__
        )

        return {
            "nimare_version": __version__,
            "studyset_id": getattr(studyset, "id", None),
            "studyset_name": getattr(studyset, "name", None),
            "space": getattr(studyset, "space", None),
            "n_analyses": int(len(ids)),
            "n_rows": int(retained.sum()),
            "kernel_transformer": {
                "class": kernel_name,
                "params": _jsonable(_kernel_params(kernel_transformer)),
            },
            "masker": type(studyset.masker).__name__,
            "missing_coordinates": self.missing_coordinates,
            "dropped_ids": dropped["no_coordinates"],
            "missing_values": self.missing_values,
            "missing_value_ids": dropped["missing_values"],
            "descriptor_fields": list(descriptor_names),
            "target_field": None if self.target_field is None else _jsonable(self.target_field),
        }


def _kernel_params(kernel_transformer):
    """Return a kernel transformer's parameters, class or instance."""
    if isinstance(kernel_transformer, type) or not hasattr(kernel_transformer, "get_params"):
        return {}
    return kernel_transformer.get_params()


def _memory_location(memory):
    """Return the location a joblib Memory writes to, or None when it is a no-op."""
    if memory is None:
        return None
    if isinstance(memory, Memory):
        return memory.location
    return str(memory)


def _has_multiple_labels(values):
    """Report whether any value holds more than one label."""
    return any(isinstance(value, (list, tuple, set, np.ndarray)) for value in values)


def _align_map_rows(maps, ids, has_coordinates):
    """Return map rows in ``ids`` order, with all-zero rows where there are no foci.

    Kernel transformers return one row per analysis that has coordinates,
    ordered by analysis id rather than by the Studyset's own row order, and
    ``return_type="sparse"`` drops the ids that name them. Rebuilding the id
    order here is what keeps every row's map with its own target and study;
    pairing the two by position only works while the Studyset happens to be in
    sorted order.
    """
    map_ids = np.sort(ids[has_coordinates])

    if len(map_ids) != maps.shape[0]:
        raise ValueError(
            f"The kernel transformer returned {maps.shape[0]} maps for {len(map_ids)} "
            "analyses with coordinates. Map features cannot be aligned to analyses."
        )

    # One extra all-zero row that every coordinate-less analysis points at.
    padded = sparse.vstack(
        [maps, sparse.csr_matrix((1, maps.shape[1]), dtype=maps.dtype)], format="csr"
    )
    rows = np.full(len(ids), maps.shape[0], dtype=int)
    rows[has_coordinates] = np.searchsorted(map_ids, ids[has_coordinates])

    return padded[rows].tocsr()


# --------------------------------------------------------------- reduction


class AtlasAggregator(TransformerMixin, BaseEstimator):
    """Aggregate masked voxel features into atlas regions.

    Turns each row back into an image in the source masker's space and lets a
    nilearn masker summarise it, so region definitions, resampling and the
    aggregation strategy stay nilearn's business.

    Parameters
    ----------
    atlas_masker : :class:`nilearn.maskers.NiftiLabelsMasker` or \
:class:`nilearn.maskers.NiftiMapsMasker`
        The masker defining the regions. It is cloned and fitted in the source
        mask's space, so the instance passed in is left alone.
    masker : :class:`nilearn.maskers.NiftiMasker`, optional
        The masker defining the voxel order of the incoming features, normally
        :attr:`MAFeatureDataset.masker`, by default None.
    batch_size : :obj:`int`, default=10
        How many rows are held in dense image form at once. Ten rows of a 2 mm
        whole-brain mask is roughly 18 MB.

    Examples
    --------
    >>> reducer = AtlasAggregator(  # doctest: +SKIP
    ...     atlas_masker=NiftiLabelsMasker(labels_img=atlas.maps),
    ...     masker=dataset.masker,
    ... )
    """

    def __init__(self, atlas_masker=None, masker=None, batch_size=10):
        self.atlas_masker = atlas_masker
        self.masker = masker
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Fit the atlas masker in the source mask's space.

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
        if self.atlas_masker is None:
            raise ValueError("AtlasAggregator requires an atlas_masker.")
        if self.masker is None:
            raise ValueError(
                "AtlasAggregator requires the masker that defines the voxel order of the "
                "features, normally MAFeatureDataset.masker."
            )

        self.mask_img_ = self.masker.mask_img
        atlas_masker = clone(self.atlas_masker)
        atlas_masker.set_params(mask_img=self.mask_img_)
        self.atlas_masker_ = atlas_masker.fit(self.mask_img_)
        self.n_features_in_ = X.shape[1]
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

        return np.vstack(batches)

    def get_feature_names_out(self, input_features=None):
        """Return the region names, as the atlas masker reports them.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        :obj:`numpy.ndarray` of :obj:`str`
            One name per region column.
        """
        check_is_fitted(self, ["atlas_masker_"])
        names = _region_names(self.atlas_masker_)
        if names is None:
            names = [f"region_{idx}" for idx in range(self._n_regions())]
        return np.asarray(names, dtype=str)

    def _n_regions(self):
        """Return how many regions the fitted atlas masker produces."""
        maps_img = getattr(self.atlas_masker_, "maps_img_", None)
        if maps_img is not None:
            return maps_img.shape[-1]
        labels = list(getattr(self.atlas_masker_, "labels_", []) or [])
        background = getattr(self.atlas_masker_, "background_label", 0)
        return len([label for label in labels if label != background])


def _region_names(atlas_masker):
    """Return the region names a fitted nilearn masker reports, or None."""
    try:
        names = atlas_masker.get_feature_names_out()
    except (AttributeError, NotFittedError, ValueError, TypeError):
        names = None

    # nilearn hands back a zero-dimensional array wrapping ``dict_values``.
    if isinstance(names, np.ndarray) and names.ndim == 0:
        names = names.item()
    if names is not None:
        names = [str(name) for name in names]
        if names:
            return names

    region_names = getattr(atlas_masker, "region_names_", None)
    if isinstance(region_names, Mapping) and region_names:
        return [str(name) for name in region_names.values()]

    return None


def make_map_reducer(method, masker=None, **kwargs):
    """Build a reduction workflow for voxelwise map features.

    All three workflows are ordinary scikit-learn transformers: put one in a
    pipeline, or fit it on a training dataset with
    :meth:`MAFeatureDataset.fit_transform_maps`.

    Parameters
    ----------
    method : {"variance_threshold", "truncated_svd", "atlas_aggregation"}
        ``"variance_threshold"`` drops map features that barely vary and keeps
        the matrix sparse; ``"truncated_svd"`` is a sparse-compatible low-rank
        decomposition; ``"atlas_aggregation"`` summarises each map over the
        regions of an atlas.
    masker : :class:`nilearn.maskers.NiftiMasker`, optional
        The masker defining the voxel order of the features, by default None.
        Required for ``"atlas_aggregation"``.
    **kwargs
        Passed to the underlying transformer, for example ``n_components`` for
        ``"truncated_svd"`` or ``atlas_masker`` for ``"atlas_aggregation"``.

    Returns
    -------
    estimator
        An unfitted scikit-learn transformer.

    Raises
    ------
    :obj:`ValueError`
        If ``method`` is not one of the supported workflows, or if atlas
        aggregation is requested without the source masker.

    See Also
    --------
    sklearn.feature_selection.VarianceThreshold
    sklearn.decomposition.TruncatedSVD
    AtlasAggregator
    """
    if method == "variance_threshold":
        return VarianceThreshold(**kwargs)

    if method == "truncated_svd":
        return TruncatedSVD(**kwargs)

    if method == "atlas_aggregation":
        if masker is None:
            raise ValueError(
                "Atlas aggregation needs the source masker that defines the voxel order "
                "of the map features, normally MAFeatureDataset.masker."
            )
        return AtlasAggregator(masker=masker, **kwargs)

    raise ValueError(
        f"Unknown map reducer {method!r}. Supported workflows are "
        f"{', '.join(MAP_REDUCERS)}, or pass a scikit-learn transformer directly."
    )
