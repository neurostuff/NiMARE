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
:meth:`FeatureSet.make_preprocessor` builds the piece that goes in it.

The public surface is :meth:`FeatureSet.from_studyset`, which builds the
container, and :class:`AtlasAggregator`, which reduces its voxelwise features
over the regions of an atlas. Every other reduction is an ordinary
scikit-learn transformer, used as scikit-learn documents it.
"""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory
from nibabel.spatialimages import SpatialImage
from nilearn.image import load_img
from nilearn.maskers import BaseMasker, NiftiLabelsMasker, NiftiMapsMasker
from nilearn.masking import unmask
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from nimare.base import NiMAREBase
from nimare.studyset import normalize_collection
from nimare.studyset.columns import ID_COLS
from nimare.studyset.requirements import PerAnalysis

LGR = logging.getLogger(__name__)

__all__ = [
    "AtlasAggregator",
    "FeatureSet",
]

#: Studyset tables a descriptor or target field may be selected from.
FIELD_SOURCES = ("metadata", "annotations", "texts")

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


def _source_attribute(source):
    """Return the Studyset attribute a source's raw values are read from."""
    return "annotations_df" if source == "annotations" else source


def _is_pattern(field):
    """Report whether a field selector names a set of labels rather than one field."""
    return any(character in field for character in "*?[")


def _matching_labels(studyset, pattern):
    """Return the annotation labels matching ``pattern``, in the Studyset's order."""
    return [
        label for label in _selectable_columns(studyset, "annotations") if fnmatch(label, pattern)
    ]


def _resolve_selector(studyset, selector, what="descriptor"):
    """Return ``(source, field)``, inferring the source from a bare field name."""
    source, field = _as_selector(selector)

    if source is not None and _is_pattern(field):
        if source != "annotations":
            raise ValueError(
                f"A pattern selects annotation labels, so {field!r} cannot come from "
                f"{source}. Name a {source} field exactly."
            )
        if not _matching_labels(studyset, field):
            raise ValueError(
                f"{what.capitalize()} pattern {field!r} matches no annotation label. This "
                f"Studyset annotates with "
                f"{_preview(_selectable_columns(studyset, 'annotations'))}."
            )
        return source, field

    if source is not None:
        if field not in _selectable_columns(studyset, source):
            raise ValueError(
                f"{what.capitalize()} field {field!r} was not found in the Studyset "
                f"{source}. Available {source} fields: "
                f"{_preview(_selectable_columns(studyset, source))}."
            )
        return source, field

    if _is_pattern(field):
        if not _matching_labels(studyset, field):
            raise ValueError(
                f"{what.capitalize()} pattern {field!r} matches no annotation label. This "
                f"Studyset annotates with "
                f"{_preview(_selectable_columns(studyset, 'annotations'))}."
            )
        return "annotations", field

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


class _DescriptorBlock(NamedTuple):
    """One descriptor selection: its column names, its values, what it lacks."""

    names: list
    values: Any  # (n_rows, n_names), dense for a field and sparse for labels
    missing: Any  # boolean mask over rows, or None where absence means zero


def _read_labels(studyset, pattern):
    """Return ``(names, sparse values)`` for the annotation labels a pattern names.

    Read from the Studyset's :class:`~nimare.studyset.blocks.LabelBlock`, which
    is the sparse form the annotation is stored in: the Neurosynth release
    annotates 115,747 analyses with 794 labels, and a dense read of that is
    92 million cells holding 3 million values.
    """
    block = studyset.label_block()
    names = _matching_labels(studyset, pattern)
    columns = [block.col(name) for name in names]
    return names, sparse.csc_matrix(block.values)[:, columns].tocsr()


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


def _to_dense(block):
    """Return a dense view of a feature block."""
    return block.toarray() if sparse.issparse(block) else block


def _dense_step(transformer):
    """Wrap a descriptor transformer so that it is handed dense columns.

    The descriptor block is stored dense and is a handful of numeric columns,
    but it arrives here sparse because it sits beside the voxels in one matrix.
    Most scikit-learn transformers expect dense input for this kind of column --
    ``StandardScaler`` refuses to centre sparse data at all -- so the block is
    densified for them. The map block stays sparse.
    """
    if isinstance(transformer, str):
        return transformer
    return Pipeline(
        [
            ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("transform", transformer),
        ]
    )


def _is_passthrough(transformer):
    """Report whether a transformer argument asks for nothing to be done."""
    return isinstance(transformer, str) and transformer == "passthrough"


def _step_name(name, position, used):
    """Return a ColumnTransformer step name, which may not contain ``__``.

    Annotation fields often do -- ``Neurosynth_TFIDF__pain`` -- so the name is
    softened, and disambiguated by position if softening made it collide.
    """
    safe = name.replace("__", "_") or f"descriptor_{position}"
    if safe in used:
        safe = f"{safe}_{position}"
    used.add(safe)
    return safe


def _hstack_blocks(blocks):
    """Stack descriptor blocks side by side, staying sparse if any of them is."""
    if any(sparse.issparse(block) for block in blocks):
        return sparse.hstack([_as_sparse(block) for block in blocks], format="csr")
    return np.hstack(blocks)


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


class FeatureSet(NiMAREBase):
    """Aligned modeled activation features, descriptors, target and provenance.

    Build one from a Studyset with :meth:`from_studyset`. The constructor below
    takes the blocks directly, which is how :meth:`split`,
    :meth:`select_analyses`, :meth:`copy` and the map-reduction methods build
    their results, and what to use for features generated some other way.

    Every array here shares one row order: row ``i`` is analysis ``ids[i]`` from
    study ``study_ids[i]``. That order is fixed when the feature set is built
    and is preserved by :meth:`split`, :meth:`select_analyses` and the
    map-reduction methods.

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
    AtlasAggregator : Reduce the voxelwise features over the regions of an atlas.
    """

    def __init__(
        self,
        map_features: Any,
        ids: Sequence[str],
        study_ids: Sequence[str],
        *,
        descriptor_features: Any | None = None,
        descriptor_names: Sequence[str] | None = None,
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
            if not sparse.issparse(descriptor_features):
                descriptor_features = np.asarray(descriptor_features)
            if len(descriptor_features.shape) != 2:
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
        self.target = None if target is None else np.asarray(target)
        self.provenance = {} if provenance is None else provenance
        self.masker = masker

    @classmethod
    def from_studyset(
        cls,
        studyset,
        kernel_transformer,
        *,
        descriptor_fields=None,
        target_field=None,
        target_transformer=None,
        missing_coordinates="drop",
        missing_values="raise",
        memory=None,
        memory_level=2,
    ):
        """Build a feature set from a Studyset.

        Generates one modeled activation (MA) map per analysis through the
        kernel transformer, appends any numeric descriptor fields, extracts any
        target, and returns them aligned to the analyses they came from, with
        the study labels that keep analyses from one study out of two different
        partitions.

        The work happens here rather than in ``__init__``, which stays a plain
        data constructor: it is also how :meth:`split`, :meth:`select_analyses`
        and the map-reduction methods build their results, from blocks that
        already exist.

        Parameters
        ----------
        studyset : :class:`~nimare.nimads.Studyset`
            The Studyset to convert. One analysis becomes one row.
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
        memory : :class:`joblib.Memory`, :obj:`str` or :class:`pathlib.Path`, optional
            Cache location for MA map generation, by default None. Repeated calls
            over the same Studyset then reuse the maps instead of regenerating
            them, across processes as well as within one. Used only when the kernel
            transformer does not define its own cache; the kernel's own ``memory``
            always wins.
        memory_level : :obj:`int`, default=2
            How eagerly ``memory`` caches, following the NiMARE convention. Kernel
            transformers cache their maps at level 2, which is why that is the
            default here; a lower level asks for them not to be cached.

        Returns
        -------
        :class:`FeatureSet`
            One row per retained analysis, with map features, any descriptor
            features, any target, study groups and provenance. Call
            :meth:`to_sklearn` for the scikit-learn bundle.

        Notes
        -----
        Descriptor fields must be numeric, because the exported feature matrix
        is numeric. A categorical or text field raises and says where its raw
        values are -- ``studyset.metadata``, ``studyset.annotations_df`` or
        ``studyset.texts`` -- so that you can encode them and select the numeric
        result. Encoding here would fit the encoder on every analysis, including
        the ones you are about to hold out.

        An annotation is selected a label at a time by name, or many at a time
        by pattern: ``("annotations", "Neurosynth_TFIDF__*")`` takes every
        matching label, under its own name, and keeps the block sparse, which
        matters when an annotation runs to thousands of labels. A label no
        analysis carries is a zero rather than a gap, so ``missing_values`` has
        nothing to report about one.

        Generating the maps does not leak: an analysis's MA map is a function of
        that analysis's own foci, so it never sees ``y`` or another row. Everything
        that learns *across* rows belongs in a
        :class:`~sklearn.pipeline.Pipeline`, which :meth:`make_preprocessor`
        builds the piece for.

        Examples
        --------
        >>> features = FeatureSet.from_studyset(  # doctest: +SKIP
        ...     studyset,
        ...     kernel_transformer=MKDAKernel(r=10),
        ...     target_field=("metadata", "comparison_task"),
        ... )
        >>> train, test = features.split(test_size=0.25, random_state=13)  # doctest: +SKIP
        >>> bunch = features.to_sklearn()  # doctest: +SKIP

        See Also
        --------
        AtlasAggregator : Reduce the map features over the regions of an atlas.
        """
        return _FeatureExtractor(
            kernel_transformer=kernel_transformer,
            descriptor_fields=descriptor_fields,
            target_field=target_field,
            target_transformer=target_transformer,
            missing_coordinates=missing_coordinates,
            missing_values=missing_values,
            memory=memory,
            memory_level=memory_level,
        ).transform(studyset, container=cls)

    def __repr__(self):
        """Show the shape, and whether any columns are descriptors."""
        n_rows, n_features = self.shape
        descriptors = ""
        if self._descriptor_features is not None:
            descriptors = f", n_descriptors={self._descriptor_features.shape[1]}"
        return (
            f"{self.__class__.__name__}(n_rows={n_rows}, n_features={n_features}, "
            f"n_studies={len(np.unique(self.study_ids))}{descriptors})"
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
    def descriptor_names(self):
        """:obj:`list` of :obj:`str`: the descriptor columns, in order."""
        return list(self._descriptor_names or [])

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
        (:class:`FeatureSet`, :class:`FeatureSet`)
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
        map_reducer,
        descriptor_transformer="passthrough",
        **reducer_params,
    ):
        """Apply a reducer to the map columns and something else to the rest.

        This is :class:`~sklearn.compose.ColumnTransformer`, with the column
        boundary filled in, the masker bound into an atlas reducer, and
        ``sparse_threshold=1.0`` so that a map block denser than scikit-learn's
        default threshold is not quietly densified.

        When there are no descriptor columns there is nothing to keep the
        reducer away from, so the reducer is returned as it is: put a
        scikit-learn transformer straight into your pipeline and this method is
        not needed at all.

        Which is also the rule for when it *is* needed. A transformer placed
        directly in a pipeline sees every column it is given, so once there are
        descriptor columns a bare reducer decomposes them along with the
        voxels, quietly. Going through this method costs nothing on a map-only
        feature set and keeps a pipeline correct if descriptor fields are added
        later.

        Parameters
        ----------
        map_reducer : estimator, :obj:`type`, atlas or None
            A scikit-learn transformer, a transformer class built here from
            ``**reducer_params``, any atlas :class:`AtlasAggregator` accepts
            (this feature set supplies the voxel order), or None to leave the
            map columns alone.
        descriptor_transformer : estimator, :obj:`str` or :obj:`dict`, default="passthrough"
            What to apply to the descriptor columns: one transformer for all of
            them, for example :class:`~sklearn.impute.SimpleImputer` when
            descriptors were kept with missing values, or a mapping from
            descriptor name to transformer when they need different treatment.
            Descriptors the mapping does not name are passed through, and the
            column order is the one they came in with. Transformers are handed
            the descriptor columns dense, since that is what most of them
            expect of a handful of numeric columns; the map block stays sparse.
        **reducer_params
            Passed to ``map_reducer`` when it is a class.

        Returns
        -------
        estimator
            A :class:`~sklearn.compose.ColumnTransformer` when there are
            descriptor columns, and the reducer itself when there are not.

        Examples
        --------
        >>> pipeline = make_pipeline(  # doctest: +SKIP
        ...     features.make_preprocessor(TruncatedSVD(n_components=50)),
        ...     LogisticRegression(),
        ... )
        >>> pipeline = make_pipeline(  # doctest: +SKIP
        ...     features.make_preprocessor(fetch_atlas_difumo(dimension=64)),
        ...     LogisticRegression(),
        ... )
        >>> preprocessor = features.make_preprocessor(  # doctest: +SKIP
        ...     TruncatedSVD(n_components=50),
        ...     descriptor_transformer={
        ...         "sample_sizes": SimpleImputer(strategy="median"),
        ...         "year": StandardScaler(),
        ...     },
        ... )

        Notes
        -----
        Written out, the two-block case is the ordinary scikit-learn recipe,
        and :attr:`map_columns` and :attr:`descriptor_columns` are public so
        that you can write it yourself::

            ColumnTransformer(
                [
                    ("maps", TruncatedSVD(n_components=50), features.map_columns),
                    ("descriptors", SimpleImputer(), features.descriptor_columns),
                ],
                sparse_threshold=1.0,
            )
        """
        if map_reducer is None or (isinstance(map_reducer, str) and map_reducer == "passthrough"):
            reducer = "passthrough"
        else:
            reducer = _resolve_map_reducer(map_reducer, masker=self.masker, **reducer_params)

        if self._descriptor_features is None:
            if not _is_passthrough(descriptor_transformer):
                raise ValueError(
                    "This feature set has no descriptor columns, so there is nothing for "
                    "descriptor_transformer to act on."
                )
            return reducer

        return ColumnTransformer(
            [
                ("maps", reducer, self.map_columns),
                (
                    "descriptors",
                    self._descriptor_step(descriptor_transformer),
                    self.descriptor_columns,
                ),
            ],
            sparse_threshold=1.0,
        )

    def _descriptor_step(self, descriptor_transformer):
        """Return the step applied to the descriptor block."""
        if not isinstance(descriptor_transformer, Mapping):
            return _dense_step(descriptor_transformer)

        names = self.descriptor_names
        unknown = [name for name in descriptor_transformer if name not in names]
        if unknown:
            raise ValueError(
                f"No descriptor called {_preview(unknown)}. This feature set has "
                f"{_preview(names)}."
            )

        # Column indices here are relative to the descriptor block, and every
        # descriptor gets a step of its own so the columns come out in the order they
        # went in rather than in the order the mapping happened to name them.
        used = set()
        steps = []
        for position, name in enumerate(names):
            steps.append(
                (
                    _step_name(name, position, used),
                    _dense_step(descriptor_transformer.get(name, "passthrough")),
                    [position],
                )
            )

        return ColumnTransformer(steps, sparse_threshold=1.0)

    # -------------------------------------------------------- map reduction

    def fit_transform_maps(self, reducer):
        """Fit a reducer on this dataset's map features and apply it.

        Fit it on the training dataset only, then pass the same, now fitted,
        reducer to :meth:`transform_maps` for held-out data.

        Parameters
        ----------
        reducer : estimator
            A scikit-learn transformer, or an :class:`AtlasAggregator` built
            with this feature set's masker.

        Returns
        -------
        :class:`FeatureSet`
            A feature set with reduced map features and everything else
            unchanged.
        """
        if _is_atlas_like(reducer):
            raise TypeError(
                "Build the aggregator first, as AtlasAggregator(atlas, "
                "masker=features.masker), so that the fitted one can be reused on "
                "held-out rows with transform_maps."
            )
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
        :class:`FeatureSet`
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
            # str(), because numpy's str_ prints as np.str_('...') in numpy 2.
            names = [str(name) for name in reducer.get_feature_names_out()]
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
        :class:`FeatureSet`
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

        return self._rebuild(
            map_features=_take_rows(self._map_features, rows),
            descriptor_features=_take_rows(self._descriptor_features, rows),
            ids=self.ids[rows],
            study_ids=self.study_ids[rows],
            target=None if self.target is None else self.target[rows],
        )

    def copy(self):
        """Return an independent copy of the dataset.

        Returns
        -------
        :class:`FeatureSet`
            A copy that shares no mutable state with this dataset.
        """
        return self._rebuild(
            map_features=self._map_features.copy(),
            descriptor_features=(
                None if self._descriptor_features is None else self._descriptor_features.copy()
            ),
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
            "target": self.target,
            "provenance": self.provenance,
            "masker": self.masker,
            "map_feature_names": self._map_feature_names,
        }
        kwargs.update(changes)
        positional = (kwargs.pop("map_features"), kwargs.pop("ids"), kwargs.pop("study_ids"))
        return FeatureSet(*positional, **kwargs)


# ------------------------------------------------------------- the extractor


class _FeatureExtractor(NiMAREBase):
    """Carry out one conversion from a Studyset to a :class:`FeatureSet`.

    Internal. :meth:`FeatureSet.from_studyset` is the public entry point and
    documents the parameters. The class exists so that the stages of one conversion --
    field selection, target handling, row retention, map generation,
    provenance -- stay separate methods over shared configuration, rather than
    one long function threading nine arguments through itself.
    """

    def __init__(
        self,
        kernel_transformer: Any,
        descriptor_fields: Sequence[Any] | None = None,
        target_field: Any | None = None,
        target_transformer: Any | None = None,
        missing_coordinates: str = "drop",
        missing_values: str = "raise",
        memory: Any = None,
        memory_level: int = 2,
    ):
        self.kernel_transformer = kernel_transformer
        self.descriptor_fields = descriptor_fields
        self.target_field = target_field
        self.target_transformer = target_transformer
        self.missing_coordinates = missing_coordinates
        self.missing_values = missing_values
        self.memory = memory
        self.memory_level = memory_level

    # ------------------------------------------------------------- public API

    def transform(self, studyset, container=None):
        """Convert a Studyset into ``container``, by default a :class:`FeatureSet`.

        Parameters
        ----------
        studyset : :class:`~nimare.nimads.Studyset`
            The Studyset to convert.
        container : :obj:`type`, optional
            The class to build, so that a subclass calling
            :meth:`FeatureSet.from_studyset` gets its own type back.

        Returns
        -------
        :class:`FeatureSet`
            One row per retained analysis, with map features, any descriptor
            features, any target, study groups and provenance.
        """
        container = FeatureSet if container is None else container
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

        blocks = self._read_descriptors(studyset)
        target, target_missing = self._read_target(studyset)

        retained, dropped = self._retained_rows(ids, has_coordinates, blocks, target_missing)

        studyset_rows = studyset.select_analyses(retained)
        map_features = self._map_matrix(studyset_rows, ids[retained], has_coordinates[retained])

        descriptor_names = [name for block in blocks for name in block.names]
        descriptor_matrix = None
        if blocks:
            kept = [_take_rows(block.values, retained) for block in blocks]
            descriptor_matrix = kept[0] if len(kept) == 1 else _hstack_blocks(kept)

        return container(
            map_features,
            ids=ids[retained],
            study_ids=study_ids[retained],
            descriptor_features=descriptor_matrix,
            descriptor_names=descriptor_names,
            target=None if target is None else target[retained],
            provenance=self._provenance(studyset, ids, retained, dropped, descriptor_names),
            masker=studyset.masker,
        )

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
        """Return one :class:`_DescriptorBlock` per selector."""
        selectors = self.descriptor_fields
        if selectors is None:
            return []
        # A tuple is one ``(source, field)`` selector; a list holds several.
        if isinstance(selectors, (str, Mapping, tuple)):
            selectors = [selectors]

        blocks, seen = [], set()
        for selector in selectors:
            source, field = _resolve_selector(studyset, selector, what="descriptor")

            if _is_pattern(field):
                names, values = _read_labels(studyset, field)
                # An annotation is sparse by nature: a label no analysis carries is a
                # zero, not a gap, so there is nothing for missing_values to report.
                block = _DescriptorBlock(names, values, None)
            else:
                values, kind = _read_field(studyset, source, field)
                if kind != "numeric":
                    raise ValueError(
                        f"Descriptor field {field!r} from {source} is {kind}, and the "
                        "feature matrix is numeric. Encode it yourself -- its raw "
                        f"values are in studyset.{_source_attribute(source)} -- and "
                        "select the numeric result."
                    )
                block = _DescriptorBlock(
                    [field],
                    np.asarray(values, dtype=float).reshape(-1, 1),
                    _missing_mask(values, kind),
                )

            repeated = [name for name in block.names if name in seen]
            if repeated:
                raise ValueError(
                    f"Descriptor field {_preview(repeated)} was selected more than once."
                )
            seen.update(block.names)
            blocks.append(block)

        return blocks

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

    def _retained_rows(self, ids, has_coordinates, blocks, target_missing):
        """Return the retained-row mask and a record of what was dropped."""
        retained = np.ones(len(ids), dtype=bool)
        dropped = {"no_coordinates": [], "missing_values": {}}

        if self.missing_coordinates == "drop":
            retained &= has_coordinates
            dropped["no_coordinates"] = ids[~has_coordinates].tolist()

        missing_by_field = {}
        for block in blocks:
            if block.missing is not None and block.missing.any():
                missing_by_field[block.names[0]] = ids[block.missing].tolist()
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
        maps = self._resolve_kernel().transform(studyset, return_type="sparse")
        return _align_map_rows(_as_sparse(maps).tocsr(), ids, has_coordinates)

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
            self.memory
            if isinstance(self.memory, Memory)
            else Memory(location=self.memory, verbose=0)
        )
        # Passed through as given: a kernel transformer caches its maps at level 2, so a
        # lower level is a request not to cache them.
        kernel_transformer.memory_level = int(self.memory_level)
        return kernel_transformer

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
            "descriptor_fields": _jsonable(self.descriptor_fields),
            "n_descriptor_features": int(len(descriptor_names)),
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
    """Aggregate masked voxel features into the regions of a nilearn atlas.

    Takes any atlas nilearn can load -- a fetched atlas, an image, a file, the
    name of a nilearn fetcher, or a masker you built yourself -- and turns each
    row of map features into one value per region. Rows are converted back into
    images in the source mask's space in batches and summarised by a nilearn
    masker, so region definitions, resampling and the aggregation strategy stay
    nilearn's business.

    Parameters
    ----------
    atlas : object, optional
        The atlas, in any of these forms, by default None:

        - a :class:`~sklearn.utils.Bunch` from a ``nilearn.datasets.fetch_atlas_*``
          function, whose ``maps`` and ``labels`` are read;
        - a 3D (deterministic) or 4D (probabilistic) atlas image, or a path to one;
        - the name of a nilearn fetcher, such as ``"harvard_oxford"``, with any
          arguments it needs in ``atlas_kwargs``;
        - a fitted or unfitted :class:`~nilearn.maskers.NiftiLabelsMasker` or
          :class:`~nilearn.maskers.NiftiMapsMasker`, when the defaults chosen
          here are not the ones you want. It is cloned, never modified.

        A 4D atlas is summarised with a :class:`~nilearn.maskers.NiftiMapsMasker`
        and a 3D one with a :class:`~nilearn.maskers.NiftiLabelsMasker`, both
        with ``resampling_target="data"``.
    masker : :class:`~nilearn.maskers.NiftiMasker` or img_like, optional
        The masker defining the voxel order of the incoming features, normally
        :attr:`FeatureSet.masker`, by default None.
    atlas_kwargs : :obj:`dict`, optional
        Arguments for the nilearn fetcher when ``atlas`` names one, by default
        None.
    batch_size : :obj:`int`, default=10
        How many rows are held in dense image form at once. Ten rows of a 2 mm
        whole-brain mask is roughly 18 MB; larger batches use more memory and
        call nilearn fewer times.

    Attributes
    ----------
    atlas_masker_ : :class:`~nilearn.maskers.BaseMasker`
        The fitted nilearn masker doing the aggregation.
    region_names_ : :obj:`list` of :obj:`str` or None
        Region names read from the atlas, when it carries any.
    n_features_out_ : :obj:`int`
        How many regions the fitted masker actually reports, known once
        anything has been transformed or named.

    Notes
    -----
    How many regions an atlas yields depends on the nilearn version as well as
    on the atlas: regions that fall outside the mask are kept by nilearn 0.12
    and dropped by 0.13. The region count and names reported here follow
    whichever nilearn is installed, so a feature matrix is comparable across
    environments only when the nilearn version is.

    Examples
    --------
    >>> from nilearn.datasets import fetch_atlas_difumo  # doctest: +SKIP
    >>> reducer = AtlasAggregator(  # doctest: +SKIP
    ...     atlas=fetch_atlas_difumo(dimension=64),
    ...     masker=dataset.masker,
    ... )
    """

    def __init__(self, atlas=None, masker=None, atlas_kwargs=None, batch_size=10):
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
        0.13, where a region that falls outside the mask is dropped from the
        output but not from either of them. One all-zero row costs a single
        masker call and is exact.
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
    """Return the region names a fetched nilearn atlas carries, or None."""
    labels = getattr(atlas, "labels", None)
    if labels is None:
        return None

    if hasattr(labels, "columns"):
        # A frame of region attributes, as DiFuMo returns.
        named = [column for column in labels.columns if "name" in str(column).lower()]
        if not named:
            named = [
                column
                for column in labels.columns
                if pd.api.types.is_object_dtype(labels[column])
                or pd.api.types.is_string_dtype(labels[column])
            ]
        if not named:
            return None
        labels = labels[named[0]].tolist()
    elif getattr(getattr(labels, "dtype", None), "names", None):
        fields = [field for field in labels.dtype.names if "name" in field.lower()]
        labels = labels[(fields or list(labels.dtype.names))[0]].tolist()

    names = [str(label) for label in labels]
    return names or None


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
