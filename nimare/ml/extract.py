"""Reading a Studyset into the bundle :meth:`~nimare.studyset.Studyset.to_bunch` returns."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import Bunch

from nimare.base import NiMAREBase
from nimare.ml._helpers import (
    _as_sparse,
    _hstack,
    _hstack_blocks,
    _jsonable,
    _missing_mask,
    _preview,
    _take_rows,
)
from nimare.studyset import normalize_collection
from nimare.studyset.blocks import label_block_for
from nimare.studyset.columns import ID_COLS
from nimare.studyset.frames import _is_numeric
from nimare.studyset.requirements import PerAnalysis

FIELD_SOURCES = ("metadata", "annotations", "texts")

FIELD_REPORT_COLUMNS = ("source", "field", "kind", "coverage", "n_unique", "example")

TARGET_KEY = "<target>"

MISSING_POLICIES = ("raise", "drop", "keep")

MISSING_ROLES = ("target", "descriptors")


_SOURCE_ALIASES = {
    "annotations_df": "annotations",
    "annotation": "annotations",
    "text": "texts",
    "metadata": "metadata",
    "annotations": "annotations",
    "texts": "texts",
}


def _as_selector(selector):
    """Return ``(source, field)`` for a field selector, with ``source`` optional.

    A selector is a bare field name, or a ``(source, field)`` pair matching the
    ``(kind, field)`` vocabulary NiMARE estimators already use in
    ``_required_inputs``. The pair is needed only when a bare name is ambiguous,
    which the resolver says when it is.
    """
    source, field = None, None
    if isinstance(selector, str):
        field = selector
    elif isinstance(selector, Sequence) and len(selector) == 2:
        source, field = selector
    else:
        raise TypeError(
            f"Field selector {selector!r} must be a field name or a (source, field) pair."
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


class _Fields:
    """The fields a Studyset offers a selector, and how to read them.

    Annotation labels come from the sparse label block, which is also where
    their values come from: one source of truth for what a label is called,
    whether it is numeric, and what it holds.
    """

    def __init__(self, studyset):
        self._studyset = studyset
        self._frames = {}
        self._label_block = None
        self._entries = {}

    # ------------------------------------------------------------ what exists
    @property
    def label_block(self):
        """:class:`~nimare.studyset.blocks.LabelBlock` or None: every annotation's labels."""
        if self._label_block is None and self._studyset.annotations:
            self._label_block = label_block_for(self._studyset.view)
        return self._label_block

    def frame(self, source):
        """Return the Studyset table a source names."""
        if source not in self._frames:
            self._frames[source] = getattr(
                self._studyset, "annotations_df" if source == "annotations" else source
            )
        return self._frames[source]

    def names(self, source):
        """Return the fields a selector may name in one source."""
        if source == "annotations":
            block = self.label_block
            return [] if block is None else [str(label) for label in block.labels]
        return [column for column in self.frame(source).columns if column not in ID_COLS]

    def matching(self, pattern):
        """Return the annotation labels a pattern names, in the Studyset's order.

        A pattern that matches nothing is retried with its brackets escaped,
        because an extractor names its repeated fields ``groups[0].count`` and
        a bracket is a character class to :mod:`fnmatch`.
        """
        labels = self.names("annotations")
        hits = [label for label in labels if fnmatch(label, pattern)]
        if hits or "[" not in pattern:
            return hits
        literal = _escape_brackets(pattern)
        return [label for label in labels if fnmatch(label, literal)]

    # ------------------------------------------------------------- resolution
    def resolve(self, selector, what):
        """Return ``(source, field, is_pattern)`` for one selector."""
        source, field = _as_selector(selector)
        sources = FIELD_SOURCES if source is None else (source,)

        # An exact name wins over pattern matching, so that a label called
        # ``groups[0].BMI`` is selectable at all.
        exact = [name for name in sources if field in set(self.names(name))]
        if len(exact) > 1:
            raise ValueError(
                f"{what.capitalize()} field {field!r} is ambiguous: it appears in "
                f"{', '.join(exact)}. Name the source explicitly, for example "
                f"('{exact[0]}', '{field}')."
            )
        if exact:
            return exact[0], field, False

        if _looks_like_pattern(field):
            if source not in (None, "annotations"):
                raise ValueError(
                    f"A pattern selects annotation labels, so {field!r} cannot come from "
                    f"{source}. Name a {source} field exactly."
                )
            if self.matching(field):
                return "annotations", field, True
            raise ValueError(
                f"{what.capitalize()} pattern {field!r} matches no annotation label. This "
                f"Studyset annotates with {_preview(self.names('annotations'))}."
            )

        if source is not None:
            raise ValueError(
                f"{what.capitalize()} field {field!r} was not found in the Studyset "
                f"{source}. Available {source} fields: {_preview(self.names(source))}."
            )
        raise ValueError(
            f"{what.capitalize()} field {field!r} was not found in the Studyset metadata, "
            "annotations or texts. Name the source explicitly with a (source, field) "
            "selector if it should be there."
        )

    # ---------------------------------------------------------------- reading
    def value(self, source, field):
        """Return ``(values aligned to the analyses, kind)`` for one field.

        ``kind`` is ``"numeric"``, ``"categorical"`` or ``"text"``, and an
        absent value is missing, which is what a named field means.
        """
        if source == "texts":
            return self.frame(source)[field].to_numpy(dtype=object), "text"

        if source == "annotations":
            columns = self._annotation_store(field)
            if columns is None or _is_numeric(columns.entries(field)[1]):
                return self._annotation_values(field, columns), "numeric"
            return columns.get(field, sel=self._studyset.view.index), "categorical"

        # The frame, rather than PerAnalysis, because it merges study-level metadata
        # into the analyses that inherit it even when a sibling declares its own.
        raw = self.frame(source)[field]
        present = raw.notna()
        numbers = pd.to_numeric(raw, errors="coerce")
        if not present.any() or numbers[present].notna().all():
            return numbers.to_numpy(dtype=float), "numeric"

        # Not numbers as they stand, which is what a list of per-group sample sizes
        # looks like; PerAnalysis reduces those the way the rest of NiMARE does.
        reduced = PerAnalysis(field).values(self._studyset.store)[self._studyset.view.index]
        if np.isfinite(reduced).any():
            return np.asarray(reduced, dtype=float), "numeric"

        return raw.to_numpy(dtype=object), "categorical"

    def matrix(self, pattern):
        """Return ``(names, sparse values)`` for the labels a pattern names.

        An absent label is a zero rather than a gap, which is what a sparse
        annotation means.
        """
        names = self.matching(pattern)
        unusable = [name for name in names if not self.is_numeric_label(name)]
        if unusable:
            raise ValueError(
                f"Pattern {pattern!r} matches {len(unusable)} label(s) that are not "
                f"numeric, and the feature matrix is numeric: {_preview(unusable)}. "
                "Narrow the pattern, or encode those labels and select the result. "
                "describe_fields(studyset, source='annotations') reports every label's "
                "kind, so its 'field' column filtered to kind == 'numeric' is a "
                "descriptor_fields list."
            )

        block = self.label_block
        columns = [block.col(name) for name in names]
        return names, sparse.csc_matrix(block.values)[:, columns].tocsr()

    def is_numeric_label(self, label):
        """Report whether an annotation label holds numbers.

        A label no analysis carries holds nothing to disagree with, and reads
        as a column of zeros.
        """
        columns = self._annotation_store(label)
        if columns is None:
            # Renamed past a collision by the union, so it is only in the block.
            return True
        values = columns.entries(label)[1]
        return not len(values) or _is_numeric(values)

    def _annotation_store(self, label):
        """Return the column store that owns a label, or None."""
        if not self._entries:
            self._entries = {
                name: annotation.columns
                for annotation in self._studyset.annotations
                for name in annotation.columns.keys()
            }
        return self._entries.get(label)

    def _annotation_values(self, label, columns):
        """Return one numeric label's values, missing where the label is absent."""
        if columns is None:
            return self.label_block.column(label)
        return columns.get_numeric(label, sel=self._studyset.view.index)


def _escape_brackets(pattern):
    """Return ``pattern`` with ``[`` and ``]`` taken literally rather than as a class.

    One pass, because escaping each bracket introduces brackets of its own.
    ``*`` and ``?`` are left alone: they are why the caller wrote a pattern.
    """
    escaped = {"[": "[[]", "]": "[]]"}
    return "".join(escaped.get(character, character) for character in pattern)


def _looks_like_pattern(field):
    """Report whether a field selector reads as a glob."""
    return any(character in field for character in "*?[")


def describe_fields(studyset, source=None, min_coverage=0.0):
    """Return the fields :meth:`~nimare.studyset.Studyset.to_bunch` can read.

    A release-scale Studyset offers hundreds of metadata columns and hundreds
    of annotation labels, most of which no analysis fills in. This reports what
    each one holds, using the same reader
    :meth:`~nimare.studyset.Studyset.to_bunch` uses, so a field described as
    numeric here is numeric there.

    Parameters
    ----------
    studyset : :class:`~nimare.nimads.Studyset`
        The Studyset to describe.
    source : {"metadata", "annotations", "texts"}, optional
        Restrict the report to one source, by default None, meaning all three.
    min_coverage : :obj:`float`, optional
        Drop fields reported by a smaller fraction of analyses than this, by
        default 0.0, which keeps every field.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per field, ordered by coverage, with columns ``source``,
        ``field``, ``kind``, ``coverage``, ``n_unique`` and ``example``.

    Examples
    --------
    >>> fields = describe_fields(studyset, min_coverage=0.5)  # doctest: +SKIP
    >>> fields[fields.kind == "numeric"].head()               # doctest: +SKIP
    """
    studyset = normalize_collection(studyset)
    fields = _Fields(studyset)
    sources = FIELD_SOURCES if source is None else (_as_selector((source, "x"))[0],)

    n_rows = len(studyset.ids)
    rows = []
    for name in sources:
        for field in fields.names(name):
            values, kind = fields.value(name, field)
            missing = _missing_mask(values, kind)
            coverage = 0.0 if not n_rows else float((~missing).sum()) / n_rows
            if coverage < min_coverage:
                continue
            present = np.asarray(values, dtype=object)[~missing]
            rows.append(
                {
                    "source": name,
                    "field": field,
                    "kind": kind,
                    "coverage": coverage,
                    "n_unique": len({str(value) for value in present}),
                    "example": None if not len(present) else str(present[0])[:60],
                }
            )

    table = pd.DataFrame(rows, columns=FIELD_REPORT_COLUMNS)
    return table.sort_values(["coverage", "source", "field"], ascending=[False, True, True])


def _held_out_studies(test_size, n_studies):
    """Return the number of studies to hold out, or explain why there is none."""
    if n_studies < 2:
        raise ValueError(
            f"A grouped split needs at least 2 studies, but this Studyset has {n_studies}. "
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


class _DescriptorBlock(NamedTuple):
    """One descriptor selection: its column names, its values, what it lacks."""

    names: list
    values: Any  # (n_rows, n_names), dense for a field and sparse for labels
    missing: Any  # boolean mask over rows, or None where absence means zero


def _missing_by_field(ids, blocks, target_missing, retained):
    """Return ``{field: analyses missing it}``, counting only the rows being kept.

    A row the coordinate policy already removed is not in the output to be
    missing from.
    """
    fields = [(block.names[0], block.missing) for block in blocks]
    fields.append((TARGET_KEY, target_missing))

    return {
        field: ids[missing & retained].tolist()
        for field, missing in fields
        if missing is not None and (missing & retained).any()
    }


class _FeatureExtractor(NiMAREBase):
    """Carry out one conversion from a Studyset to a scikit-learn bundle.

    Internal. :meth:`~nimare.studyset.Studyset.to_bunch` is the public entry
    point and documents the parameters. The class exists so that the stages of one conversion --
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
        test_size=None,
        random_state=None,
        memory: Any = None,
        memory_level: int = 2,
    ):
        self.kernel_transformer = kernel_transformer
        self.descriptor_fields = descriptor_fields
        self.target_field = target_field
        self.target_transformer = target_transformer
        self.missing_coordinates = missing_coordinates
        self.missing_values = missing_values
        self.test_size = test_size
        self.random_state = random_state
        self.memory = memory
        self.memory_level = memory_level

    # ------------------------------------------------------------- public API

    def transform(self, studyset):
        """Convert a Studyset into the bundle scikit-learn expects.

        Parameters
        ----------
        studyset : :class:`~nimare.nimads.Studyset`
            The Studyset to convert.

        Returns
        -------
        :class:`sklearn.utils.Bunch`
            One row per retained analysis. See
            :meth:`~nimare.studyset.Studyset.to_bunch`, which documents it.
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

        fields = _Fields(studyset)
        blocks = self._read_descriptors(fields)
        target, target_missing = self._read_target(fields)

        retained, dropped = self._retained_rows(ids, has_coordinates, blocks, target_missing)

        self._check_target(target, retained, ids)

        studyset_rows = studyset.select_analyses(retained)
        map_features = self._map_matrix(studyset_rows, ids[retained], has_coordinates[retained])

        descriptor_names = [name for block in blocks for name in block.names]
        bundle = self._bundle(
            studyset,
            map_features,
            self._descriptor_matrix(blocks, retained),
            descriptor_names,
            ids=ids[retained],
            groups=study_ids[retained],
            target=None if target is None else target[retained],
            provenance=self._provenance(studyset, ids, retained, dropped, descriptor_names),
        )
        if self.test_size is not None:
            bundle.train, bundle.test = self._grouped_split(bundle.groups)
        return bundle

    @staticmethod
    def _descriptor_matrix(blocks, retained):
        """Return the descriptor block for the retained rows, or None."""
        if not blocks:
            return None
        kept = [_take_rows(block.values, retained) for block in blocks]
        return kept[0] if len(kept) == 1 else _hstack_blocks(kept)

    @staticmethod
    def _bundle(studyset, map_features, descriptors, descriptor_names, **aligned):
        """Assemble the bundle, with the column boundary the blocks imply."""
        n_map = map_features.shape[1]
        n_descriptors = 0 if descriptors is None else descriptors.shape[1]
        return Bunch(
            data=_hstack(map_features, descriptors),
            feature_names=[f"voxel_{index}" for index in range(n_map)] + list(descriptor_names),
            map_columns=slice(0, n_map),
            descriptor_columns=slice(n_map, n_map + n_descriptors),
            descriptor_names=list(descriptor_names),
            masker=studyset.masker,
            **aligned,
        )

    def _grouped_split(self, groups):
        """Return the row positions of a grouped holdout over ``groups``.

        A study belongs to exactly one partition, which is the whole point of
        splitting here rather than with a plain shuffle.
        """
        n_test = _held_out_studies(self.test_size, len(np.unique(groups)))
        splitter = GroupShuffleSplit(n_splits=1, test_size=n_test, random_state=self.random_state)
        train, test = next(splitter.split(np.zeros(len(groups)), groups=groups))
        return train, test

    # ------------------------------------------------------------- validation

    def _validate_options(self):
        """Check the option vocabulary before any work is done."""
        if self.missing_coordinates not in ("drop", "include"):
            raise ValueError(
                "missing_coordinates must be 'drop' or 'include', not "
                f"{self.missing_coordinates!r}."
            )
        policies = (
            self.missing_values.values()
            if isinstance(self.missing_values, Mapping)
            else (self.missing_values,)
        )
        if isinstance(self.missing_values, Mapping):
            unknown = set(self.missing_values) - set(MISSING_ROLES)
            if unknown:
                raise ValueError(
                    f"missing_values names {_preview(sorted(unknown))}, but a mapping "
                    f"sets a policy per role: {', '.join(MISSING_ROLES)}."
                )
        for policy in policies:
            if policy not in MISSING_POLICIES:
                raise ValueError(
                    "missing_values must be 'raise', 'drop' or 'keep', or a mapping "
                    f"from role to one of those, not {policy!r}."
                )

    # -------------------------------------------------------------- selection

    def _read_descriptors(self, fields):
        """Return one :class:`_DescriptorBlock` per selector."""
        selectors = self.descriptor_fields
        if selectors is None:
            return []
        # A tuple is one ``(source, field)`` selector; a list holds several.
        if isinstance(selectors, (str, Mapping, tuple)):
            selectors = [selectors]

        blocks, seen = [], set()
        for selector in selectors:
            source, field, is_pattern = fields.resolve(selector, what="descriptor")

            if is_pattern:
                names, values = fields.matrix(field)
                block = _DescriptorBlock(names, values, None)
            else:
                values, kind = fields.value(source, field)
                if kind != "numeric":
                    raise ValueError(
                        f"Descriptor field {field!r} from {source} is {kind}, and the "
                        "feature matrix is numeric. Encode it yourself -- its raw values "
                        f"are in the Studyset's {source} -- and select the numeric result."
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

    def _read_target(self, fields):
        """Return ``(target values, missing mask)`` for the selected target."""
        if self.target_field is None:
            return None, None

        source, field, is_pattern = fields.resolve(self.target_field, what="target")
        if is_pattern:
            raise ValueError(
                f"Target pattern {field!r} names a set of labels, and a target is one "
                "value per analysis. Name one label."
            )
        values, kind = fields.value(source, field)
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

        return values, missing

    def _check_target(self, target, retained, ids):
        """Refuse a target that says the same thing about every analysis kept."""
        if target is None:
            return
        kept = np.asarray(target)[retained]
        present = kept[~_missing_mask(kept, "numeric" if kept.dtype.kind in "fiu" else "other")]
        if len(present) and len(np.unique(present)) == 1:
            raise ValueError(
                f"The target has the single value {present[0]!r} for every analysis that "
                f"was kept ({len(kept)} of {len(ids)}), so there is nothing to predict."
            )

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
        retained = (
            has_coordinates.copy()
            if self.missing_coordinates == "drop"
            else (np.ones(len(ids), dtype=bool))
        )
        dropped = {
            "no_coordinates": ids[~retained].tolist(),
            "missing_values": _missing_by_field(ids, blocks, target_missing, retained),
        }

        by_policy = {policy: {} for policy in MISSING_POLICIES}
        for field, affected in dropped["missing_values"].items():
            by_policy[self._missing_policy(field)][field] = affected

        if by_policy["raise"]:
            raise ValueError(
                "Missing values in "
                + "; ".join(
                    f"{field} ({len(affected)} analyses: {_preview(affected, 3)})"
                    for field, affected in by_policy["raise"].items()
                )
                + ". Fix the Studyset, or choose missing_values='drop' to remove those "
                "analyses or missing_values='keep' to impute them in your pipeline. "
                "A mapping sets the two roles apart, as in "
                "{'target': 'drop', 'descriptors': 'keep'}, since a target cannot be "
                "imputed."
            )
        for affected in by_policy["drop"].values():
            retained &= ~np.isin(ids, affected)

        if not retained.any():
            raise ValueError(
                "No analyses are left after applying missing_coordinates="
                f"{self.missing_coordinates!r} and missing_values={self.missing_values!r}."
            )

        return retained, dropped

    # ------------------------------------------------------------ map features

    def _missing_policy(self, field):
        """Return the missing-value policy that governs one field."""
        if not isinstance(self.missing_values, Mapping):
            return self.missing_values
        role = "target" if field == TARGET_KEY else "descriptors"
        return self.missing_values.get(role, "raise")

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

    rows = np.empty(len(ids), dtype=int)
    rows[has_coordinates] = np.searchsorted(map_ids, ids[has_coordinates])

    if not has_coordinates.all():
        # One extra all-zero row for every coordinate-less analysis to point at.
        maps = sparse.vstack(
            [maps, sparse.csr_matrix((1, maps.shape[1]), dtype=maps.dtype)], format="csr"
        )
        rows[~has_coordinates] = maps.shape[0] - 1

    if np.array_equal(rows, np.arange(len(ids))):
        return maps.tocsr()

    return maps[rows].tocsr()
