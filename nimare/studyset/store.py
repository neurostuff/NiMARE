"""The studyset store: one immutable columnar value.

Levels, and how the links between them are encoded:

===============  =====================================  =========================
level            parent -> children                     child -> parent
===============  =====================================  =========================
study            ``analysis_offsets``                   --
analysis         ``point_offsets``, ``image_offsets``,   ``study_idx``
                 ``condition_offsets``
point            --                                     ``point_analysis``
image            ``image_offsets``                      ``image_attrs["analysis_idx"]``
===============  =====================================  =========================

Both directions are stored on purpose. Offsets make "the foci of analysis *i*" a
zero-copy slice; the parent column makes "which analysis is focus *j* in" a
lookup. Callers want both constantly, and the reverse column costs 4 bytes a
focus -- 3.5 MB on the whole of neurostore, 0.6% of the store.

Invariants, checked by :func:`nimare.studyset.layout.check_invariants`:

I1  every array is read-only, including after unpickling.
I2  offsets are compiled from the parent columns, never permuted independently.
I3  reordering enumerates the declared level stores, so a column added later
    cannot be missed.
I4  children are sorted by parent and the offsets cover them exactly.
I5  derived values are memoised against the store and never invalidated, which
    is sound because the store cannot change.
I6  document order lives in ``*_source_order``, not in row order.
I7  metadata stays at the level NIMADS declares it at.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nimare.studyset.columns import ColumnStore, Dict8

__all__ = ["StudysetStore", "derived", "freeze", "replace"]


@dataclass
class StudysetStore:
    """Immutable columnar representation of a studyset."""

    id: Optional[str] = None
    name: str = ""

    # ---- level 0: studies
    study_key: np.ndarray = None  # object[n_s] NIMADS ids
    study_attrs: ColumnStore = None  # name, doi, pmid, authors, ...
    study_metadata: ColumnStore = None  # study-level metadata (I7)
    study_has_metadata: np.ndarray = None  # bool[n_s]: source had a dict
    study_source_order: np.ndarray = None  # int32[n_s] (I6)
    analysis_offsets: np.ndarray = None  # int64[n_s + 1]

    # ---- level 1: analyses
    analysis_key: np.ndarray = None  # object[n_a] NIMADS ids
    analysis_full_key: np.ndarray = None  # object[n_a] "study-analysis"
    study_idx: np.ndarray = None  # int32[n_a] parent study row
    analysis_attrs: ColumnStore = None  # name, description
    metadata: ColumnStore = None  # analysis-level metadata (I7)
    analysis_has_metadata: np.ndarray = None  # bool[n_a]
    analysis_source_order: np.ndarray = None  # int32[n_a] (I6)
    texts: ColumnStore = None
    point_offsets: np.ndarray = None  # int64[n_a + 1]
    image_offsets: np.ndarray = None  # int64[n_a + 1]
    condition_offsets: np.ndarray = None  # int64[n_a + 1]

    # ---- level 2: points
    point_analysis: np.ndarray = None  # int32[n_p] parent analysis row
    xyz: np.ndarray = None  # float64[n_p, 3] contiguous
    point_key: np.ndarray = None  # object[n_p], may be all None
    point_space: np.ndarray = None  # int16[n_p] -> space_dict
    point_kind: np.ndarray = None  # int16[n_p] -> kind_dict
    point_values: ColumnStore = None  # sparse z_stat, t_stat, ...
    space_dict: Dict8 = None
    kind_dict: Dict8 = None
    coordinate_metadata_columns: frozenset = frozenset()

    # ---- level 2: images and conditions
    image_attrs: ColumnStore = None  # analysis_idx, url, filename, ...
    condition_code: np.ndarray = None  # int32[n_c] -> condition_dict
    condition_weight: np.ndarray = None  # float64[n_c]
    condition_dict: Dict8 = None
    condition_descriptions: dict = field(default_factory=dict)

    # ---- annotations
    annotations: dict = field(default_factory=dict)  # id -> AnnotationSet

    # ------------------------------------------------------------------ sizes
    @property
    def n_studies(self):
        """Return the number of studies."""
        return 0 if self.study_key is None else len(self.study_key)

    @property
    def n_analyses(self):
        """Return the number of analyses."""
        return 0 if self.analysis_key is None else len(self.analysis_key)

    @property
    def n_points(self):
        """Return the number of foci."""
        return 0 if self.xyz is None else len(self.xyz)

    @property
    def n_images(self):
        """Return the number of images."""
        if self.image_attrs is None:
            return 0
        return self.image_attrs.n_rows

    # ------------------------------------------------- declared level stores
    def study_level_stores(self):
        """Every ColumnStore indexed by study row (I3)."""
        return [cs for cs in (self.study_attrs, self.study_metadata) if cs is not None]

    def analysis_level_stores(self):
        """Every ColumnStore indexed by analysis row (I3)."""
        out = [cs for cs in (self.analysis_attrs, self.metadata, self.texts) if cs is not None]
        out.extend(a.columns for a in self.annotations.values())
        return out

    def point_level_stores(self):
        """Every ColumnStore indexed by point row (I3)."""
        return [cs for cs in (self.point_values,) if cs is not None]

    # -------------------------------------------------------------- identity
    def __repr__(self):  # pragma: no cover - debugging aid
        """Return a debugging representation naming the level sizes."""
        return (
            f"<StudysetStore {self.id!r}: {self.n_studies} studies, "
            f"{self.n_analyses} analyses, {self.n_points} foci>"
        )

    def __setstate__(self, state):
        """Restore, then re-freeze (I1).

        numpy does not carry ``writeable=False`` through pickle, so a store that
        came back from ``pickle.loads`` -- or from a joblib process worker -- was
        silently mutable again. Memo tables are dropped rather than shipped:
        they are cheap to rebuild and would otherwise bloat every dispatch.
        """
        self.__dict__.update(state)
        self.__dict__.pop("_derived", None)
        freeze(self)

    def __getstate__(self):
        """Restore the columns and re-freeze them, which pickle does not preserve."""
        state = dict(self.__dict__)
        state.pop("_derived", None)
        return state


def derived(store):
    """Return the store's memo table for values that are pure functions of it (I5).

    Sound without invalidation because the store is immutable: a memo cannot go
    stale if nothing can change underneath it.
    """
    got = getattr(store, "_derived", None)
    if got is None:
        got = {}
        object.__setattr__(store, "_derived", got)
    return got


def freeze(store):
    """Mark every array read-only (I1). O(number of columns), not of rows."""
    for name, value in list(vars(store).items()):
        if name == "_derived":
            continue
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
        elif isinstance(value, ColumnStore):
            value.freeze()
    for ann in (store.annotations or {}).values():
        if ann.columns is not None:
            ann.columns.freeze()
    return store


def replace(store, **changes):
    """Return a new store sharing every column not being replaced (copy-on-write).

    What makes "views cannot go stale" structural rather than policed: there is
    no mutation to observe, so there is no revision counter, no dirty flag and no
    mutation-tracking wrapper.
    """
    new = dataclasses.replace(store, **changes)
    new.__dict__.pop("_derived", None)
    return freeze(new)
