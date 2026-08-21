"""Typed requirements: what an algorithm declares it needs.

Replaces the stringly-typed ``_required_inputs = {"coordinates": ("coordinates",
None)}`` mapping and the dict-of-parallel-lists it produced. A requirement can
say which analyses satisfy it, and can turn a view into the block it describes;
:meth:`nimare.studyset.view.View.resolve` intersects the validity masks and
narrows once, so every block comes back aligned to the same analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from nimare.studyset.store import derived

__all__ = ["Coordinates", "Images", "Labels", "PerAnalysis", "Texts"]


@dataclass(frozen=True)
class Coordinates:
    """Foci, grouped by analysis, in ``space``."""

    space: Optional[str] = None
    name: str = "coordinates"

    def validity(self, view):
        store = view.store
        sizes = store.point_offsets[view.index + 1] - store.point_offsets[view.index]
        return sizes > 0

    def resolve(self, view):
        if self.space is not None and view.context.space != self.space:
            view = view.with_context(space=self.space)
        return view.coordinate_block()


@dataclass(frozen=True)
class PerAnalysis:
    """One number per analysis, from metadata at either level."""

    field: str
    reduce: str = "mean"
    name: str = "per_analysis"

    def dense(self, store):
        key = ("per_analysis", self.field, self.reduce)
        cache = derived(store)
        got = cache.get(key)
        if got is None:
            reducer = {"mean": np.mean, "max": np.max, "min": np.min}[self.reduce]
            if self.field in ("sample_sizes", "sample_size"):
                out = _normalized_sample_sizes(store, reducer)
            else:
                out = store.metadata.get_numeric(self.field, reduce=reducer)
                if not np.isfinite(out).any() and store.study_metadata is not None:
                    per_study = store.study_metadata.get_numeric(
                        self.field, reduce=reducer
                    )
                    out = per_study[store.study_idx]
            out = np.asarray(out, dtype=np.float64)
            out.flags.writeable = False
            cache[key] = out
            got = out
        return got

    def validity(self, view):
        return np.isfinite(self.dense(view.store)[view.index])

    def resolve(self, view):
        return self.dense(view.store)[view.index]


@dataclass(frozen=True)
class Images:
    """One statistic map per analysis, of type ``imtype``."""

    imtype: str
    policy: str = "all"
    name: str = "images"

    def validity(self, view):
        store = view.store
        ia = store.image_attrs
        if ia is None or not ia.n_rows:
            return np.zeros(len(view.index), dtype=bool)
        present = np.zeros(store.n_analyses, dtype=bool)
        match = ia.dense["value_type"] == self.imtype
        present[ia.dense["analysis_idx"][match]] = True
        return present[view.index]

    def resolve(self, view):
        return view.image_block(self.imtype, policy=self.policy)


@dataclass(frozen=True)
class Labels:
    """The annotation matrix."""

    annotation: Optional[str] = None
    name: str = "labels"

    def validity(self, view):
        store = view.store
        if not store.annotations:
            return np.zeros(len(view.index), dtype=bool)
        covered = np.zeros(store.n_analyses, dtype=bool)
        sets = (
            [store.annotations[self.annotation]]
            if self.annotation is not None
            else list(store.annotations.values())
        )
        for annotation in sets:
            for label in annotation.columns.keys():
                if label in annotation.columns.dense:
                    covered[:] = True
                    break
                idx, _ = annotation.columns.sparse[label]
                covered[np.asarray(idx, dtype=np.int64)] = True
        return covered[view.index]

    def resolve(self, view):
        from nimare.studyset.blocks import label_block, label_block_union

        if self.annotation is not None or len(view.store.annotations) == 1:
            return label_block(view, self.annotation)
        return label_block_union(view)[0]


@dataclass(frozen=True)
class Texts:
    """Documents from a text field."""

    field: str = "abstract"
    name: str = "texts"

    def validity(self, view):
        store = view.store
        cs = store.texts
        if cs is None or self.field not in cs:
            return np.zeros(len(view.index), dtype=bool)
        present = np.zeros(store.n_analyses, dtype=bool)
        if self.field in cs.dense:
            col = cs.dense[self.field]
            present[np.flatnonzero(np.asarray([bool(v) for v in col]))] = True
        else:
            idx, values = cs.sparse[self.field]
            rows = [int(i) for i, v in zip(idx, values) if v]
            present[rows] = True
        return present[view.index]

    def resolve(self, view):
        return view.text_block(self.field)


def _normalized_sample_sizes(store, reducer):
    """Sample sizes from whichever level declares them.

    The raw ``sample_size`` / ``sample_sizes`` keys stay where NIMADS put them so
    that export is lossless, while consumers see one normalised field.
    """
    from nimare.io import _extract_coerced_sample_sizes

    n_a = store.n_analyses
    out = np.full(n_a, np.nan, dtype=np.float64)
    a_md, s_md = store.metadata, store.study_metadata
    a_vals = {
        key: a_md.get(key) for key in ("sample_sizes", "sample_size") if key in a_md
    }
    s_vals = {}
    if s_md is not None:
        s_vals = {
            key: s_md.get(key) for key in ("sample_sizes", "sample_size") if key in s_md
        }
    def listify(value):
        # A parquet list column reads back as an ndarray, which the coercion
        # helper rejects outright.
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    for row in range(n_a):
        s_row = int(store.study_idx[row])
        candidates = [
            (
                "sample_sizes",
                listify(a_vals["sample_sizes"][row]) if "sample_sizes" in a_vals else None,
            ),
            ("sample_size", a_vals["sample_size"][row] if "sample_size" in a_vals else None),
            (
                "sample_sizes",
                listify(s_vals["sample_sizes"][s_row]) if "sample_sizes" in s_vals else None,
            ),
            ("sample_size", s_vals["sample_size"][s_row] if "sample_size" in s_vals else None),
        ]
        sizes = _extract_coerced_sample_sizes(candidates)
        if sizes:
            out[row] = reducer(np.asarray(sizes, dtype=np.float64))
    return out
