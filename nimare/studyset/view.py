"""A view: which analyses, which foci, and the execution context.

A view is ``(store, analysis index, point mask, context)`` and nothing else.
Narrowing is index arithmetic, so it never touches the data columns.

Two selection levels rather than one, because they are both real: estimators
select *analyses*, while :class:`~nimare.diagnostics.FocusFilter` selects *foci*
and keeps every analysis.

The context -- target space, masker, base path -- lives here rather than on the
store because it is not a property of the data. The same studyset is the same
studyset whichever template you analyse it in.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Optional

import numpy as np

from nimare.studyset import layout
from nimare.studyset.store import derived

__all__ = ["Context", "View"]


@functools.lru_cache(maxsize=None)
def _default_masker(target):
    from nimare.utils import get_masker, get_template

    return get_masker(get_template(target, mask="brain"))


@dataclass(frozen=True)
class Context:
    """Execution setup: not data, so it lives on the view."""

    space: Optional[str] = None
    masker: Optional[object] = None
    basepath: Optional[str] = None

    def resolved_masker(self):
        """The caller's masker, or the cached default for ``space``."""
        if self.masker is not None:
            return self.masker
        if not isinstance(self.space, str):
            return None
        return _default_masker(self.space)

    def with_(self, **changes):
        return dataclasses.replace(
            self, **{k: v for k, v in changes.items() if v is not None}
        )


class View:
    """A selection of analyses over one immutable store."""

    __slots__ = ("store", "index", "point_mask", "context", "_cache")

    def __init__(self, store, index=None, point_mask=None, context=None):
        self.store = store
        self.index = (
            np.arange(store.n_analyses, dtype=np.int64)
            if index is None
            else np.asarray(index, dtype=np.int64)
        )
        self.point_mask = point_mask
        self.context = context if context is not None else Context()
        self._cache = {}

    # ------------------------------------------------------------- identity
    def __len__(self):
        return len(self.index)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<View {len(self)}/{self.store.n_analyses} analyses>"

    @property
    def keys(self):
        """The full ``study-analysis`` ids of the selected analyses."""
        got = self._cache.get("keys")
        if got is None:
            got = self.store.analysis_full_key[self.index]
            self._cache["keys"] = got
        return got

    @property
    def study_keys(self):
        """Unique study ids of the selected analyses."""
        got = self._cache.get("study_keys")
        if got is None:
            store = self.store
            got = np.unique(store.study_key[store.study_idx[self.index]])
            self._cache["study_keys"] = got
        return got

    # ------------------------------------------------------------ narrowing
    def select(self, mask_or_positions):
        """Narrow to a subset of the selected analyses. O(k)."""
        idx = np.asarray(mask_or_positions)
        if idx.dtype == bool:
            idx = self.index[idx]
        return View(self.store, idx, self.point_mask, self.context)

    def select_keys(self, keys, *, allow_short=True):
        """Narrow to the named analyses, by full id or short analysis id."""
        wanted = np.unique(np.asarray([str(k) for k in np.atleast_1d(keys)], dtype=str))
        rows = _resolve_key_rows(self.store, wanted, allow_short=allow_short)
        keep = np.isin(self.index, rows)
        return self.select(keep)

    def drop_keys(self, keys):
        """Narrow to everything except the named analyses."""
        wanted = np.unique(np.asarray([str(k) for k in np.atleast_1d(keys)], dtype=str))
        rows = _resolve_key_rows(self.store, wanted, allow_short=True)
        return self.select(~np.isin(self.index, rows))

    def select_studies(self, study_keys, *, exclude=False):
        """Narrow by parent study."""
        store = self.store
        wanted = np.asarray([str(k) for k in np.atleast_1d(study_keys)], dtype=str)
        hit = np.isin(store.study_key.astype(str), wanted)
        per_analysis = hit[store.study_idx[self.index]]
        return self.select(~per_analysis if exclude else per_analysis)

    def select_points(self, mask):
        """Narrow to a subset of foci, keeping every analysis."""
        mask = np.asarray(mask, dtype=bool)
        combined = mask if self.point_mask is None else (mask & self.point_mask)
        return View(self.store, self.index, combined, self.context)

    def with_context(self, **changes):
        """A view with different execution context. No data is touched."""
        return View(self.store, self.index, self.point_mask, self.context.with_(**changes))

    # --------------------------------------------------------------- blocks
    def coordinate_block(self):
        """Foci for the selection, grouped by analysis."""
        from nimare.studyset.blocks import CoordinateBlock

        got = self._cache.get("coordinates")
        if got is None:
            store = self.store
            xyz, space, space_dict = layout.harmonized_coordinates(store, self.context.space)
            p_idx, offsets = layout.ranges_to_indices(
                store.point_offsets[self.index], store.point_offsets[self.index + 1]
            )
            if self.point_mask is not None:
                keep = self.point_mask[p_idx]
                counts = np.diff(offsets)
                groups = np.repeat(np.arange(len(counts)), counts)
                p_idx = p_idx[keep]
                offsets = np.concatenate(
                    ([0], np.cumsum(np.bincount(groups[keep], minlength=len(counts))))
                )
            got = CoordinateBlock(
                xyz=xyz[p_idx],
                offsets=offsets,
                group_keys=self.keys,
                space=space[p_idx],
                space_categories=list(space_dict.categories),
                point_rows=p_idx,
            )
            self._cache["coordinates"] = got
        return got

    def image_block(self, imtype, *, policy="all"):
        from nimare.studyset.blocks import image_block

        return image_block(self, imtype, policy=policy)

    def label_block(self, annotation=None):
        from nimare.studyset.blocks import label_block

        return label_block(self, annotation)

    def text_block(self, field="abstract"):
        from nimare.studyset.blocks import text_block

        return text_block(self, field)

    # ---------------------------------------------------- spatial questions
    def points_in_mask(self, mask_data, affine):
        """Point-level boolean for foci inside an image mask.

        Uses the coordinates in the context's space: a Talairach focus and its
        MNI projection are not the same voxel.
        """
        xyz, _, _ = layout.harmonized_coordinates(self.store, self.context.space)
        inv = np.linalg.inv(affine)
        with np.errstate(invalid="ignore"):
            ijk = (xyz @ inv[:3, :3].T + inv[:3, 3]).astype(np.int32)
        ijk = np.clip(ijk, 0, np.asarray(mask_data.shape) - 1)
        return mask_data[ijk[:, 0], ijk[:, 1], ijk[:, 2]] > 0

    def points_near(self, xyz, radius):
        """Point-level boolean for foci within ``radius`` mm of any of ``xyz``."""
        from scipy.spatial.distance import cdist

        coords, _, _ = layout.harmonized_coordinates(self.store, self.context.space)
        if not len(coords):
            return np.zeros(0, dtype=bool)
        query = np.atleast_2d(np.asarray(xyz, dtype=float))
        return (cdist(query, coords) <= radius).any(axis=0)

    def analyses_with_points(self, point_bool):
        """Analyses with at least one flagged focus, as a view.

        Counted with ``bincount`` over the point-to-analysis column rather than
        ``add.reduceat`` over the offsets: ``reduceat`` returns the element *at*
        the index when two offsets coincide, so analyses with no foci came back
        as hits.
        """
        store = self.store
        if store.n_points == 0:
            return self.select(np.zeros(len(self.index), dtype=bool))
        parents = layout.point_parents(store)
        flagged = np.asarray(point_bool, dtype=bool)
        hits = np.bincount(parents[flagged].astype(np.int64), minlength=store.n_analyses) > 0
        return self.select(hits[self.index])

    # ---------------------------------------------------- pandas conversion
    def frame(self, name):
        """A memoised pandas frame. Compatibility surface, not a hot path."""
        got = self._cache.get(("frame", name))
        if got is None:
            from nimare.studyset import frames as _frames

            got = getattr(_frames, name)(self)
            self._cache[("frame", name)] = got
        # A shallow copy, so one caller's edit is not the next caller's read.
        return got.copy(deep=False)

    # ------------------------------------------------- requirement handling
    def resolve(self, requirements, drop_invalid=True):
        """Intersect each requirement's validity, then resolve against the result.

        Returns ``(narrowed view, {name: block})``. Because the narrowing
        produces a view, every block is aligned to the same analyses by
        construction -- there is no parallel-list bookkeeping to get wrong.
        """
        requirements = tuple(requirements)
        valid = np.ones(len(self.index), dtype=bool)
        for requirement in requirements:
            valid &= requirement.validity(self)
        if not valid.all():
            if not drop_invalid:
                missing = int((~valid).sum())
                raise ValueError(
                    f"{missing} of {len(valid)} analyses lack required data; pass "
                    "drop_invalid=True to analyse the rest"
                )
            narrowed = self.select(valid)
        else:
            narrowed = self
        blocks = {r.name: r.resolve(narrowed) for r in requirements}
        return narrowed, blocks


def _resolve_key_rows(store, wanted, *, allow_short=True):
    """Rows whose full id -- or short analysis id -- is in ``wanted``."""
    cache = derived(store)
    rows = []
    kinds = ("full", "short") if allow_short else ("full",)
    for kind in kinds:
        key = ("sorted_keys", kind)
        got = cache.get(key)
        if got is None:
            if kind == "full":
                keys = store.analysis_full_key.astype(str)
            else:
                keys = np.asarray(
                    [str(k).rsplit("-", 1)[-1] for k in store.analysis_full_key], dtype=str
                )
            order = np.argsort(keys, kind="stable")
            got = (keys[order], order)
            cache[key] = got
        keys, order = got
        if not len(keys):
            continue
        pos = np.searchsorted(keys, wanted)
        ok = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == wanted)
        rows.append(order[pos[ok]])
    if not rows:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(rows))
