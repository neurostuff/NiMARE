"""Blocks: the shape each algorithm family actually wants.

A block is derived from a view, read-only, and knows nothing about the model that
consumes it. The point of naming them is that consumers stop rebuilding the same
structures by hand -- ``CorrelationDecoder`` kept its own ``{id: row}`` map,
``GCLDAModel`` built its own document index and CSR-style token arrays, and
``IBMAEstimator`` cached masked images against a fit rather than a studyset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nimare.studyset.store import derived

__all__ = [
    "Comparison",
    "CoordinateBlock",
    "DesignBlock",
    "ImageBlock",
    "LabelBlock",
    "TextBlock",
    "design_block",
    "group_by",
    "image_block",
    "label_block",
    "label_block_union",
    "text_block",
]


@dataclass
class CoordinateBlock:
    """Foci grouped by analysis.

    ``offsets`` are CSR group boundaries, so "the foci of group *g*" is a slice
    rather than a boolean mask over every focus.
    """

    xyz: np.ndarray
    offsets: np.ndarray
    group_keys: np.ndarray
    space: np.ndarray
    space_categories: list = field(default_factory=list)
    point_rows: Optional[np.ndarray] = None
    _ijk_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __len__(self):
        """Return the number of foci in the block."""
        return len(self.xyz)

    @property
    def n_groups(self):
        """Return the number of analyses the foci are grouped into."""
        return len(self.offsets) - 1

    def group(self, g):
        """Return the foci of group ``g``, as a view into ``xyz``."""
        return self.xyz[self.offsets[g] : self.offsets[g + 1]]

    def group_sizes(self):
        """Return the number of foci in each group."""
        return np.diff(self.offsets)

    def group_of_point(self):
        """Return the group index of every focus."""
        return np.repeat(np.arange(self.n_groups, dtype=np.int32), self.group_sizes())

    def ijk(self, affine):
        """Matrix indices for ``affine``, memoised.

        Truncates rather than rounds, matching :func:`nimare.utils.mm2vox`.
        """
        key = np.asarray(affine).tobytes()
        got = self._ijk_cache.get(key)
        if got is None:
            inv = np.linalg.inv(affine)
            with np.errstate(invalid="ignore"):
                got = (self.xyz @ inv[:3, :3].T + inv[:3, 3]).astype(np.int32)
            got.flags.writeable = False
            self._ijk_cache[key] = got
        return got


@dataclass(frozen=True)
class ImageBlock:
    """One row per *image*, with its parent analysis and study alongside.

    An analysis carrying two z maps yields two rows. ``study_pos`` is the
    dependence group an image-based meta-analysis needs, so the grouping falls
    out of the same array rather than being rebuilt from a dict.
    """

    refs: np.ndarray
    analysis_pos: np.ndarray
    study_pos: np.ndarray
    imtype: str
    space: np.ndarray
    metadata: np.ndarray
    values: Optional[np.ndarray] = None
    raw_refs: Optional[np.ndarray] = None

    def __len__(self):
        """Return the number of images in the block."""
        return len(self.refs)

    def dependence_groups(self):
        """Integer group codes, densely recoded, one per image."""
        if not len(self.study_pos):
            return np.zeros(0, dtype=np.int64)
        return np.unique(self.study_pos, return_inverse=True)[1]

    def rows(self, positions):
        """Row-select an already-masked matrix. A refit reads no NIfTIs."""
        if self.values is None:
            raise ValueError("images have not been masked yet")
        return self.values[positions]

    def with_values(self, values):
        """Return a copy of the block carrying masked image data."""
        return ImageBlock(
            refs=self.refs,
            analysis_pos=self.analysis_pos,
            study_pos=self.study_pos,
            imtype=self.imtype,
            space=self.space,
            metadata=self.metadata,
            values=values,
            raw_refs=self.raw_refs,
        )


@dataclass(frozen=True)
class LabelBlock:
    """The annotation matrix, column-major.

    Decoders address it by *label*, so it is stored CSC: one column is a
    contiguous slice. Stored CSR, ``getcol`` is O(total nonzeros) and the
    per-feature cost collapses as the label count grows.
    """

    values: object  # scipy.sparse.csc_matrix
    labels: np.ndarray
    rows: np.ndarray
    _col_of: dict = field(default_factory=dict, compare=False, repr=False)

    def col(self, label):
        """Return the column index of ``label``."""
        if not self._col_of:
            self._col_of.update({lab: i for i, lab in enumerate(self.labels)})
        try:
            return self._col_of[label]
        except KeyError:
            raise KeyError(f"unknown label {label!r}") from None

    def column(self, label):
        """One label's values as a dense float array over the view's analyses."""
        j = self.col(label)
        lo, hi = self.values.indptr[j], self.values.indptr[j + 1]
        out = np.zeros(self.values.shape[0], dtype=np.float64)
        out[self.values.indices[lo:hi]] = self.values.data[lo:hi]
        return out

    def above(self, label, threshold):
        """Boolean mask: this label at or above ``threshold``."""
        j = self.col(label)
        lo, hi = self.values.indptr[j], self.values.indptr[j + 1]
        rows = self.values.indices[lo:hi]
        vals = self.values.data[lo:hi]
        out = np.zeros(self.values.shape[0], dtype=bool)
        out[rows[vals >= threshold]] = True
        if threshold <= 0:
            # absent entries are implicit zeros, which clear the threshold too
            out[:] = True
            out[rows[vals < threshold]] = False
        return out

    def dense(self, labels=None):
        """Build a dense matrix for the requested labels."""
        if labels is None:
            return self.values.toarray()
        cols = [self.col(label) for label in labels]
        return self.values[:, cols].toarray()

    def counts_above(self, threshold):
        """Per-label count of analyses at or above ``threshold``, one pass."""
        return np.asarray((self.values >= threshold).sum(axis=0)).ravel()


@dataclass(frozen=True)
class TextBlock:
    """Documents, with the document index already integral."""

    text: list
    rows: np.ndarray
    field: str

    def __len__(self):
        """Return the number of documents in the block."""
        return len(self.text)

    def counts(self, vectorizer):
        """``(documents x terms)`` sparse counts and the vocabulary."""
        weights = vectorizer.fit_transform(self.text)
        names = np.asarray([str(n) for n in vectorizer.get_feature_names_out()], dtype=object)
        return weights.tocsr(), names

    @staticmethod
    def token_indices(counts):
        """Token-level ``(document, term)`` index arrays from sparse counts."""
        coo = counts.tocoo()
        repeats = coo.data.astype(np.int64)
        return np.repeat(coo.row, repeats), np.repeat(coo.col, repeats)


@dataclass(frozen=True)
class DesignBlock:
    """A moderator design matrix, aligned to a view by construction."""

    matrix: np.ndarray
    columns: np.ndarray

    def standardized(self):
        """Centre and scale each column."""
        out = self.matrix - self.matrix.mean(axis=0)
        sd = out.std(axis=0)
        sd[sd == 0] = 1.0
        return DesignBlock(out / sd, self.columns)


@dataclass(frozen=True)
class Comparison:
    """A pairwise input: two views that provably share one store.

    Naming the pair is what lets the obvious checks happen -- that both sides
    come from the same studyset, and that they do not overlap. Neither is
    checkable when a pairwise estimator simply takes two collections.
    """

    group1: object
    group2: object

    def __post_init__(self):
        """Reject a pair drawn from two stores, or one that overlaps."""
        if self.group1.store is not self.group2.store:
            raise ValueError("a Comparison must be between two views of the same studyset")
        overlap = np.intersect1d(self.group1.index, self.group2.index)
        if overlap.size:
            raise ValueError(f"groups overlap in {overlap.size} analyses")

    @property
    def n1(self):
        """Return the number of analyses on the first side."""
        return len(self.group1)

    def pooled(self):
        """Return the union as a view. Only indices are concatenated."""
        from nimare.studyset.view import View

        return View(
            self.group1.store,
            np.concatenate([self.group1.index, self.group2.index]),
            self.group1.point_mask,
            self.group1.context,
        )

    def permute(self, seed):
        """Permute the pair in index space. No data is moved."""
        from nimare.studyset.view import View

        pool = np.concatenate([self.group1.index, self.group2.index])
        order = np.arange(len(pool))
        np.random.default_rng(seed).shuffle(order)
        store, ctx, pm = self.group1.store, self.group1.context, self.group1.point_mask
        return (
            View(store, np.sort(pool[order[: self.n1]]), pm, ctx),
            View(store, np.sort(pool[order[self.n1 :]]), pm, ctx),
        )


# ------------------------------------------------------------------ builders


def image_block(view, imtype, *, policy="all"):
    """Collect every image of ``imtype`` for the view.

    ``policy="all"`` keeps every image, which is what NIMADS permits;
    ``"first"`` keeps one per analysis; ``"error"`` refuses to choose. The legacy
    one-column-per-type table silently applied ``"first"``.
    """
    store = view.store
    ia = store.image_attrs
    empty_o = np.empty(0, dtype=np.int64)
    if ia is None or not ia.n_rows:
        return ImageBlock(
            np.empty(0, dtype=object),
            empty_o,
            empty_o,
            imtype,
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
        )

    pos_of_analysis = np.full(store.n_analyses, -1, dtype=np.int64)
    pos_of_analysis[view.index] = np.arange(len(view.index))

    match = (ia.dense["value_type"] == imtype) & (pos_of_analysis[ia.dense["analysis_idx"]] >= 0)
    rows = np.flatnonzero(match)
    if rows.size == 0:
        return ImageBlock(
            np.empty(0, dtype=object),
            empty_o,
            empty_o,
            imtype,
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
        )

    parents = ia.dense["analysis_idx"][rows]
    if policy != "all":
        order = np.argsort(parents, kind="stable")
        rows, parents = rows[order], parents[order]
        first = np.flatnonzero(np.r_[True, parents[1:] != parents[:-1]])
        dropped = len(parents) - len(first)
        if dropped and policy == "error":
            raise ValueError(
                f"{dropped} analyses carry more than one {imtype!r} image; pass "
                "policy='all' to keep them or policy='first' to choose"
            )
        rows, parents = rows[first], parents[first]

    raw, refs = _resolve_refs(ia, rows, view.context.basepath)
    return ImageBlock(
        refs=refs,
        analysis_pos=pos_of_analysis[parents],
        study_pos=store.study_idx[parents].astype(np.int64),
        imtype=imtype,
        space=ia.dense["space"][rows],
        metadata=ia.dense["metadata"][rows],
        raw_refs=raw,
    )


def _resolve_refs(ia, rows, basepath):
    """Return ``(stored reference, resolved reference)`` for each image row.

    Both are kept: consumers want a path they can open, while a frame reports the
    relative form the studyset actually stores.
    """
    import os

    from nimare.io import _select_image_path
    from nimare.utils import _try_prepend

    urls = ia.dense["url"][rows]
    files = ia.dense["filename"][rows]
    raw = np.empty(len(rows), dtype=object)
    resolved = np.empty(len(rows), dtype=object)
    for i, (url, filename) in enumerate(zip(urls, files)):
        ref = _select_image_path(url, filename)
        raw[i] = ref
        if isinstance(ref, str) and ref and "://" not in ref and not os.path.isabs(ref):
            resolved[i] = _try_prepend(ref, basepath) if basepath else ref
        else:
            resolved[i] = ref
    return raw, resolved


def label_block(view, annotation=None):
    """Build the annotation matrix for one annotation set.

    Refuses to guess when the studyset carries several. The legacy behaviour --
    merging every annotation into one frame -- let two sets that share a label
    name overwrite each other, and compose's default note key is ``included``, so
    any two compose annotations collide completely.
    """
    import scipy.sparse as sp

    store = view.store
    name = annotation
    if name is None:
        if not store.annotations:
            raise ValueError("studyset has no annotations")
        if len(store.annotations) > 1:
            raise ValueError(
                "studyset has several annotations "
                f"({', '.join(sorted(store.annotations))}); name the one to use, "
                "or call label_block_union() to combine them deliberately"
            )
        name = next(iter(store.annotations))
    if name not in store.annotations:
        raise KeyError(f"unknown annotation {name!r}")

    cache = derived(store)
    key = ("label_matrix", name)
    got = cache.get(key)
    if got is None:
        cs = store.annotations[name].columns
        labels = sorted(cs.keys())
        rows_acc, cols_acc, vals_acc = [], [], []
        for j, label in enumerate(labels):
            if label in cs.dense:
                col = cs.dense[label]
                idx = np.arange(len(col))
                values = list(col)
            else:
                idx, values = cs.sparse[label]
                idx = np.asarray(idx, dtype=np.int64)
                values = list(values)
            keep_rows, keep_vals = [], []
            for row, value in zip(idx, values):
                if not isinstance(value, (bool, int, float, np.number)):
                    continue
                numeric = float(value)
                if numeric == 0.0 or numeric != numeric:
                    continue  # an explicit zero is not an annotation
                keep_rows.append(int(row))
                keep_vals.append(numeric)
            if keep_rows:
                rows_acc.append(np.asarray(keep_rows, dtype=np.int64))
                cols_acc.append(np.full(len(keep_rows), j, dtype=np.int32))
                vals_acc.append(np.asarray(keep_vals, dtype=np.float32))
        shape = (store.n_analyses, len(labels))
        if rows_acc:
            matrix = sp.csc_matrix(
                (
                    np.concatenate(vals_acc),
                    (np.concatenate(rows_acc), np.concatenate(cols_acc)),
                ),
                shape=shape,
                dtype=np.float32,
            )
        else:
            matrix = sp.csc_matrix(shape, dtype=np.float32)
        got = (matrix, np.asarray(labels, dtype=object))
        cache[key] = got
    matrix, labels = got
    return LabelBlock(sp.csc_matrix(matrix[view.index]), labels, view.index)


def label_block_union(view, annotations=None, *, on_collision="prefix"):
    """Combine several annotations into one matrix, deliberately.

    Returns ``(block, collisions)``. ``on_collision="prefix"`` qualifies clashing
    labels with their annotation id; ``"error"`` refuses. Either way the caller
    learns that two sets disagree about a name.
    """
    import scipy.sparse as sp

    store = view.store
    names = sorted(store.annotations) if annotations is None else list(annotations)
    seen, collisions, parts, labels_out = {}, [], [], []
    for name in names:
        block = label_block(view, name)
        for j, label in enumerate(block.labels):
            out_label = label
            if label in seen:
                collisions.append((label, seen[label], name))
                if on_collision == "error":
                    raise ValueError(
                        f"label {label!r} appears in annotations {seen[label]!r} and "
                        f"{name!r}; pass on_collision='prefix' to keep both"
                    )
                out_label = f"{name}.{label}"
            else:
                seen[label] = name
            parts.append(block.values[:, [j]])
            labels_out.append(out_label)
    if parts:
        matrix = sp.hstack(parts, format="csc")
    else:
        matrix = sp.csc_matrix((len(view), 0), dtype=np.float32)
    return LabelBlock(matrix, np.asarray(labels_out, dtype=object), view.index), collisions


def text_block(view, field="abstract"):
    """Documents with text in ``field``. Rows without text are absent."""
    cs = view.store.texts
    if cs is None or field not in cs:
        raise ValueError(f"studyset has no text field {field!r}")
    if field in cs.dense:
        col = cs.dense[field]
        idx = np.flatnonzero(np.asarray([bool(v) for v in col]))
        values = [col[i] for i in idx]
    else:
        idx, values = cs.sparse[field]
        idx = np.asarray(idx, dtype=np.int64)
        keep = [i for i, v in enumerate(values) if v]
        idx, values = idx[keep], [list(values)[i] for i in keep]
    pos_of = np.full(view.store.n_analyses, -1, dtype=np.int64)
    pos_of[view.index] = np.arange(len(view.index))
    selected = pos_of[idx] >= 0
    return TextBlock(
        [v for v, keep in zip(values, selected) if keep],
        pos_of[idx[selected]],
        field,
    )


def design_block(view, moderators, annotation=None):
    """Build a moderator design matrix from annotation labels."""
    block = label_block(view, annotation)
    matrix = (
        np.column_stack([block.column(m) for m in moderators])
        if moderators
        else np.zeros((len(view), 0))
    )
    return DesignBlock(matrix.astype(np.float64), np.asarray(moderators, dtype=object))


def group_by(view, labels, annotation=None):
    """Partition the view by the values of one or more annotation labels."""
    block = label_block(view, annotation)
    columns = np.column_stack([block.column(label) for label in labels])
    keys, codes = np.unique(columns, axis=0, return_inverse=True)
    return {tuple(keys[c]): view.select(np.flatnonzero(codes == c)) for c in range(len(keys))}
