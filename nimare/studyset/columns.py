"""Column containers for the studyset store.

Three small value types, none of which know anything about NIMADS, algorithms or
pandas:

``Dict8``
    dictionary encoding for low-cardinality strings (coordinate space, image
    type, condition name).
``ColumnStore``
    a set of columns aligned to one level's row index, each either dense (one
    value per row) or sparse (row indices plus values). Studyset metadata is
    overwhelmingly sparse: the neurostore release annotation table holds 3.0M
    values in a shape that would be 92M dense cells.
``AnnotationSet``
    one annotation: its identity, its declared ``note_keys`` types, and its
    columns, kept together so that several annotations on one studyset stay
    distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["Dict8", "ColumnStore", "AnnotationSet"]


class Dict8:
    """Dictionary encoder for a low-cardinality string column."""

    __slots__ = ("categories", "_lookup")

    def __init__(self, categories=None):
        self.categories = list(categories or [])
        self._lookup = {value: i for i, value in enumerate(self.categories)}

    def code(self, value):
        """Return the code for ``value``, assigning a new one if unseen."""
        got = self._lookup.get(value, -1)
        if got == -1:
            got = len(self.categories)
            self.categories.append(value)
            self._lookup[value] = got
        return got

    def decode(self, codes):
        """Map an array of codes back to values."""
        if not self.categories:
            return np.full(len(codes), None, dtype=object)
        cats = np.asarray(self.categories, dtype=object)
        return cats[np.asarray(codes, dtype=np.int64)]

    def __len__(self):
        return len(self.categories)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Dict8({len(self.categories)} categories)"

    def __getstate__(self):
        return {"categories": list(self.categories)}

    def __setstate__(self, state):
        self.categories = list(state["categories"])
        self._lookup = {v: i for i, v in enumerate(self.categories)}


@dataclass
class ColumnStore:
    """Columns aligned to one level's row index, dense or sparse."""

    n_rows: int
    dense: dict = field(default_factory=dict)
    sparse: dict = field(default_factory=dict)

    def keys(self):
        """Every column name, dense and sparse."""
        return list(self.dense) + list(self.sparse)

    def __contains__(self, name):
        return name in self.dense or name in self.sparse

    def add_dense(self, name, values):
        values = np.asarray(values)
        if len(values) != self.n_rows:
            raise ValueError(
                f"column {name!r} has {len(values)} values, expected {self.n_rows}"
            )
        self.dense[name] = values

    def add_sparse(self, name, idx, values):
        """Record a column present only on ``idx``.

        An empty ``idx`` still registers the column: a field declared everywhere
        but populated nowhere is still a declared field, and dropping it would
        change what ``get_metadata()`` reports and lose it on export.
        """
        self.sparse[name] = (np.asarray(idx, dtype=np.int64), list(values))

    def copy(self):
        return ColumnStore(self.n_rows, dict(self.dense), dict(self.sparse))

    def get(self, name, sel=None, fill=None):
        """Values for ``sel`` (or every row) as an object array."""
        if name in self.dense:
            col = self.dense[name]
            return col if sel is None else col[sel]
        idx, values = self.sparse[name]
        n = self.n_rows if sel is None else len(sel)
        out = np.full(n, fill, dtype=object)
        if sel is None:
            for i, value in zip(idx, values):
                out[int(i)] = value
            return out
        # Assign element-wise so that list-valued entries (sample_sizes, for
        # one) stay single objects rather than being broadcast into a 2-D array.
        pos = np.searchsorted(sel, idx)
        ok = (pos < len(sel)) & (sel[np.minimum(pos, max(len(sel) - 1, 0))] == idx)
        for p, keep, value in zip(pos, ok, values):
            if keep:
                out[int(p)] = value
        return out

    def get_numeric(self, name, sel=None, fill=np.nan, reduce=np.mean):
        """Values for ``sel`` as float64, reducing list-valued entries.

        This is what blocks want: no ``None``, no object dtype, missing rows as
        NaN. The object-dtype :meth:`get` stays available for callers that need
        to tell ``None`` from ``nan``.
        """
        n = self.n_rows if sel is None else len(sel)
        out = np.full(n, fill, dtype=np.float64)
        if name not in self:
            return out
        raw = self.get(name, sel=sel)
        for i, value in enumerate(raw):
            if value is None:
                continue
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value):
                    out[i] = reduce(np.asarray(value, dtype=np.float64))
                continue
            try:
                out[i] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def rows(self, wanted):
        """``{row: {name: value}}`` for the requested rows, skipping nulls."""
        wanted = {int(r) for r in wanted}
        out = {}
        for name in self.keys():
            if name in self.dense:
                col = self.dense[name]
                pairs = ((i, col[i]) for i in wanted)
            else:
                idx, values = self.sparse[name]
                pairs = zip(idx, values)
            for row, value in pairs:
                row = int(row)
                if row not in wanted or value is None:
                    continue
                out.setdefault(row, {})[name] = value
        return out

    def reorder(self, order, inverse):
        """Permute in place: dense columns gather, sparse indices remap."""
        for name, col in list(self.dense.items()):
            self.dense[name] = col[order]
        for name, (idx, values) in list(self.sparse.items()):
            new_idx = inverse[np.asarray(idx, dtype=np.int64)]
            perm = np.argsort(new_idx, kind="stable")
            values = list(values)
            self.sparse[name] = (new_idx[perm], [values[i] for i in perm])

    def freeze(self):
        for col in self.dense.values():
            if isinstance(col, np.ndarray):
                col.flags.writeable = False
        return self


@dataclass
class AnnotationSet:
    """One annotation: identity, declared types, and columns over the analyses.

    NIMADS gives an annotation its own id, name and ``note_keys`` types, and a
    studyset may carry several. Keeping the set intact is what makes that
    representable -- and prevents two annotations that share a label name from
    overwriting one another, which matters because compose's default note key is
    ``included``, so any two compose annotations on one studyset collide.
    """

    id: str
    name: str = ""
    columns: ColumnStore = None
    note_key_types: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    description: Optional[str] = None

    def keys(self):
        return self.columns.keys()

    def dtype(self, label):
        """The type NIMADS declared for ``label``, if any."""
        return self.note_key_types.get(label)

    def copy(self):
        return AnnotationSet(
            id=self.id,
            name=self.name,
            columns=self.columns.copy(),
            note_key_types=dict(self.note_key_types),
            metadata=dict(self.metadata),
            description=self.description,
        )
