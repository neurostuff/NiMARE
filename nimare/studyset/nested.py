"""Read-only nested accessors over the columns.

Convenient for inspection and for code that reads a studyset the way the document
is shaped. These are *views*, not storage: they hold a store and a row, and every
attribute is a column lookup. The previous implementation kept an equivalent
object graph as a second copy of the data, which is what the revision counters
and mutation tracking existed to keep in step.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Analysis", "Image", "Point", "Study", "studies_of"]


class _Row:
    __slots__ = ("_store", "_row", "_context")

    def __init__(self, store, row, context=None):
        self._store = store
        self._row = int(row)
        # Coordinates are stored raw and projected on demand, so an accessor
        # needs to know which space it is being read in.
        self._context = context

    @property
    def row(self):
        """The store row this accessor points at."""
        return self._row


class Study(_Row):
    """One study."""

    @property
    def id(self):
        return str(self._store.study_key[self._row])

    @property
    def name(self):
        return self._attr("name")

    @property
    def authors(self):
        return self._attr("authors")

    @property
    def publication(self):
        return self._attr("publication")

    @property
    def description(self):
        return self._attr("description")

    @property
    def doi(self):
        return self._attr("doi")

    @property
    def pmid(self):
        return self._attr("pmid")

    @property
    def year(self):
        return self._attr("year")

    @property
    def metadata(self):
        cs = self._store.study_metadata
        return {} if cs is None else cs.rows([self._row]).get(self._row, {})

    @property
    def analyses(self):
        store = self._store
        lo, hi = store.analysis_offsets[self._row], store.analysis_offsets[self._row + 1]
        return [Analysis(store, r, self._context) for r in range(int(lo), int(hi))]

    def _attr(self, name):
        col = self._store.study_attrs.dense.get(name)
        return None if col is None else col[self._row]

    def __repr__(self):
        return f"<Study: {self.id}>"

    def __str__(self):
        return f"{self.name}"


class Analysis(_Row):
    """One analysis (a contrast)."""

    @property
    def id(self):
        return str(self._store.analysis_key[self._row])

    @property
    def full_id(self):
        """The ``study-analysis`` identifier."""
        return str(self._store.analysis_full_key[self._row])

    @property
    def name(self):
        return self._store.analysis_attrs.dense["name"][self._row]

    @property
    def description(self):
        return self._store.analysis_attrs.dense["description"][self._row]

    @property
    def study(self):
        return Study(self._store, int(self._store.study_idx[self._row]), self._context)

    @property
    def metadata(self):
        return self._store.metadata.rows([self._row]).get(self._row, {})

    def get_metadata(self, field=None):
        """This analysis' metadata, or one field of it."""
        merged = dict(self.study.metadata)
        merged.update(self.metadata)
        if field is None:
            return merged
        return merged.get(field)

    @property
    def texts(self):
        cs = self._store.texts
        return {} if cs is None else cs.rows([self._row]).get(self._row, {})

    @property
    def annotations(self):
        """``{label: value}`` across every annotation on the studyset."""
        out = {}
        for annotation in self._store.annotations.values():
            out.update(annotation.columns.rows([self._row]).get(self._row, {}))
        return out

    @property
    def points(self):
        store = self._store
        lo, hi = store.point_offsets[self._row], store.point_offsets[self._row + 1]
        return [Point(store, r, self._context) for r in range(int(lo), int(hi))]

    @property
    def images(self):
        store = self._store
        ia = store.image_attrs
        if ia is None or not ia.n_rows:
            return []
        rows = np.flatnonzero(ia.dense["analysis_idx"] == self._row)
        return [Image(store, r, self._context) for r in rows]

    @property
    def conditions(self):
        store = self._store
        lo = int(store.condition_offsets[self._row])
        hi = int(store.condition_offsets[self._row + 1])
        return [
            {
                "name": store.condition_dict.categories[int(store.condition_code[r])],
                "weight": float(store.condition_weight[r]),
            }
            for r in range(lo, hi)
        ]

    @property
    def weights(self):
        store = self._store
        lo = int(store.condition_offsets[self._row])
        hi = int(store.condition_offsets[self._row + 1])
        return store.condition_weight[lo:hi].tolist()

    def __repr__(self):
        return f"<Analysis: {self.id}>"

    def __str__(self):
        return f"{self.name}: {len(self.points)} points"


class Point(_Row):
    """One focus, reported in the space its studyset is being read in."""

    def _projected(self):
        from nimare.studyset.layout import harmonized_coordinates

        target = getattr(self._context, "space", None)
        return harmonized_coordinates(self._store, target)

    @property
    def coordinates(self):
        xyz, _, _ = self._projected()
        return xyz[self._row].tolist()

    @property
    def x(self):
        return float(self._projected()[0][self._row, 0])

    @property
    def y(self):
        return float(self._projected()[0][self._row, 1])

    @property
    def z(self):
        return float(self._projected()[0][self._row, 2])

    @property
    def space(self):
        _, codes, space_dict = self._projected()
        return space_dict.categories[int(codes[self._row])]

    @property
    def kind(self):
        return self._store.kind_dict.categories[int(self._store.point_kind[self._row])]

    @property
    def id(self):
        return self._store.point_key[self._row]

    @property
    def values(self):
        return self._store.point_values.rows([self._row]).get(self._row, {})

    def __repr__(self):
        return f"<Point: {self.coordinates}>"


class Image(_Row):
    """One statistic map."""

    @property
    def value_type(self):
        return self._store.image_attrs.dense["value_type"][self._row]

    @property
    def url(self):
        return self._store.image_attrs.dense["url"][self._row]

    @property
    def filename(self):
        return self._store.image_attrs.dense["filename"][self._row]

    @property
    def space(self):
        return self._store.image_attrs.dense["space"][self._row]

    @property
    def metadata(self):
        return self._store.image_attrs.dense["metadata"][self._row]

    @property
    def analysis(self):
        return Analysis(
            self._store,
            int(self._store.image_attrs.dense["analysis_idx"][self._row]),
            self._context,
        )

    def __repr__(self):
        return f"<Image: {self.value_type}>"


def studies_of(view):
    """Studies represented in ``view``.

    A study with no analyses at all cannot be excluded by an analysis selection,
    so it is always present -- the same rule export uses.
    """
    store = view.store
    selected = np.zeros(store.n_analyses, dtype=bool)
    selected[view.index] = True
    counts = np.diff(store.analysis_offsets)
    rows = [
        r
        for r in range(store.n_studies)
        if counts[r] == 0
        or selected[store.analysis_offsets[r] : store.analysis_offsets[r + 1]].any()
    ]
    return [Study(store, r, view.context) for r in rows]
