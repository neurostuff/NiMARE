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

    #: The ColumnStore on the store that this accessor's attributes live in.
    _attrs = None

    @property
    def row(self):
        """The store row this accessor points at."""
        return self._row

    def _attr(self, name):
        """Return one dense attribute at this row, or ``None`` when absent.

        Reading a column at a row was spelled three ways -- a helper that
        tolerated absence, a direct index that raised, and five inlined
        subscripts. Absence is a property of the document, not a bug, so it is
        ``None``.
        """
        cs = getattr(self._store, self._attrs)
        col = None if cs is None else cs.dense.get(name)
        return None if col is None else col[self._row]


class Study(_Row):
    """One study."""

    _attrs = "study_attrs"

    @property
    def id(self):
        """Return this study's id."""
        return str(self._store.study_key[self._row])

    @property
    def name(self):
        """Return this study's title."""
        return self._attr("name")

    @property
    def authors(self):
        """Return this study's authors."""
        return self._attr("authors")

    @property
    def publication(self):
        """Return the publication this study appeared in."""
        return self._attr("publication")

    @property
    def description(self):
        """Return this study's description."""
        return self._attr("description")

    @property
    def doi(self):
        """Return this study's DOI."""
        return self._attr("doi")

    @property
    def pmid(self):
        """Return this study's PubMed id."""
        return self._attr("pmid")

    @property
    def year(self):
        """Return this study's publication year."""
        return self._attr("year")

    @property
    def metadata(self):
        """Return this study's study-level metadata."""
        cs = self._store.study_metadata
        return {} if cs is None else cs.rows([self._row]).get(self._row, {})

    @property
    def analyses(self):
        """Return this study's analyses."""
        store = self._store
        lo, hi = store.analysis_offsets[self._row], store.analysis_offsets[self._row + 1]
        return [Analysis(store, r, self._context) for r in range(int(lo), int(hi))]

    def __repr__(self):
        """Return a debugging representation naming the study."""
        return f"<Study: {self.id}>"

    def __str__(self):
        """Return this study's title."""
        return f"{self.name}"


class Analysis(_Row):
    """One analysis (a contrast)."""

    _attrs = "analysis_attrs"

    @property
    def id(self):
        """Return this analysis' id."""
        return str(self._store.analysis_key[self._row])

    @property
    def full_id(self):
        """The ``study-analysis`` identifier."""
        return str(self._store.analysis_full_key[self._row])

    @property
    def name(self):
        """Return this analysis' name."""
        return self._attr("name")

    @property
    def description(self):
        """Return this analysis' description."""
        return self._attr("description")

    @property
    def study(self):
        """Return the study this analysis belongs to."""
        return Study(self._store, int(self._store.study_idx[self._row]), self._context)

    @property
    def metadata(self):
        """Return this analysis' metadata."""
        return self._store.metadata.rows([self._row]).get(self._row, {})

    def get_metadata(self, field=None):
        """Return this analysis' metadata, or one field of it."""
        merged = dict(self.study.metadata)
        merged.update(self.metadata)
        if field is None:
            return merged
        return merged.get(field)

    @property
    def texts(self):
        """Return the texts attached to this analysis."""
        cs = self._store.texts
        return {} if cs is None else cs.rows([self._row]).get(self._row, {})

    @property
    def annotations(self):
        """``{annotation id: {label: value}}`` for this analysis.

        Keyed by annotation because a studyset may carry several, and two of them
        may use the same label name.
        """
        out = {}
        for ann_id, annotation in self._store.annotations.items():
            note = annotation.columns.rows([self._row]).get(self._row, {})
            if note:
                out[ann_id] = note
        return out

    @property
    def labels(self):
        """``{label: value}`` merged across every annotation."""
        out = {}
        for annotation in self._store.annotations.values():
            out.update(annotation.columns.rows([self._row]).get(self._row, {}))
        return out

    @property
    def points(self):
        """Return this analysis' foci."""
        store = self._store
        lo, hi = store.point_offsets[self._row], store.point_offsets[self._row + 1]
        return [Point(store, r, self._context) for r in range(int(lo), int(hi))]

    @property
    def images(self):
        """Return this analysis' images."""
        store = self._store
        ia = store.image_attrs
        if ia is None or not ia.n_rows:
            return []
        rows = np.flatnonzero(ia.dense["analysis_idx"] == self._row)
        return [Image(store, r, self._context) for r in rows]

    @property
    def conditions(self):
        """Return the conditions this analysis declares."""
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
        """Return the weight of each condition."""
        store = self._store
        lo = int(store.condition_offsets[self._row])
        hi = int(store.condition_offsets[self._row + 1])
        return store.condition_weight[lo:hi].tolist()

    def __repr__(self):
        """Return a debugging representation naming the analysis."""
        return f"<Analysis: {self.id}>"

    def __str__(self):
        """Return this analysis' name."""
        return f"{self.name}: {len(self.points)} points"


class Point(_Row):
    """One focus, reported in the space its studyset is being read in."""

    def _projected(self):
        from nimare.studyset.layout import harmonized_coordinates

        target = getattr(self._context, "space", None)
        return harmonized_coordinates(self._store, target)

    @property
    def coordinates(self):
        """Return this focus as an ``(x, y, z)`` tuple."""
        xyz, _, _ = self._projected()
        return xyz[self._row].tolist()

    @property
    def x(self):
        """Return this focus' x coordinate."""
        return float(self._projected()[0][self._row, 0])

    @property
    def y(self):
        """Return this focus' y coordinate."""
        return float(self._projected()[0][self._row, 1])

    @property
    def z(self):
        """Return this focus' z coordinate."""
        return float(self._projected()[0][self._row, 2])

    @property
    def space(self):
        """Return the space this focus is stored in."""
        _, codes, space_dict = self._projected()
        return space_dict.categories[int(codes[self._row])]

    @property
    def kind(self):
        """Return what this focus measures, when the document says."""
        return self._store.kind_dict.categories[int(self._store.point_kind[self._row])]

    @property
    def id(self):
        """Return this focus' id."""
        return self._store.point_key[self._row]

    @property
    def values(self):
        """Return the point-level values attached to this focus."""
        return self._store.point_values.rows([self._row]).get(self._row, {})

    def __repr__(self):
        """Return a debugging representation naming the focus."""
        return f"<Point: {self.coordinates}>"


class Image(_Row):
    """One statistic map."""

    _attrs = "image_attrs"

    @property
    def value_type(self):
        """Return what this image holds, such as ``z`` or ``varcope``."""
        return self._attr("value_type")

    @property
    def url(self):
        """Return the URL this image was fetched from."""
        return self._attr("url")

    @property
    def filename(self):
        """Return the path this image is stored at."""
        return self._attr("filename")

    @property
    def space(self):
        """Return the space this image is in."""
        return self._attr("space")

    @property
    def metadata(self):
        """Return this image's metadata."""
        return self._attr("metadata")

    @property
    def analysis(self):
        """Return the analysis this image belongs to."""
        return Analysis(self._store, int(self._attr("analysis_idx")), self._context)

    def __repr__(self):
        """Return a debugging representation naming the image."""
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
