"""The :class:`Studyset` a user holds: one store, one view, one context.

Deliberately thin. Everything it does is either a selection (index arithmetic),
a block (a derived shape), a frame (pandas compatibility) or a codec call. What
it no longer has is worth listing, because it is most of what the previous
implementation was: no retained source document, no ``Study``/``Analysis`` object
graph as storage, no projected-table cache, no table overrides, no revision
counters and no mutation tracking. Those existed to keep several copies of the
same data agreed with one another.
"""

from __future__ import annotations

import operator
import os

import numpy as np
import pandas as pd

from nimare.studyset import blocks as _blocks
from nimare.studyset import edit, requirements
from nimare.studyset.io import from_nimads, from_parquet, to_nimads_dict, write_nimads
from nimare.studyset.io.parquet import write_parquet
from nimare.studyset.layout import harmonize_space
from nimare.studyset.store import replace
from nimare.studyset.view import Context, View

__all__ = ["Studyset"]

_OPS = {
    "==": operator.eq,
    "eq": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
    ">": operator.gt,
    "gt": operator.gt,
    ">=": operator.ge,
    "ge": operator.ge,
    "<": operator.lt,
    "lt": operator.lt,
    "<=": operator.le,
    "le": operator.le,
    "in": lambda series, value: series.isin(value),
    "contains": lambda series, value: series.astype(str).str.contains(value, na=False),
}


class Studyset:
    """A collection of studies for meta-analysis.

    Parameters
    ----------
    source : :obj:`dict`, :obj:`str`, or :class:`~nimare.studyset.store.StudysetStore`
        A NIMADS studyset dictionary, a path to one as JSON, a path to a parquet
        release directory, or an existing store.
    target : :obj:`str` or None, default="mni152_2mm"
        Space to report coordinates in. The raw coordinates are always kept, so
        re-targeting is exact rather than cumulative.
    mask : Niimg-like or :class:`~nilearn.maskers.NiftiMasker`, optional
        Masker for execution. Defaults to the template brain mask for ``target``.
    annotations : :obj:`list` of :obj:`dict`, optional
        Extra NIMADS annotation payloads to attach.
    basepath : :obj:`str`, optional
        Directory that relative image paths are resolved against.
    """

    _id_cols = ["id", "study_id", "contrast_id"]

    def __init__(
        self,
        source,
        target="mni152_2mm",
        mask=None,
        annotations=None,
        basepath=None,
    ):
        from nimare.studyset.store import StudysetStore

        if isinstance(source, Studyset):
            store, view = source.store, source.view
            self._view = View(store, view.index, view.point_mask, view.context)
            return
        if isinstance(source, StudysetStore):
            store = source
        elif isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
            store = from_parquet(str(source))
        else:
            store = from_nimads(source, annotations=annotations)
        self._view = View(
            store,
            context=Context(space=target, masker=mask, basepath=basepath),
        )

    # ----------------------------------------------------------- construction
    @classmethod
    def _wrap(cls, view):
        obj = cls.__new__(cls)
        obj._view = view
        return obj

    @classmethod
    def from_nimads(cls, source, **kwargs):
        """Read a NIMADS studyset from a dict or a JSON path."""
        return cls(source, **kwargs)

    @classmethod
    def from_parquet(cls, directory, **kwargs):
        """Read a parquet studyset release directory."""
        return cls(from_parquet(str(directory)), **kwargs)

    @classmethod
    def from_sleuth(cls, sleuth_file, **kwargs):
        """Read a Sleuth text file."""
        from nimare.io import convert_sleuth_to_nimads_dict

        return cls(convert_sleuth_to_nimads_dict(sleuth_file), **kwargs)

    @classmethod
    def from_dataset(cls, dataset):
        """Convert a legacy :class:`~nimare.dataset.Dataset`.

        Not optimised: ``Dataset`` is deprecated, and this is the boundary where
        it becomes a studyset before anything else touches it.
        """
        from nimare.io import convert_dataset_to_nimads_dict

        return cls(
            convert_dataset_to_nimads_dict(dataset),
            target=dataset.space,
            mask=dataset.masker,
            basepath=_infer_basepath(dataset),
        )

    # --------------------------------------------------------------- identity
    @property
    def store(self):
        """The immutable store behind this studyset."""
        return self._view.store

    @property
    def view(self):
        """The selection over that store."""
        return self._view

    @property
    def id(self):
        return self.store.id

    @property
    def name(self):
        return self.store.name

    @property
    def ids(self):
        """:obj:`numpy.ndarray`: full ``study-analysis`` identifiers."""
        return self._view.keys.astype(str)

    @property
    def study_ids(self):
        """:obj:`numpy.ndarray`: unique study identifiers."""
        return self._view.study_keys.astype(str)

    @property
    def space(self):
        return self._view.context.space

    @property
    def masker(self):
        return self._view.context.resolved_masker()

    @property
    def basepath(self):
        return self._view.context.basepath

    @property
    def annotations(self):
        """:obj:`list` of :class:`~nimare.studyset.columns.AnnotationSet`."""
        return list(self.store.annotations.values())

    @property
    def studies(self):
        """Read-only nested accessors for the selected studies.

        Views over the columns rather than a second copy of the data, so reading
        them cannot drift from what the store holds.
        """
        from nimare.studyset.nested import studies_of

        return studies_of(self._view)

    @property
    def analyses(self):
        """Read-only nested accessors for the selected analyses."""
        from nimare.studyset.nested import Analysis

        return [Analysis(self.store, r, self._view.context) for r in self._view.index]

    def __len__(self):
        return len(self._view)

    def __repr__(self):
        return f"<Studyset: {self.id}>"

    def __str__(self):
        return f"Studyset: {self.name} :: studies: {len(self.study_ids)}"

    # ----------------------------------------------------------------- frames
    @property
    def coordinates(self):
        """One row per focus."""
        return self._view.frame("coordinates")

    @property
    def images(self):
        """One row per analysis, one column per image type."""
        return self._view.frame("images")

    @property
    def metadata(self):
        """One row per analysis, study metadata merged in."""
        return self._view.frame("metadata")

    @property
    def texts(self):
        return self._view.frame("texts")

    @property
    def annotations_df(self):
        """Every annotation flattened into one frame.

        Collisions between annotations are qualified with the annotation id and
        recorded in ``frame.attrs["annotation_collisions"]`` rather than one
        silently overwriting the other.
        """
        return self._view.frame("annotations")

    # ----------------------------------------------------------------- blocks
    def resolve(self, requirements_, drop_invalid=True):
        """Narrow to the analyses that satisfy every requirement, and build blocks."""
        narrowed, resolved = self._view.resolve(requirements_, drop_invalid=drop_invalid)
        return Studyset._wrap(narrowed), resolved

    def coordinate_block(self):
        return self._view.coordinate_block()

    def image_block(self, imtype, *, policy="all"):
        return self._view.image_block(imtype, policy=policy)

    def label_block(self, annotation=None):
        return self._view.label_block(annotation)

    def text_block(self, field="abstract"):
        return self._view.text_block(field)

    def sample_sizes(self, reduce="mean"):
        """One sample size per analysis, from whichever level declares it."""
        return requirements.PerAnalysis("sample_sizes", reduce=reduce).resolve(self._view)

    # -------------------------------------------------------------- selection
    def slice(self, ids=None, *, analyses=None, filter_level="analysis"):
        """A studyset with only the requested ids."""
        if ids is None and analyses is not None:
            ids = analyses
        elif ids is None:
            raise TypeError("slice() requires 'ids'")
        if filter_level == "study":
            return self.filter_study_ids(ids)
        if filter_level != "analysis":
            raise ValueError(
                f"filter_level must be 'analysis' or 'study', got {filter_level!r}"
            )
        return self.filter_ids(ids)

    def filter_ids(self, ids):
        """Keep the named analyses. Full ids or short analysis ids both work."""
        return Studyset._wrap(self._view.select_keys(ids))

    def filter_study_ids(self, study_ids):
        return Studyset._wrap(self._view.select_studies(study_ids))

    def exclude_study_ids(self, study_ids):
        return Studyset._wrap(self._view.select_studies(study_ids, exclude=True))

    def filter_annotations(self, labels, threshold=0.001, match="all", annotation=None):
        """Keep analyses whose annotation labels reach ``threshold``."""
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            raise ValueError(f"Argument 'labels' cannot be {type(labels)}")
        if match not in ("all", "any"):
            raise ValueError("match must be 'all' or 'any'")
        block = self._label_block_for(labels, annotation)
        masks = np.vstack([block.above(label, threshold) for label in labels])
        keep = masks.all(axis=0) if match == "all" else masks.any(axis=0)
        return Studyset._wrap(self._view.select(keep))

    def filter_metadata(self, field, op, value):
        """Keep analyses whose metadata ``field`` satisfies ``op value``."""
        frame = self.metadata
        if field not in frame.columns:
            raise ValueError(f"Unknown metadata field: {field}")
        if op not in _OPS:
            raise ValueError(f"Unsupported metadata operator: {op}")
        keep = _OPS[op](frame[field], value)
        if not isinstance(keep, pd.Series):
            keep = pd.Series(keep, index=frame.index)
        return Studyset._wrap(
            self._view.select(keep.fillna(False).to_numpy(dtype=bool))
        )

    def select_points(self, point_mask):
        """Keep a subset of foci and every analysis."""
        return Studyset._wrap(self._view.select_points(point_mask))

    def with_context(self, **changes):
        """A studyset with different space, masker or basepath."""
        return Studyset._wrap(self._view.with_context(**changes))

    def update_path(self, new_path):
        """Resolve relative image paths against ``new_path``."""
        return self.with_context(basepath=os.path.abspath(new_path))

    def copy(self):
        """A new handle on the same immutable store."""
        return Studyset._wrap(
            View(self.store, self._view.index, self._view.point_mask, self._view.context)
        )

    # ---------------------------------------------------------------- queries
    def _label_block_for(self, labels, annotation=None):
        store = self.store
        if annotation is not None:
            block = self._view.label_block(annotation)
        elif len(store.annotations) <= 1:
            block = self._view.label_block()
        else:
            block = _blocks.label_block_union(self._view)[0]
        known = set(block.labels.tolist())
        missing = [label for label in labels if label not in known]
        if missing:
            raise ValueError(f"Missing label(s): {', '.join(map(str, missing))}")
        return block

    def get_labels(self, ids=None):
        """Labels present in the studyset's annotations."""
        if not self.store.annotations:
            return []
        view = self._view if ids is None else self._view.select_keys(ids)
        if len(self.store.annotations) == 1:
            block = view.label_block()
        else:
            block = _blocks.label_block_union(view)[0]
        if ids is None:
            return block.labels.tolist()
        counts = block.counts_above(-np.inf) if len(view) else np.zeros(len(block.labels))
        present = np.asarray((block.values != 0).sum(axis=0)).ravel() > 0
        return [label for label, keep in zip(block.labels.tolist(), present) if keep]

    def get_studies_by_label(self, labels=None, label_threshold=0.001, annotation=None):
        """Full analysis ids whose labels reach the threshold."""
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            raise ValueError(f"Argument 'labels' cannot be {type(labels)}")
        block = self._label_block_for(labels, annotation)
        masks = np.vstack([block.above(label, label_threshold) for label in labels])
        return list(self.ids[masks.all(axis=0)])

    def get_analyses_by_label(self, labels=None, label_threshold=0.001, annotation=None):
        """Short analysis ids whose labels reach the threshold."""
        return [
            str(i).rsplit("-", 1)[-1]
            for i in self.get_studies_by_label(labels, label_threshold, annotation)
        ]

    def get_studies_by_mask(self, mask):
        """Full analysis ids with at least one focus inside ``mask``."""
        from nilearn.image import load_img

        from nimare.utils import _mask_img_to_bool

        if self.store.n_points == 0:
            return []
        mask = load_img(mask)
        flagged = self._view.points_in_mask(_mask_img_to_bool(mask), mask.affine)
        return list(self._view.analyses_with_points(flagged).keys.astype(str))

    def get_analyses_by_mask(self, mask):
        return [str(i).rsplit("-", 1)[-1] for i in self.get_studies_by_mask(mask)]

    def get_studies_by_coordinate(self, xyz, r=20):
        """Full analysis ids with a focus within ``r`` mm of any of ``xyz``."""
        if self.store.n_points == 0:
            return []
        flagged = self._view.points_near(xyz, r)
        return list(self._view.analyses_with_points(flagged).keys.astype(str))

    def get_analyses_by_coordinate(self, xyz, r=None, n=None):
        """Short analysis ids near ``xyz``, by radius or by count."""
        if (r is None) == (n is None):
            raise ValueError("Exactly one of r or n must be provided.")
        xyz = np.asarray(xyz).ravel()
        if xyz.shape != (3,):
            raise ValueError("xyz must be a 1 x 3 array-like object.")
        if self.store.n_points == 0:
            return []
        block = self.coordinate_block()
        distances = np.sqrt(((block.xyz - xyz) ** 2).sum(axis=1))
        groups = block.group_of_point()
        if r is not None:
            hit = np.unique(groups[distances <= r])
        else:
            hit = np.unique(groups[np.argsort(distances)[:n]])
        return [str(k).rsplit("-", 1)[-1] for k in block.group_keys[hit]]

    def get_metadata(self, field=None, ids=None):
        """Metadata field names, or one field's values."""
        frame = self.metadata if ids is None else self.slice(ids).metadata
        available = [c for c in frame.columns if c not in self._id_cols]
        if field is None:
            return available
        if field not in available:
            raise ValueError(
                f"{field} not found in metadata.\nAvailable: {', '.join(available)}"
            )
        return frame[field].tolist()

    def get_images(self, imtype=None, ids=None, policy="first"):
        """Image types present, or one type's paths."""
        view = self._view if ids is None else self._view.select_keys(ids)
        if imtype is None:
            ia = self.store.image_attrs
            if ia is None or not ia.n_rows:
                return []
            return sorted({t for t in ia.dense["value_type"] if t})
        block = _blocks.image_block(view, imtype, policy=policy)
        out = [None] * len(view)
        for pos, ref in zip(block.analysis_pos, block.refs):
            if out[pos] is None:
                out[pos] = ref
        return out

    def get_texts(self, text_type=None, ids=None):
        """Text field names, or one field's values."""
        frame = self.texts if ids is None else self.slice(ids).texts
        available = [c for c in frame.columns if c not in self._id_cols]
        if text_type is None:
            return available
        if text_type not in available:
            raise ValueError(f"{text_type} not found in texts.")
        return frame[text_type].tolist()

    def get_points(self, analyses=None):
        """``{analysis id: [point dicts]}``."""
        store = self.store
        rows = self._rows_for(analyses)
        cache = _points_by_row(store)
        return {str(store.analysis_key[r]): cache.get(int(r), []) for r in rows}

    def get_analyses_by_metadata(self, key, value=None):
        """``{analysis id: {key: value}}`` for analyses carrying ``key``."""
        store = self.store
        out = {}
        for level in (store.metadata, store.study_metadata):
            if level is None or key not in level:
                continue
            rows = (
                self._view.index
                if level is store.metadata
                else store.study_idx[self._view.index]
            )
            values = level.get(key, sel=None)
            for pos, row in enumerate(rows):
                found = values[int(row)]
                if found is None:
                    continue
                if value is None or found == value:
                    analysis_row = self._view.index[pos]
                    out[str(store.analysis_key[analysis_row])] = {key: found}
            if out:
                break
        return out

    def get_analyses_by_annotations(self, key, value=None):
        """Analyses carrying ``key``, which may name a label or a whole annotation.

        Accepts either form: the previous implementation keyed on the annotation
        id here while ``get_studies_by_label`` keyed on the label.
        """
        store = self.store
        out = {}
        if key in store.annotations:
            notes = store.annotations[key].columns.rows(self._view.index)
            for row in self._view.index:
                note = notes.get(int(row))
                if note and (value is None or note == value):
                    out[str(store.analysis_key[row])] = {key: note}
            return out
        for annotation in store.annotations.values():
            if key not in annotation.columns:
                continue
            values = annotation.columns.get(key, sel=None)
            for row in self._view.index:
                found = values[int(row)]
                if found is None:
                    continue
                if value is None or found == value:
                    out[str(store.analysis_key[row])] = {key: found}
        return out

    def get_annotations(self, analyses=None):
        """``{analysis id: {label: value}}`` across every annotation."""
        store = self.store
        rows = self._rows_for(analyses)
        cache = _notes_by_row(store)
        return {str(store.analysis_key[r]): cache.get(int(r), {}) for r in rows}

    def _rows_for(self, analyses):
        if analyses is None:
            return list(self._view.index)
        selected = self._view.select_keys(analyses)
        return list(selected.index)

    # ------------------------------------------------------------- structural
    def combine_analyses(self):
        """One analysis per study, foci and images concatenated.

        Annotation notes name pre-merge analyses, so they cannot be carried over
        and are dropped.
        """
        doc = to_nimads_dict(self.store, self._view.index, include_annotations=False)
        for study in doc.get("studies", []):
            analyses = study.get("analyses", [])
            if len(analyses) <= 1:
                continue
            merged = {
                "id": "_".join(str(a["id"]) for a in analyses),
                "name": "; ".join(str(a.get("name") or "") for a in analyses),
                "description": None,
                "metadata": _merge_dicts(a.get("metadata") for a in analyses),
                "texts": _merge_dicts(a.get("texts") for a in analyses),
                "conditions": [c for a in analyses for c in (a.get("conditions") or [])],
                "weights": [w for a in analyses for w in (a.get("weights") or [])],
                "images": [i for a in analyses for i in (a.get("images") or [])],
                "points": [p for a in analyses for p in (a.get("points") or [])],
            }
            study["analyses"] = [merged]
        return Studyset(
            doc,
            target=self.space,
            mask=self._view.context.masker,
            basepath=self.basepath,
        )

    def merge(self, right):
        """Merge another studyset in, preferring this one on conflicts."""
        if not isinstance(right, Studyset):
            raise ValueError("Can only merge with another Studyset")
        left_doc = to_nimads_dict(self.store, self._view.index)
        right_doc = to_nimads_dict(right.store, right.view.index)
        left_doc["id"] = f"{self.id}_{right.id}"
        left_doc["name"] = f"Merged: {self.name} + {right.name}"
        by_id = {str(s["id"]): s for s in left_doc["studies"]}
        for study in right_doc["studies"]:
            sid = str(study["id"])
            if sid not in by_id:
                left_doc["studies"].append(study)
                continue
            left = by_id[sid]
            for field in ("name", "authors", "publication", "doi", "pmid", "year"):
                if not left.get(field):
                    left[field] = study.get(field)
            left["metadata"] = _merge_dicts([study.get("metadata"), left.get("metadata")])
            have = {str(a["id"]) for a in left["analyses"]}
            left["analyses"].extend(
                a for a in study["analyses"] if str(a["id"]) not in have
            )
        existing = {a["id"] for a in (left_doc.get("annotations") or [])}
        left_doc.setdefault("annotations", []).extend(
            a for a in (right_doc.get("annotations") or []) if a["id"] not in existing
        )
        return Studyset(
            left_doc,
            target=self.space,
            mask=self._view.context.masker,
            basepath=self.basepath,
        )

    # ------------------------------------------------------------------ edits
    def with_annotation(self, name, labels, matrix, rows=None, note_key_types=None):
        """A studyset carrying an extra annotation. Copy-on-write."""
        if rows is None:
            rows = self._view.index
        store = edit.with_annotation(
            self.store, name, labels, matrix, rows, note_key_types
        )
        return Studyset._wrap(
            View(store, self._view.index, self._view.point_mask, self._view.context)
        )

    def with_points(self, analysis_positions, xyz, **kwargs):
        """A studyset with extra foci. Copy-on-write.

        An active point mask is materialised first: a mask is indexed against
        the store it was computed for, so appending foci would leave it stale.
        """
        store = self.store
        if self._view.point_mask is not None:
            store = edit.keep_points(store, self._view.point_mask)
        store = edit.with_points(store, analysis_positions, xyz, **kwargs)
        return Studyset._wrap(View(store, self._view.index, None, self._view.context))

    def with_images(self, analysis_positions, refs, imtype, **kwargs):
        """A studyset with extra images. Copy-on-write."""
        store = edit.with_images(self.store, analysis_positions, refs, imtype, **kwargs)
        return Studyset._wrap(
            View(store, self._view.index, self._view.point_mask, self._view.context)
        )

    def materialize_points(self):
        """A studyset whose store holds only the currently selected foci."""
        if self._view.point_mask is None:
            return self
        store = edit.keep_points(self.store, self._view.point_mask)
        return Studyset._wrap(View(store, self._view.index, None, self._view.context))

    def with_metadata(self, name, values, *, level="analysis"):
        """A studyset with one extra metadata column. Copy-on-write."""
        store = edit.with_metadata(self.store, name, values, level=level)
        return Studyset._wrap(
            View(store, self._view.index, self._view.point_mask, self._view.context)
        )

    def with_texts(self, rows, field, values):
        """A studyset with text added. Copy-on-write."""
        store = edit.with_texts(self.store, rows, field, values)
        return Studyset._wrap(
            View(store, self._view.index, self._view.point_mask, self._view.context)
        )

    def with_annotation_payload(self, payload):
        """A studyset carrying a NIMADS annotation payload. Copy-on-write."""
        store = edit.with_annotation_payload(self.store, payload)
        return Studyset._wrap(
            View(store, self._view.index, self._view.point_mask, self._view.context)
        )

    def harmonized(self, target):
        """A studyset whose stored coordinates are in ``target``."""
        return Studyset._wrap(
            View(
                harmonize_space(self.store, target),
                self._view.index,
                self._view.point_mask,
                self._view.context.with_(space=target),
            )
        )

    # ------------------------------------------------------------------- io
    def to_dict(self):
        """The nested NIMADS document for the selected analyses."""
        return to_nimads_dict(self.store, self._view.index)

    def to_nimads(self, filename):
        """Write NIMADS JSON."""
        write_nimads(self.store, filename, self._view.index)

    def to_parquet(self, directory):
        """Write a parquet studyset release directory."""
        return write_parquet(
            self.store if len(self._view) == self.store.n_analyses else _materialize(self),
            directory,
        )

    def to_dataset(self):
        """Convert to a legacy :class:`~nimare.dataset.Dataset`.

        Not optimised: ``Dataset`` is deprecated and this exists so that existing
        data can still be written out.
        """
        from nimare.io import convert_nimads_to_dataset

        return convert_nimads_to_dataset(self.to_dict())

    def save(self, filename):
        """Pickle the studyset."""
        import pickle

        with open(filename, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(filename):
        """Load a pickled studyset."""
        import pickle

        with open(filename, "rb") as fh:
            return pickle.load(fh)


def _infer_basepath(dataset):
    """Where a Dataset's relative image paths were resolved against."""
    basepath = getattr(dataset, "basepath", None)
    if basepath:
        return basepath
    images = getattr(dataset, "images", None)
    if images is None or not len(images):
        return None
    for column in images.columns:
        if not column.endswith("__relative"):
            continue
        absolute_col = column[: -len("__relative")]
        if absolute_col not in images.columns:
            continue
        for relative, absolute in zip(images[column], images[absolute_col]):
            if isinstance(relative, str) and isinstance(absolute, str):
                if absolute.endswith(relative):
                    return absolute[: -len(relative)].rstrip(os.sep) or os.sep
    return None


def _materialize(studyset):
    """A store containing only the selected analyses."""
    return from_nimads(studyset.to_dict())


def _merge_dicts(dicts):
    out = {}
    for d in dicts:
        if isinstance(d, dict):
            out.update(d)
    return out or None


def _points_by_row(store):
    """``{analysis row: [point dicts]}``, built once per store."""
    from nimare.studyset.store import derived

    cache = derived(store)
    got = cache.get("points_by_row")
    if got is None:
        coords = store.xyz.tolist()
        spaces = store.point_space.tolist()
        kinds = store.point_kind.tolist()
        space_cats = store.space_dict.categories
        kind_cats = store.kind_dict.categories
        offsets = store.point_offsets.tolist()
        got = {}
        for row in range(store.n_analyses):
            lo, hi = offsets[row], offsets[row + 1]
            if lo == hi:
                continue
            got[row] = [
                {
                    "coordinates": coords[i],
                    "space": space_cats[spaces[i]],
                    "kind": kind_cats[kinds[i]],
                }
                for i in range(lo, hi)
            ]
        cache["points_by_row"] = got
    return got


def _notes_by_row(store):
    """``{analysis row: {label: value}}`` across annotations, built once."""
    from nimare.studyset.store import derived

    cache = derived(store)
    got = cache.get("notes_by_row")
    if got is None:
        got = {}
        for annotation in store.annotations.values():
            for row, note in annotation.columns.rows(range(store.n_analyses)).items():
                got.setdefault(row, {}).update(note)
        cache["notes_by_row"] = got
    return got
