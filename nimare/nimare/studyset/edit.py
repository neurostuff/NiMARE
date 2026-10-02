"""Copy-on-write growth: ``store -> store``.

Each operation returns a new store sharing every column it did not touch. The
parent-first discipline applies here too: build the new parent column, sort the
children by it, then compile the offsets from it.
"""

from __future__ import annotations

import numpy as np
from pandas.api.types import is_numeric_dtype

from nimare.studyset.columns import ID_COLS, AnnotationSet, ColumnStore, is_missing
from nimare.studyset.layout import (
    inverse_permutation,
    note_row_resolver,
    offsets_from_parents,
    point_parents,
)
from nimare.studyset.store import replace as replace_store

__all__ = [
    "keep_images",
    "keep_points",
    "with_annotation",
    "with_annotation_payload",
    "with_annotations_frame",
    "with_images",
    "with_metadata",
    "with_points",
    "with_texts",
]


def with_metadata(store, name, values, *, level="analysis"):
    """Return a new store with one extra dense metadata column."""
    values = np.asarray(values)
    target = store.metadata if level == "analysis" else store.study_metadata
    expected = store.n_analyses if level == "analysis" else store.n_studies
    if len(values) != expected:
        raise ValueError(f"expected {expected} values, got {len(values)}")
    updated = target.copy()
    updated.add_dense(name, values)
    return replace_store(
        store, **{"metadata" if level == "analysis" else "study_metadata": updated}
    )


def with_annotation(store, name, labels, matrix, rows, note_key_types=None):
    """Return a new store carrying an extra annotation.

    Replaces the ``dset.copy()`` plus inner-merge idiom the annotators use: no
    deep copy, and analyses without a value are simply absent rather than dropped
    from the table.
    """
    import scipy.sparse as sp

    matrix = sp.csc_matrix(matrix)
    rows = np.asarray(rows, dtype=np.int64)
    columns = ColumnStore(store.n_analyses)
    for j, label in enumerate(labels):
        col = matrix[:, j]
        nz = col.nonzero()[0]
        if len(nz):
            columns.add_sparse(
                str(label), rows[nz], np.asarray(col.toarray().ravel()[nz], dtype=float)
            )
        else:
            columns.add_sparse(str(label), [], [])
    annotations = dict(store.annotations)
    annotations[name] = AnnotationSet(
        id=name,
        name=name,
        columns=columns,
        note_key_types=note_key_types or {str(label): "number" for label in labels},
    )
    return replace_store(store, annotations=annotations)


def with_points(store, analysis_positions, xyz, *, space=None, kind=None, values=None):
    """Return a new store with extra foci, maintaining both traversal directions."""
    from nimare.studyset.columns import Dict8

    analysis_positions = np.asarray(analysis_positions, dtype=np.int32)
    new_xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    if len(analysis_positions) != len(new_xyz):
        raise ValueError("analysis_positions and xyz must have the same length")

    parents = np.concatenate([point_parents(store).astype(np.int32), analysis_positions])
    order = np.argsort(parents, kind="stable")

    space_dict = Dict8(store.space_dict.categories)
    kind_dict = Dict8(store.kind_dict.categories)
    all_xyz = np.concatenate([store.xyz, new_xyz])
    all_key = np.concatenate([store.point_key, np.full(len(new_xyz), None, dtype=object)])
    all_space = np.concatenate(
        [store.point_space, np.full(len(new_xyz), space_dict.code(space), dtype=np.int16)]
    )
    all_kind = np.concatenate(
        [store.point_kind, np.full(len(new_xyz), kind_dict.code(kind), dtype=np.int16)]
    )

    point_values = ColumnStore(len(all_xyz))
    inverse = inverse_permutation(order)
    for name, (idx, vals) in store.point_values.sparse.items():
        new_idx = inverse[np.asarray(idx, dtype=np.int64)]
        perm = np.argsort(new_idx, kind="stable")
        vals = list(vals)
        point_values.add_sparse(name, new_idx[perm], [vals[i] for i in perm])
    if values:
        offset = len(store.xyz)
        for name, column in values.items():
            rows = inverse[offset + np.arange(len(new_xyz))]
            keep = [i for i, v in enumerate(column) if v is not None]
            if keep:
                point_values.add_sparse(name, rows[keep], [column[i] for i in keep])

    parents = parents[order]
    return replace_store(
        store,
        xyz=np.ascontiguousarray(all_xyz[order]),
        point_key=all_key[order],
        point_space=all_space[order],
        point_kind=all_kind[order],
        point_analysis=parents,
        point_offsets=offsets_from_parents(parents, store.n_analyses),
        point_values=point_values,
        space_dict=space_dict,
        kind_dict=kind_dict,
    )


def with_images(store, analysis_positions, refs, imtype, *, space=None, metadata=None):
    """Return a new store with extra images.

    A derived map sits alongside its source rather than overwriting it, which the
    one-column-per-type table could not express.
    """
    analysis_positions = np.asarray(analysis_positions, dtype=np.int32)
    refs = list(refs)
    n_add = len(refs)
    if len(analysis_positions) != n_add:
        raise ValueError("analysis_positions and refs must have the same length")

    ia = store.image_attrs
    parents = np.concatenate([ia.dense["analysis_idx"], analysis_positions])
    order = np.argsort(parents, kind="stable")
    dense = {
        "analysis_idx": parents[order].astype(np.int32),
        "url": np.concatenate([ia.dense["url"], np.full(n_add, None, dtype=object)])[order],
        "filename": np.concatenate([ia.dense["filename"], np.asarray(refs, dtype=object)])[order],
        "value_type": np.concatenate(
            [ia.dense["value_type"], np.full(n_add, imtype, dtype=object)]
        )[order],
        "space": np.concatenate([ia.dense["space"], np.full(n_add, space, dtype=object)])[order],
        "metadata": np.concatenate(
            [
                ia.dense["metadata"],
                np.asarray(metadata if metadata is not None else [None] * n_add, dtype=object),
            ]
        )[order],
    }
    return replace_store(
        store,
        image_attrs=ColumnStore(len(dense["url"]), dense=dense),
        image_offsets=offsets_from_parents(dense["analysis_idx"], store.n_analyses),
    )


def keep_images(store, mask):
    """Return a new store holding only the flagged images.

    The image counterpart of :func:`keep_points`. Images are level-2 children of
    an analysis just as foci are, so dropping them is the same operation: subset
    the columns, then recompile the offsets from the surviving parents.
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != store.n_images:
        raise ValueError(f"mask has {len(mask)} entries, expected {store.n_images}")
    keep = np.flatnonzero(mask)
    attrs = store.image_attrs.subset(keep)
    parents = np.asarray(attrs.dense["analysis_idx"], dtype=np.int32)
    return replace_store(
        store,
        image_attrs=attrs,
        image_offsets=offsets_from_parents(parents, store.n_analyses),
    )


def with_texts(store, rows, field, values):
    """Return a new store with text added to ``field`` for ``rows``."""
    texts = store.texts.copy() if store.texts is not None else ColumnStore(store.n_analyses)
    rows = np.asarray(rows, dtype=np.int64)
    values = list(values)
    if field in texts:
        existing_rows, existing_values = texts.entries(field)
        merged = {
            int(row): value
            for row, value in zip(existing_rows, existing_values)
            if value is not None
        }
        # The merged column is written back sparse, so a dense one has to go.
        texts.dense.pop(field, None)
    else:
        merged = {}
    for row, value in zip(rows, values):
        if value is not None:
            merged[int(row)] = value
    order = sorted(merged)
    texts.add_sparse(field, order, [merged[row] for row in order])
    return replace_store(store, texts=texts)


def with_annotations_frame(store, frame, name=None, id_column="id", replace=False):
    """Return a new store carrying one annotation built from a frame of labels.

    The write counterpart of the ``annotations_df`` read: a frame with an id
    column and one column per label, which is the shape the annotators and the
    text tools produce. Rows naming analyses the studyset does not hold are
    ignored, as they are for a NIMADS payload.

    ``replace`` discards the existing annotations rather than adding alongside
    them, which is what a caller who round-tripped ``annotations_df`` means.
    """
    resolve = note_row_resolver(
        store.study_key, store.analysis_key, store.analysis_full_key, store.study_idx
    )
    ids = frame[id_column].tolist()
    targets = np.fromiter(
        (-1 if (row := resolve(None, key)) is None else row for key in ids),
        dtype=np.int64,
        count=len(ids),
    )
    matched = targets >= 0

    columns = ColumnStore(store.n_analyses)
    note_key_types = {}
    for label in frame.columns:
        if label == id_column or label in ID_COLS:
            continue
        values = frame[label].to_numpy()
        keep = matched & np.asarray([not is_missing(v) for v in values], dtype=bool)
        rows = targets[keep]
        order = np.argsort(rows, kind="stable")
        columns.add_sparse(
            str(label), rows[order], [values[i] for i in np.flatnonzero(keep)[order]]
        )
        # pandas extension dtypes (StringDtype, for one) are not numpy dtypes,
        # so ask pandas rather than np.issubdtype.
        note_key_types[str(label)] = "number" if is_numeric_dtype(frame[label]) else "string"

    ann_id = name or "annotations"
    annotations = {} if replace else dict(store.annotations)
    annotations[ann_id] = AnnotationSet(
        id=ann_id, name=ann_id, columns=columns, note_key_types=note_key_types
    )
    return replace_store(store, annotations=annotations)


def with_annotation_payload(store, payload):
    """Return a new store carrying a NIMADS annotation payload.

    Notes are matched to analyses by id; notes naming analyses the studyset does
    not hold are ignored rather than fatal.
    """
    payload = dict(payload)
    ann_id = payload.get("id") or f"annotation{len(store.annotations)}"
    resolve = note_row_resolver(
        store.study_key, store.analysis_key, store.analysis_full_key, store.study_idx
    )
    buckets = {}
    for note in payload.get("notes") or []:
        row = resolve(note.get("study"), note.get("analysis"))
        if row is None:
            continue
        for key, value in (note.get("note") or {}).items():
            bucket = buckets.setdefault(key, ([], []))
            if value is None:
                continue
            bucket[0].append(row)
            bucket[1].append(value)
    declared = {
        key: (value.get("type") if isinstance(value, dict) else value)
        for key, value in (payload.get("note_keys") or {}).items()
    }
    for key in declared:
        buckets.setdefault(key, ([], []))
    columns = ColumnStore(store.n_analyses)
    for key, (rows, values) in buckets.items():
        columns.add_sparse(str(key), rows, values)
    annotations = dict(store.annotations)
    annotations[ann_id] = AnnotationSet(
        id=ann_id,
        name=payload.get("name") or "",
        columns=columns,
        note_key_types=declared,
        metadata=payload.get("metadata") or {},
        description=payload.get("description"),
    )
    return replace_store(store, annotations=annotations)


def keep_points(store, mask):
    """Return a new store holding only the flagged foci.

    A point mask belongs to the store it was computed against, so any edit that
    changes the point set invalidates it. Materialising the mask first keeps that
    from happening silently.
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != store.n_points:
        raise ValueError(f"mask has {len(mask)} entries, expected {store.n_points}")
    keep = np.flatnonzero(mask)
    parents = point_parents(store)[keep].astype(np.int32)
    point_values = store.point_values.subset(keep)
    return replace_store(
        store,
        xyz=np.ascontiguousarray(store.xyz[keep]),
        point_key=store.point_key[keep],
        point_space=store.point_space[keep],
        point_kind=store.point_kind[keep],
        point_analysis=parents,
        point_offsets=offsets_from_parents(parents, store.n_analyses),
        point_values=point_values,
    )
