"""Copy-on-write growth: ``store -> store``.

Each operation returns a new store sharing every column it did not touch. The
parent-first discipline applies here too: build the new parent column, sort the
children by it, then compile the offsets from it.
"""

from __future__ import annotations

import numpy as np

from nimare.studyset.columns import AnnotationSet, ColumnStore
from nimare.studyset.layout import offsets_from_parents, point_parents
from nimare.studyset.store import replace

__all__ = ["with_annotation", "with_images", "with_metadata", "with_points"]


def with_metadata(store, name, values, *, level="analysis"):
    """A new store with one extra dense metadata column."""
    values = np.asarray(values)
    target = store.metadata if level == "analysis" else store.study_metadata
    expected = store.n_analyses if level == "analysis" else store.n_studies
    if len(values) != expected:
        raise ValueError(f"expected {expected} values, got {len(values)}")
    updated = target.copy()
    updated.add_dense(name, values)
    return replace(store, **{"metadata" if level == "analysis" else "study_metadata": updated})


def with_annotation(store, name, labels, matrix, rows, note_key_types=None):
    """A new store carrying an extra annotation.

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
    return replace(store, annotations=annotations)


def with_points(store, analysis_positions, xyz, *, space=None, kind=None, values=None):
    """A new store with extra foci, maintaining both traversal directions."""
    from nimare.studyset.columns import Dict8

    analysis_positions = np.asarray(analysis_positions, dtype=np.int32)
    new_xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    if len(analysis_positions) != len(new_xyz):
        raise ValueError("analysis_positions and xyz must have the same length")

    parents = np.concatenate(
        [point_parents(store).astype(np.int32), analysis_positions]
    )
    order = np.argsort(parents, kind="stable")

    space_dict = Dict8(store.space_dict.categories)
    kind_dict = Dict8(store.kind_dict.categories)
    all_xyz = np.concatenate([store.xyz, new_xyz])
    all_key = np.concatenate(
        [store.point_key, np.full(len(new_xyz), None, dtype=object)]
    )
    all_space = np.concatenate(
        [store.point_space, np.full(len(new_xyz), space_dict.code(space), dtype=np.int16)]
    )
    all_kind = np.concatenate(
        [store.point_kind, np.full(len(new_xyz), kind_dict.code(kind), dtype=np.int16)]
    )

    point_values = ColumnStore(len(all_xyz))
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
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
    return replace(
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
    """A new store with extra images.

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
        "filename": np.concatenate(
            [ia.dense["filename"], np.asarray(refs, dtype=object)]
        )[order],
        "value_type": np.concatenate(
            [ia.dense["value_type"], np.full(n_add, imtype, dtype=object)]
        )[order],
        "space": np.concatenate(
            [ia.dense["space"], np.full(n_add, space, dtype=object)]
        )[order],
        "metadata": np.concatenate(
            [
                ia.dense["metadata"],
                np.asarray(metadata if metadata is not None else [None] * n_add, dtype=object),
            ]
        )[order],
    }
    return replace(
        store,
        image_attrs=ColumnStore(len(dense["url"]), dense=dense),
        image_offsets=offsets_from_parents(dense["analysis_idx"], store.n_analyses),
    )
