"""Layout rules: canonical ordering, offsets, and coordinate space.

Everything here maintains the store's structural invariants. The discipline that
matters, learned the hard way: **offsets are a compiled index over the parent
columns, never a second source of truth** (I2). Permuting the offsets directly
and patching the parent columns separately left 779 of 915 analyses linked to
their pre-sort parents on a real studyset, and export silently paired analyses
with the wrong studies.
"""

from __future__ import annotations

import numpy as np

from nimare.studyset.columns import ColumnStore
from nimare.studyset.store import derived, freeze

__all__ = [
    "canonicalize",
    "check_invariants",
    "harmonize_space",
    "harmonized_coordinates",
    "offsets_from_parents",
    "point_parents",
    "ranges_to_indices",
]


def ranges_to_indices(starts, stops):
    """Concatenate the ranges ``[starts[i], stops[i])`` in O(total)."""
    starts = np.asarray(starts, dtype=np.int64)
    stops = np.asarray(stops, dtype=np.int64)
    counts = stops - starts
    total = int(counts.sum())
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if total == 0:
        return np.empty(0, dtype=np.int64), offsets
    out = np.ones(total, dtype=np.int64)
    nz = counts > 0
    s, e, c = starts[nz], stops[nz], counts[nz]
    bounds = np.cumsum(c)
    out[0] = s[0]
    out[bounds[:-1]] = s[1:] - e[:-1] + 1
    np.cumsum(out, out=out)
    return out, offsets


def offsets_from_parents(parents, n_parents):
    """CSR offsets compiled from a parent-index column sorted ascending (I2)."""
    parents = np.asarray(parents, dtype=np.int64)
    return np.concatenate(([0], np.cumsum(np.bincount(parents, minlength=n_parents))))


def point_parents(store):
    """Return the point -> analysis column, stored rather than re-derived."""
    if store.point_analysis is not None:
        return store.point_analysis
    counts = np.diff(store.point_offsets)
    return np.repeat(np.arange(store.n_analyses, dtype=np.int32), counts)


def canonicalize(store):
    """Sort studies by id and analyses by full id, moving every child level.

    Order within an analysis is preserved (the sorts are stable), and the order
    the source document had is kept in ``*_source_order`` (I6) so that anything
    order-dependent -- a generated id, a condition/weight pairing -- does not
    depend on the storage layout.
    """
    s_order = np.argsort(store.study_key, kind="stable")
    s_inv = np.empty(len(s_order), dtype=np.int64)
    s_inv[s_order] = np.arange(len(s_order))
    store.study_key = store.study_key[s_order]
    for cs in store.study_level_stores():
        cs.reorder(s_order, s_inv)
    if store.study_source_order is not None and len(store.study_source_order):
        store.study_source_order = store.study_source_order[s_order]
    if store.study_has_metadata is not None and len(store.study_has_metadata):
        store.study_has_metadata = store.study_has_metadata[s_order]
    store.study_idx = s_inv[store.study_idx.astype(np.int64)].astype(np.int32)

    store = reorder_analyses(store, np.argsort(store.analysis_full_key, kind="stable"))

    # The analysis order now follows the study order, so study -> analyses is
    # just per-study counts.
    store.analysis_offsets = offsets_from_parents(store.study_idx, store.n_studies)
    # Reordering builds new arrays, so re-establish I1 before handing it back.
    return freeze(store)


def reorder_analyses(store, order):
    """Permute the analysis level, carrying every child with it (I2, I3)."""
    order = np.asarray(order, dtype=np.int64)
    n_a = len(order)
    inv = np.empty(n_a, dtype=np.int64)
    inv[order] = np.arange(n_a)

    # points: remap parents, sort by parent, recompile the offsets
    if store.n_points:
        new_parent = inv[store.point_analysis.astype(np.int64)]
        p_order = np.argsort(new_parent, kind="stable")
        store.point_analysis = new_parent[p_order].astype(np.int32)
        store.xyz = np.ascontiguousarray(store.xyz[p_order])
        store.point_key = store.point_key[p_order]
        store.point_space = store.point_space[p_order]
        store.point_kind = store.point_kind[p_order]
        p_inv = np.empty(len(p_order), dtype=np.int64)
        p_inv[p_order] = np.arange(len(p_order))
        for cs in store.point_level_stores():
            cs.reorder(p_order, p_inv)
        store.point_offsets = offsets_from_parents(store.point_analysis, n_a)
    else:
        store.point_offsets = np.zeros(n_a + 1, dtype=np.int64)

    # conditions: no parent column, so gather by range then recompile
    if store.condition_code is not None and len(store.condition_code):
        c_idx, _ = ranges_to_indices(
            store.condition_offsets[order], store.condition_offsets[order + 1]
        )
        counts = store.condition_offsets[order + 1] - store.condition_offsets[order]
        store.condition_code = store.condition_code[c_idx]
        store.condition_weight = store.condition_weight[c_idx]
        store.condition_offsets = np.concatenate(([0], np.cumsum(counts)))
    else:
        store.condition_offsets = np.zeros(n_a + 1, dtype=np.int64)

    # images: parent column, same discipline as points
    ia = store.image_attrs
    if ia is not None and ia.n_rows:
        new_parent = inv[ia.dense["analysis_idx"].astype(np.int64)]
        i_order = np.argsort(new_parent, kind="stable")
        for name, col in list(ia.dense.items()):
            ia.dense[name] = col[i_order]
        ia.dense["analysis_idx"] = new_parent[i_order].astype(np.int32)
        i_inv = np.empty(len(i_order), dtype=np.int64)
        i_inv[i_order] = np.arange(len(i_order))
        for name, (idx, values) in list(ia.sparse.items()):
            ni = i_inv[np.asarray(idx, dtype=np.int64)]
            perm = np.argsort(ni, kind="stable")
            values = list(values)
            ia.sparse[name] = (ni[perm], [values[i] for i in perm])
        store.image_offsets = offsets_from_parents(ia.dense["analysis_idx"], n_a)
    else:
        store.image_offsets = np.zeros(n_a + 1, dtype=np.int64)

    for cs in store.analysis_level_stores():
        cs.reorder(order, inv)

    store.analysis_key = store.analysis_key[order]
    store.analysis_full_key = store.analysis_full_key[order]
    store.study_idx = store.study_idx[order]
    if store.analysis_source_order is not None and len(store.analysis_source_order):
        store.analysis_source_order = store.analysis_source_order[order]
    if store.analysis_has_metadata is not None and len(store.analysis_has_metadata):
        store.analysis_has_metadata = store.analysis_has_metadata[order]
    return freeze(store)


def check_invariants(store):
    """Return a list of invariant violations. Empty means the store is sound."""
    problems = []
    n_s, n_a, n_p = store.n_studies, store.n_analyses, store.n_points

    # I1
    for name, value in vars(store).items():
        if name == "_derived":
            continue
        if isinstance(value, np.ndarray) and value.flags.writeable:
            problems.append(f"I1: {name} is writeable")
        elif isinstance(value, ColumnStore):
            for col_name, col in value.dense.items():
                if isinstance(col, np.ndarray) and col.flags.writeable:
                    problems.append(f"I1: {name}.{col_name} is writeable")

    # I4 / I2
    for label, offsets, n_children, parents in (
        ("analysis_offsets", store.analysis_offsets, n_a, store.study_idx),
        ("point_offsets", store.point_offsets, n_p, store.point_analysis),
        (
            "image_offsets",
            store.image_offsets,
            store.n_images,
            (
                None
                if store.image_attrs is None or not store.n_images
                else store.image_attrs.dense["analysis_idx"]
            ),
        ),
        (
            "condition_offsets",
            store.condition_offsets,
            0 if store.condition_code is None else len(store.condition_code),
            None,
        ),
    ):
        if offsets is None:
            problems.append(f"I4: {label} missing")
            continue
        expected_len = (n_s if label == "analysis_offsets" else n_a) + 1
        if len(offsets) != expected_len:
            problems.append(f"I4: {label} has length {len(offsets)}, expected {expected_len}")
            continue
        if offsets[0] != 0:
            problems.append(f"I4: {label} does not start at 0")
        if np.any(np.diff(offsets) < 0):
            problems.append(f"I4: {label} is not monotonic")
        if offsets[-1] != n_children:
            problems.append(f"I4: {label} ends at {offsets[-1]}, expected {n_children}")
        if parents is not None and len(parents):
            parents = np.asarray(parents, dtype=np.int64)
            if np.any(np.diff(parents) < 0):
                problems.append(f"I4: children of {label} are not sorted by parent")
            recompiled = offsets_from_parents(parents, expected_len - 1)
            if not np.array_equal(recompiled, np.asarray(offsets)):
                problems.append(f"I2: {label} disagrees with its parent column")

    # I3 / shapes
    for cs, n_rows, label in (
        (store.study_metadata, n_s, "study_metadata"),
        (store.metadata, n_a, "metadata"),
        (store.texts, n_a, "texts"),
        (store.point_values, n_p, "point_values"),
    ):
        if cs is not None and cs.n_rows != n_rows:
            problems.append(f"I3: {label}.n_rows is {cs.n_rows}, expected {n_rows}")
    for ann_id, ann in (store.annotations or {}).items():
        if ann.columns is not None and ann.columns.n_rows != n_a:
            problems.append(f"I3: annotation {ann_id} has {ann.columns.n_rows} rows")

    # I6
    for label, seq, n in (
        ("study_source_order", store.study_source_order, n_s),
        ("analysis_source_order", store.analysis_source_order, n_a),
    ):
        if seq is not None and len(seq) not in (0, n):
            problems.append(f"I6: {label} has length {len(seq)}, expected {n}")

    return problems


def _space_transforms(target):
    from nimare.utils import mni2tal, tal2mni

    lowered = str(target).lower()
    if "mni" in lowered or "ale" in lowered:
        return {"MNI": None, "TAL": tal2mni, "Talairach": tal2mni}
    if "tal" in lowered:
        return {"MNI": mni2tal, "TAL": None, "Talairach": None}
    raise ValueError(f"Unrecognized space: {target}")


def harmonize_space(store, target):
    """Return a store whose foci are all in ``target``.

    One vectorized transform per *space category* rather than per row, because
    the space column is dictionary-encoded. Unrecognized spaces are relabelled
    without being transformed, matching the legacy behaviour.
    """
    import logging

    from nimare.studyset.store import replace

    if target is None:
        return store
    transform = _space_transforms(target)
    xyz = np.array(store.xyz, copy=True)
    unknown = []
    for code, category in enumerate(store.space_dict.categories):
        if category not in transform:
            unknown.append(category)
            continue
        alg = transform[category]
        if alg is None:
            continue
        rows = store.point_space == code
        if rows.any():
            xyz[rows] = alg(xyz[rows])
    if unknown:
        logging.getLogger(__name__).warning(
            "Not applying transforms to coordinates in unrecognized space(s): %s",
            ", ".join(repr(u) for u in unknown),
        )
    from nimare.studyset.columns import Dict8

    space_dict = Dict8()
    code = space_dict.code(target)
    return replace(
        store,
        xyz=xyz,
        point_space=np.full(len(xyz), code, dtype=np.int16),
        space_dict=space_dict,
    )


def harmonized_coordinates(store, target):
    """``(xyz, space codes, space dict)`` in ``target``, memoised (I5).

    Derived rather than baked in at load, so re-targeting stays exact: projecting
    in place would make TAL -> MNI -> TAL lossy. The raw coordinates remain, and
    each requested target is one memoised array.
    """
    if target is None:
        return store.xyz, store.point_space, store.space_dict
    cache = derived(store)
    key = ("xyz", target)
    got = cache.get(key)
    if got is None:
        projected = harmonize_space(store, target)
        got = (projected.xyz, projected.point_space, projected.space_dict)
        cache[key] = got
    return got
