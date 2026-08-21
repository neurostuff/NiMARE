"""pandas frames derived from a view.

Kept in their own module because nothing in the core needs pandas. These are the
shapes reports, ``to_dataset`` and user-facing inspection want; algorithms should
take blocks instead. The annotation frame in particular is the wrong shape for
the data -- the neurostore release annotation is 794 labels over 115,747
analyses, 92M dense cells holding 3.0M values -- which is what
:class:`~nimare.studyset.blocks.LabelBlock` exists to avoid.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from nimare.studyset import layout

__all__ = ["annotations", "coordinates", "images", "metadata", "texts"]

ID_COLS = ["id", "study_id", "contrast_id"]


def _id_columns(view):
    store = view.store
    sel = view.index
    return {
        "id": store.analysis_full_key[sel].astype(str),
        "study_id": store.study_key[store.study_idx[sel]].astype(str),
        "contrast_id": store.analysis_key[sel].astype(str),
    }


def _is_numeric(values):
    """True when every value present is a number (or a bool)."""
    seen = False
    for value in values:
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
            seen = True
            continue
        return False
    return seen


def _build(cols):
    """Build a frame from a column dict in one shot.

    Assigning columns one at a time is O(columns squared) in pandas' block
    manager: the release annotation frame has 794 label columns and took 21 s
    that way against 1.3 s built at once.
    """
    return pd.DataFrame(cols, copy=False)


def coordinates(view):
    """One row per focus: ids, x/y/z, space, then any point-level extras."""
    store = view.store
    block = view.coordinate_block()
    counts = block.group_sizes()
    codes = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    sel = view.index
    cats = list(block.space_categories) or list(store.space_dict.categories)
    # Plain repeats rather than pandas Categoricals: ids are not guaranteed
    # unique across a studyset (a study may appear more than once), and
    # `Categorical.from_codes` rejects duplicate categories. Repeating the string
    # arrays is also marginally faster.
    cols = {
        "id": np.repeat(block.group_keys.astype(str), counts),
        "study_id": np.repeat(store.study_key[store.study_idx[sel]].astype(str), counts),
        "contrast_id": np.repeat(store.analysis_key[sel].astype(str), counts),
        "x": block.xyz[:, 0],
        "y": block.xyz[:, 1],
        "z": block.xyz[:, 2],
    }
    cols["space"] = (
        np.asarray(cats, dtype=object)[block.space]
        if cats
        else np.full(len(block.xyz), None, dtype=object)
    )
    rows = block.point_rows
    if rows is None:
        rows = np.arange(len(block.xyz))
    kinds = store.kind_dict.categories
    if any(k is not None for k in kinds):
        cols["kind"] = store.kind_dict.decode(store.point_kind[rows])
    for name in sorted(store.point_values.keys()):
        cols[name] = store.point_values.get(name, sel=None)[rows]
    return _build(cols)


def images(view, *, policy="first"):
    """One row per analysis, one column per image type.

    The legacy shape. ``policy="first"`` is an explicit narrowing of a store that
    still holds every image, rather than a ceiling baked into storage.
    """
    from nimare.studyset.blocks import image_block

    store = view.store
    cols = _id_columns(view)
    n = len(view.index)
    ia = store.image_attrs
    types = sorted(
        {t for t in (ia.dense["value_type"] if ia is not None and ia.n_rows else []) if t}
    )
    space_col = np.full(n, None, dtype=object)
    for imtype in types:
        block = image_block(view, imtype, policy=policy)
        absolute = np.full(n, None, dtype=object)
        relative = np.full(n, None, dtype=object)
        raw = block.raw_refs if block.raw_refs is not None else block.refs
        for pos, ref, stored, sp in zip(
            block.analysis_pos, block.refs, raw, block.space
        ):
            if isinstance(ref, str) and ref:
                absolute[pos] = ref
            if (
                isinstance(stored, str)
                and stored
                and "://" not in stored
                and not os.path.isabs(stored)
            ):
                relative[pos] = stored
            if space_col[pos] is None:
                space_col[pos] = sp
        cols[imtype] = absolute
        cols[f"{imtype}__relative"] = relative
    cols["space"] = space_col
    return _build(cols)


def metadata(view):
    """Analysis metadata, merged with the study metadata it inherits."""
    store = view.store
    sel = view.index
    cols = _id_columns(view)
    if len(sel) == 0:
        return _build(cols)

    study_rows = store.study_idx[sel]
    present = np.sort(np.unique(study_rows))
    study_name = store.study_attrs.dense.get("name")
    analysis_name = store.analysis_attrs.dense.get("name")
    sname = (
        np.asarray(study_name, dtype=object)[study_rows]
        if study_name is not None
        else cols["study_id"]
    )
    aname = (
        np.asarray(analysis_name, dtype=object)[sel]
        if analysis_name is not None
        else cols["contrast_id"]
    )
    sname = np.array(
        [n if n else sid for n, sid in zip(sname, cols["study_id"])], dtype=object
    )
    aname = np.array(
        [n if n else cid for n, cid in zip(aname, cols["contrast_id"])], dtype=object
    )
    cols["study_name"] = sname
    cols["analysis_name"] = aname
    for src, dst in (("authors", "authors"), ("publication", "journal")):
        col = store.study_attrs.dense.get(src)
        cols[dst] = (
            np.asarray(col, dtype=object)[study_rows]
            if col is not None
            else np.full(len(sel), None, dtype=object)
        )
    cols["name"] = np.array([f"{a}-{b}" for a, b in zip(sname, aname)], dtype=object)

    # A study-level field counts as present only if a study with selected
    # analyses declares it, matching a table built from analysis rows.
    if store.study_metadata is not None:
        any_declared = bool(
            store.study_has_metadata is not None
            and len(store.study_has_metadata)
            and store.study_has_metadata[present].any()
        )
        for name in sorted(store.study_metadata.keys()):
            if name in ("sample_size", "sample_sizes"):
                continue
            idx, _ = store.study_metadata.sparse.get(name, (np.array([]), []))
            has_value = np.isin(np.asarray(idx, dtype=np.int64), present).any()
            if not has_value and not (any_declared and len(idx) == 0):
                continue
            per_study = store.study_metadata.get(name, sel=present)
            cols[name] = per_study[np.searchsorted(present, study_rows)]

    for name in sorted(store.metadata.keys()):
        if name in ("sample_size", "sample_sizes"):
            continue
        values = store.metadata.get(name, sel=sel)
        if name in cols:
            base = cols[name]
            cols[name] = np.array(
                [a if a is not None else b for a, b in zip(values, base)], dtype=object
            )
        else:
            cols[name] = values

    from nimare.studyset.requirements import coerced_sample_sizes

    sizes = coerced_sample_sizes(store)
    if any(sizes[int(row)] is not None for row in sel):
        # Build the object array explicitly: np.array of equal-length lists
        # collapses into a 2-D array, which pandas rejects.
        column = np.empty(len(sel), dtype=object)
        for i, row in enumerate(sel):
            column[i] = sizes[int(row)]
        cols["sample_sizes"] = column
    return _build(cols)


def texts(view):
    store = view.store
    cols = _id_columns(view)
    if store.texts is not None:
        for name in sorted(store.texts.keys()):
            cols[name] = store.texts.get(name, sel=view.index)
    return _build(cols)


def annotations(view, annotation=None):
    """The merged annotation frame, with collisions reported rather than silent."""
    store = view.store
    cols = _id_columns(view)
    if not store.annotations:
        return _build(cols)
    names = [annotation] if annotation is not None else sorted(store.annotations)
    seen, collisions = {}, []
    for name in names:
        cs = store.annotations[name].columns
        for label in sorted(cs.keys()):
            # Numeric labels come back as float64 with NaN for absent rows.
            # Annotation values are weights, and consumers do arithmetic on
            # them -- an object column reaches numpy as dtype=object and
            # `np.sqrt` has nothing to call.
            values = cs.get(label, sel=view.index)
            if _is_numeric(values):
                values = cs.get_numeric(label, sel=view.index)
            if label in seen:
                collisions.append((label, seen[label], name))
                cols[f"{name}.{label}"] = values
            else:
                seen[label] = name
                cols[label] = values
    out = _build(cols)
    out.attrs["annotation_collisions"] = collisions
    return out
