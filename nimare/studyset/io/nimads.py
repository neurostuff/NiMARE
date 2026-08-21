"""NIMADS reader and writer.

Reading is a single streaming pass into typed Python lists, then one conversion
per column: no intermediate DataFrames, no per-row dicts, no structural deep copy
of the source document. Points arrive already grouped by analysis, so the CSR
offsets are cumulative counts.

Writing reconstructs the nested document from the columns. It is allowed to be
slower than reading -- and is -- because nothing in an analysis pipeline exports
in a loop.
"""

from __future__ import annotations

import json

import numpy as np

from nimare.studyset.columns import AnnotationSet, ColumnStore, Dict8
from nimare.studyset.layout import canonicalize, offsets_from_parents
from nimare.studyset.store import StudysetStore, freeze

__all__ = ["from_nimads", "to_nimads_dict", "write_nimads"]

_STUDY_ATTRS = (
    "name",
    "doi",
    "pmid",
    "authors",
    "year",
    "publication",
    "description",
)


def _denull(value):
    """Treat the literal string ``"None"`` as missing.

    Neurostore exports carry it where a null was meant, and a studyset that
    reports the string "None" as a metadata value poisons every comparison
    downstream.
    """
    return None if value == "None" else value


def from_nimads(source, *, canonical_order=True, annotations=None):
    """Build a store from a NIMADS studyset document.

    Parameters
    ----------
    source : :obj:`dict` or :obj:`str`
        A NIMADS studyset dictionary, or a path to one as JSON.
    canonical_order : :obj:`bool`, default=True
        Sort studies by id and analyses by ``study-analysis``. Document order is
        retained in the ``*_source_order`` columns either way.
    annotations : :obj:`list` of :obj:`dict`, optional
        Extra annotation payloads to attach, in addition to any the document
        carries.
    """
    from nimare.io import (
        _extract_coerced_sample_sizes,
        _extract_coordinate_row_metadata,
        _normalize_image_type,
        _point_value_kind_to_coordinate_column,
    )

    if isinstance(source, (str, bytes)):
        with open(source) as fh:
            source = json.load(fh)
    if not isinstance(source, dict):
        raise TypeError(f"NIMADS source must be a dict or a path, not {type(source)}")

    payloads = list(source.get("annotations") or [])
    if annotations is not None:
        payloads.extend(
            annotations if isinstance(annotations, (list, tuple)) else [annotations]
        )

    studies = source.get("studies") or []

    study_key = []
    study_attrs = {name: [] for name in _STUDY_ATTRS}
    study_has_md, study_md_rows, study_md_payload = [], [], []
    analysis_offsets = [0]

    analysis_key, analysis_full_key, study_idx, analysis_name = [], [], [], []
    analysis_desc, analysis_has_md = [], []
    md_rows, md_payload = [], []
    text_rows, text_payload = [], []

    coord_flat, point_parent, point_key = [], [], []
    point_space, point_kind = [], []
    space_dict, kind_dict = Dict8(), Dict8()
    pv_rows, pv_columns, pv_values = [], [], []

    img_parent, img_url, img_file, img_type, img_space, img_meta = [], [], [], [], [], []
    cond_code, cond_weight = [], []
    cond_dict, cond_desc = Dict8(), {}
    point_counts, image_counts, condition_counts = [], [], []

    inline_notes = {}

    a_row = 0
    for s_row, study in enumerate(studies):
        # NIMADS does not require ids; generate stable positional ones so that
        # a document without them still round-trips and can be selected from.
        sid = str(study["id"]) if study.get("id") is not None else f"study-{s_row}"
        study_key.append(sid)
        for name in _STUDY_ATTRS:
            study_attrs[name].append(_denull(study.get(name)))
        smd = study.get("metadata")
        study_has_md.append(isinstance(smd, dict))
        if isinstance(smd, dict) and smd:
            study_md_rows.append(s_row)
            study_md_payload.append(smd)

        for analysis in study.get("analyses") or []:
            aid = (
                str(analysis["id"])
                if analysis.get("id") is not None
                else f"analysis-{a_row}"
            )
            analysis_key.append(aid)
            analysis_full_key.append(f"{sid}-{aid}")
            study_idx.append(s_row)
            analysis_name.append(analysis.get("name"))
            analysis_desc.append(analysis.get("description"))

            points = analysis.get("points") or []
            for point in points:
                coords = point.get("coordinates") or [np.nan, np.nan, np.nan]
                coord_flat.extend(coords[:3])
                point_parent.append(a_row)
                point_key.append(point.get("id"))
                point_space.append(space_dict.code(point.get("space")))
                point_kind.append(kind_dict.code(point.get("kind")))
                for value in point.get("values") or []:
                    if not isinstance(value, dict) or value.get("value") is None:
                        continue
                    column = _point_value_kind_to_coordinate_column(value.get("kind"))
                    if column is None:
                        continue
                    pv_rows.append(len(point_key) - 1)
                    pv_columns.append(column)
                    pv_values.append(value["value"])
            point_counts.append(len(points))

            images = analysis.get("images") or []
            kept_images = 0
            for image in images:
                imtype = _normalize_image_type(image.get("value_type"))
                if imtype is None:
                    continue
                img_parent.append(a_row)
                img_url.append(image.get("url"))
                img_file.append(image.get("filename"))
                img_type.append(imtype)
                img_space.append(image.get("space"))
                img_meta.append(image.get("metadata"))
                kept_images += 1
            image_counts.append(kept_images)

            conditions = analysis.get("conditions") or []
            weights = analysis.get("weights") or []
            for i, condition in enumerate(conditions):
                if isinstance(condition, dict):
                    name, description = condition.get("name"), condition.get("description")
                else:
                    name, description = condition, None
                code = cond_dict.code(name)
                if description is not None:
                    cond_desc.setdefault(code, description)
                cond_code.append(code)
                cond_weight.append(weights[i] if i < len(weights) else 1.0)
            condition_counts.append(len(conditions))

            amd = analysis.get("metadata")
            analysis_has_md.append(isinstance(amd, dict))
            if isinstance(amd, dict) and amd:
                md_rows.append(a_row)
                md_payload.append(amd)

            texts = analysis.get("texts")
            if isinstance(texts, dict) and texts:
                text_rows.append(a_row)
                text_payload.append({k: _denull(v) for k, v in texts.items()})

            inline = analysis.get("annotations")
            if isinstance(inline, dict) and inline:
                flat = {}
                for ann_id, note in inline.items():
                    if isinstance(note, dict):
                        rows, notes = inline_notes.setdefault(ann_id, ([], []))
                        rows.append(a_row)
                        notes.append({k: _denull(v) for k, v in note.items()})
                    else:
                        flat[ann_id] = _denull(note)
                if flat:
                    rows, notes = inline_notes.setdefault("_inline", ([], []))
                    rows.append(a_row)
                    notes.append(flat)

            a_row += 1
        analysis_offsets.append(a_row)

    n_s, n_a, n_p = len(study_key), a_row, len(point_key)

    # top-level annotation payloads, resolved against the analysis ids
    ann_meta = {}
    row_of_analysis = None
    for payload in payloads:
        ann_id = payload.get("id")
        ann_meta[ann_id] = {
            "name": payload.get("name") or "",
            "description": payload.get("description"),
            "metadata": payload.get("metadata") or {},
            "note_key_types": {
                key: (value.get("type") if isinstance(value, dict) else value)
                for key, value in (payload.get("note_keys") or {}).items()
            },
        }
        notes = payload.get("notes") or []
        if not notes:
            inline_notes.setdefault(ann_id, ([], []))
            continue
        if row_of_analysis is None:
            row_of_analysis = {key: i for i, key in enumerate(analysis_key)}
        rows, collected = inline_notes.setdefault(ann_id, ([], []))
        for note in notes:
            row = row_of_analysis.get(str(note.get("analysis")))
            if row is None:
                continue
            rows.append(row)
            collected.append(
                {k: _denull(v) for k, v in (note.get("note") or {}).items()}
            )

    xyz = np.asarray(coord_flat, dtype=np.float64).reshape(n_p, 3) if n_p else np.zeros((0, 3))
    point_analysis = np.asarray(point_parent, dtype=np.int32)

    store = StudysetStore(
        id=source.get("id"),
        name=source.get("name") or "",
        study_key=np.asarray(study_key, dtype=object),
        study_attrs=ColumnStore(
            n_s,
            dense={k: np.asarray(v, dtype=object) for k, v in study_attrs.items()},
        ),
        study_metadata=ColumnStore(n_s),
        study_has_metadata=np.asarray(study_has_md, dtype=bool),
        study_source_order=np.arange(n_s, dtype=np.int32),
        analysis_offsets=np.asarray(analysis_offsets, dtype=np.int64),
        analysis_key=np.asarray(analysis_key, dtype=object),
        analysis_full_key=np.asarray(analysis_full_key, dtype=object),
        study_idx=np.asarray(study_idx, dtype=np.int32),
        analysis_attrs=ColumnStore(
            n_a,
            dense={
                "name": np.asarray(analysis_name, dtype=object),
                "description": np.asarray(analysis_desc, dtype=object),
            },
        ),
        metadata=ColumnStore(n_a),
        analysis_has_metadata=np.asarray(analysis_has_md, dtype=bool),
        analysis_source_order=np.arange(n_a, dtype=np.int32),
        texts=ColumnStore(n_a),
        point_offsets=offsets_from_parents(point_analysis, n_a),
        image_offsets=np.concatenate(([0], np.cumsum(image_counts, dtype=np.int64)))
        if n_a
        else np.zeros(1, dtype=np.int64),
        condition_offsets=np.concatenate(([0], np.cumsum(condition_counts, dtype=np.int64)))
        if n_a
        else np.zeros(1, dtype=np.int64),
        point_analysis=point_analysis,
        xyz=xyz,
        point_key=np.asarray(point_key, dtype=object),
        point_space=np.asarray(point_space, dtype=np.int16),
        point_kind=np.asarray(point_kind, dtype=np.int16),
        point_values=ColumnStore(n_p),
        space_dict=space_dict,
        kind_dict=kind_dict,
        image_attrs=ColumnStore(
            len(img_url),
            dense={
                "analysis_idx": np.asarray(img_parent, dtype=np.int32),
                "url": np.asarray(img_url, dtype=object),
                "filename": np.asarray(img_file, dtype=object),
                "value_type": np.asarray(img_type, dtype=object),
                "space": np.asarray(img_space, dtype=object),
                "metadata": np.asarray(img_meta, dtype=object),
            },
        ),
        condition_code=np.asarray(cond_code, dtype=np.int32),
        condition_weight=np.asarray(cond_weight, dtype=np.float64),
        condition_dict=cond_dict,
        condition_descriptions=cond_desc,
    )

    # point-value columns, grouped by target column name
    if pv_rows:
        columns = np.asarray(pv_columns, dtype=object)
        rows_arr = np.asarray(pv_rows, dtype=np.int64)
        values_arr = np.asarray(pv_values, dtype=object)
        for column in dict.fromkeys(pv_columns):
            mask = columns == column
            store.point_values.add_sparse(str(column), rows_arr[mask], values_arr[mask])

    # analysis metadata: one sparse column per field, plus the coordinate-level
    # arrays NIMADS smuggles through `coordinate_*` keys
    coordinate_columns = set()
    if md_rows:
        md_cols, coord_extra = {}, {}
        for row, payload in zip(md_rows, md_payload):
            n_pts = int(store.point_offsets[row + 1] - store.point_offsets[row])
            coord_rows, coord_keys = _extract_coordinate_row_metadata(payload, n_pts)
            lo = int(store.point_offsets[row])
            for column, values in coord_rows.items():
                bucket = coord_extra.setdefault(column, ([], []))
                for i, value in enumerate(values):
                    if value is not None:
                        bucket[0].append(lo + i)
                        bucket[1].append(value)
            for key, value in payload.items():
                if key in coord_keys:
                    continue
                bucket = md_cols.setdefault(str(key), ([], []))
                value = _denull(value)
                if value is None:
                    continue
                bucket[0].append(row)
                bucket[1].append(value)
        for key, (rows, values) in md_cols.items():
            store.metadata.add_sparse(key, rows, values)
        for column, (rows, values) in coord_extra.items():
            store.point_values.add_sparse(column, rows, values)
            coordinate_columns.add(column)
    store.coordinate_metadata_columns = frozenset(coordinate_columns)

    if study_md_rows:
        study_cols = {}
        for row, payload in zip(study_md_rows, study_md_payload):
            for key, value in payload.items():
                bucket = study_cols.setdefault(str(key), ([], []))
                value = _denull(value)
                if value is None:
                    continue
                bucket[0].append(row)
                bucket[1].append(value)
        for key, (rows, values) in study_cols.items():
            store.study_metadata.add_sparse(key, rows, values)

    if text_rows:
        fields = set()
        for payload in text_payload:
            fields.update(payload)
        for field_name in fields:
            rows = [r for r, p in zip(text_rows, text_payload) if p.get(field_name)]
            values = [p[field_name] for p in text_payload if p.get(field_name)]
            store.texts.add_sparse(field_name, rows, values)

    for ann_id, (rows, notes) in inline_notes.items():
        # One pass over the notes into per-label buckets. Looping labels on the
        # outside probes every note once per label: 794 labels x 22k notes is
        # 17.5M dict lookups, which made loading superlinear.
        buckets = {}
        for row, note in zip(rows, notes):
            for key, value in note.items():
                bucket = buckets.setdefault(key, ([], []))
                if value is None:
                    continue
                bucket[0].append(row)
                bucket[1].append(value)
        columns = ColumnStore(n_a)
        for key, (krows, kvalues) in buckets.items():
            columns.add_sparse(key, krows, kvalues)
        meta = ann_meta.get(ann_id, {})
        store.annotations[ann_id] = AnnotationSet(
            id=ann_id,
            name=meta.get("name", ""),
            columns=columns,
            note_key_types=meta.get("note_key_types", {}),
            metadata=meta.get("metadata", {}),
            description=meta.get("description"),
        )

    if canonical_order:
        store = canonicalize(store)
    return freeze(store)


# --------------------------------------------------------------------- writing


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _point_values(store):
    """``{point row: [NIMADS point-value dicts]}`` from the sparse columns."""
    from nimare.io import _coordinate_column_to_point_value_kind

    out = {}
    for column in store.point_values.keys():
        if column in store.coordinate_metadata_columns:
            continue  # these belong to analysis metadata, not to the point
        kind = _coordinate_column_to_point_value_kind(column)
        if kind is None:
            kind = column[len("value_"):] if column.startswith("value_") else column
        idx, values = store.point_values.sparse.get(column, (np.array([]), []))
        for row, value in zip(np.asarray(idx, dtype=np.int64), list(values)):
            if value is None:
                continue
            out.setdefault(int(row), []).append({"kind": kind, "value": _jsonable(value)})
    return out


def _coordinate_metadata(store, analysis_row):
    """``coordinate_*`` analysis metadata, reassembled from the point columns."""
    if not store.coordinate_metadata_columns:
        return {}
    lo = int(store.point_offsets[analysis_row])
    hi = int(store.point_offsets[analysis_row + 1])
    if lo == hi:
        return {}
    out = {}
    for column in store.coordinate_metadata_columns:
        idx, values = store.point_values.sparse.get(column, (np.array([]), []))
        series = [None] * (hi - lo)
        for i, value in zip(np.asarray(idx, dtype=np.int64), list(values)):
            if lo <= int(i) < hi:
                series[int(i) - lo] = _jsonable(value)
        if any(v is not None for v in series):
            out[f"coordinate_{column}"] = series
    return out


def _rows_with_declared(cs, rows, declared_for=None):
    """``{row: {field: value}}``, optionally filling declared-but-empty fields.

    ``declared_for`` is a boolean mask of rows whose source carried a metadata
    dict. Those rows get every declared field, null where absent, so a field that
    happens to be empty everywhere still survives a round trip.
    """
    if cs is None:
        return {}
    out = cs.rows(rows)
    if declared_for is not None and len(declared_for):
        declared = [name for name in cs.keys()]
        for row in rows:
            row = int(row)
            if row < len(declared_for) and declared_for[row]:
                entry = out.setdefault(row, {})
                for name in declared:
                    entry.setdefault(name, None)
    return {row: {k: _jsonable(v) for k, v in entry.items()} for row, entry in out.items()}


def annotations_to_nimads(store, analysis_rows):
    """Every annotation, as NIMADS annotation objects, restricted to ``analysis_rows``."""
    out = []
    for ann_id, annotation in store.annotations.items():
        cs = annotation.columns
        per_row = {}
        for label in cs.keys():
            if label in cs.dense:
                col = cs.dense[label]
                rows, values = np.arange(len(col)), list(col)
            else:
                rows, values = cs.sparse[label]
                rows = np.asarray(rows, dtype=np.int64)
                values = np.asarray(values)
                values = values.tolist() if values.dtype != object else list(values)
            for row, value in zip(rows, values):
                if value is None or (isinstance(value, float) and value != value):
                    continue
                per_row.setdefault(int(row), {})[label] = value

        notes = []
        for row in analysis_rows:
            note = per_row.get(int(row))
            if not note:
                continue
            notes.append(
                {
                    "analysis": store.analysis_key[row],
                    "study": store.study_key[store.study_idx[row]],
                    "note": note,
                }
            )
        if not notes:
            continue
        out.append(
            {
                "id": ann_id,
                "name": annotation.name,
                "description": annotation.description,
                "metadata": annotation.metadata,
                "note_keys": annotation.note_key_types
                or {label: None for label in sorted(cs.keys())},
                "notes": notes,
            }
        )
    return out


def to_nimads_dict(
    store,
    analysis_rows=None,
    *,
    include_annotations=True,
    order_by_source=True,
):
    """Rebuild the nested NIMADS document for the selected analyses."""
    if analysis_rows is None:
        analysis_rows = np.arange(store.n_analyses)
    analysis_rows = np.asarray(analysis_rows, dtype=np.int64)
    keep = np.zeros(store.n_analyses, dtype=bool)
    keep[analysis_rows] = True
    keep_l = keep.tolist()

    xyz_l = store.xyz.tolist()
    pkey_l = store.point_key.tolist()
    space_l = store.space_dict.decode(store.point_space).tolist()
    kind_l = store.kind_dict.decode(store.point_kind).tolist()
    p_off = store.point_offsets.tolist()
    a_off = store.analysis_offsets.tolist()
    c_off = store.condition_offsets.tolist()
    akey_l = store.analysis_key.tolist()
    aname_l = store.analysis_attrs.dense["name"].tolist()
    adesc_l = store.analysis_attrs.dense["description"].tolist()
    skey_l = store.study_key.tolist()
    study_cols = {
        name: (col.tolist() if hasattr(col, "tolist") else list(col))
        for name, col in store.study_attrs.dense.items()
    }

    values_by_point = _point_values(store)
    conditions = [
        {
            "name": store.condition_dict.categories[int(code)],
            "description": store.condition_descriptions.get(int(code)),
        }
        for code in store.condition_code
    ]
    weights = store.condition_weight.tolist() if len(store.condition_weight) else []
    md_by_analysis = _rows_with_declared(
        store.metadata, analysis_rows, declared_for=store.analysis_has_metadata
    )
    texts_by_analysis = _rows_with_declared(store.texts, analysis_rows)
    smd_by_study = _rows_with_declared(
        store.study_metadata,
        range(store.n_studies),
        declared_for=store.study_has_metadata,
    )

    images_by_analysis = {}
    ia = store.image_attrs
    if ia is not None and ia.n_rows:
        for i, parent in enumerate(ia.dense["analysis_idx"].tolist()):
            images_by_analysis.setdefault(parent, []).append(i)

    a_seq = store.analysis_source_order if order_by_source else None
    s_seq = store.study_source_order if order_by_source else None
    study_rows = (
        np.argsort(s_seq, kind="stable")
        if s_seq is not None and len(s_seq)
        else np.arange(store.n_studies)
    )

    studies_out = []
    for s_row in study_rows:
        s_row = int(s_row)
        a_rows = list(range(a_off[s_row], a_off[s_row + 1]))
        if a_seq is not None and len(a_seq):
            a_rows.sort(key=lambda r: int(a_seq[r]))
        analyses_out = []
        for a in a_rows:
            if not keep_l[a]:
                continue
            lo, hi = p_off[a], p_off[a + 1]
            c_lo, c_hi = c_off[a], c_off[a + 1]
            analysis_md = dict(md_by_analysis.get(a, {}))
            analysis_md.update(_coordinate_metadata(store, a))
            analyses_out.append(
                {
                    "id": akey_l[a],
                    "name": aname_l[a],
                    "description": adesc_l[a],
                    "metadata": analysis_md or None,
                    "texts": texts_by_analysis.get(a) or None,
                    "conditions": conditions[c_lo:c_hi],
                    "weights": weights[c_lo:c_hi],
                    "images": [
                        {
                            "url": _jsonable(ia.dense["url"][i]),
                            "filename": _jsonable(ia.dense["filename"][i]),
                            "value_type": _jsonable(ia.dense["value_type"][i]),
                            "space": _jsonable(ia.dense["space"][i]),
                            "metadata": _jsonable(ia.dense["metadata"][i]),
                        }
                        for i in images_by_analysis.get(a, [])
                    ],
                    "points": [
                        {
                            "id": pkey_l[i],
                            "coordinates": xyz_l[i],
                            "space": space_l[i],
                            "kind": kind_l[i],
                            "values": values_by_point.get(i, []),
                        }
                        for i in range(lo, hi)
                    ],
                }
            )
        had_analyses = a_off[s_row + 1] > a_off[s_row]
        if had_analyses and not analyses_out:
            continue  # every analysis of this study was filtered out
        study_out = {name: _jsonable(col[s_row]) for name, col in study_cols.items()}
        study_out["metadata"] = smd_by_study.get(s_row) or None
        study_out["id"] = skey_l[s_row]
        study_out["analyses"] = analyses_out
        studies_out.append(study_out)

    doc = {"id": store.id, "name": store.name, "studies": studies_out}
    if include_annotations:
        doc["annotations"] = annotations_to_nimads(store, analysis_rows)
    return doc


def write_nimads(store, path, analysis_rows=None, **kwargs):
    """Write the studyset as NIMADS JSON."""
    with open(path, "w") as fh:
        json.dump(to_nimads_dict(store, analysis_rows, **kwargs), fh)
