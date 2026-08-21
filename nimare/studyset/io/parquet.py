"""Parquet reader and writer for the neurostore studyset release format.

The release tables are already columnar, so reading is a set of column reads and
one ``searchsorted`` join. This is the intended path for large studysets: the
whole of neurostore -- 32,444 studies, 115,747 analyses, 871,660 foci -- loads in
about 4 seconds.
"""

from __future__ import annotations

import json
import os

import numpy as np

from nimare.studyset.columns import AnnotationSet, ColumnStore, Dict8
from nimare.studyset.layout import canonicalize, offsets_from_parents
from nimare.studyset.store import StudysetStore, freeze

__all__ = ["from_parquet", "write_parquet"]

_TABLES = ("studies", "analyses", "coordinates", "images", "metadata", "texts", "annotations")


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Parquet studyset IO requires pyarrow. Install it with "
            "`python -m pip install pyarrow`."
        ) from exc


def _resolve_rows(keys, wanted):
    """Row of each ``wanted`` value in ``keys``, or -1. Vectorized."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    pos = np.searchsorted(sorted_keys, wanted)
    ok = (pos < len(keys)) & (sorted_keys[np.minimum(pos, max(len(keys) - 1, 0))] == wanted)
    rows = np.full(len(wanted), -1, dtype=np.int64)
    rows[ok] = order[pos[ok]]
    return rows


def _column_store_from(table, n_rows, keys, *, skip=("id", "study_id", "contrast_id")):
    """Sparse columns filtered with Arrow validity bitmaps, no per-cell Python."""
    import pyarrow as pa

    cs = ColumnStore(n_rows)
    if table is None or table.num_rows == 0:
        return cs
    rows = _resolve_rows(keys, np.asarray(table.column("id").to_pylist(), dtype=object))
    for name in table.schema.names:
        if name in skip:
            continue
        col = table.column(name).combine_chunks()
        if col.null_count == len(col):
            cs.add_sparse(name, [], [])
            continue
        valid = np.asarray(col.is_valid())
        keep = valid & (rows >= 0)
        if not keep.any():
            cs.add_sparse(name, [], [])
            continue
        taken = col.filter(pa.array(keep))
        try:
            values = taken.to_numpy(zero_copy_only=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
            values = np.asarray(taken.to_pylist(), dtype=object)
        cs.add_sparse(name, rows[keep], values)
    return cs


def from_parquet(directory, *, load_annotations=True, canonical_order=True):
    """Build a store from a parquet studyset release directory."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    def read(name):
        path = os.path.join(directory, f"{name}.parquet")
        return pq.read_table(path) if os.path.exists(path) else None

    manifest = {}
    manifest_path = os.path.join(directory, "studyset.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    studies_t, analyses_t, coords_t = read("studies"), read("analyses"), read("coordinates")
    if studies_t is None or analyses_t is None:
        raise ValueError(f"{directory} is not a studyset parquet release")

    study_key = np.asarray(studies_t.column("study_id").to_pylist(), dtype=object)
    study_attrs = ColumnStore(
        len(study_key),
        dense={
            name: np.asarray(studies_t.column(name).to_pylist(), dtype=object)
            for name in studies_t.schema.names
            if name != "study_id"
        },
    )

    a_full = np.asarray(analyses_t.column("id").to_pylist(), dtype=object)
    a_study = np.asarray(analyses_t.column("study_id").to_pylist(), dtype=object)
    a_key = np.asarray(analyses_t.column("contrast_id").to_pylist(), dtype=object)
    a_name = np.asarray(analyses_t.column("name").to_pylist(), dtype=object)
    n_a = len(a_full)
    study_idx = _resolve_rows(study_key, a_study).astype(np.int32)

    if coords_t is not None and coords_t.num_rows:
        c_id = np.asarray(coords_t.column("id").to_pylist(), dtype=object)
        c_analysis = _resolve_rows(a_full, c_id)
        keep = c_analysis >= 0
        c_analysis = c_analysis[keep]
        order = np.argsort(c_analysis, kind="stable")
        c_analysis = c_analysis[order]
        xyz = np.column_stack(
            [
                np.asarray(coords_t.column(axis).to_numpy(zero_copy_only=False), dtype=np.float64)[
                    keep
                ][order]
                for axis in ("x", "y", "z")
            ]
        )

        def encode(colname):
            if colname not in coords_t.schema.names:
                return Dict8(), np.zeros(len(xyz), dtype=np.int16)
            raw = np.asarray(coords_t.column(colname).to_pylist(), dtype=object)[keep][order]
            d = Dict8()
            return d, np.asarray([d.code(v) for v in raw], dtype=np.int16)

        space_dict, space_codes = encode("space")
        kind_dict, kind_codes = encode("kind")
        point_analysis = c_analysis.astype(np.int32)
        point_values = ColumnStore(len(xyz))
        skip = {"id", "study_id", "contrast_id", "x", "y", "z", "space", "kind"}
        for name in coords_t.schema.names:
            if name in skip:
                continue
            col = coords_t.column(name).combine_chunks()
            if col.null_count == len(col):
                continue
            values = np.asarray(col.to_numpy(zero_copy_only=False), dtype=object)[keep][order]
            present = np.flatnonzero(
                np.asarray([v is not None and v == v for v in values], dtype=bool)
            )
            if present.size:
                point_values.add_sparse(name, present, values[present])
    else:
        xyz = np.zeros((0, 3))
        point_analysis = np.zeros(0, dtype=np.int32)
        space_dict, kind_dict = Dict8(), Dict8()
        space_codes = np.zeros(0, dtype=np.int16)
        kind_codes = np.zeros(0, dtype=np.int16)
        point_values = ColumnStore(0)

    # images: the release stores them as wide "<type>__source" columns, so the
    # one-per-type ceiling has already been applied upstream
    images_t = read("images")
    img_parent, img_ref, img_type, img_space = [], [], [], []
    if images_t is not None and images_t.num_rows:
        i_rows = _resolve_rows(a_full, np.asarray(images_t.column("id").to_pylist(), dtype=object))
        spaces = (
            np.asarray(images_t.column("space").to_pylist(), dtype=object)
            if "space" in images_t.schema.names
            else np.full(len(i_rows), None, dtype=object)
        )
        for name in images_t.schema.names:
            if not name.endswith("__source"):
                continue
            imtype = name[: -len("__source")]
            values = np.asarray(images_t.column(name).to_pylist(), dtype=object)
            present = np.flatnonzero(
                np.asarray([v is not None for v in values], dtype=bool) & (i_rows >= 0)
            )
            img_parent.extend(i_rows[present].tolist())
            img_ref.extend(values[present].tolist())
            img_type.extend([imtype] * len(present))
            img_space.extend(spaces[present].tolist())

    annotations = {}
    if load_annotations:
        ann_cs = _column_store_from(read("annotations"), n_a, a_full)
        if ann_cs.keys():
            declared = [a.get("id") for a in (manifest.get("annotations") or []) if a.get("id")]
            ann_id = declared[0] if declared else "release"
            annotations[ann_id] = AnnotationSet(
                id=ann_id, name="neurostore release annotation", columns=ann_cs
            )

    store = StudysetStore(
        id=manifest.get("id"),
        name=manifest.get("name", ""),
        study_key=study_key,
        study_attrs=study_attrs,
        study_metadata=ColumnStore(len(study_key)),
        study_has_metadata=np.zeros(len(study_key), dtype=bool),
        study_source_order=np.arange(len(study_key), dtype=np.int32),
        analysis_offsets=offsets_from_parents(study_idx, len(study_key)),
        analysis_key=a_key,
        analysis_full_key=a_full,
        study_idx=study_idx,
        analysis_attrs=ColumnStore(
            n_a,
            dense={"name": a_name, "description": np.full(n_a, None, dtype=object)},
        ),
        metadata=_column_store_from(read("metadata"), n_a, a_full),
        analysis_has_metadata=np.ones(n_a, dtype=bool),
        analysis_source_order=np.arange(n_a, dtype=np.int32),
        texts=_column_store_from(read("texts"), n_a, a_full),
        point_offsets=offsets_from_parents(point_analysis, n_a),
        image_offsets=(
            offsets_from_parents(np.asarray(img_parent, dtype=np.int64), n_a)
            if img_parent
            else np.zeros(n_a + 1, dtype=np.int64)
        ),
        condition_offsets=np.zeros(n_a + 1, dtype=np.int64),
        point_analysis=point_analysis,
        xyz=xyz,
        point_key=np.full(len(xyz), None, dtype=object),
        point_space=space_codes,
        point_kind=kind_codes,
        point_values=point_values,
        space_dict=space_dict,
        kind_dict=kind_dict,
        image_attrs=ColumnStore(
            len(img_ref),
            dense={
                "analysis_idx": np.asarray(img_parent, dtype=np.int32),
                "url": np.full(len(img_ref), None, dtype=object),
                "filename": np.asarray(img_ref, dtype=object),
                "value_type": np.asarray(img_type, dtype=object),
                "space": np.asarray(img_space, dtype=object),
                "metadata": np.full(len(img_ref), None, dtype=object),
            },
        ),
        condition_code=np.zeros(0, dtype=np.int32),
        condition_weight=np.zeros(0, dtype=np.float64),
        condition_dict=Dict8(),
        annotations=annotations,
    )
    if canonical_order:
        store = canonicalize(store)
    return freeze(store)


def write_parquet(store, directory, *, annotation=None):
    """Write the studyset as a parquet release directory."""
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nimare.studyset.view import View

    os.makedirs(directory, exist_ok=True)
    view = View(store)
    frames = {
        "studies": _studies_table(store),
        "analyses": _analyses_table(store),
        "coordinates": view.frame("coordinates"),
        "images": view.frame("images"),
        # The raw metadata columns, at the level they were declared. The
        # compatibility frame merges the levels and derives a normalised
        # `sample_sizes`, which would lose the raw keys on a round trip.
        "metadata": _metadata_table(store, view),
        "texts": view.frame("texts"),
    }
    if store.annotations:
        frames["annotations"] = view.frame("annotations")
    written = {}
    for name, frame in frames.items():
        path = os.path.join(directory, f"{name}.parquet")
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        written[name] = f"{name}.parquet"
    manifest = {
        "format": "nimare-studyset-parquet",
        "version": 1,
        "id": store.id,
        "name": store.name,
        "tables": written,
        "annotations": [{"id": key} for key in store.annotations],
    }
    with open(os.path.join(directory, "studyset.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return directory


def _metadata_table(store, view):
    import pandas as pd

    cols = {
        "id": store.analysis_full_key.astype(str),
        "study_id": store.study_key[store.study_idx].astype(str),
        "contrast_id": store.analysis_key.astype(str),
    }
    for name in sorted(store.metadata.keys()):
        cols[name] = store.metadata.get(name)
    if store.study_metadata is not None:
        for name in sorted(store.study_metadata.keys()):
            if name in cols:
                continue
            per_study = store.study_metadata.get(name)
            cols[name] = per_study[store.study_idx]
    return pd.DataFrame(cols, copy=False)


def _studies_table(store):
    import pandas as pd

    cols = {"study_id": store.study_key.astype(str)}
    for name in ("name", "description", "authors", "publication"):
        if name in store.study_attrs.dense:
            cols[name] = store.study_attrs.dense[name].astype(object)
    return pd.DataFrame(cols)


def _analyses_table(store):
    import pandas as pd

    return pd.DataFrame(
        {
            "id": store.analysis_full_key.astype(str),
            "study_id": store.study_key[store.study_idx].astype(str),
            "contrast_id": store.analysis_key.astype(str),
            "name": store.analysis_attrs.dense["name"].astype(object),
        }
    )


def convert_neurostore_json_to_parquet(
    studyset_source,
    output_dir,
    *,
    annotation_source=None,
    manifest_source=None,
    studyset_id=None,
    studyset_name=None,
    annotation_id=None,
    overwrite=False,
):
    """Convert a NeuroStore release JSON export into parquet tables.

    Reads the document into a store and writes it back out columnar. The previous
    implementation streamed the JSON to keep peak memory down; this one does not,
    because the store it builds is an order of magnitude smaller than the parsed
    document it came from, so the parse is the high-water mark either way.

    Parameters
    ----------
    studyset_source : :obj:`str`, :obj:`pathlib.Path`, or :obj:`dict`
        A NeuroStore/NIMADS studyset JSON path, or a loaded dict.
    output_dir : :obj:`str` or :obj:`pathlib.Path`
        Directory to write the parquet tables and ``studyset.json`` into.
    annotation_source : :obj:`str`, :obj:`pathlib.Path`, or :obj:`dict`, optional
        A NeuroStore annotation JSON path, or a loaded dict, to attach.
    manifest_source : :obj:`str`, :obj:`pathlib.Path`, or :obj:`dict`, optional
        A release manifest, used to fill in missing ids.
    studyset_id, studyset_name, annotation_id : :obj:`str`, optional
        Explicit identifiers, overriding anything the manifest supplies.
    overwrite : :obj:`bool`, default=False
        If False, refuse to write into a directory that already has files.

    Returns
    -------
    :obj:`str`
        ``output_dir``.
    """
    from nimare.studyset.io.nimads import from_nimads
    from nimare.studyset.store import replace

    def load(source):
        if source is None or isinstance(source, dict):
            return source
        with open(source) as fh:
            return json.load(fh)

    studyset = load(studyset_source)
    annotation = load(annotation_source)
    manifest = load(manifest_source) or {}

    output_dir = str(output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains files; pass overwrite=True to replace them"
        )

    def manifest_field(entity, field):
        value = manifest.get(entity)
        return value.get(field) if isinstance(value, dict) else None

    resolved_id = studyset_id or studyset.get("id") or manifest_field("studyset", "id")
    resolved_name = (
        studyset_name or studyset.get("name") or manifest_field("studyset", "name") or ""
    )
    if annotation is not None:
        annotation = dict(annotation)
        annotation["id"] = (
            annotation_id
            or annotation.get("id")
            or manifest_field("annotation", "id")
            or "annotation"
        )

    # Ids are the join keys of the parquet tables, so unlike a general NIMADS
    # read this path insists on them rather than generating positional ones.
    for i, study in enumerate(studyset.get("studies") or []):
        if study.get("id") is None:
            raise ValueError(f"Could not infer an id for study at position {i}.")
        for analysis in study.get("analyses") or []:
            if analysis.get("id") is None:
                raise ValueError(f"An analysis of study {study['id']!r} has no id.")

    store = from_nimads(studyset, annotations=[annotation] if annotation else None)
    store = replace(store, id=resolved_id, name=resolved_name)
    os.makedirs(output_dir, exist_ok=True)
    return write_parquet(store, output_dir)
