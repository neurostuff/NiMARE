"""Input/Output operations."""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from nimare.dataset import Dataset
from nimare.exceptions import InvalidStudysetError
from nimare.extract.utils import _get_dataset_dir
from nimare.utils import _transform_coordinates_to_space, load_nimads, mni2tal, tal2mni

LGR = logging.getLogger(__name__)

DEFAULT_MAP_TYPE_CONVERSION = {
    "T map": "t",
    "variance": "varcope",
    "univariate-beta map": "beta",
    "Z map": "z",
    "p map": "p",
}

COORDINATE_METADATA_PREFIX = "coordinate_"
POINT_RELATIONSHIP_COLUMNS = ("kind", "label_id", "image")
_COORDINATE_VALUE_KIND_MAP = {
    "z_stat": "Z",
    "t_stat": "T",
    "beta": "beta",
    "variance": "variance",
    "varcope": "variance",
    "se": "SE",
    "p": "p",
}
_POINT_VALUE_COLUMN_MAP = {
    "z": "z_stat",
    "z-stat": "z_stat",
    "z_stat": "z_stat",
    "t": "t_stat",
    "t-stat": "t_stat",
    "t_stat": "t_stat",
    "beta": "beta",
    "variance": "variance",
    "varcope": "variance",
    "se": "se",
    "p": "p",
}


def _parse_feature_filename_entities(features_file):
    """Extract vocab/source/type entities from a Neurosynth-style feature filename."""
    patterns = {
        "vocab": r"vocab-([A-Za-z0-9]+)_",
        "source": r"source-([A-Za-z0-9]+)_",
        "type": r"type-([A-Za-z0-9]+)_",
    }
    matches = {}
    for entity, pattern in patterns.items():
        match = re.search(pattern, features_file)
        if match is None:
            raise ValueError(
                "Could not parse feature filename entity "
                f"'{entity}' from '{features_file}'. Expected a Neurosynth-style filename "
                "containing vocab-*, source-*, and type-* segments."
            )
        matches[entity] = match.group(1)
    return matches["vocab"], matches["source"], matches["type"]


def _is_missing_sample_size_value(value):
    """Return True when a sample-size value should be treated as missing."""
    return value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value))


def _coerce_sample_sizes_list(value):
    """Coerce a ``sample_sizes`` value to a numeric list when possible."""
    if _is_missing_sample_size_value(value):
        return None

    if not isinstance(value, (list, tuple)):
        LGR.warning(f"Expected sample_sizes to be list or tuple, but got {type(value)}.")
        return None

    if not value:
        return None

    coerced = []
    for i, sample_size_value in enumerate(value):
        if _is_missing_sample_size_value(sample_size_value):
            LGR.warning(f"Expected sample_sizes[{i}] to be numeric, but got missing data.")
            return None
        if not isinstance(sample_size_value, (int, float)):
            LGR.warning(
                f"Expected sample_sizes[{i}] to be numeric, but got "
                f"{type(sample_size_value)}. Attempting to convert to numeric."
            )
        try:
            coerced.append(int(sample_size_value))
        except (ValueError, TypeError):
            try:
                coerced.append(float(sample_size_value))
            except (ValueError, TypeError):
                LGR.warning(
                    f"Could not convert {sample_size_value} to numeric from type "
                    f"{type(sample_size_value)}."
                )
                return None

    return coerced


def _coerce_sample_size_scalar(value):
    """Coerce a scalar ``sample_size`` value to a one-element numeric list."""
    if _is_missing_sample_size_value(value) or not value:
        return None

    if not isinstance(value, (int, float)):
        LGR.warning(
            f"Expected sample_size to be numeric, but got {type(value)}."
            " Attempting to convert to numeric."
        )
    try:
        return [int(value)]
    except (ValueError, TypeError):
        try:
            return [float(value)]
        except (ValueError, TypeError):
            LGR.warning(f"Could not convert {value} to numeric from type {type(value)}.")
            return None


def _extract_coerced_sample_sizes(candidates):
    """Return the first valid sample-size candidate as a numeric list."""
    sample_sizes = None
    for key, raw_value in candidates:
        if key == "sample_sizes":
            sample_sizes = _coerce_sample_sizes_list(raw_value)
        else:
            sample_sizes = _coerce_sample_size_scalar(raw_value)

        if sample_sizes:
            return sample_sizes

    return None


def _coordinate_metadata_key(column):
    """Return the reserved metadata key used to persist per-coordinate extras."""
    return f"{COORDINATE_METADATA_PREFIX}{column}"


def _point_value_kind_to_coordinate_column(kind):
    """Map a NIMADS point-value kind to a Dataset-style coordinate column."""
    if _is_missing(kind):
        return None

    normalized = str(kind).strip().lower().replace(" ", "_")
    return _POINT_VALUE_COLUMN_MAP.get(normalized, f"value_{normalized}")


def _coordinate_column_to_point_value_kind(column):
    """Map a Dataset coordinate column to a NIMADS point-value kind."""
    return _COORDINATE_VALUE_KIND_MAP.get(column)


def _extract_coordinate_row_metadata(metadata, n_points):
    """Extract per-coordinate arrays serialized into analysis metadata."""
    coordinate_rows = {}
    coordinate_keys = set()
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or not key.startswith(COORDINATE_METADATA_PREFIX):
            continue
        if isinstance(value, list) and len(value) == n_points:
            coordinate_rows[key[len(COORDINATE_METADATA_PREFIX) :]] = value
            coordinate_keys.add(key)
    return coordinate_rows, coordinate_keys


def convert_nimads_to_dataset(studyset, annotation=None):
    """Convert nimads studyset to a dataset.

    .. versionadded:: 0.0.14

    Parameters
    ----------
    studyset : :obj:`str`, :obj:`dict`, :obj:`nimare.nimads.StudySet`
        Path to a JSON file containing a nimads studyset, a dictionary containing a nimads
        studyset, or a nimads studyset object.
    annotation : :obj:`str`, :obj:`dict`, :obj:`nimare.nimads.Annotation`, optional
        Optional path to a JSON file containing a nimads annotation, a dictionary containing a
        nimads annotation, or a nimads annotation object.

    Returns
    -------
    dset : :obj:`nimare.dataset.Dataset`
        NiMARE Dataset object containing experiment information from nimads studyset.

    .. warning::
        :class:`~nimare.dataset.Dataset` is deprecated and will be removed in a future release.
        Prefer keeping data in :class:`~nimare.nimads.Studyset` form and using
        :meth:`~nimare.nimads.Studyset.view` when a Dataset-like tabular view is needed.
    """

    def _analysis_to_dict(study, analysis):
        study_name = study.name or study.id
        analysis_name = analysis.name or analysis.id
        n_points = len(analysis.points)
        point_space = analysis.points[0].space if analysis.points else "UNKNOWN"
        if isinstance(point_space, str):
            point_space_lower = point_space.lower()
            if "mni" in point_space_lower or "ale" in point_space_lower:
                point_space = "MNI"
            elif "tal" in point_space_lower:
                point_space = "TAL"
        metadata = {
            "authors": study.authors,
            "journal": study.publication,
            "study_name": study_name,
            "analysis_name": analysis_name,
            "name": f"{study_name}-{analysis_name}",
        }
        combined_metadata = analysis.get_metadata().copy()
        coordinate_metadata, coordinate_metadata_keys = _extract_coordinate_row_metadata(
            combined_metadata,
            n_points,
        )
        # Preserve existing sample-size parsing behavior below by avoiding direct passthrough.
        combined_metadata.pop("sample_sizes", None)
        combined_metadata.pop("sample_size", None)
        for key in coordinate_metadata_keys:
            combined_metadata.pop(key, None)
        metadata.update(combined_metadata)

        coords = {
            "space": point_space,
            "x": [p.x for p in analysis.points] or [None],
            "y": [p.y for p in analysis.points] or [None],
            "z": [p.z for p in analysis.points] or [None],
        }
        for column in POINT_RELATIONSHIP_COLUMNS:
            values = [getattr(point, column, None) for point in analysis.points]
            if any(not _is_missing(value) for value in values):
                coords[column] = values

        point_value_columns = {}
        for i_point, point in enumerate(analysis.points):
            for point_value in getattr(point, "values", []) or []:
                if not isinstance(point_value, dict):
                    continue
                column = _point_value_kind_to_coordinate_column(point_value.get("kind"))
                value = point_value.get("value")
                if column is None or _is_missing(value):
                    continue
                point_value_columns.setdefault(column, [None] * n_points)
                point_value_columns[column][i_point] = value
        for column, values in point_value_columns.items():
            coords[column] = values

        for column, values in coordinate_metadata.items():
            if any(not _is_missing(value) for value in values):
                coords[column] = values

        result = {
            "metadata": metadata,
            "coords": coords,
        }

        # Carry image paths through conversion when available.
        images = {}
        for image in analysis.images:
            image_type = image.value_type
            if image_type in DEFAULT_MAP_TYPE_CONVERSION:
                image_type = DEFAULT_MAP_TYPE_CONVERSION[image_type]
            elif isinstance(image_type, str):
                image_type = image_type.lower().strip()
                if image_type.endswith(" map"):
                    image_type = image_type[: -len(" map")]
            if image_type == "variance":
                image_type = "varcope"
            image_path = image.url or image.filename
            if image_path:
                images[image_type] = image_path
        if images:
            if analysis.images and getattr(analysis.images[0], "space", None):
                images["space"] = analysis.images[0].space
            result["images"] = images

        # Handle annotations if present
        labels = None
        if analysis.annotations:
            labels = {}
            try:
                for key, annotation in analysis.annotations.items():
                    if isinstance(annotation, dict):
                        labels.update(annotation)
                    else:
                        labels[key] = annotation
            except (TypeError, AttributeError) as e:
                raise ValueError(f"Invalid annotation format: {str(e)}") from e
            result["labels"] = labels

        if analysis.texts:
            result["text"] = analysis.texts

        # Sample size priority order:
        # 1) sample_size in annotations
        # 2) sample_sizes in annotations
        # 3) sample_size(s) in analysis metadata
        # 4) sample_size(s) in study metadata
        candidates = [
            ("sample_size", labels.get("sample_size") if labels else None),
            ("sample_sizes", labels.get("sample_sizes") if labels else None),
            ("sample_sizes", analysis.metadata.get("sample_sizes")),
            ("sample_size", analysis.metadata.get("sample_size")),
            ("sample_sizes", study.metadata.get("sample_sizes")),
            ("sample_size", study.metadata.get("sample_size")),
        ]
        sample_sizes = _extract_coerced_sample_sizes(candidates)
        if sample_sizes:
            result["metadata"]["sample_sizes"] = sample_sizes

        return result

    def _study_to_dict(study):
        study_name = study.name or study.id
        metadata = {
            "authors": study.authors,
            "journal": study.publication,
            "title": study_name,
            "study_name": study_name,
        }
        metadata.update(study.metadata or {})
        return {
            "metadata": metadata,
            "contrasts": {a.id: _analysis_to_dict(study, a) for a in study.analyses},
        }

    # load nimads studyset
    studyset = load_nimads(studyset, annotation)
    return Dataset({s.id: _study_to_dict(s) for s in list(studyset.studies)})


def convert_neurosynth_to_dict(
    coordinates_file,
    metadata_file,
    annotations_files=None,
    feature_groups=None,
):
    """Convert Neurosynth/NeuroQuery database files to a dictionary.

    .. versionchanged:: 0.0.10

        * Use new format for Neurosynth and NeuroQuery files.

    .. versionchanged:: 0.0.9

        * Support annotations files organized in a dictionary.

    Parameters
    ----------
    coordinates_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's coordinates.
    metadata_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's metadata.
    annotations_files : :obj:`dict`, :obj:`list` of :obj:`dict`, or None, optional
        Optional file(s) with Neurosynth/NeuroQuery's annotations.
        This should consist of a dictionary with two keys: "features" and "vocabulary".
        "features" should have an NPZ file containing a sparse matrix of feature values.
        "vocabulary" should have a TXT file containing labels.
        The vocabulary corresponds to the columns of the feature matrix, while study IDs are
        inferred from the metadata file, which MUST be in the same order as the features matrix.
        Multiple sets of annotations may be provided, in which case "annotations_files" should be
        a list of dictionaries. The appropriate name of each annotation set will be inferred from
        the "features" filename, but this can be overwritten by using the "feature_groups"
        parameter.
        Default is None.
    feature_groups : :obj:`list` of :obj:`str`, or None, optional
        An optional list of names of annotation sets defined in "annotations_files".
        This should only be used if "annotations_files" is used and the users wants to override
        the automatically-extracted annotation set names.
        Default is None.

    Returns
    -------
    dset_dict : :obj:`dict`
        NiMARE-organized dictionary containing experiment information from text files.

    Warnings
    --------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    coords_df = pd.read_table(coordinates_file)
    metadata_df = pd.read_table(metadata_file)
    assert metadata_df["id"].is_unique, "Metadata file must have one row per ID."

    coords_df["id"] = coords_df["id"].astype(str)
    metadata_df["id"] = metadata_df["id"].astype(str)
    metadata_df = metadata_df.set_index("id", drop=False)
    ids = metadata_df["id"].tolist()

    if "space" not in metadata_df.columns:
        LGR.warning("No 'space' column detected. Defaulting to 'UNKNOWN'.")
        metadata_df["space"] = "UNKNOWN"

    if isinstance(annotations_files, dict):
        annotations_files = [annotations_files]

    if isinstance(feature_groups, str):
        feature_groups = [feature_groups]

    # Load labels into a single DataFrame
    if annotations_files is not None:
        label_dfs = []
        if feature_groups is not None:
            if len(feature_groups) != len(annotations_files):
                raise ValueError(
                    "feature_groups and annotations_files must have the same length. "
                    f"Got {len(feature_groups)} feature group names and "
                    f"{len(annotations_files)} annotation file sets."
                )

        for i_feature_group, annotations_dict in enumerate(annotations_files):
            features_file = annotations_dict["features"]
            vocabulary_file = annotations_dict["vocabulary"]

            vocab, source, value_type = _parse_feature_filename_entities(features_file)

            if feature_groups is not None:
                feature_group = feature_groups[i_feature_group]
                feature_group = feature_group.rstrip("_") + "__"
            else:
                feature_group = f"{vocab}_{source}_{value_type}__"

            features = sparse.load_npz(features_file).toarray()
            vocab = np.loadtxt(vocabulary_file, dtype=str, delimiter="\t")

            labels = [feature_group + label for label in vocab]

            temp_label_df = pd.DataFrame(features, index=ids, columns=labels)
            temp_label_df.index.name = "study_id"

            label_dfs.append(temp_label_df)

        label_df = pd.concat(label_dfs, axis=1)
    else:
        label_df = None

    # Compile (pseudo-)NIMADS-format dictionary
    coords_grouped = coords_df.groupby("id")[["x", "y", "z"]].agg(list)
    label_dict = label_df.to_dict(orient="index") if label_df is not None else None

    dset_dict = {}

    has_authors = "authors" in metadata_df.columns
    has_journal = "journal" in metadata_df.columns
    has_year = "year" in metadata_df.columns
    has_title = "title" in metadata_df.columns

    for row in metadata_df.itertuples():
        # Use index for study ID to support datasets without an explicit "id" column.
        sid = row.Index
        coord_row = coords_grouped.loc[sid] if sid in coords_grouped.index else None
        xs = coord_row["x"] if coord_row is not None else []
        ys = coord_row["y"] if coord_row is not None else []
        zs = coord_row["z"] if coord_row is not None else []

        authors = row.authors if has_authors else "n/a"
        journal = row.journal if has_journal else "n/a"
        year = row.year if has_year else "n/a"
        title = row.title if has_title else "n/a"

        study_dict = {
            "metadata": {
                "authors": authors,
                "journal": journal,
                "year": year,
                "title": title,
            },
            "contrasts": {
                "1": {
                    "metadata": {
                        "authors": authors,
                        "journal": journal,
                        "year": year,
                        "title": title,
                    },
                    "coords": {
                        "space": row.space,
                        "x": xs,
                        "y": ys,
                        "z": zs,
                    },
                }
            },
        }

        if label_dict is not None:
            study_dict["contrasts"]["1"]["labels"] = label_dict.get(sid, {})

        dset_dict[sid] = study_dict

    return dset_dict


def convert_neurosynth_to_json(
    coordinates_file,
    metadata_file,
    out_file,
    annotations_files=None,
    feature_groups=None,
):
    """Convert Neurosynth/NeuroQuery dataset text file to a NiMARE json file.

    .. versionchanged:: 0.0.10

        * Use new format for Neurosynth and NeuroQuery files.

    .. versionchanged:: 0.0.9

        * Support annotations files organized in a dictionary.

    Parameters
    ----------
    coordinates_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's coordinates and metadata.
    metadata_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's metadata.
    out_file : :obj:`str`
        Output NiMARE-format json file.
    annotations_files : :obj:`dict`, :obj:`list` of :obj:`dict`, or None, optional
        Optional file(s) with Neurosynth/NeuroQuery's annotations.
        This should consist of a dictionary with two keys: "features" and "vocabulary".
        "features" should have an NPZ file containing a sparse matrix of feature values.
        "vocabulary" should have a TXT file containing labels.
        The vocabulary corresponds to the columns of the feature matrix, while study IDs are
        inferred from the metadata file, which MUST be in the same order as the features matrix.
        Multiple sets of annotations may be provided, in which case "annotations_files" should be
        a list of dictionaries. The appropriate name of each annotation set will be inferred from
        the "features" filename, but this can be overwritten by using the "feature_groups"
        parameter.
        Default is None.
    feature_groups : :obj:`list` of :obj:`str`, or None, optional
        An optional list of names of annotation sets defined in "annotations_files".
        This should only be used if "annotations_files" is used and the users wants to override
        the automatically-extracted annotation set names.
        Default is None.

    Warnings
    --------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    dset_dict = convert_neurosynth_to_dict(
        coordinates_file, metadata_file, annotations_files, feature_groups
    )
    with open(out_file, "w") as fo:
        json.dump(dset_dict, fo, indent=4, sort_keys=True)


def convert_neurosynth_to_dataset(
    coordinates_file,
    metadata_file,
    annotations_files=None,
    feature_groups=None,
    target="mni152_2mm",
):
    """Convert Neurosynth/NeuroQuery database files into NiMARE Dataset.

    .. versionchanged:: 0.0.10

        * Use new format for Neurosynth and NeuroQuery files.

    .. versionchanged:: 0.0.9

        * Support annotations files organized in a dictionary.

    Parameters
    ----------
    coordinates_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's coordinates and metadata.
    metadata_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's metadata.
    annotations_files : :obj:`dict`, :obj:`list` of :obj:`dict`, or None, optional
        Optional file(s) with Neurosynth/NeuroQuery's annotations.
        This should consist of a dictionary with two keys: "features" and "vocabulary".
        "features" should have an NPZ file containing a sparse matrix of feature values.
        "vocabulary" should have a TXT file containing labels.
        The vocabulary corresponds to the columns of the feature matrix, while study IDs are
        inferred from the metadata file, which MUST be in the same order as the features matrix.
        Multiple sets of annotations may be provided, in which case "annotations_files" should be
        a list of dictionaries. The appropriate name of each annotation set will be inferred from
        the "features" filename, but this can be overwritten by using the "feature_groups"
        parameter.
        Default is None.
    feature_groups : :obj:`list` of :obj:`str`, or None, optional
        An optional list of names of annotation sets defined in "annotations_files".
        This should only be used if "annotations_files" is used and the users wants to override
        the automatically-extracted annotation set names.
        Default is None.
    target : {'mni152_2mm', 'ale_2mm'}, optional
        Target template space for coordinates. Default is 'mni152_2mm'.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        Dataset object containing experiment information from text_file.

    .. warning::
        :class:`~nimare.dataset.Dataset` output is deprecated and will be removed in a future
        release. When possible, prefer :func:`~nimare.extract.fetch_neurosynth` or
        :func:`~nimare.extract.fetch_neuroquery`, which return
        :class:`~nimare.nimads.Studyset` objects by default.

    .. warning::
        Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
        format. Old code using this function **will not work** with the new version.
    """
    dset_dict = convert_neurosynth_to_dict(
        coordinates_file,
        metadata_file,
        annotations_files,
        feature_groups,
    )
    return Dataset(dset_dict, target=target)


def convert_neurosynth_to_studyset(
    coordinates_file,
    metadata_file,
    annotations_files=None,
    feature_groups=None,
    target="mni152_2mm",
    *,
    materialize=False,
    studyset_id="nimads_from_neurosynth",
    studyset_name="",
):
    """Convert Neurosynth/NeuroQuery database files into a Studyset.

    Parameters
    ----------
    coordinates_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery coordinates.
    metadata_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery metadata.
    annotations_files : :obj:`dict`, :obj:`list` of :obj:`dict`, or None, optional
        Optional feature matrix/vocabulary inputs in the same format accepted by
        :func:`convert_neurosynth_to_dataset`.
    feature_groups : :obj:`list` of :obj:`str`, or None, optional
        Optional override names for annotation feature groups.
    target : {'mni152_2mm', 'ale_2mm'}, optional
        Target template space for coordinates. Default is 'mni152_2mm'.
    materialize : :obj:`bool`, optional
        If True, build a fully materialized nested Studyset through the legacy Dataset
        conversion path. If False, return a lightweight Studyset backed by cached
        Dataset-style tables for faster execution-oriented loading. Default is False.
    studyset_id : :obj:`str`, optional
        Identifier for the returned Studyset.
    studyset_name : :obj:`str`, optional
        Human-readable name for the returned Studyset.

    Returns
    -------
    :obj:`~nimare.nimads.Studyset`
        Studyset object containing experiment information from the Neurosynth files.
    """
    from nimare.nimads import Studyset

    if materialize:
        dataset = convert_neurosynth_to_dataset(
            coordinates_file,
            metadata_file,
            annotations_files=annotations_files,
            feature_groups=feature_groups,
            target=target,
        )
        return Studyset.from_dataset(dataset)

    metadata_df = pd.read_table(metadata_file)
    metadata_df["study_id"] = metadata_df["id"].astype(str)
    metadata_df["contrast_id"] = "1"
    metadata_df["id"] = metadata_df["study_id"] + "-1"

    study_names = metadata_df["study_id"].copy()
    if "title" in metadata_df.columns:
        study_names = metadata_df["title"].where(metadata_df["title"].notna(), study_names)
    metadata_df["study_name"] = study_names
    metadata_df["analysis_name"] = "1"
    metadata_df["name"] = metadata_df["study_name"]
    metadata_columns = ["id", "study_id", "contrast_id"] + [
        col for col in metadata_df.columns if col not in {"id", "study_id", "contrast_id"}
    ]
    metadata_df = metadata_df[metadata_columns].sort_values("id").reset_index(drop=True)
    full_ids = metadata_df["id"].to_numpy(dtype=str)
    study_ids = metadata_df["study_id"].to_numpy(dtype=str)

    coords_df = pd.read_table(coordinates_file)
    coords_df["study_id"] = coords_df["id"].astype(str)
    coords_df["contrast_id"] = "1"
    coords_df["id"] = coords_df["study_id"] + "-1"
    if "space" in metadata_df.columns:
        coord_space_map = metadata_df.set_index("study_id")["space"]
        coords_df["space"] = coords_df["study_id"].map(coord_space_map).fillna("UNKNOWN")
    else:
        coords_df["space"] = "UNKNOWN"
    coords_df = coords_df[["id", "study_id", "contrast_id", "x", "y", "z", "space"]]
    if target is not None:
        coords_df = _transform_coordinates_to_space(coords_df, target)
    coords_df = coords_df.sort_values("id").reset_index(drop=True)

    if isinstance(annotations_files, dict):
        annotations_files = [annotations_files]

    if isinstance(feature_groups, str):
        feature_groups = [feature_groups]

    annotation_tables = []
    if annotations_files is not None:
        if feature_groups is not None:
            if len(feature_groups) != len(annotations_files):
                raise ValueError(
                    "feature_groups and annotations_files must have the same length. "
                    f"Got {len(feature_groups)} feature group names and "
                    f"{len(annotations_files)} annotation file sets."
                )

        for i_feature_group, annotations_dict in enumerate(annotations_files):
            features_file = annotations_dict["features"]
            vocabulary_file = annotations_dict["vocabulary"]

            vocab, source, value_type = _parse_feature_filename_entities(features_file)

            if feature_groups is not None:
                feature_group = feature_groups[i_feature_group].rstrip("_") + "__"
            else:
                feature_group = f"{vocab}_{source}_{value_type}__"

            feature_matrix = sparse.load_npz(features_file).astype(np.float32).toarray()
            if feature_matrix.shape[0] != len(full_ids):
                raise ValueError(
                    "Feature matrix row count does not match metadata rows: "
                    f"{feature_matrix.shape[0]} != {len(full_ids)}"
                )

            vocabulary = np.loadtxt(vocabulary_file, dtype=str, delimiter="\t")
            labels = [feature_group + label for label in vocabulary.tolist()]
            annotation_tables.append(pd.DataFrame(feature_matrix, columns=labels, copy=False))

    if annotation_tables:
        annotations_df = pd.concat(annotation_tables, axis=1)
        annotations_df.insert(0, "contrast_id", "1")
        annotations_df.insert(0, "study_id", study_ids)
        annotations_df.insert(0, "id", full_ids)
    else:
        annotations_df = pd.DataFrame(
            {
                "id": full_ids,
                "study_id": study_ids,
                "contrast_id": "1",
            }
        )
    annotations_df = annotations_df.sort_values("id").reset_index(drop=True)

    if "abstract" in metadata_df.columns:
        texts_df = metadata_df[["id", "study_id", "contrast_id", "abstract"]].copy()
    else:
        texts_df = pd.DataFrame(columns=["id", "study_id", "contrast_id"])

    table_cache = {
        "space": target,
        "masker": None,
        "basepath": None,
        "ids": np.sort(full_ids),
        "coordinates": coords_df,
        "images": pd.DataFrame(columns=["id", "study_id", "contrast_id"]),
        "metadata": metadata_df,
        "annotations": annotations_df,
        "texts": texts_df,
    }

    def _materializer():
        dataset = convert_neurosynth_to_dataset(
            coordinates_file,
            metadata_file,
            annotations_files=annotations_files,
            feature_groups=feature_groups,
            target=target,
        )
        return convert_dataset_to_nimads_dict(
            dataset,
            studyset_id=studyset_id,
            studyset_name=studyset_name,
        )

    return Studyset.from_table_cache(
        table_cache,
        studyset_id=studyset_id,
        studyset_name=studyset_name,
        target=target,
        materializer=_materializer,
    )


def convert_nimads_to_sleuth(
    studyset: Union[str, Dict[str, Any], Any],
    output_dir: Union[str, Path],
    target: Union[Literal["MNI", "TAL"], str] = "MNI",
    decimal_precision: Optional[int] = 2,
    annotation: Optional[Union[str, Dict[str, Any], Any]] = None,
    export_metadata: Optional[
        List[Literal["authors", "publication", "year", "doi", "pmid", "affiliations"]]
    ] = None,
    *,
    annotation_columns: Optional[Sequence[str]] = None,
    annotation_values: Optional[Dict[str, Sequence[Any]]] = None,
    split_booleans: Optional[bool] = True,
):
    """Convert a NIMADS Studyset to Sleuth text file(s).

    Parameters
    ----------
    studyset : :obj:`str` or :obj:`dict` or :obj:`nimare.nimads.Studyset`
        Path to a Studyset JSON, a Studyset-like dictionary, or a Studyset object.
    output_dir : :obj:`str` or :obj:`pathlib.Path`
        Directory for Sleuth output. Created if missing.
    target : str, optional
        Target space for coordinates. Accepts "MNI" or "TAL", but also commonly-used
        dataset target strings such as "ale_2mm" or "mni152_2mm". These are normalized
        to either "MNI" or "TAL" internally. Default is "MNI".
    decimal_precision : :obj:`int`, optional
        Decimal places for output coordinates. Default is 2.
    annotation : :obj:`str` or :obj:`dict` or :obj:`nimare.nimads.Annotation`, optional
        Optional annotation to split output files by annotation values.
    export_metadata : :obj:`list` of {"authors", "publication", "year", "doi",
        "pmid", "affiliations"}, optional
        Metadata fields to export as comments. Default is ["doi", "pmid"].
    annotation_columns : :obj:`list` of :obj:`str`, optional
        Restrict splitting to these annotation keys.
    annotation_values : :obj:`dict`, optional
        Filter annotation splits to these values per key.
    split_booleans : :obj:`bool`, optional
        When True, boolean columns produce ``_true``/``_false`` files. Default is True.

    Raises
    ------
    InvalidStudysetError
        On invalid inputs or schema/structure mismatch.
    """
    if export_metadata is None:
        export_metadata = ["doi", "pmid"]
    if decimal_precision < 0:
        raise InvalidStudysetError("decimal_precision must be non-negative")

    # Normalize target string to canonical Sleuth reference-space ("MNI" or "TAL")
    try:
        tgt_str = str(target).lower()
    except Exception:
        tgt_str = "mni"
    if tgt_str in ("mni", "mni152_2mm", "ale_2mm", "mni152"):
        target_norm = "MNI"
    elif "tal" in tgt_str:
        target_norm = "TAL"
    else:
        # Fallback to MNI for unknown values
        target_norm = "MNI"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load studyset object and dict for validation
    try:
        # Local import to avoid circular import at module import time.
        from nimare.nimads import Studyset as _Studyset

        studyset_obj = studyset if isinstance(studyset, _Studyset) else _Studyset(studyset)
    except Exception as e:
        raise InvalidStudysetError(f"Failed to load studyset: {e}") from e

    if annotation is not None:
        try:
            studyset_obj.annotations = annotation
        except Exception as e:
            raise InvalidStudysetError(f"Failed to load annotation: {e}") from e

    # Handle annotations
    if studyset_obj.annotations:
        # Process with annotations - create separate files
        _process_with_annotations(
            studyset_obj,
            output_dir,
            target_norm,
            decimal_precision,
            export_metadata,
            annotation_columns,
            annotation_values,
            split_booleans,
        )
    else:
        # Process without annotations - single file
        _process_without_annotations(
            studyset_obj,
            output_dir,
            target_norm,
            decimal_precision,
            export_metadata,
        )


def _write_analysis(
    f,
    study,
    analysis,
    target: str,
    decimal_precision: int,
    export_metadata,
):
    """Write a single analysis block (header, metadata, coordinates) to an open file handle.

    This centralizes the previously duplicated code for writing study/analysis headers,
    sample sizes, metadata comment lines, and coordinates (including named-space transforms).
    """
    # Validate points
    if not hasattr(analysis, "points"):
        raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")
    if not analysis.points:
        return

    for point in analysis.points:
        if not (hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z")):
            raise InvalidStudysetError(f"Point in analysis {analysis.id} is missing coordinates")

    # Generate hierarchical label with metadata
    study_name = study.name or f"Study_{study.id}"
    analysis_name = analysis.name or f"Analysis_{analysis.id}"

    if getattr(study, "authors", ""):
        label = f"{study.authors} ({study_name}): {analysis_name}"
    else:
        label = f"{study_name}: {analysis_name}"

    # Write study header
    f.write(f"//{label}\n")

    # Write sample size if available
    sample_size = _get_sample_size(analysis, study)
    if sample_size:
        f.write(f"//Subjects={sample_size}\n")

    # Write additional metadata as comments
    metadata_lines = _generate_metadata_comments(study, export_metadata)
    for line in metadata_lines:
        f.write(f"//{line}\n")

    # Transform and write coordinates
    coords = []
    source_space = None
    for point in analysis.points:
        if source_space is None:
            source_space = getattr(point, "space", None) or "UNKNOWN"
        coords.append([point.x, point.y, point.z])

    if coords:
        coords = np.array(coords)
        # Validate coordinates are numeric
        if not np.issubdtype(coords.dtype, np.number):
            raise InvalidStudysetError("Coordinates must be numeric")
        # Apply named-space normalization if requested
        if target == "MNI" and source_space == "TAL":
            coords = tal2mni(coords)
        elif target == "TAL" and source_space == "MNI":
            coords = mni2tal(coords)

        # Write coordinates with specified precision
        for coord in coords:
            f.write(
                f"{coord[0]:.{decimal_precision}f}\t"
                f"{coord[1]:.{decimal_precision}f}\t"
                f"{coord[2]:.{decimal_precision}f}\n"
            )

    f.write("\n")  # Empty line between analyses


def _process_without_annotations(
    studyset,
    output_dir,
    target,
    decimal_precision,
    export_metadata,
):
    """Process studyset without annotations (single Sleuth file)."""
    output_file = output_dir / "nimads_sleuth_file.txt"

    with open(output_file, "w") as f:
        f.write(f"//Reference={target}\n")

        # If no studies, leave just the header
        if not studyset.studies:
            return

        # Iterate deterministically and use helper to write each analysis
        for study in sorted(studyset.studies, key=lambda s: s.id):
            if not hasattr(study, "analyses"):
                raise InvalidStudysetError(f"Study {study.id} is missing analyses")

            for analysis in sorted(study.analyses, key=lambda a: a.id):
                if not hasattr(analysis, "points"):
                    raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")
                if not analysis.points:
                    continue

                _write_analysis(
                    f,
                    study,
                    analysis,
                    target=target,
                    decimal_precision=decimal_precision,
                    export_metadata=export_metadata,
                )


def _process_with_annotations(
    studyset,
    output_dir,
    target,
    decimal_precision,
    export_metadata,
    annotation_columns,
    annotation_values,
    split_booleans,
):
    """Process studyset with annotations, creating multiple files."""
    annotation = studyset.annotations[0]  # Use first annotation

    # Validate annotation
    if not hasattr(annotation, "notes"):
        raise InvalidStudysetError("Annotation is missing notes")

    # Group analyses by annotation values
    analysis_groups = defaultdict(list)

    # Process each analysis and group by annotation values
    for study in sorted(studyset.studies, key=lambda s: s.id):
        # Validate study
        if not hasattr(study, "analyses"):
            raise InvalidStudysetError(f"Study {study.id} is missing analyses")

        for analysis in sorted(study.analyses, key=lambda a: a.id):
            # Validate analysis
            if not hasattr(analysis, "points"):
                raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")

            # Lookup this annotation's note (annotation.id -> note dict) on the analysis
            note = None
            if getattr(analysis, "annotations", None):
                note = analysis.annotations.get(annotation.id)

            if note:
                # Group by annotation values (restricted and filtered if requested)
                keyvals = list(note.items())
                if annotation_columns is not None:
                    keyvals = [(k, v) for k, v in keyvals if k in set(annotation_columns)]
                for key, value in sorted(keyvals, key=lambda kv: (kv[0], str(kv[1]))):
                    if (
                        annotation_values
                        and key in annotation_values
                        and value not in annotation_values[key]
                    ):
                        continue
                    if isinstance(value, bool) and split_booleans:
                        group_name = f"{key}_{str(value).lower()}"
                    else:
                        group_name = f"{key}_{value}"
                    analysis_groups[group_name].append((study, analysis))
            else:
                # Add to default group if no annotation for this analysis
                analysis_groups["default"].append((study, analysis))

    # Create separate files for each group (sorted deterministically)
    for group_name in sorted(analysis_groups.keys()):
        analyses = analysis_groups[group_name]
        # Sanitize filename
        safe_name = "".join(c for c in group_name if c.isalnum() or c in "._-").rstrip()
        if not safe_name:
            safe_name = "unnamed_group"

        output_file = output_dir / f"{safe_name}.txt"

        with open(output_file, "w") as f:
            f.write(f"//Reference={target}\n")

            # Process each analysis in this group
            for study, analysis in analyses:
                # Validate analysis
                if not hasattr(analysis, "points"):
                    raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")

                if not analysis.points:
                    continue

                # Validate points
                for point in analysis.points:
                    if (
                        not hasattr(point, "x")
                        or not hasattr(point, "y")
                        or not hasattr(point, "z")
                    ):
                        raise InvalidStudysetError(
                            f"Point in analysis {analysis.id} is missing coordinates"
                        )

                _write_analysis(
                    f,
                    study,
                    analysis,
                    target=target,
                    decimal_precision=decimal_precision,
                    export_metadata=export_metadata,
                )


def _generate_metadata_comments(study, export_metadata):
    """Generate metadata comment lines for Sleuth file."""
    comments = []

    # Authors and publication info (prefer top-level attributes)
    if getattr(study, "authors", "") and "authors" in export_metadata:
        comments.append(f"Authors={study.authors}")
    if getattr(study, "publication", "") and "publication" in export_metadata:
        comments.append(f"Publication={study.publication}")

    # Collect metadata from Study object and any top-level attributes that may
    # have been provided in the original JSON but not stored in study.metadata.
    md = getattr(study, "metadata", {}) or {}

    # Year may exist as a top-level attribute or inside metadata
    year = getattr(study, "year", None) or md.get("year")
    if year and "year" in export_metadata:
        comments.append(f"Year={year}")

    # DOI and PMID may also be top-level, in __dict__, or in metadata
    doi = getattr(study, "doi", None) or md.get("doi")
    if doi and "doi" in export_metadata:
        comments.append(f"DOI={doi}")

    pmid = getattr(study, "pmid", None) or md.get("pmid")
    if pmid and "pmid" in export_metadata:
        comments.append(f"PubMedId={pmid}")

    # Affiliations/institutions may be in metadata or as a top-level attribute
    aff = (
        md.get("affiliations")
        or md.get("institutions")
        or md.get("institution")
        or getattr(study, "affiliations", None)
    )
    if aff and "affiliations" in export_metadata:
        comments.append(f"Affiliations={aff}")

    return comments


def _get_sample_size(analysis, study):
    """Extract sample size from analysis or study metadata."""
    # Check analysis metadata first
    sample_sizes = analysis.metadata.get("sample_sizes")
    sample_size = analysis.metadata.get("sample_size")

    # Fall back to study metadata
    if not sample_sizes and not sample_size:
        sample_sizes = study.metadata.get("sample_sizes")
        sample_size = study.metadata.get("sample_size")

    # Return appropriate sample size value
    if sample_sizes:
        if isinstance(sample_sizes, (list, tuple)):
            return min(sample_sizes) if sample_sizes else None
        else:
            return sample_sizes
    elif sample_size:
        return sample_size

    return None


def convert_sleuth_to_dict(text_file):
    """Convert Sleuth text file to a dictionary.

    Parameters
    ----------
    text_file : :obj:`str` or :obj:`list` of :obj:`str`
        Path to Sleuth-format text file.
        More than one text file may be provided.

    Returns
    -------
    :obj:`dict`
        NiMARE-organized dictionary containing experiment information from text file.
    """
    if isinstance(text_file, list):
        dset_dict = {}
        for tf in text_file:
            temp_dict = convert_sleuth_to_dict(tf)
            for sid in temp_dict.keys():
                if sid in dset_dict.keys():
                    dset_dict[sid]["contrasts"] = {
                        **dset_dict[sid]["contrasts"],
                        **temp_dict[sid]["contrasts"],
                    }
                else:
                    dset_dict[sid] = temp_dict[sid]
        return dset_dict

    with open(text_file, "r") as file_object:
        data = file_object.read()

    data = [line.rstrip() for line in re.split("\n\r|\r\n|\n|\r", data)]
    data = [line for line in data if line]
    # First line indicates space. The rest are studies, ns, and coords
    space = data[0].replace(" ", "").replace("//Reference=", "")

    SPACE_OPTS = ["MNI", "TAL", "Talairach"]
    if space not in SPACE_OPTS:
        raise ValueError(f"Space {space} unknown. Options supported: {', '.join(SPACE_OPTS)}.")

    # Split into experiments
    data = data[1:]
    exp_idx = []
    header_lines = [i for i in range(len(data)) if data[i].startswith("//")]

    # Get contiguous header lines to define contrasts
    ranges = []
    for k, g in groupby(enumerate(header_lines), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
        if "Subjects" not in data[group[-1]]:
            raise ValueError(f"Sample size line missing for {data[group[0] : group[-1] + 1]}")
    start_idx = [r[0] for r in ranges]
    end_idx = start_idx[1:] + [len(data) + 1]
    split_idx = zip(start_idx, end_idx)

    def _parse_sleuth_study_info(study_info):
        """Parse Sleuth header text into study and contrast names.

        Heuristics:
        - Prefer splitting on the first ";" or ":" that occurs *after* a
          four-digit year (e.g., "Smith et al., 2010; Condition"). This
          avoids splitting on separators that might appear in journal titles
          or other parts of the citation.
        - If no year is present, fall back to the first ";" or ":" in the
          string (backwards compatible with the original behavior).
        - If no separator is found, treat the entire string as the study
          name and use "analysis_1" as the contrast name.
        """
        study_info = study_info.strip()
        if not study_info:
            return "", "analysis_1"

        # Prefer a separator that appears after a year-like token.
        year_match = re.search(r"(19|20)\d{2}", study_info)
        if year_match is not None:
            candidate_indices = []
            for sep in [";", ":"]:
                idx = study_info.find(sep, year_match.end())
                if idx != -1:
                    candidate_indices.append(idx)

            if candidate_indices:
                split_idx = min(candidate_indices)
                left = study_info[:split_idx].strip()
                right = study_info[split_idx + 1 :].strip()
                return left or study_info, right or "analysis_1"

        # Fallback: first semicolon, then colon anywhere in the string.
        for sep in [";", ":"]:
            if sep in study_info:
                left, _, right = study_info.partition(sep)
                left = left.strip()
                right = right.strip()
                return left or study_info, right or "analysis_1"

        # No usable separator found.
        return study_info, "analysis_1"

    dset_dict = {}
    for i_exp, exp_idx in enumerate(split_idx):
        exp_data = data[exp_idx[0] : exp_idx[1]]
        if exp_data:
            header_idx = [i for i in range(len(exp_data)) if exp_data[i].startswith("//")]
            study_info_idx = header_idx[:-1]
            n_idx = header_idx[-1]
            study_info = [exp_data[i].replace("//", "").strip() for i in study_info_idx]
            study_info = " ".join(study_info)
            study_name, contrast_name = _parse_sleuth_study_info(study_info)
            sample_size = int(exp_data[n_idx].replace(" ", "").replace("//Subjects=", ""))
            xyz = exp_data[n_idx + 1 :]  # Coords are everything after study info and n
            xyz = [row.split() for row in xyz]
            correct_shape = np.all([len(coord) == 3 for coord in xyz])
            if not correct_shape:
                all_shapes = np.unique([len(coord) for coord in xyz]).astype(str)
                raise ValueError(
                    f"Coordinates for study '{study_info}' are not all "
                    f"correct length. Lengths detected: {', '.join(all_shapes)}."
                )

            try:
                xyz = np.array(xyz, dtype=float)
            except:
                # Prettify xyz before reporting error
                strs = [[str(e) for e in row] for row in xyz]
                lens = [max(map(len, col)) for col in zip(*strs)]
                fmt = "\t".join("{{:{}}}".format(x) for x in lens)
                table = "\n".join([fmt.format(*row) for row in strs])
                raise ValueError(
                    f"Conversion to numpy array failed for study '{study_info}'. Coords:\n{table}"
                )

            x, y, z = list(xyz[:, 0]), list(xyz[:, 1]), list(xyz[:, 2])

            if study_name not in dset_dict.keys():
                dset_dict[study_name] = {"contrasts": {}}
            dset_dict[study_name]["contrasts"][contrast_name] = {"coords": {}, "metadata": {}}
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["space"] = space
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["x"] = x
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["y"] = y
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["z"] = z
            dset_dict[study_name]["contrasts"][contrast_name]["metadata"]["sample_sizes"] = [
                sample_size
            ]
    return dset_dict


def convert_sleuth_to_json(text_file, out_file):
    """Convert Sleuth output text file into json.

    Parameters
    ----------
    text_file : :obj:`str` or :obj:`list` of :obj:`str`
        Path to Sleuth-format text file.
        More than one text file may be provided.
    out_file : :obj:`str`
        Path to output json file.
    """
    if not isinstance(text_file, str) and not isinstance(text_file, list):
        raise ValueError(f"Unsupported type for parameter 'text_file': {type(text_file)}")
    dset_dict = convert_sleuth_to_dict(text_file)

    with open(out_file, "w") as fo:
        json.dump(dset_dict, fo, indent=4, sort_keys=True)


def convert_sleuth_to_dataset(text_file, target="ale_2mm"):
    """Convert Sleuth output text file into NiMARE Dataset.

    .. warning::
        :class:`~nimare.dataset.Dataset` output is deprecated and will be removed in a future
        release. Prefer :func:`~nimare.io.convert_sleuth_to_studyset`.

    Parameters
    ----------
    text_file : :obj:`str` or :obj:`list` of :obj:`str`
        Path to Sleuth-format text file.
        More than one text file may be provided.
    target : {'ale_2mm', 'mni152_2mm'} or None, optional
        Target template space for coordinates. If None,
        coordinates remain in the reference space indicated by the Sleuth
        file's //Reference= header and no template-based masker is created.
        If a template name is provided, coordinates are transformed into
        that space and a corresponding Dataset masker is created.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        Dataset object containing experiment information from text_file.
    """
    if not isinstance(text_file, str) and not isinstance(text_file, list):
        raise ValueError(f"Unsupported type for parameter 'text_file': {type(text_file)}")
    dset_dict = convert_sleuth_to_dict(text_file)
    return Dataset(dset_dict, target=target)


def convert_sleuth_to_studyset(text_file, target="ale_2mm"):
    """Convert Sleuth output text file into a NiMARE Studyset.

    This is the Studyset-native companion to
    :func:`convert_sleuth_to_dataset` and accepts the same arguments.

    Parameters
    ----------
    text_file : :obj:`str` or :obj:`list` of :obj:`str`
        Path to Sleuth-format text file.
        More than one text file may be provided.
    target : {'ale_2mm', 'mni152_2mm'} or None, optional
        Target template space for coordinates. If None,
        coordinates remain in the reference space indicated by the Sleuth
        file's //Reference= header and no template-based masker is created.
        If a template name is provided, coordinates are transformed into
        that space and a corresponding Studyset view can be created.

    Returns
    -------
    :obj:`~nimare.nimads.Studyset`
        Studyset object containing experiment information from ``text_file``.
    """
    from nimare.nimads import Studyset

    dataset = convert_sleuth_to_dataset(text_file, target=target)
    return Studyset.from_dataset(dataset)


def _is_missing(value: Any) -> bool:
    """Return True when value should be treated as missing."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except Exception:
        return False
    return False


def _to_serializable(value: Any) -> Any:
    """Convert values to JSON-serializable Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return {k: _to_serializable(v) for k, v in value.to_dict().items()}
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _coerce_coordinate_extra_value(value: Any) -> Any:
    """Best-effort conversion of Dataset coordinate extras to JSON scalars."""
    value = _to_serializable(value)
    if not isinstance(value, str):
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def convert_dataset_to_nimads_dict(
    dataset: Dataset,
    *,
    studyset_id: str = "nimads_from_dataset",
    studyset_name: str = "",
    out_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert a NiMARE Dataset to a NIMADS Studyset dictionary.

    .. warning::
        :class:`~nimare.dataset.Dataset` input is deprecated and will be removed in a future
        release. Prefer operating on :class:`~nimare.nimads.Studyset` directly, or convert once
        with :func:`~nimare.io.convert_dataset_to_studyset`.

    Parameters
    ----------
    dataset : :obj:`~nimare.dataset.Dataset`
        Dataset instance derived from Sleuth or other sources.
    studyset_id : :obj:`str`, optional
        Identifier for the resulting Studyset. Default is 'nimads_from_dataset'.
    studyset_name : :obj:`str`, optional
        Human-readable name for the Studyset. Default is ''.
    out_file : :obj:`str` or :obj:`pathlib.Path`, optional
        Optional path to write the Studyset JSON. Default is None.

    Returns
    -------
    dict
        NIMADS Studyset dictionary.
    """
    id_cols = {"id", "study_id", "contrast_id"}
    coordinate_core_cols = id_cols | {"x", "y", "z", "space"}

    # Build studies in the deterministic order of dataset.ids (sorted in Dataset)
    studies: Dict[str, Dict[str, Any]] = {}

    # Prepare lookups
    md = dataset.metadata
    coords = dataset.coordinates
    images_df = dataset.images
    annotations_df = dataset.annotations
    texts_df = dataset.texts

    # Ensure minimal required columns exist even if empty
    if md is None or md.empty:
        md = md if md is not None else pd.DataFrame(columns=["id", "study_id", "contrast_id"])
    if coords is None or coords.empty:
        coords = (
            coords
            if coords is not None
            else pd.DataFrame(columns=["id", "study_id", "contrast_id", "x", "y", "z", "space"])
        )
    if images_df is None or images_df.empty:
        images_df = (
            images_df
            if images_df is not None
            else pd.DataFrame(columns=["id", "study_id", "contrast_id"])
        )
    if annotations_df is None or annotations_df.empty:
        annotations_df = (
            annotations_df
            if annotations_df is not None
            else pd.DataFrame(columns=["id", "study_id", "contrast_id"])
        )
    if texts_df is None or texts_df.empty:
        texts_df = (
            texts_df
            if texts_df is not None
            else pd.DataFrame(columns=["id", "study_id", "contrast_id"])
        )

    def _first_rows_by_id(df):
        if df is None or df.empty:
            return {}
        return (
            df.drop_duplicates(subset="id", keep="first")
            .set_index("id", drop=False)
            .to_dict(orient="index")
        )

    md_by_id = _first_rows_by_id(md)
    image_by_id = _first_rows_by_id(images_df)
    annotation_by_id = _first_rows_by_id(annotations_df)
    text_by_id = _first_rows_by_id(texts_df)
    coords_by_id = {
        id_: group.reset_index(drop=True) for id_, group in coords.groupby("id", sort=False)
    }

    for id_ in dataset.ids:
        md_row = md_by_id.get(id_)
        crows = coords_by_id.get(id_)
        if md_row is None and crows is None:
            continue

        row = md_row if md_row is not None else crows.iloc[0].to_dict()
        study_id = str(row["study_id"])
        contrast_id = str(row["contrast_id"])

        study_name = study_id
        if md_row is not None:
            for study_name_col in ("study_name", "title"):
                value = md_row.get(study_name_col)
                if isinstance(value, str) and value.strip():
                    study_name = value
                    break

        # Study object (create if needed)
        if study_id not in studies:
            studies[study_id] = {
                "id": study_id,
                "name": study_name,
                "authors": "",
                "publication": "",
                "metadata": {},
                "analyses": [],
            }

        if md_row is not None:
            authors = md_row.get("authors")
            if isinstance(authors, str) and authors.strip():
                studies[study_id]["authors"] = authors
            publication = md_row.get("journal", md_row.get("publication"))
            if isinstance(publication, str) and publication.strip():
                studies[study_id]["publication"] = publication

        analysis_name = contrast_id
        if md_row is not None:
            for analysis_name_col in ("analysis_name", "contrast_name", "name"):
                value = md_row.get(analysis_name_col)
                if isinstance(value, str) and value.strip():
                    analysis_name = value
                    break

        # Analysis object
        analysis: Dict[str, Any] = {
            "id": contrast_id,
            "name": analysis_name,
            "conditions": [{"name": "default", "description": ""}],
            "weights": [1.0],
            "images": [],
            "points": [],
            "metadata": {},
        }

        if md_row is not None:
            for col, value in md_row.items():
                if col in id_cols:
                    continue
                if _is_missing(value):
                    continue
                analysis["metadata"][col] = _to_serializable(value)

        # Collect annotations for this analysis.
        annotation_row = annotation_by_id.get(id_)
        if annotation_row is not None:
            annotation_dict = {}
            for col, value in annotation_row.items():
                if col in id_cols:
                    continue
                if _is_missing(value):
                    continue
                annotation_dict[col] = _to_serializable(value)
            if annotation_dict:
                analysis["annotations"] = annotation_dict

        # Collect texts for this analysis.
        text_row = text_by_id.get(id_)
        if text_row is not None:
            text_dict = {}
            for col, value in text_row.items():
                if col in id_cols:
                    continue
                if _is_missing(value):
                    continue
                text_dict[col] = _to_serializable(value)
            if text_dict:
                analysis["texts"] = text_dict

        # Collect image metadata for this analysis.
        image_row = image_by_id.get(id_)
        if image_row is not None:
            image_space = image_row.get("space", dataset.space)
            if isinstance(image_space, str):
                image_space_lower = image_space.lower()
                if "tal" in image_space_lower:
                    image_space = "TAL"
                elif "mni" in image_space_lower or "ale" in image_space_lower:
                    image_space = "MNI"

            image_type_map = {}
            for col in images_df.columns:
                if col in ("id", "study_id", "contrast_id", "space"):
                    continue

                image_type = col.replace("__relative", "")
                image_path = image_row.get(col, None)
                if _is_missing(image_path):
                    continue
                if isinstance(image_path, os.PathLike):
                    image_path = os.fspath(image_path)
                if not isinstance(image_path, str) or not image_path:
                    continue

                existing_path, existing_is_relative = image_type_map.get(
                    image_type,
                    (None, False),
                )
                is_relative = col.endswith("__relative")
                # Prefer relative paths so serialized Studysets remain portable.
                if existing_path is not None and existing_is_relative and not is_relative:
                    continue
                if existing_path is not None and not existing_is_relative and not is_relative:
                    continue
                image_type_map[image_type] = (image_path, is_relative)

            for image_type, (image_path, _) in image_type_map.items():
                analysis["images"].append(
                    {
                        "url": image_path,
                        "filename": Path(image_path).name,
                        "space": image_space or "UNKNOWN",
                        "value_type": image_type,
                    }
                )

        # Collect points for this analysis in order of appearance
        if crows is None:
            crows = pd.DataFrame(columns=list(coordinate_core_cols))

        coordinate_extra_cols = [col for col in crows.columns if col not in coordinate_core_cols]
        point_value_cols = {
            col: _coordinate_column_to_point_value_kind(col)
            for col in coordinate_extra_cols
            if _coordinate_column_to_point_value_kind(col) is not None
        }
        coordinate_metadata_cols = [
            col
            for col in coordinate_extra_cols
            if col not in POINT_RELATIONSHIP_COLUMNS and col not in point_value_cols
        ]
        for col in coordinate_metadata_cols:
            serialized = [
                None if _is_missing(value) else _coerce_coordinate_extra_value(value)
                for value in crows[col].tolist()
            ]
            if any(value is not None for value in serialized):
                analysis["metadata"][_coordinate_metadata_key(col)] = serialized

        for _, crow in crows.iterrows():
            if (
                _is_missing(crow.get("x"))
                or _is_missing(crow.get("y"))
                or _is_missing(crow.get("z"))
            ):
                continue

            sp = crow.get("space", None)
            if isinstance(sp, str):
                spl = sp.lower()
                if "mni" in spl or "ale" in spl:
                    sp = "MNI"
                elif "tal" in spl:
                    sp = "TAL"
            point = {
                "space": sp,
                "coordinates": [float(crow["x"]), float(crow["y"]), float(crow["z"])],
            }
            for col in POINT_RELATIONSHIP_COLUMNS:
                value = crow.get(col)
                if not _is_missing(value):
                    point[col] = _to_serializable(value)

            point_values = []
            for col, kind in point_value_cols.items():
                value = crow.get(col)
                if _is_missing(value):
                    continue
                point_values.append({"kind": kind, "value": _coerce_coordinate_extra_value(value)})
            if point_values:
                point["values"] = point_values

            analysis["points"].append(point)

        studies[study_id]["analyses"].append(analysis)

    studyset: Dict[str, Any] = {
        "id": studyset_id,
        "name": studyset_name,
        "studies": list(studies.values()),
    }

    if out_file is not None:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(studyset, f, indent=2, sort_keys=False)

    return studyset


def convert_dataset_to_studyset(
    dataset: Dataset,
    *,
    studyset_id: str = "nimads_from_dataset",
    studyset_name: str = "",
    out_file: Optional[Union[str, Path]] = None,
):
    """Convert a NiMARE Dataset into a nimads.Studyset object.

    This is a convenience wrapper around :func:`convert_dataset_to_nimads_dict` that
    returns a fully-constructed :class:`nimare.nimads.Studyset` instance instead of a
    plain dictionary. If ``out_file`` is provided, the underlying NIMADS dictionary will
    also be written to disk (same behavior as :func:`convert_dataset_to_nimads_dict`).

    .. warning::
        :class:`~nimare.dataset.Dataset` input is deprecated and will be removed in a future
        release. For new workflows, prefer starting from :class:`~nimare.nimads.Studyset`
        directly.

    Parameters
    ----------
    dataset
        NiMARE Dataset to convert.
    studyset_id
        Identifier for the resulting Studyset.
    studyset_name
        Human-readable name for the Studyset.
    out_file
        Optional path to write the intermediate Studyset JSON.

    Returns
    -------
    nimads.Studyset
        Constructed Studyset object.
    """
    nimads_dict = convert_dataset_to_nimads_dict(
        dataset, studyset_id=studyset_id, studyset_name=studyset_name, out_file=out_file
    )
    # Local import to avoid circular import at module import time.
    from nimare.nimads import Studyset as _Studyset

    studyset = _Studyset(nimads_dict)
    studyset._attach_dataset_context(dataset)
    return studyset


def convert_sleuth_to_nimads_dict(
    text_file: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    target: str = None,
    studyset_id: str = None,
    studyset_name: str = "",
) -> Dict[str, Any]:
    """Convert Sleuth text file(s) to a NIMADS Studyset dictionary.

    Parameters
    ----------
    text_file : :obj:`str`, :obj:`pathlib.Path`, or sequence of such
        Path(s) to Sleuth text file(s).
    target : :obj:`str`, optional
        Target space for Dataset loader. If None (default), uses the space
        specified in the Sleuth file's //Reference= tag without conversion.
        Accepts common dataset targets (e.g., "ale_2mm", "mni152_2mm") or
        user-friendly strings like "MNI", "TAL", or variants such as
        "Talairach". These are normalized to a Dataset-supported template
        string internally. Default is None.
    studyset_id : :obj:`str`, optional
        Identifier for the resulting Studyset. Default is 'nimads_from_sleuth'.
    studyset_name : :obj:`str`, optional
        Human-readable name for the Studyset. Default is ''.

    Returns
    -------
    dict
        NIMADS Studyset dictionary.
    """
    # Original behavior when target is specified
    # Normalize incoming target string to dataset template keys accepted by
    # Dataset (mni152_2mm or ale_2mm). Accept common variants including misspellings.
    try:
        tgt_str = str(target).lower()
    except Exception:
        tgt_str = "mni"

    # Map common/variant user inputs to Dataset-supported template keys.
    if "tal" in tgt_str:
        # Represent Talairach-like Sleuth targets using the MNI template
        # when converting Sleuth -> Dataset (Dataset expects 'mni152_2mm' or 'ale_2mm').
        ds_target = "mni152_2mm"
    elif "ale" in tgt_str:
        ds_target = "ale_2mm"
    elif "mni" in tgt_str:
        ds_target = "mni152_2mm"
    else:
        ds_target = None

    dset = convert_sleuth_to_dataset(text_file, target=ds_target)
    return convert_dataset_to_nimads_dict(
        dset, studyset_id=studyset_id, studyset_name=studyset_name
    )


def convert_neurovault_to_dataset(
    collection_ids,
    contrasts,
    img_dir=None,
    map_type_conversion=None,
    **dset_kwargs,
):
    """Convert a group of NeuroVault collections into a NiMARE Dataset.

    .. versionadded:: 0.0.8

    .. warning::
        :class:`~nimare.dataset.Dataset` output is deprecated and will be removed in a future
        release. Prefer :func:`~nimare.generate.create_neurovault_studyset`.

    Parameters
    ----------
    collection_ids : :obj:`list` of :obj:`int` or :obj:`dict`
        A list of collections on neurovault specified by their id.
        The collection ids can accessed through the neurovault API
        (i.e., https://neurovault.org/api/collections) or
        their main website (i.e., https://neurovault.org/collections).
        For example, in this URL https://neurovault.org/collections/8836/,
        `8836` is the collection id.
        collection_ids can also be a dictionary whose keys are the informative
        study name and the values are collection ids to give the collections
        more informative names in the dataset.
    contrasts : :obj:`dict`
        Dictionary whose keys represent the name of the contrast in
        the dataset and whose values represent a regular expression that would
        match the names represented in NeuroVault.
        For example, under the ``Name`` column in this URL
        https://neurovault.org/collections/8836/,
        a valid contrast could be "as-Animal", which will be called "animal" in the created
        dataset if the contrasts argument is ``{'animal': "as-Animal"}``.
    img_dir : :obj:`str` or None, optional
        Base path to save all the downloaded images, by default the images
        will be saved to a temporary directory with the prefix "neurovault".
    map_type_conversion : :obj:`dict` or None, optional
        Dictionary whose keys are what you expect the `map_type` name to
        be in neurovault and the values are the name of the respective
        statistic map in a nimare dataset. Default = None.
    **dset_kwargs : keyword arguments passed to Dataset
        Keyword arguments to pass in when creating the Dataset object.
        see :obj:`~nimare.dataset.Dataset` for details.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        Dataset object containing experiment information from neurovault.
    """
    img_dir = Path(_get_dataset_dir("_".join(contrasts.keys()), data_dir=img_dir))

    if map_type_conversion is None:
        map_type_conversion = DEFAULT_MAP_TYPE_CONVERSION

    if not isinstance(collection_ids, dict):
        collection_ids = {nv_coll: nv_coll for nv_coll in collection_ids}

    dataset_dict = {}
    for coll_name, nv_coll in collection_ids.items():
        nv_url = f"https://neurovault.org/api/collections/{nv_coll}/images/?format=json"
        images = requests.get(nv_url).json()
        if "Not found" in images.get("detail", ""):
            raise ValueError(
                f"Collection {nv_coll} not found. "
                "Three likely causes are (1) the collection doesn't exist, "
                "(2) the collection is private, or "
                "(3) the provided ID corresponds to an image instead of a collection."
            )

        dataset_dict[f"study-{coll_name}"] = {"contrasts": {}}
        for contrast_name, contrast_regex in contrasts.items():
            dataset_dict[f"study-{coll_name}"]["contrasts"][contrast_name] = {
                "images": {
                    "beta": None,
                    "t": None,
                    "varcope": None,
                },
                "metadata": {"sample_sizes": None},
            }

            sample_sizes = []
            no_images = True
            for img_dict in images["results"]:
                if not (
                    re.match(contrast_regex, img_dict["name"])
                    and img_dict["map_type"] in map_type_conversion
                    and img_dict["analysis_level"] == "group"
                ):
                    continue

                no_images = False
                filename = img_dir / (
                    f"collection-{nv_coll}_id-{img_dict['id']}_" + Path(img_dict["file"]).name
                )

                if not filename.exists():
                    r = requests.get(img_dict["file"])
                    with open(filename, "wb") as f:
                        f.write(r.content)

                (
                    dataset_dict[f"study-{coll_name}"]["contrasts"][contrast_name]["images"][
                        map_type_conversion[img_dict["map_type"]]
                    ]
                ) = filename.as_posix()

                # aggregate sample sizes (should all be the same)
                sample_sizes.append(img_dict["number_of_subjects"])

            if no_images:
                raise ValueError(
                    f"No images were found for contrast {contrast_name}. "
                    f"Please check the contrast regular expression: {contrast_regex}"
                )
            # take modal sample size (raise warning if there are multiple values)
            if len(set(sample_sizes)) > 1:
                sample_size = _resolve_sample_size(sample_sizes)
                LGR.warning(
                    (
                        f"Multiple sample sizes were found for neurovault collection: {nv_coll}"
                        f"for contrast: {contrast_name}, sample sizes: {set(sample_sizes)}"
                        f", selecting modal sample size: {sample_size}"
                    )
                )
            else:
                sample_size = sample_sizes[0]
            (
                dataset_dict[f"study-{coll_name}"]["contrasts"][contrast_name]["metadata"][
                    "sample_sizes"
                ]
            ) = [sample_size]

    dataset = Dataset(dataset_dict, **dset_kwargs)

    return dataset


def _resolve_sample_size(sample_sizes):
    """Choose modal sample_size if there are multiple sample_sizes to choose from."""
    sample_size_counts = Counter(sample_sizes)
    if None in sample_size_counts:
        sample_size_counts.pop(None)

    return sample_size_counts.most_common()[0][0]
