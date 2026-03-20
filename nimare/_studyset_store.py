"""Internal storage and execution helpers for Studyset."""

from __future__ import annotations

import copy
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from nimare.io import DEFAULT_MAP_TYPE_CONVERSION
from nimare.utils import _transform_coordinates_to_space, _try_prepend, get_masker, get_template

_ID_COLS = ["id", "study_id", "contrast_id"]
_TABLE_ATTRS = ("ids", "coordinates", "images", "metadata", "annotations", "texts")


def _coordinate_space_family(space):
    """Normalize a coordinate-space label to its broad family."""
    if not isinstance(space, str):
        return None

    space_lower = space.lower()
    if "mni" in space_lower or "ale" in space_lower:
        return "MNI"
    if "tal" in space_lower:
        return "TAL"
    return None


def _rows_to_df(rows, columns):
    """Build a sorted DataFrame from row dictionaries."""
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    sort_col = "id" if "id" in df.columns else columns[0]
    return df.sort_values(by=sort_col).reset_index(drop=True)


def _normalize_image_type(value_type):
    """Convert NIMADS image type names into NiMARE Dataset image column names."""
    if value_type in DEFAULT_MAP_TYPE_CONVERSION:
        return DEFAULT_MAP_TYPE_CONVERSION[value_type]

    if not isinstance(value_type, str):
        return None

    value_type = value_type.strip().lower()
    if value_type.endswith(" map"):
        value_type = value_type[: -len(" map")]
    if value_type == "variance":
        return "varcope"
    return value_type


def _coerce_annotation_payloads(annotations):
    """Normalize top-level annotation payloads to dictionaries."""
    if annotations is None:
        return []
    if isinstance(annotations, dict):
        return [copy.deepcopy(annotations)]
    if isinstance(annotations, (list, tuple)):
        return [copy.deepcopy(annotation) for annotation in annotations]
    return [copy.deepcopy(annotations.to_dict())]


@lru_cache(maxsize=None)
def _cached_default_masker(target):
    """Build and cache the default fitted masker for a target space."""
    return get_masker(get_template(target, mask="brain"))


def _apply_annotation_payloads(source_dict, annotation_payloads):
    """Apply top-level annotation notes into analysis-level annotation dictionaries."""
    analysis_map = {}
    for study in source_dict.get("studies", []):
        for analysis in study.get("analyses", []):
            analysis_map[str(analysis["id"])] = analysis

    for annotation in annotation_payloads:
        for note in annotation.get("notes", []):
            analysis_id = str(note["analysis"])
            if analysis_id not in analysis_map:
                continue
            analysis = analysis_map[analysis_id]
            annotations = analysis.setdefault("annotations", {}) or {}
            annotations[annotation["id"]] = copy.deepcopy(note["note"])
            analysis["annotations"] = annotations

    source_dict["annotations"] = copy.deepcopy(annotation_payloads)
    return source_dict


def _study_sample_space(study):
    """Return one representative coordinate space for a study, if available."""
    for analysis in study.get("analyses", []):
        points = analysis.get("points", []) or []
        if points:
            return points[0].get("space")
    return None


def _study_spaces_match_target(source_dict, target):
    """Check one representative space family per study against a target space."""
    target_family = _coordinate_space_family(target)
    if target_family is None:
        return True

    found_space = False
    for study in source_dict.get("studies", []):
        study_space_family = _coordinate_space_family(_study_sample_space(study))
        if study_space_family is None:
            continue
        found_space = True
        if study_space_family != target_family:
            return False
    return found_space


def _set_source_dict_coordinate_space(source_dict, target):
    """Relabel all point spaces in a source dictionary without changing coordinates."""
    for study in source_dict.get("studies", []):
        for analysis in study.get("analyses", []):
            for point in analysis.get("points", []) or []:
                point["space"] = target
    return source_dict


def _table_spaces_match_target(spaces, target):
    """Check whether all table space labels fall in the target family."""
    target_family = _coordinate_space_family(target)
    if target_family is None:
        return False

    found_space = False
    for space in spaces:
        space_family = _coordinate_space_family(space)
        if space_family is None:
            return False
        found_space = True
        if space_family != target_family:
            return False
    return found_space


def _harmonize_source_dict_coordinates(source_dict, target, *, harmonize_coordinates=True):
    """Transform all point coordinates in a source dictionary into the target space."""
    if target is None or not harmonize_coordinates:
        return source_dict

    if _study_spaces_match_target(source_dict, target):
        return _set_source_dict_coordinate_space(source_dict, target)

    point_rows = []
    point_refs = []
    for study in source_dict.get("studies", []):
        for analysis in study.get("analyses", []):
            for point in analysis.get("points", []) or []:
                coords = point.get("coordinates", [None, None, None])
                point_rows.append(
                    {
                        "x": float(coords[0]),
                        "y": float(coords[1]),
                        "z": float(coords[2]),
                        "space": point.get("space"),
                    }
                )
                point_refs.append(point)

    if not point_rows:
        return source_dict

    transformed = _transform_coordinates_to_space(pd.DataFrame(point_rows), target)
    for point, row in zip(point_refs, transformed.itertuples(index=False)):
        point["coordinates"] = [float(row.x), float(row.y), float(row.z)]
        point["space"] = row.space

    return source_dict


def _build_tables_from_source(source_dict):
    """Build canonical Studyset tables from a NIMADS-like source dictionary."""
    studies_rows = []
    analyses_rows = []
    ids = []
    coordinate_rows = []
    image_rows = []
    metadata_rows = []
    annotation_rows = []
    text_rows = []

    for study in source_dict.get("studies", []):
        study_id = str(study["id"])
        studies_rows.append(
            {
                "study_id": study_id,
                "name": study.get("name", ""),
                "authors": study.get("authors", ""),
                "publication": study.get("publication", ""),
            }
        )

        for analysis in study.get("analyses", []):
            contrast_id = str(analysis["id"])
            full_id = f"{study_id}-{contrast_id}"
            base_row = {"id": full_id, "study_id": study_id, "contrast_id": contrast_id}
            ids.append(full_id)
            analyses_rows.append(
                {
                    **base_row,
                    "name": analysis.get("name", ""),
                }
            )

            study_name = study.get("name") or study_id
            analysis_name = analysis.get("name") or contrast_id
            metadata_row = {
                **base_row,
                "study_name": study_name,
                "analysis_name": analysis_name,
                "authors": study.get("authors", ""),
                "journal": study.get("publication", ""),
                "name": f"{study_name}-{analysis_name}",
            }
            combined_metadata = copy.deepcopy(study.get("metadata", {}) or {})
            combined_metadata.update(copy.deepcopy(analysis.get("metadata", {}) or {}))
            metadata_row.update(combined_metadata)
            metadata_rows.append(metadata_row)

            annotation_row = dict(base_row)
            for key, note in (analysis.get("annotations", {}) or {}).items():
                if isinstance(note, dict):
                    annotation_row.update(note)
                else:
                    annotation_row[key] = note
            annotation_rows.append(annotation_row)

            text_row = dict(base_row)
            text_row.update(copy.deepcopy(analysis.get("texts", {}) or {}))
            text_rows.append(text_row)

            image_row = dict(base_row)
            for image in analysis.get("images", []) or []:
                image_type = _normalize_image_type(image.get("value_type"))
                if image_type is None:
                    continue

                image_value = image.get("url") or image.get("filename")
                if not isinstance(image_value, str) or not image_value:
                    continue

                image_row[f"{image_type}__source"] = image_value
                if image.get("space") and "space" not in image_row:
                    image_row["space"] = image.get("space")
            image_rows.append(image_row)

            for point in analysis.get("points", []) or []:
                coords = point.get("coordinates", [None, None, None])
                coordinate_rows.append(
                    {
                        **base_row,
                        "x": float(coords[0]),
                        "y": float(coords[1]),
                        "z": float(coords[2]),
                        "space": point.get("space"),
                    }
                )

    ids = np.sort(np.asarray(ids, dtype=str))
    return {
        "studies": _rows_to_df(studies_rows, ["study_id", "name", "authors", "publication"]),
        "analyses": _rows_to_df(analyses_rows, _ID_COLS + ["name"]),
        "ids": ids,
        "coordinates": _rows_to_df(coordinate_rows, _ID_COLS + ["x", "y", "z", "space"]),
        "images": _rows_to_df(image_rows, _ID_COLS),
        "metadata": _rows_to_df(metadata_rows, _ID_COLS),
        "annotations": _rows_to_df(annotation_rows, _ID_COLS),
        "texts": _rows_to_df(text_rows, _ID_COLS),
    }


def _resolve_image_table(images_df, basepath):
    """Resolve relative image paths for the current base path."""
    if images_df is None:
        return None

    if images_df.empty:
        return images_df.copy()

    resolved = images_df.copy()
    source_cols = [col for col in resolved.columns if col.endswith("__source")]
    for source_col in source_cols:
        image_type = source_col[: -len("__source")]
        image_values = resolved[source_col]
        resolved[f"{image_type}__relative"] = None
        resolved[image_type] = None

        for idx, image_value in image_values.items():
            if not isinstance(image_value, str) or not image_value:
                continue

            is_remote = "://" in image_value
            is_relative = not is_remote and not image_value.startswith("/")
            if is_relative:
                resolved.at[idx, f"{image_type}__relative"] = image_value
                if basepath:
                    resolved.at[idx, image_type] = _try_prepend(image_value, basepath)
            else:
                resolved.at[idx, image_type] = image_value

        resolved = resolved.drop(columns=[source_col])

    return resolved.sort_values(by="id").reset_index(drop=True)


@dataclass
class StudysetExecutionProfile:
    """Execution configuration for one Studyset projection."""

    target: Optional[str] = None
    masker: Optional[object] = None
    basepath: Optional[str] = None
    coordinate_space_policy: str = "explicit"

    def __post_init__(self):
        if self.masker is not None:
            self.masker = get_masker(self.masker)
        elif isinstance(self.target, str):
            self.masker = copy.copy(_cached_default_masker(self.target))

    @property
    def is_ready(self):
        return self.masker is not None or self.target is not None

    def cache_key(self):
        mask_img = getattr(self.masker, "mask_img", None) if self.masker is not None else None
        return (
            self.target,
            id(mask_img),
            self.basepath,
            self.coordinate_space_policy,
        )


class StudysetStore:
    """Canonical Studyset storage with normalized tables and optional lazy source materialization."""

    def __init__(
        self,
        studyset_id,
        studyset_name="",
        *,
        source_dict=None,
        tables=None,
        annotation_payloads=None,
        materializer: Optional[Callable[[], dict]] = None,
    ):
        self.studyset_id = studyset_id
        self.studyset_name = studyset_name
        self._source_dict = source_dict
        self._tables = tables or _build_tables_from_source(source_dict or {"studies": []})
        self._annotation_payloads = _coerce_annotation_payloads(annotation_payloads)
        self._materializer = materializer

    @classmethod
    def from_source_dict(
        cls,
        source_dict,
        annotation_payloads=None,
        target=None,
        *,
        harmonize_coordinates=True,
    ):
        """Create a store from a source dictionary."""
        source_dict = copy.deepcopy(source_dict)
        source_dict = _harmonize_source_dict_coordinates(
            source_dict,
            target,
            harmonize_coordinates=harmonize_coordinates,
        )
        annotation_payloads = (
            _coerce_annotation_payloads(annotation_payloads)
            if annotation_payloads is not None
            else _coerce_annotation_payloads(source_dict.get("annotations", []))
        )
        source_dict = _apply_annotation_payloads(source_dict, annotation_payloads)
        return cls(
            source_dict["id"],
            source_dict.get("name", ""),
            source_dict=source_dict,
            annotation_payloads=annotation_payloads,
        )

    @classmethod
    def from_table_cache(
        cls,
        studyset_id,
        studyset_name,
        table_cache,
        *,
        annotation_payloads=None,
        materializer=None,
    ):
        """Create a store directly from Dataset-like tables."""
        tables = {
            "studies": pd.DataFrame(columns=["study_id", "name", "authors", "publication"]),
            "analyses": pd.DataFrame(columns=_ID_COLS + ["name"]),
            "ids": np.sort(np.asarray(table_cache.get("ids", []), dtype=str)),
            "coordinates": table_cache.get("coordinates"),
            "images": table_cache.get("images"),
            "metadata": table_cache.get("metadata"),
            "annotations": table_cache.get("annotations"),
            "texts": table_cache.get("texts"),
        }
        if tables["metadata"] is not None and not tables["metadata"].empty:
            analyses = tables["metadata"][_ID_COLS].copy()
            if "analysis_name" in tables["metadata"].columns:
                analyses["name"] = tables["metadata"]["analysis_name"]
            else:
                analyses["name"] = analyses["contrast_id"]
            tables["analyses"] = analyses.sort_values(by="id").reset_index(drop=True)

        return cls(
            studyset_id,
            studyset_name,
            tables=tables,
            annotation_payloads=annotation_payloads,
            materializer=materializer,
        )

    @property
    def annotation_payloads(self):
        return copy.deepcopy(self._annotation_payloads)

    @property
    def ids(self):
        return self._tables["ids"]

    def selected_ids(self, selected_full_ids=None):
        if selected_full_ids is None:
            return self.ids
        selected_full_ids = np.sort(np.asarray(selected_full_ids, dtype=str))
        return self.ids[np.isin(self.ids, selected_full_ids)]

    def raw_tables(self, selected_full_ids=None, *, basepath=None):
        """Return raw Dataset-like tables for the selected analyses."""
        ids = self.selected_ids(selected_full_ids)
        id_set = set(ids.tolist())

        table_cache = {"ids": ids}
        for attr in _TABLE_ATTRS[1:]:
            table = self._tables.get(attr)
            if table is None:
                table_cache[attr] = None
            elif table.empty:
                table_cache[attr] = table.copy()
            else:
                table_cache[attr] = table.loc[table["id"].isin(id_set)].copy()

        table_cache["images"] = _resolve_image_table(table_cache["images"], basepath)
        return table_cache

    def projected_tables(self, execution_profile, selected_full_ids=None):
        """Return execution-ready Dataset-like tables for the selected analyses."""
        table_cache = self.raw_tables(selected_full_ids, basepath=execution_profile.basepath)
        coordinates = table_cache["coordinates"]
        if (
            coordinates is not None
            and execution_profile.target is not None
            and not coordinates.empty
            and execution_profile.coordinate_space_policy == "harmonize"
        ):
            existing_spaces = set(coordinates["space"].dropna().astype(str).unique().tolist())
            if not existing_spaces:
                pass
            elif _table_spaces_match_target(existing_spaces, execution_profile.target):
                if existing_spaces != {execution_profile.target}:
                    coordinates = coordinates.copy()
                    coordinates.loc[:, "space"] = execution_profile.target
            else:
                coordinates = _transform_coordinates_to_space(
                    coordinates.copy(),
                    execution_profile.target,
                )
        table_cache["coordinates"] = coordinates
        table_cache["space"] = execution_profile.target
        table_cache["masker"] = execution_profile.masker
        table_cache["basepath"] = execution_profile.basepath
        return table_cache

    def resolve_full_ids(self, values, *, allow_short_ids=False, selected_full_ids=None):
        """Resolve explicit full IDs or short analysis IDs to full analysis IDs."""
        ids = self.selected_ids(selected_full_ids)
        values = [str(value) for value in values]
        full_ids = set(ids.tolist())
        resolved = []
        short_id_map = None
        if allow_short_ids:
            short_id_map = {}
            for full_id in ids:
                _, contrast_id = full_id.rsplit("-", 1)
                short_id_map.setdefault(contrast_id, []).append(full_id)

        for value in values:
            if value in full_ids:
                resolved.append(value)
                continue
            if allow_short_ids and value in short_id_map:
                resolved.extend(short_id_map[value])

        return np.sort(np.asarray(sorted(set(resolved)), dtype=str))

    def selected_source_dict(self, selected_full_ids=None):
        """Materialize and filter the canonical source dictionary."""
        source_dict = self._source_dict
        if source_dict is None and self._materializer is not None:
            source_dict = self._materializer()
            source_dict = _apply_annotation_payloads(source_dict, self._annotation_payloads)
            self._source_dict = source_dict

        if source_dict is None:
            source_dict = {
                "id": self.studyset_id,
                "name": self.studyset_name,
                "studies": [],
                "annotations": self.annotation_payloads,
            }

        if selected_full_ids is None:
            return copy.deepcopy(source_dict)

        selected_ids = set(self.selected_ids(selected_full_ids).tolist())
        if not selected_ids:
            return {
                "id": self.studyset_id,
                "name": self.studyset_name,
                "studies": [],
                "annotations": [],
            }

        selected_source = {
            "id": source_dict["id"],
            "name": source_dict.get("name", ""),
            "studies": [],
        }
        retained_analysis_ids = set()
        for study in source_dict.get("studies", []):
            kept_analyses = []
            for analysis in study.get("analyses", []):
                full_id = f"{study['id']}-{analysis['id']}"
                if full_id in selected_ids:
                    kept_analyses.append(copy.deepcopy(analysis))
                    retained_analysis_ids.add(str(analysis["id"]))

            if kept_analyses:
                kept_study = copy.deepcopy(study)
                kept_study["analyses"] = kept_analyses
                selected_source["studies"].append(kept_study)

        filtered_annotations = []
        for annotation in self._annotation_payloads:
            filtered_notes = [
                copy.deepcopy(note)
                for note in annotation.get("notes", [])
                if str(note["analysis"]) in retained_analysis_ids
            ]
            if filtered_notes:
                filtered_annotation = copy.deepcopy(annotation)
                filtered_annotation["notes"] = filtered_notes
                filtered_annotations.append(filtered_annotation)

        if filtered_annotations:
            selected_source["annotations"] = filtered_annotations
        return selected_source
