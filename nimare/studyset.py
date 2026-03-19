"""Studyset-backed execution views for estimators and workflows."""

from __future__ import annotations

import copy
import logging
import os.path as op

import numpy as np
import pandas as pd
from nilearn.image import load_img

from nimare.base import NiMAREBase
from nimare.io import DEFAULT_MAP_TYPE_CONVERSION
from nimare.utils import (
    _listify,
    _mask_img_to_bool,
    _try_prepend,
    get_masker,
    get_template,
    load_nimads,
    mm2vox,
)

LGR = logging.getLogger(__name__)
_TABLE_ATTRS = ("ids", "coordinates", "images", "metadata", "annotations", "texts")


def _normalize_point_space(space):
    """Normalize point space names to Dataset-compatible target strings."""
    if not isinstance(space, str):
        return None

    space_lower = space.lower()
    if "tal" in space_lower:
        return "mni152_2mm"
    if "ale" in space_lower:
        return "ale_2mm"
    if "mni" in space_lower:
        return "mni152_2mm"
    return None


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


def _snapshot_dataset_tables(dataset, copy_tables=False):
    """Capture Dataset tables for StudysetView construction or Studyset caching."""
    table_cache = {
        "space": dataset.space,
        "masker": dataset.masker,
        "basepath": dataset.basepath,
    }
    for attr in _TABLE_ATTRS:
        value = getattr(dataset, attr)
        table_cache[attr] = value.copy() if copy_tables else value
    return table_cache


def _slice_table_cache(table_cache, ids):
    """Filter a Dataset-style table cache down to the requested analysis IDs."""
    ids = np.sort(np.asarray(_listify(ids), dtype=str))
    id_set = set(ids.tolist())

    sliced_cache = {
        "space": table_cache.get("space"),
        "masker": table_cache.get("masker"),
        "basepath": table_cache.get("basepath"),
        "ids": ids,
    }
    for attr in _TABLE_ATTRS[1:]:
        value = table_cache.get(attr)
        if value is None:
            sliced_cache[attr] = None
        else:
            sliced_cache[attr] = value.loc[value["id"].isin(id_set)].copy()
    return sliced_cache


class StudysetView(NiMAREBase):
    """Dataset-like, cached tabular view of a NIMADS Studyset."""

    _id_cols = ["id", "study_id", "contrast_id"]

    def __init__(self, studyset, target=None, mask=None, basepath=None, materialize_ids=True):
        self._dataset = None
        self.studyset = load_nimads(studyset)
        context = self.studyset._get_execution_context()
        self.space = target or context["space"]
        self.basepath = basepath if basepath is not None else context["basepath"]

        mask = mask if mask is not None else context["masker"]
        self.masker = get_masker(mask) if mask is not None else None
        self._ensure_masker_from_space()

        self._ids = None
        self._coordinates = None
        self._images = None
        self._metadata = None
        self._annotations = None
        self._texts = None
        table_cache = context["table_cache"]
        if table_cache is not None:
            self._load_table_cache(table_cache)
        if materialize_ids:
            # Materialize IDs eagerly for compatibility with Dataset-like private access patterns.
            _ = self.ids

    @classmethod
    def from_dataset(cls, dataset):
        """Create a Dataset-backed StudysetView without rebuilding nested Studyset objects."""
        view = cls.from_table_cache(
            _snapshot_dataset_tables(dataset, copy_tables=True),
            target=dataset.space,
            mask=dataset.masker,
            basepath=dataset.basepath,
        )
        view._dataset = dataset
        return view

    @classmethod
    def from_table_cache(cls, table_cache, target=None, mask=None, basepath=None):
        """Create a StudysetView directly from pre-materialized Dataset-style tables."""
        view = object.__new__(cls)
        view._dataset = None
        view.studyset = None
        view.space = target if target is not None else table_cache.get("space")
        view.basepath = basepath if basepath is not None else table_cache.get("basepath")
        view.masker = get_masker(mask) if mask is not None else table_cache.get("masker")
        view._ids = None
        view._coordinates = None
        view._images = None
        view._metadata = None
        view._annotations = None
        view._texts = None
        view._load_table_cache(table_cache)
        return view

    def _load_table_cache(self, table_cache):
        """Attach pre-materialized Dataset-compatible tables to this view."""
        if self.space is None:
            self.space = table_cache.get("space")
        if self.basepath is None:
            self.basepath = table_cache.get("basepath")
        if self.masker is None:
            self.masker = table_cache.get("masker")

        for attr in _TABLE_ATTRS:
            setattr(self, f"_{attr}", table_cache.get(attr))

        self._ensure_masker_from_space()

    def _iter_flat_analyses(self):
        """Yield analyses with their parent study and full Dataset-style ID."""
        for study, analysis in self.studyset._iter_analyses():
            yield study, analysis, f"{study.id}-{analysis.id}"

    @classmethod
    def _analysis_row(cls, study, analysis, full_id):
        """Create the shared ID columns for a single analysis row."""
        return {
            "id": full_id,
            "study_id": study.id,
            "contrast_id": analysis.id,
        }

    def _spawn_cached_clone(self):
        """Return a lightweight clone with shared cached tables."""
        clone = object.__new__(StudysetView)
        clone._dataset = self._dataset
        clone.studyset = self.studyset
        clone.space = self.space
        clone.basepath = self.basepath
        clone.masker = self.masker
        clone._ids = self._ids
        clone._coordinates = self._coordinates
        clone._images = self._images
        clone._metadata = self._metadata
        clone._annotations = self._annotations
        clone._texts = self._texts
        return clone

    def _get_available_table_cache(self):
        """Return the best available Dataset-style cache for this view."""
        table_cache = {
            "space": self.space,
            "masker": self.masker,
            "basepath": self.basepath,
        }
        if self.studyset is not None:
            table_cache.update(self.studyset._get_execution_context()["table_cache"] or {})

        for attr in _TABLE_ATTRS:
            value = getattr(self, f"_{attr}")
            if value is not None:
                table_cache[attr] = value

        return table_cache

    def __deepcopy__(self, memo):
        """Avoid recursively copying the underlying Dataset/Studyset graph."""
        existing = memo.get(id(self))
        if existing is not None:
            return existing

        clone = self._spawn_cached_clone()
        memo[id(self)] = clone
        return clone

    @staticmethod
    def _rows_to_df(rows, columns):
        """Build a sorted DataFrame from row dictionaries."""
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows).sort_values(by="id")

    def _update_studyset_cache(self, *attrs):
        """Persist newly materialized tables back onto the source Studyset."""
        if self.studyset is None:
            return

        table_cache = dict(self.studyset._get_execution_context()["table_cache"] or {})
        table_cache.update(
            {
                "space": self.space,
                "masker": self.masker,
                "basepath": self.basepath,
            }
        )
        for attr in attrs:
            table_cache[attr] = getattr(self, f"_{attr}")
        self.studyset._set_table_cache(table_cache)

    def _materialize_tables(self, *table_names):
        """Populate requested tables from the underlying Studyset in one pass."""
        if self.studyset is None:
            return

        requested = tuple(dict.fromkeys(table_names))
        if not requested:
            return

        pending = [name for name in requested if getattr(self, f"_{name}") is None]
        if not pending:
            return

        build_ids = self._ids is None
        build_coordinates = "coordinates" in pending
        build_images = "images" in pending
        build_metadata = "metadata" in pending
        build_annotations = "annotations" in pending
        build_texts = "texts" in pending

        ids = [] if build_ids else None
        coordinate_rows = [] if build_coordinates else None
        image_rows = [] if build_images else None
        metadata_rows = [] if build_metadata else None
        annotation_rows = [] if build_annotations else None
        text_rows = [] if build_texts else None
        spaces = [] if build_coordinates and self.space is None else None

        for study, analysis, full_id in self._iter_flat_analyses():
            base_row = self._analysis_row(study, analysis, full_id)

            if build_ids:
                ids.append(full_id)

            if build_metadata:
                study_name = study.name or study.id
                analysis_name = analysis.name or analysis.id
                metadata_row = {
                    **base_row,
                    "study_name": study_name,
                    "analysis_name": analysis_name,
                    "authors": study.authors,
                    "journal": study.publication,
                    "name": f"{study_name}-{analysis_name}",
                }
                metadata_row.update(analysis.get_metadata())
                metadata_rows.append(metadata_row)

            if build_annotations:
                annotation_row = dict(base_row)
                for key, note in (analysis.annotations or {}).items():
                    if isinstance(note, dict):
                        annotation_row.update(note)
                    else:
                        annotation_row[key] = note
                annotation_rows.append(annotation_row)

            if build_images:
                image_row = dict(base_row)
                for image in analysis.images:
                    image_type = _normalize_image_type(image.value_type)
                    if image_type is None:
                        continue
                    image_value = image.url if image.url else image.filename
                    if not isinstance(image_value, str) or not image_value:
                        continue

                    is_remote = "://" in image_value
                    is_relative = not op.isabs(image_value) and not is_remote
                    if is_relative:
                        image_row[f"{image_type}__relative"] = image_value
                        if self.basepath:
                            image_row[image_type] = _try_prepend(image_value, self.basepath)
                    else:
                        image_row[image_type] = image_value

                    if image.space and "space" not in image_row:
                        image_row["space"] = image.space
                image_rows.append(image_row)

            if build_texts:
                text_row = dict(base_row)
                text_row.update(analysis.texts or {})
                text_rows.append(text_row)

            if build_coordinates:
                for point in analysis.points:
                    coordinate_rows.append(
                        {
                            **base_row,
                            "x": float(point.x),
                            "y": float(point.y),
                            "z": float(point.z),
                            "space": point.space,
                        }
                    )
                    if spaces is not None:
                        spaces.append(_normalize_point_space(point.space))

        built_attrs = []
        if build_ids:
            self._ids = np.sort(np.asarray(ids))
            built_attrs.append("ids")

        if build_coordinates:
            self._coordinates = self._rows_to_df(
                coordinate_rows,
                self._id_cols + ["x", "y", "z", "space"],
            )
            built_attrs.append("coordinates")

        if build_images:
            self._images = self._rows_to_df(image_rows, self._id_cols)
            built_attrs.append("images")

        if build_metadata:
            self._metadata = self._rows_to_df(metadata_rows, self._id_cols)
            built_attrs.append("metadata")

        if build_annotations:
            self._annotations = self._rows_to_df(annotation_rows, self._id_cols)
            built_attrs.append("annotations")

        if build_texts:
            self._texts = self._rows_to_df(text_rows, self._id_cols)
            built_attrs.append("texts")

        if spaces is not None and self.space is None:
            inferred_spaces = {space for space in spaces if space is not None}
            if len(inferred_spaces) == 1:
                self.space = inferred_spaces.pop()
                self._ensure_masker_from_space()

        self._update_studyset_cache(*built_attrs)

    def _ensure_masker_from_space(self):
        """Initialize a default masker from the inferred/declared Studyset space when possible."""
        if self.masker is not None or not isinstance(self.space, str):
            return

        try:
            self.masker = get_masker(get_template(self.space, mask="brain"))
        except Exception:
            LGR.warning(
                "Could not initialize masker from Studyset space '%s'. "
                "Provide a mask explicitly for coordinate-based analyses.",
                self.space,
            )

    @property
    def ids(self):
        """numpy.ndarray: 1D array of full analysis identifiers."""
        if self._ids is None:
            if self.studyset is None:
                raise ValueError(
                    "StudysetView is missing both cached tables and a Studyset source."
                )
            self._materialize_tables("ids")
        return self._ids

    @property
    def coordinates(self):
        """pandas.DataFrame: Coordinates table."""
        if self._coordinates is None:
            self._materialize_tables("coordinates")
        return self._coordinates

    @coordinates.setter
    def coordinates(self, df):
        self._coordinates = df.sort_values(by="id")
        self._ids = np.sort(df["id"].unique().astype(str))

    @property
    def metadata(self):
        """pandas.DataFrame: Metadata table."""
        if self._metadata is None:
            self._materialize_tables("metadata")
        return self._metadata

    @metadata.setter
    def metadata(self, df):
        self._metadata = df.sort_values(by="id")

    @property
    def annotations(self):
        """pandas.DataFrame: Flattened analysis annotation table."""
        if self._annotations is None:
            self._materialize_tables("annotations")
        return self._annotations

    @annotations.setter
    def annotations(self, df):
        self._annotations = df.sort_values(by="id")

    @property
    def images(self):
        """pandas.DataFrame: Image file table."""
        if self._images is None:
            self._materialize_tables("images")
        return self._images

    @images.setter
    def images(self, df):
        self._images = df.sort_values(by="id")

    @property
    def texts(self):
        """pandas.DataFrame: Flattened analysis texts table."""
        if self._texts is None:
            self._materialize_tables("texts")
        return self._texts

    def copy(self):
        """Create a copy of the StudysetView."""
        return copy.deepcopy(self)

    def slice(self, ids):
        """Create a new view with only requested full analysis IDs."""
        ids = set(_listify(ids))
        if self._dataset is not None:
            return StudysetView.from_dataset(self._dataset.slice(sorted(ids)))

        if self.studyset is None or (
            not self.studyset.is_materialized and not getattr(self.studyset, "_studies", None)
        ):
            sliced_cache = _slice_table_cache(self._get_available_table_cache(), sorted(ids))
            return StudysetView.from_table_cache(
                sliced_cache,
                target=self.space,
                mask=self.masker,
                basepath=self.basepath,
            )

        studyset_dict = self.studyset.to_dict()
        for study in studyset_dict["studies"]:
            study_id = study["id"]
            study["analyses"] = [
                analysis
                for analysis in study.get("analyses", [])
                if f"{study_id}-{analysis['id']}" in ids
            ]

        studyset_dict["studies"] = [
            study for study in studyset_dict["studies"] if study["analyses"]
        ]

        sliced_view = StudysetView(
            studyset_dict,
            target=self.space,
            mask=self.masker,
            basepath=self.basepath,
            materialize_ids=False,
        )

        cached_attrs = []
        if self._ids is not None:
            sliced_view._ids = np.sort(self._ids[np.isin(self._ids, list(ids))])
            cached_attrs.append("ids")

        for attr in _TABLE_ATTRS[1:]:
            cached_table = getattr(self, f"_{attr}")
            if cached_table is None:
                continue
            setattr(
                sliced_view,
                f"_{attr}",
                cached_table[cached_table["id"].isin(ids)].copy(),
            )
            cached_attrs.append(attr)

        if cached_attrs:
            sliced_view._update_studyset_cache(*cached_attrs)

        return sliced_view

    def _generic_column_getter(self, attr, ids=None, column=None, ignore_columns=None):
        if ignore_columns is None:
            ignore_columns = list(self._id_cols)
        else:
            ignore_columns += self._id_cols

        df = getattr(self, attr)
        return_first = False

        if isinstance(ids, str) and column is not None:
            return_first = True
        ids = _listify(ids)

        available_types = [c for c in df.columns if c not in ignore_columns]
        if (column is not None) and (column not in available_types):
            raise ValueError(
                f"{column} not found in {attr}.\nAvailable types: {', '.join(available_types)}"
            )

        if column is not None:
            if ids is not None:
                result = df[column].loc[df["id"].isin(ids)].tolist()
            else:
                result = df[column].tolist()
            result = [None if pd.isna(val) else val for val in result]
        else:
            if ids is not None:
                result = {v: df[v].loc[df["id"].isin(ids)].tolist() for v in available_types}
                result = {k: v for k, v in result.items() if any(v)}
            else:
                result = {v: df[v].tolist() for v in available_types}
            result = list(result.keys())

        if return_first:
            return result[0]
        return result

    def get_labels(self, ids=None):
        """Extract list of labels for which analyses have annotations."""
        if not isinstance(ids, list) and ids is not None:
            ids = _listify(ids)

        result = [c for c in self.annotations.columns if c not in self._id_cols]
        if ids is not None and result:
            temp_annotations = self.annotations.loc[self.annotations["id"].isin(ids)]
            res = temp_annotations[result].any(axis=0)
            result = res.loc[res].index.tolist()
        return result

    def get_studies_by_label(self, labels=None, label_threshold=0.001):
        """Extract list of analyses with a given label."""
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            raise ValueError(f"Argument 'labels' cannot be {type(labels)}")

        missing_labels = [label for label in labels if label not in self.annotations.columns]
        if missing_labels:
            raise ValueError(f"Missing label(s): {', '.join(missing_labels)}")

        temp_annotations = self.annotations[self._id_cols + labels]
        found_rows = (temp_annotations[labels] >= label_threshold).all(axis=1)
        if any(found_rows):
            found_ids = temp_annotations.loc[found_rows, "id"].tolist()
        else:
            found_ids = []

        return found_ids

    def get_texts(self, ids=None, text_type=None):
        """Get texts for selected analyses."""
        return self._generic_column_getter("texts", ids=ids, column=text_type)

    def get_metadata(self, ids=None, field=None):
        """Get metadata values for selected analyses."""
        return self._generic_column_getter("metadata", ids=ids, column=field)

    def get_images(self, ids=None, imtype=None):
        """Get image paths for selected analyses."""
        ignore_columns = ["space"]
        ignore_columns += [c for c in self.images.columns if c.endswith("__relative")]
        return self._generic_column_getter(
            "images",
            ids=ids,
            column=imtype,
            ignore_columns=ignore_columns,
        )

    def get(self, dict_, drop_invalid=True):
        """Retrieve files and/or metadata from the current StudysetView."""
        results = {"id": self.ids}
        keep_idx = np.arange(len(self.ids), dtype=int)
        for key, vals in dict_.items():
            if vals[0] == "image":
                temp = self.get_images(imtype=vals[1])
            elif vals[0] == "metadata":
                temp = self.get_metadata(field=vals[1])
            elif vals[0] == "coordinates":
                coord_by_id = dict(iter(self.coordinates.groupby("id")))
                temp = [coord_by_id[id_] if id_ in coord_by_id else None for id_ in self.ids]
            elif vals[0] == "annotations":
                annot_by_id = dict(iter(self.annotations.groupby("id")))
                temp = [annot_by_id[id_] if id_ in annot_by_id else None for id_ in self.ids]
            else:
                raise ValueError(f"Input '{vals[0]}' not understood.")

            results[key] = temp
            temp_keep_idx = np.where([val is not None for val in temp])[0]
            keep_idx = np.intersect1d(keep_idx, temp_keep_idx)

        if drop_invalid and (len(keep_idx) != len(self.ids)):
            LGR.info(f"Retaining {len(keep_idx)}/{len(self.ids)} studies")
        elif len(keep_idx) != len(self.ids):
            raise Exception(
                f"Only {len(keep_idx)}/{len(self.ids)} in Dataset contain the necessary data. "
                "If you want to analyze the subset of studies with required data, "
                "set `drop_invalid` to True."
            )

        for key in results:
            results[key] = [results[key][i] for i in keep_idx]
            if dict_.get(key, [None])[0] in ("coordinates", "annotations"):
                if results[key]:
                    results[key] = pd.concat(results[key])
                else:
                    results[key] = pd.DataFrame()

        return results

    def get_studies_by_mask(self, mask):
        """Extract list of analyses with at least one focus in mask."""
        mask = load_img(mask)
        if self.coordinates.empty:
            return []

        if self.masker is None:
            raise ValueError("A masker is required to evaluate coordinates against a mask.")

        dset_mask = self.masker.mask_img
        if dset_mask is not None and not np.array_equal(dset_mask.affine, mask.affine):
            LGR.warning("Mask affine does not match Dataset affine. Assuming same space.")

        dset_ijk = mm2vox(self.coordinates[["x", "y", "z"]].values, mask.affine)
        mask_data = _mask_img_to_bool(mask)

        shape = mask_data.shape
        dset_ijk = np.clip(dset_ijk, 0, np.array(shape) - 1)
        in_mask = mask_data[dset_ijk[:, 0], dset_ijk[:, 1], dset_ijk[:, 2]] > 0
        found_ids = list(self.coordinates.loc[in_mask, "id"].unique())
        return found_ids

    def get_studies_by_coordinate(self, xyz, r=20):
        """Extract list of analyses with at least one focus near coordinates."""
        from scipy.spatial.distance import cdist

        if self.coordinates.empty:
            return []

        xyz = np.array(xyz)
        assert xyz.shape[1] == 3 and xyz.ndim == 2
        distances = cdist(xyz, self.coordinates[["x", "y", "z"]].values)
        distances = np.any(distances <= r, axis=0)
        found_ids = list(self.coordinates.loc[distances, "id"].unique())
        return found_ids


def ensure_studyset_view(dataset):
    """Convert an input Dataset/Studyset into a StudysetView."""
    from nimare.dataset import Dataset
    from nimare.nimads import Studyset

    if isinstance(dataset, StudysetView):
        return dataset

    if isinstance(dataset, Dataset):
        return StudysetView.from_dataset(dataset)

    if isinstance(dataset, Studyset) or isinstance(dataset, (dict, str)):
        studyset = load_nimads(dataset)
        return StudysetView(studyset=studyset, materialize_ids=False)

    raise ValueError(
        "Input must be a Dataset, Studyset, dict, or path to a NIMADS studyset JSON, "
        f"not {type(dataset)}."
    )
