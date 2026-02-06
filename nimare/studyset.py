"""Studyset-backed execution views for estimators and workflows."""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
from nilearn.image import load_img

from nimare.base import NiMAREBase
from nimare.io import DEFAULT_MAP_TYPE_CONVERSION
from nimare.utils import _listify, _mask_img_to_bool, get_masker, get_template, load_nimads, mm2vox

LGR = logging.getLogger(__name__)


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


class StudysetView(NiMAREBase):
    """Dataset-like, cached tabular view of a NIMADS Studyset."""

    _id_cols = ["id", "study_id", "contrast_id"]

    def __init__(self, studyset, target=None, mask=None, basepath=None):
        self.studyset = load_nimads(studyset)
        self.space = target or getattr(self.studyset, "_nimare_space", None)
        self.basepath = (
            basepath if basepath is not None else getattr(self.studyset, "_nimare_basepath", None)
        )

        mask = mask if mask is not None else getattr(self.studyset, "_nimare_masker", None)
        self.masker = get_masker(mask) if mask is not None else None
        self._ensure_masker_from_space()

        self._ids = None
        self._coordinates = None
        self._images = None
        self._metadata = None
        self._annotations = None
        self._texts = None
        # Materialize IDs eagerly for compatibility with Dataset-like private access patterns.
        _ = self.ids

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
            ids = []
            for study in self.studyset.studies:
                for analysis in study.analyses:
                    ids.append(f"{study.id}-{analysis.id}")
            self._ids = np.sort(np.asarray(ids))
        return self._ids

    @property
    def coordinates(self):
        """pandas.DataFrame: Coordinates table."""
        if self._coordinates is None:
            rows = []
            spaces = []
            for study in self.studyset.studies:
                for analysis in study.analyses:
                    full_id = f"{study.id}-{analysis.id}"
                    for point in analysis.points:
                        row = {
                            "id": full_id,
                            "study_id": study.id,
                            "contrast_id": analysis.id,
                            "x": float(point.x),
                            "y": float(point.y),
                            "z": float(point.z),
                            "space": point.space,
                        }
                        rows.append(row)
                        spaces.append(_normalize_point_space(point.space))

            if rows:
                coordinates = pd.DataFrame(rows).sort_values(by="id")
            else:
                coordinates = pd.DataFrame(columns=self._id_cols + ["x", "y", "z", "space"])

            if self.space is None:
                inferred_spaces = {space for space in spaces if space is not None}
                if len(inferred_spaces) == 1:
                    self.space = inferred_spaces.pop()
                    self._ensure_masker_from_space()

            self._coordinates = coordinates
        return self._coordinates

    @coordinates.setter
    def coordinates(self, df):
        self._coordinates = df.sort_values(by="id")
        self._ids = np.sort(df["id"].unique().astype(str))

    @property
    def metadata(self):
        """pandas.DataFrame: Metadata table."""
        if self._metadata is None:
            rows = []
            for study in self.studyset.studies:
                for analysis in study.analyses:
                    full_id = f"{study.id}-{analysis.id}"
                    row = {
                        "id": full_id,
                        "study_id": study.id,
                        "contrast_id": analysis.id,
                        "study_name": study.name or study.id,
                        "analysis_name": analysis.name or analysis.id,
                        "authors": study.authors,
                        "journal": study.publication,
                        "name": f"{study.name or study.id}-{analysis.name or analysis.id}",
                    }
                    row.update(analysis.get_metadata())
                    rows.append(row)

            if rows:
                self._metadata = pd.DataFrame(rows).sort_values(by="id")
            else:
                self._metadata = pd.DataFrame(columns=self._id_cols)
        return self._metadata

    @metadata.setter
    def metadata(self, df):
        self._metadata = df.sort_values(by="id")

    @property
    def annotations(self):
        """pandas.DataFrame: Flattened analysis annotation table."""
        if self._annotations is None:
            rows = []
            for study in self.studyset.studies:
                for analysis in study.analyses:
                    full_id = f"{study.id}-{analysis.id}"
                    row = {
                        "id": full_id,
                        "study_id": study.id,
                        "contrast_id": analysis.id,
                    }
                    for key, note in (analysis.annotations or {}).items():
                        if isinstance(note, dict):
                            row.update(note)
                        else:
                            row[key] = note
                    rows.append(row)
            self._annotations = pd.DataFrame(rows).sort_values(by="id")
        return self._annotations

    @annotations.setter
    def annotations(self, df):
        self._annotations = df.sort_values(by="id")

    @property
    def images(self):
        """pandas.DataFrame: Image file table."""
        if self._images is None:
            rows = []
            for study in self.studyset.studies:
                for analysis in study.analyses:
                    full_id = f"{study.id}-{analysis.id}"
                    row = {
                        "id": full_id,
                        "study_id": study.id,
                        "contrast_id": analysis.id,
                    }
                    for image in analysis.images:
                        image_type = _normalize_image_type(image.value_type)
                        if image_type is None:
                            continue
                        image_path = image.url if image.url else image.filename
                        row[image_type] = image_path
                        if image.space and "space" not in row:
                            row["space"] = image.space
                    rows.append(row)

            if rows:
                self._images = pd.DataFrame(rows).sort_values(by="id")
            else:
                self._images = pd.DataFrame(columns=self._id_cols)
        return self._images

    @images.setter
    def images(self, df):
        self._images = df.sort_values(by="id")

    @property
    def texts(self):
        """pandas.DataFrame: Flattened analysis texts table."""
        if self._texts is None:
            rows = []
            for study in self.studyset.studies:
                for analysis in study.analyses:
                    row = {
                        "id": f"{study.id}-{analysis.id}",
                        "study_id": study.id,
                        "contrast_id": analysis.id,
                    }
                    row.update(analysis.texts or {})
                    rows.append(row)
            self._texts = pd.DataFrame(rows).sort_values(by="id")
        return self._texts

    def copy(self):
        """Create a copy of the StudysetView."""
        return copy.deepcopy(self)

    def slice(self, ids):
        """Create a new view with only requested full analysis IDs."""
        ids = set(_listify(ids))
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
        )

        if self.annotations is not None and not self.annotations.empty:
            sliced_view.annotations = self.annotations[self.annotations["id"].isin(ids)].copy()

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
        # Always rebuild from the Dataset to avoid stale cached data when DataFrames
        # are modified in place between estimator calls.
        studyset = Studyset.from_dataset(dataset)
        studyset._nimare_masker = dataset.masker
        studyset._nimare_space = dataset.space
        studyset._nimare_basepath = dataset.basepath
        return StudysetView(
            studyset=studyset,
            target=dataset.space,
            mask=dataset.masker,
            basepath=dataset.basepath,
        )

    if isinstance(dataset, Studyset) or isinstance(dataset, (dict, str)):
        studyset = load_nimads(dataset)
        if isinstance(dataset, Studyset):
            cached_view = getattr(dataset, "_nimare_view_cache", None)
            if cached_view is not None:
                return cached_view
        view = StudysetView(studyset=studyset)
        if isinstance(dataset, Studyset):
            dataset._nimare_view_cache = view
        return view

    raise ValueError(
        "Input must be a Dataset, Studyset, dict, or path to a NIMADS studyset JSON, "
        f"not {type(dataset)}."
    )


def restore_input_type(input_obj, studyset_view):
    """Restore output type to match input type when possible."""
    from nimare.dataset import Dataset
    from nimare.nimads import Studyset

    if isinstance(input_obj, Dataset):
        return studyset_view

    if isinstance(input_obj, Studyset):
        return studyset_view.studyset

    return studyset_view
