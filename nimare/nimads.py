"""NIMADS-related classes for NiMARE."""

import json
import logging
import operator
import os
import weakref
from copy import deepcopy

import numpy as np
import pandas as pd
from nilearn.image import load_img

from nimare._studyset_store import StudysetExecutionProfile, StudysetStore
from nimare.exceptions import InvalidStudysetError
from nimare.io import convert_nimads_to_dataset
from nimare.utils import (
    _mask_img_to_bool,
    _validate_df,
    _validate_images_df,
    mm2vox,
)

LGR = logging.getLogger(__name__)

_UNSET = object()


def _validate_studyset_source(source):
    """Validate the minimal schema required to construct a Studyset."""
    if not isinstance(source, dict):
        raise InvalidStudysetError("Studyset source must be a dictionary or JSON path")

    missing_fields = [field for field in ("id", "studies") if field not in source]
    if missing_fields:
        raise InvalidStudysetError(
            f"Studyset is missing required field(s): {', '.join(missing_fields)}"
        )

    if not isinstance(source["studies"], list):
        raise InvalidStudysetError("Studyset 'studies' field must be a list")


def _infer_dataset_basepath(dataset):
    """Infer a Dataset base path from paired absolute and relative image columns."""
    basepath = getattr(dataset, "basepath", None)
    if basepath:
        return os.path.abspath(basepath)

    images = getattr(dataset, "images", None)
    if images is None or images.empty:
        return None

    candidate_basepaths = []
    for rel_col in [col for col in images.columns if col.endswith("__relative")]:
        abs_col = rel_col[: -len("__relative")]
        if abs_col not in images.columns:
            continue

        for relative_path, absolute_path in zip(images[rel_col], images[abs_col]):
            if not isinstance(relative_path, str) or not relative_path:
                continue
            if not isinstance(absolute_path, str) or not absolute_path:
                continue
            if not os.path.isabs(absolute_path):
                continue
            if not absolute_path.endswith(relative_path):
                continue

            candidate_basepaths.append(
                absolute_path[: -len(relative_path)].rstrip(os.sep) or os.sep
            )

    if not candidate_basepaths:
        return None

    return os.path.commonpath(candidate_basepaths)


class _NotifyDict(dict):
    """Dict subclass that fires a callback on any mutation."""

    __slots__ = ("_on_mutate",)

    def __init__(self, data, on_mutate):
        super().__init__(data)
        self._on_mutate = on_mutate

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._on_mutate()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._on_mutate()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._on_mutate()

    def pop(self, *args):
        result = super().pop(*args)
        self._on_mutate()
        return result

    def clear(self):
        super().clear()
        self._on_mutate()

    def setdefault(self, key, default=None):
        if key not in self:
            super().__setitem__(key, default)
            self._on_mutate()
            return default
        return self[key]

    def __ior__(self, other):
        super().update(other)
        self._on_mutate()
        return self

    def __reduce__(self):
        # Pickle as a plain dict so unpickled copies don't carry dead callbacks.
        return (dict, (list(self.items()),))

    def __deepcopy__(self, memo):
        # Deepcopy as a plain dict; the owning Studyset re-installs tracking.
        return dict(deepcopy(list(self.items()), memo))


class Studyset:
    """A collection of studies for meta-analysis.

    .. versionadded:: 0.0.14

    This is the primary target for Estimators and Transformers in NiMARE.

    Attributes
    ----------
    id : str
        A unique identifier for the Studyset.
    name : str
        A human-readable name for the Studyset.
    annotations : :obj:`list` of :obj:`nimare.nimads.Annotation` objects
        The Annotation objects associated with the Studyset.
    studies : :obj:`list` of :obj:`nimare.nimads.Study` objects
        The Study objects comprising the Studyset.
    """

    _id_cols = ["id", "study_id", "contrast_id"]

    def __init__(
        self,
        source,
        target=_UNSET,
        mask=None,
        annotations=None,
        basepath=None,
        harmonize_coordinates=True,
    ):
        if target is _UNSET:
            target = "mni152_2mm"

        # load source as json
        if isinstance(source, str):
            with open(source, "r+") as f:
                source = json.load(f)

        _validate_studyset_source(source)

        annotation_payloads = list(source.get("annotations", []) or [])
        if annotations is not None:
            if isinstance(annotations, (list, tuple)):
                annotation_payloads.extend(list(annotations))
            else:
                annotation_payloads.append(annotations)

        store = StudysetStore.from_source_dict(
            source,
            annotation_payloads=annotation_payloads,
            target=target,
            harmonize_coordinates=harmonize_coordinates,
        )
        execution_profile = StudysetExecutionProfile(
            target=target,
            masker=mask,
            basepath=basepath,
            coordinate_space_policy=(
                "harmonize" if target is not None and harmonize_coordinates else "preserve"
            ),
        )
        self._initialize_from_store(store, execution_profile)

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Studyset: {self.id}>")

    def __str__(self):
        """Give useful information about the Studyset."""
        if self._studies is not None:
            n_studies = len(self._studies)
        else:
            ids = self._current_store().selected_ids(self._selection_full_ids)
            if ids is not None and len(ids) > 0:
                n_studies = len(set(fid.rsplit("-", 1)[0] for fid in ids))
            else:
                n_studies = 0
        return str(" ".join(["Studyset:", self.name, "::", f"studies: {n_studies}"]))

    def __setstate__(self, state):
        """Restore state and re-install mutation tracking after unpickling."""
        self.__dict__.update(state)
        if self._studies is not None:
            self._install_mutation_tracking()

    def _initialize_from_store(self, store, execution_profile, selection_full_ids=None):
        """Initialize Studyset state from a canonical store and execution profile."""
        self.id = store.studyset_id
        self.name = store.studyset_name or ""
        self._studyset_store = store
        self._selection_full_ids = (
            None
            if selection_full_ids is None
            else np.sort(np.asarray(selection_full_ids, dtype=str))
        )
        self._execution_profile = execution_profile
        self._projection_cache = {}
        self._revision = 0
        self._studies = None
        self._annotations = None
        self._store_revision = 0

    @classmethod
    def _from_store(cls, store, execution_profile, selection_full_ids=None):
        """Create a Studyset from an existing store without reparsing source payloads."""
        studyset = object.__new__(cls)
        studyset._initialize_from_store(store, execution_profile, selection_full_ids)
        return studyset

    def _selection_key(self):
        """Return a cacheable fingerprint for the current analysis selection."""
        if self._selection_full_ids is None:
            return None
        return tuple(self._selection_full_ids.tolist())

    def _copy_execution_profile(self, *, target=None, mask=None, basepath=None):
        """Return a copy of the execution profile with optional overrides."""
        return StudysetExecutionProfile(
            target=self._execution_profile.target if target is None else target,
            masker=self._execution_profile.masker if mask is None else mask,
            basepath=self._execution_profile.basepath if basepath is None else basepath,
            coordinate_space_policy=self._execution_profile.coordinate_space_policy,
        )

    def _source_dict_from_materialized(self):
        """Build a source dictionary from the current nested Study/Annotation graph."""
        studyset_dict = {
            "id": self.id,
            "name": self.name,
            "studies": [study.to_dict() for study in (self._studies or [])],
        }
        annotations = self._annotations or []
        if annotations:
            studyset_dict["annotations"] = [annotation.to_dict() for annotation in annotations]
        return studyset_dict

    def _current_store(self):
        """Return the canonical store for projections and selection."""
        return self._studyset_store

    def _copy_table_cache(self, table_cache):
        """Return a detached copy of a Dataset-style table cache."""
        copied = {
            "space": table_cache.get("space"),
            "masker": table_cache.get("masker"),
            "basepath": table_cache.get("basepath"),
        }
        ids = table_cache.get("ids")
        copied["ids"] = None if ids is None else np.asarray(ids, dtype=str).copy()
        for attr in ("coordinates", "images", "metadata", "annotations", "texts"):
            value = table_cache.get(attr)
            copied[attr] = None if value is None else value.copy()
        return copied

    def _cache_key(self, execution_profile):
        """Build a projection-cache key for the current Studyset state."""
        return (
            self._revision,
            self._selection_key(),
            execution_profile.cache_key(),
        )

    def _get_raw_table_cache(self, *, copy_tables=False):
        """Return the raw store-backed tables for the current selection."""
        table_cache = self._current_store().raw_tables(
            self._selection_full_ids,
            basepath=self._execution_profile.basepath,
        )
        table_cache["space"] = None
        table_cache["masker"] = None
        table_cache["basepath"] = self._execution_profile.basepath
        if copy_tables:
            return self._copy_table_cache(table_cache)
        return table_cache

    def _get_projected_table_cache(
        self, *, target=None, mask=None, basepath=None, copy_tables=False
    ):
        """Return Dataset-compatible execution tables for the current selection."""
        self._ensure_store_synced()
        execution_profile = self._copy_execution_profile(
            target=target,
            mask=mask,
            basepath=basepath,
        )
        cache_key = self._cache_key(execution_profile)
        if cache_key not in self._projection_cache:
            self._projection_cache[cache_key] = self._studyset_store.projected_tables(
                execution_profile,
                self._selection_full_ids,
            )

        table_cache = self._projection_cache[cache_key]
        return self._copy_table_cache(table_cache) if copy_tables else table_cache

    @property
    def is_materialized(self):
        """bool: Whether the nested Study/Analysis graph has been materialized."""
        return self._studies is not None

    @property
    def studies(self):
        """Return the nested Study graph, materializing it on demand if needed."""
        self.materialize()
        return self._studies

    @studies.setter
    def studies(self, studies):
        self._studies = studies
        self.touch()

    @property
    def annotations(self):
        """Return existing Annotations."""
        self.materialize()
        return self._annotations

    def _coerce_annotation(self, annotation):
        """Normalize one annotation payload to an Annotation instance."""
        if isinstance(annotation, dict):
            return Annotation(annotation, self)
        if isinstance(annotation, str):
            with open(annotation, "r+") as f:
                return Annotation(json.load(f), self)
        if isinstance(annotation, Annotation):
            return annotation
        raise TypeError(f"Unsupported annotation type: {type(annotation)}")

    def _extend_annotations(self, annotations):
        """Append one or more annotations and invalidate caches once."""
        self.materialize()
        if isinstance(annotations, (list, tuple)):
            loaded_annotations = [
                self._coerce_annotation(annotation) for annotation in annotations
            ]
        else:
            loaded_annotations = [self._coerce_annotation(annotations)]

        self._annotations = (self._annotations or []) + loaded_annotations
        self.touch()

    @annotations.setter
    def annotations(self, annotation):
        self._extend_annotations(annotation)

    @annotations.deleter
    def annotations(self, annotation_id=None):
        if annotation_id:
            self._annotations = [a for a in self._annotations if a.id != annotation_id]
        else:
            self._annotations = []
        self.touch()

    def _mark_dirty(self):
        """Flag the materialized graph as modified without rebuilding the store."""
        self._revision += 1
        self._projection_cache = {}

    def touch(self):
        """Invalidate Studyset-derived caches after in-place mutation."""
        if self._studies is not None:
            # The source dict is freshly built from our own graph, so pass
            # _owned=True to skip the structural copy and skip harmonization
            # (coordinates are already in the target space).
            self._studyset_store = StudysetStore.from_source_dict(
                self._source_dict_from_materialized(),
                _owned=True,
                harmonize_coordinates=False,
            )
            self.id = self._studyset_store.studyset_id
            self.name = self._studyset_store.studyset_name or self.name
            self._selection_full_ids = None
            self._install_mutation_tracking()
        self._revision += 1
        self._store_revision = self._revision
        self._projection_cache = {}

    def materialize(self):
        """Materialize the nested Study/Analysis graph if this Studyset is lazy."""
        if self._studies is not None:
            return self

        source = self._studyset_store.selected_source_dict(self._selection_full_ids)
        _validate_studyset_source(source)
        self._studies = [Study(s) for s in source["studies"]]
        self._annotations = [
            Annotation(annotation, self) for annotation in source.get("annotations", [])
        ]
        self._install_mutation_tracking()
        self._store_revision = self._revision
        return self

    def _install_mutation_tracking(self):
        """Wire up change-detection callbacks on mutable nested containers."""
        studyset_ref = weakref.ref(self)

        def _on_mutate():
            ss = studyset_ref()
            if ss is not None:
                ss._mark_dirty()

        for study in self._studies or []:
            study._on_mutate = _on_mutate
            for analysis in study.analyses:
                analysis._on_mutate = _on_mutate
                # Wrap existing plain dicts so in-place mutations are detected.
                for attr in Analysis._DICT_ATTRS:
                    val = getattr(analysis, attr, None)
                    if isinstance(val, dict) and not isinstance(val, _NotifyDict):
                        super(Analysis, analysis).__setattr__(attr, _NotifyDict(val, _on_mutate))
            # Same for Study dict attrs.
            for attr in Study._DICT_ATTRS:
                val = getattr(study, attr, None)
                if isinstance(val, dict) and not isinstance(val, _NotifyDict):
                    super(Study, study).__setattr__(attr, _NotifyDict(val, _on_mutate))

    @property
    def space(self):
        """Execution-space label used for Dataset-like Studyset views."""
        return self._execution_profile.target

    @property
    def is_execution_ready(self):
        """bool: Whether the Studyset has an explicit execution configuration."""
        return self._execution_profile.is_ready

    @property
    def masker(self):
        """Masker used for Dataset-like Studyset views."""
        return self._execution_profile.masker

    @property
    def basepath(self):
        """Base path used for image resolution in Dataset-like Studyset views."""
        return self._execution_profile.basepath

    @property
    def ids(self):
        """numpy.ndarray: 1D array of full analysis identifiers."""
        return self._current_store().selected_ids(self._selection_full_ids)

    @property
    def study_ids(self):
        """numpy.ndarray: 1D array of unique study identifiers.

        Extracted from the full ``<study_id>-<analysis_id>`` identifiers
        without materializing the nested Study graph.
        """
        return np.unique(np.array([fid.rsplit("-", 1)[0] for fid in self.ids]))

    @property
    def coordinates(self):
        """pandas.DataFrame: Dataset-like coordinates table."""
        return self._get_projected_table_cache()["coordinates"]

    @coordinates.setter
    def coordinates(self, df):
        self._set_projected_table("coordinates", df, update_ids=True)

    @property
    def images(self):
        """pandas.DataFrame: Dataset-like images table."""
        return self._get_projected_table_cache()["images"]

    @images.setter
    def images(self, df):
        self._set_projected_table("images", df)

    @property
    def metadata(self):
        """pandas.DataFrame: Dataset-like metadata table."""
        return self._get_projected_table_cache()["metadata"]

    @metadata.setter
    def metadata(self, df):
        self._set_projected_table("metadata", df)

    @property
    def texts(self):
        """pandas.DataFrame: Dataset-like texts table."""
        return self._get_projected_table_cache()["texts"]

    @texts.setter
    def texts(self, df):
        self._set_projected_table("texts", df)

    @property
    def annotations_df(self):
        """pandas.DataFrame: Flattened analysis-level annotations table."""
        return self._get_projected_table_cache()["annotations"]

    @annotations_df.setter
    def annotations_df(self, annotations_df):
        self.set_annotations_df(annotations_df)

    def _set_projected_table(self, attr, df, update_ids=False):
        """Replace one projected execution table for this Studyset."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{attr} must be a pandas DataFrame")
        _validate_df(df)
        if attr == "images":
            df = _validate_images_df(df)

        table_cache = self._get_projected_table_cache(copy_tables=True)
        table_cache[attr] = df.sort_values(by="id").reset_index(drop=True)
        if update_ids and "id" in df.columns:
            table_cache["ids"] = np.sort(df["id"].astype(str).unique())

        default_key = self._cache_key(self._execution_profile)
        self._projection_cache[default_key] = table_cache

    def set_annotations_df(self, annotations_df, overwrite=True):
        """Update analysis-level annotations from a flattened DataFrame.

        Parameters
        ----------
        annotations_df : :class:`pandas.DataFrame`
            DataFrame with one row per analysis. Must contain either an ``id`` column with
            Dataset-style full IDs (``<study_id>-<analysis_id>``) or both ``study_id`` and
            ``contrast_id`` columns from which the full IDs can be derived.
        overwrite : :obj:`bool`, optional
            If True, replace each analysis' existing annotation dictionary with the values from
            ``annotations_df``. Analyses absent from the DataFrame will have their annotations
            cleared. If False, merge new values into existing annotation dictionaries while leaving
            untouched analyses unchanged. Default is True.
        """
        if not isinstance(annotations_df, pd.DataFrame):
            raise TypeError("annotations_df must be a pandas DataFrame")

        annotations_df = annotations_df.copy()
        if "id" not in annotations_df.columns:
            required_cols = {"study_id", "contrast_id"}
            if not required_cols.issubset(annotations_df.columns):
                raise ValueError(
                    "annotations_df must contain either an 'id' column or both "
                    "'study_id' and 'contrast_id' columns."
                )
            annotations_df["id"] = (
                annotations_df["study_id"].astype(str)
                + "-"
                + annotations_df["contrast_id"].astype(str)
            )

        duplicated = annotations_df["id"].duplicated()
        if duplicated.any():
            dup_ids = annotations_df.loc[duplicated, "id"].astype(str).tolist()
            raise ValueError(
                "annotations_df contains duplicate analysis IDs: " + ", ".join(dup_ids[:10])
            )

        analysis_map = {
            f"{study.id}-{analysis.id}": analysis
            for study in self.studies
            for analysis in study.analyses
        }
        unknown_ids = sorted(set(annotations_df["id"].astype(str)) - set(analysis_map))
        if unknown_ids:
            raise ValueError(
                "annotations_df contains IDs not present in the Studyset: "
                + ", ".join(unknown_ids[:10])
            )

        ignore_cols = {"id", "study_id", "contrast_id"}
        annotations_by_id = {}
        for _, row in annotations_df.iterrows():
            full_id = str(row["id"])
            row_annotations = {
                key: value
                for key, value in row.items()
                if key not in ignore_cols and not pd.isna(value)
            }
            annotations_by_id[full_id] = row_annotations

        for full_id, analysis in analysis_map.items():
            if full_id in annotations_by_id:
                if overwrite:
                    analysis.annotations = annotations_by_id[full_id]
                else:
                    merged = dict(analysis.annotations)
                    merged.update(annotations_by_id[full_id])
                    analysis.annotations = merged
            elif overwrite:
                analysis.annotations = {}

        self.touch()

    def _generic_column_getter(self, attr, ids=None, column=None, ignore_columns=None):
        """Get one field or all available fields from a Studyset table."""
        if ignore_columns is None:
            ignore_columns = list(self._id_cols)
        else:
            ignore_columns = list(ignore_columns) + self._id_cols

        df = getattr(self, attr)
        return_first = False
        if isinstance(ids, str) and column is not None:
            return_first = True

        if ids is not None and not isinstance(ids, list):
            ids = np.atleast_1d(ids).tolist()

        available_types = [c for c in df.columns if c not in ignore_columns]
        if column is not None and column not in available_types:
            raise ValueError(
                f"{column} not found in {attr}.\nAvailable types: {', '.join(available_types)}"
            )

        if column is not None:
            if ids is not None:
                result = df.loc[df["id"].isin(ids), column].tolist()
            else:
                result = df[column].tolist()
            result = [None if pd.isna(val) else val for val in result]
        else:
            if ids is not None:
                result = {v: df.loc[df["id"].isin(ids), v].tolist() for v in available_types}
                result = {k: v for k, v in result.items() if any(v)}
            else:
                result = {v: df[v].tolist() for v in available_types}
            result = list(result.keys())

        if return_first:
            return result[0]
        return result

    def get(self, dict_, drop_invalid=True):
        """Retrieve files and/or metadata from the current Studyset."""
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
                annot_by_id = dict(iter(self.annotations_df.groupby("id")))
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
                results[key] = pd.concat(results[key]) if results[key] else pd.DataFrame()

        return results

    def get_labels(self, ids=None):
        """Extract labels present in the Studyset's analysis-level annotations."""
        if ids is not None and not isinstance(ids, (list, tuple, np.ndarray)):
            ids = [ids]

        result = [c for c in self.annotations_df.columns if c not in self._id_cols]
        if ids is not None and result:
            temp_annotations = self.annotations_df.loc[self.annotations_df["id"].isin(ids)]
            present = temp_annotations[result].any(axis=0)
            result = present.loc[present].index.tolist()
        return result

    @staticmethod
    def _flatten_analysis_annotations(analysis):
        """Return one analysis' annotations as a flat label-to-value mapping."""
        flat_annotations = {}
        for key, note in (analysis.annotations or {}).items():
            if isinstance(note, dict):
                flat_annotations.update(note)
            else:
                flat_annotations[key] = note
        return flat_annotations

    def get_analyses_by_label(self, labels=None, label_threshold=0.001):
        """Extract analysis IDs whose annotation values exceed the threshold."""
        return [
            full_id.rsplit("-", 1)[1]
            for full_id in self.get_studies_by_label(
                labels=labels,
                label_threshold=label_threshold,
            )
        ]

    def get_studies_by_label(self, labels=None, label_threshold=0.001):
        """Extract full analysis IDs whose annotation values exceed the threshold."""
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            raise ValueError(f"Argument 'labels' cannot be {type(labels)}")

        missing_labels = [label for label in labels if label not in self.annotations_df.columns]
        if missing_labels:
            raise ValueError(f"Missing label(s): {', '.join(missing_labels)}")

        temp_annotations = self.annotations_df[self._id_cols + labels]
        found_rows = (temp_annotations[labels] >= label_threshold).all(axis=1)
        if any(found_rows):
            return temp_annotations.loc[found_rows, "id"].tolist()
        return []

    def get_studies_by_mask(self, mask):
        """Extract full analysis IDs with at least one focus inside ``mask``."""
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
        dset_ijk = np.clip(dset_ijk, 0, np.array(mask_data.shape) - 1)
        in_mask = mask_data[dset_ijk[:, 0], dset_ijk[:, 1], dset_ijk[:, 2]] > 0
        return list(self.coordinates.loc[in_mask, "id"].unique())

    def get_studies_by_coordinate(self, xyz, r=None, n=None):
        """Extract full analysis IDs with at least one focus near the requested coordinates."""
        from scipy.spatial.distance import cdist

        if self.coordinates.empty:
            return []

        xyz = np.array(xyz)
        assert xyz.ndim == 2 and xyz.shape[1] == 3, "xyz must be (n, 3)"
        if r is None and n is None:
            raise ValueError("Either 'r' or 'n' must be provided")
        distances = cdist(xyz, self.coordinates[["x", "y", "z"]].values)

# If radius-based query
        if r is not None:
             mask = np.any(distances <= r, axis=0)
             return list(self.coordinates.loc[mask, "id"].unique())

# If nearest-n query
        if n is not None:
    # get minimum distance for each coordinate point
             min_dist = distances.min(axis=0)

    # sort indices by distance
             nearest_idx = np.argsort(min_dist)[:min(n, len(min_dist))]

            selected_ids = self.coordinates.iloc[nearest_idx]["id"]
            return list(pd.unique(selected_ids))
     # NOTE: nearest_idx refers to coordinate rows, not unique studies

    def _set_execution_context(self, *, space=None, masker=None, basepath=None):
        """Update cached execution context used by Studyset projections."""
        self._execution_profile = self._copy_execution_profile(
            target=space,
            mask=masker,
            basepath=basepath,
        )
        self._projection_cache = {}

    def _iter_analyses(self):
        """Yield each analysis with its parent study."""
        for study in self.studies:
            for analysis in study.analyses:
                yield study, analysis

    def _iter_selected_analyses(self, analyses):
        """Yield analyses whose IDs appear in the requested list."""
        analyses = set(analyses)
        for _, analysis in self._iter_analyses():
            if analysis.id in analyses:
                yield analysis

    def _collect_analysis_points(self):
        """Collect all coordinate points and their analysis IDs."""
        all_points = []
        analysis_ids = []
        for _, analysis in self._iter_analyses():
            for point in analysis.points:
                if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                    all_points.append([point.x, point.y, point.z])
                    analysis_ids.append(analysis.id)
        return all_points, analysis_ids

    def _attach_dataset_context(self, dataset):
        """Cache Dataset-derived execution state for fast Studyset-backed estimators."""
        self._set_execution_context(
            space=dataset.space,
            masker=dataset.masker,
            basepath=_infer_dataset_basepath(dataset),
        )

    @classmethod
    def from_table_cache(
        cls,
        table_cache,
        *,
        studyset_id="nimads_cached_tables",
        studyset_name="",
        target=None,
        mask=None,
        basepath=None,
        materializer=None,
        normalize_metadata=True,
        seed_table_cache=False,
    ):
        """Create a lightweight Studyset backed by precomputed Dataset-style tables."""
        execution_profile = StudysetExecutionProfile(
            target=target if target is not None else table_cache.get("space"),
            masker=mask if mask is not None else table_cache.get("masker"),
            basepath=basepath if basepath is not None else table_cache.get("basepath"),
        )
        store = StudysetStore.from_table_cache(
            studyset_id,
            studyset_name,
            table_cache,
            materializer=materializer,
            normalize_metadata=normalize_metadata,
        )
        studyset = cls._from_store(store, execution_profile)
        # Seed the projection cache so the first property access doesn't rebuild.
        cache_key = studyset._cache_key(execution_profile)
        if seed_table_cache:
            studyset._projection_cache[cache_key] = {
                "ids": table_cache.get("ids"),
                "coordinates": table_cache.get("coordinates"),
                "images": table_cache.get("images"),
                "metadata": table_cache.get("metadata"),
                "annotations": table_cache.get("annotations"),
                "texts": table_cache.get("texts"),
                "space": execution_profile.target,
                "masker": execution_profile.masker,
                "basepath": execution_profile.basepath,
            }
        else:
            studyset._projection_cache[cache_key] = store.projected_tables(execution_profile)
        return studyset

    @classmethod
    def from_nimads(cls, filename):
        """Create a Studyset from a NIMADS JSON file."""
        with open(filename, "r+") as fn:
            nimads = json.load(fn)

        return cls(nimads)

    @classmethod
    def from_dataset(cls, dataset, *, materialize=True):
        """Create a Studyset from a NiMARE Dataset."""
        dataset_basepath = _infer_dataset_basepath(dataset)

        if not materialize:
            from nimare.studyset import _snapshot_dataset_tables

            dataset_ref = weakref.ref(dataset)

            def _materializer():
                dataset_obj = dataset_ref()
                if dataset_obj is None:
                    raise RuntimeError(
                        "Cannot materialize Studyset because the source Dataset is gone."
                    )

                from nimare.io import convert_dataset_to_nimads_dict

                return convert_dataset_to_nimads_dict(
                    dataset_obj,
                    studyset_id="nimads_from_dataset",
                    studyset_name="",
                )

            studyset = cls.from_table_cache(
                _snapshot_dataset_tables(dataset, copy_tables=True),
                studyset_id="nimads_from_dataset",
                target=dataset.space,
                mask=dataset.masker,
                basepath=dataset_basepath,
                materializer=_materializer,
                normalize_metadata=False,
                seed_table_cache=True,
            )
            return studyset

        from nimare.io import convert_dataset_to_nimads_dict

        nimads = convert_dataset_to_nimads_dict(dataset)
        studyset = cls(
            nimads,
            target=dataset.space,
            mask=dataset.masker,
            basepath=dataset_basepath,
        )
        studyset._attach_dataset_context(dataset)
        return studyset

    @classmethod
    def from_sleuth(cls, sleuth_file):
        """Create a Studyset from a Sleuth text file."""
        from nimare.io import convert_sleuth_to_nimads_dict

        nimads = convert_sleuth_to_nimads_dict(sleuth_file)
        return cls(nimads)

    def combine_analyses(self):
        """Combine analyses in Studyset."""
        studyset = self.copy()
        for study in studyset.studies:
            if len(study.analyses) > 1:
                source_lst = [analysis.to_dict() for analysis in study.analyses]
                ids = [source["id"] for source in source_lst]
                names = [source["name"] for source in source_lst]
                conditions = [source.get("conditions", []) for source in source_lst]
                images = [source.get("images", []) for source in source_lst]
                points = [source.get("points", []) for source in source_lst]
                weights = [source.get("weights", []) for source in source_lst]
                metadata = [source.get("metadata", {}) for source in source_lst]
                annotations = [source.get("annotations", {}) for source in source_lst]
                texts = [source.get("texts", {}) for source in source_lst]

                new_source = {
                    "id": "_".join(ids),
                    "name": "; ".join(names),
                    "conditions": [cond for c_list in conditions for cond in c_list],
                    "images": [image for i_list in images for image in i_list],
                    "points": [point for p_list in points for point in p_list],
                    "weights": [weight for w_list in weights for weight in w_list],
                    "metadata": {k: v for m_dict in metadata for k, v in m_dict.items()},
                }
                combined_annotations = {
                    k: v for annot_dict in annotations for k, v in annot_dict.items()
                }
                combined_texts = {k: v for text_dict in texts for k, v in text_dict.items()}
                if combined_annotations:
                    new_source["annotations"] = combined_annotations
                if combined_texts:
                    new_source["texts"] = combined_texts
                study.analyses = [Analysis(new_source)]

        # Old Analysis objects are gone; Annotation notes hold dead weak references.
        # Clear top-level annotations so touch() can rebuild cleanly.
        studyset._annotations = []
        studyset.touch()
        return studyset

    def to_nimads(self, filename):
        """Write the Studyset to a NIMADS JSON file."""
        with open(filename, "w+") as fn:
            json.dump(self.to_dict(), fn)

    def to_dict(self):
        """Return a dictionary representation of the Studyset."""
        self._ensure_store_synced()
        return self._studyset_store.selected_source_dict(self._selection_full_ids)

    def to_dataset(self):
        """Convert the Studyset to a NiMARE Dataset."""
        return convert_nimads_to_dataset(self)

    def load(self, filename):
        """Load a Studyset from a pickled file.

        Parameters
        ----------
        filename : str
            Path to the pickled file to load from.

        Returns
        -------
        Studyset
            The loaded Studyset object.
        """
        import pickle

        with open(filename, "rb") as f:
            loaded_data = pickle.load(f)

        self.__dict__.update(loaded_data.__dict__)
        self._projection_cache = {}
        if self._studies is not None:
            self._install_mutation_tracking()
        return self

    def save(self, filename):
        """Write the Studyset to a pickled file.

        Parameters
        ----------
        filename : str
            Path where the pickled file should be saved.
        """
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def update_path(self, new_path):
        """Prepend a base path for relative image paths in the Studyset."""
        self._set_execution_context(basepath=os.path.abspath(new_path))
        return self

    def copy(self):
        """Create a copy of the Studyset."""
        return deepcopy(self)

    def __deepcopy__(self, memo):
        """Deep-copy the Studyset, sharing the immutable store to avoid expensive recursion."""
        result = object.__new__(Studyset)
        memo[id(self)] = result
        result.id = self.id
        result.name = self.name
        # The store is never mutated in-place; touch() replaces it entirely.
        result._studyset_store = self._studyset_store
        result._selection_full_ids = (
            self._selection_full_ids.copy() if self._selection_full_ids is not None else None
        )
        result._execution_profile = self._execution_profile
        result._projection_cache = {
            key: self._copy_table_cache(table_cache)
            for key, table_cache in self._projection_cache.items()
        }
        result._revision = self._revision
        result._store_revision = self._store_revision
        if self._studies is not None:
            result._studies = deepcopy(self._studies, memo)
            result._annotations = deepcopy(self._annotations, memo)
            result._install_mutation_tracking()
        else:
            result._studies = None
            result._annotations = None
        return result

    def _ensure_store_synced(self):
        """Rebuild the store from the materialized graph if it is stale.

        The store is considered stale when the materialized graph has been
        mutated since the last rebuild, tracked by comparing
        ``_store_revision`` to ``_revision``.
        """
        if self._studies is not None and self._store_revision < self._revision:
            self.touch()

    def filter_ids(self, ids):
        """Return a Studyset filtered to the requested analysis IDs."""
        self._ensure_store_synced()
        if isinstance(ids, str):
            ids = [ids]
        resolved_ids = self._current_store().resolve_full_ids(
            ids,
            allow_short_ids=True,
            selected_full_ids=self._selection_full_ids,
        )
        execution_profile = self._copy_execution_profile()
        return self.__class__._from_store(
            self._studyset_store, execution_profile, selection_full_ids=resolved_ids
        )

    def filter_annotations(self, labels, threshold=0.001, match="all"):
        """Return a Studyset filtered by annotation labels."""
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            raise ValueError(f"Argument 'labels' cannot be {type(labels)}")

        annotations_df = self.annotations_df
        missing_labels = [label for label in labels if label not in annotations_df.columns]
        if missing_labels:
            raise ValueError(f"Missing label(s): {', '.join(missing_labels)}")

        comparator = annotations_df[labels].fillna(float("-inf")) >= threshold
        if match == "all":
            keep = comparator.all(axis=1)
        elif match == "any":
            keep = comparator.any(axis=1)
        else:
            raise ValueError("match must be 'all' or 'any'")

        return self.filter_ids(annotations_df.loc[keep, "id"].tolist())

    def filter_study_ids(self, study_ids):
        """Return a Studyset keeping only analyses belonging to the given study IDs.

        Parameters
        ----------
        study_ids : :obj:`str` or :obj:`list` of :obj:`str`
            One or more study-level identifiers to keep.

        Returns
        -------
        Studyset
            A new Studyset containing only analyses from the specified studies.
        """
        if isinstance(study_ids, str):
            study_ids = [study_ids]
        keep = set(study_ids)
        full_ids = [fid for fid in self.ids if fid.rsplit("-", 1)[0] in keep]
        return self.filter_ids(full_ids)

    def exclude_study_ids(self, study_ids):
        """Return a Studyset excluding analyses belonging to the given study IDs.

        Parameters
        ----------
        study_ids : :obj:`str` or :obj:`list` of :obj:`str`
            One or more study-level identifiers to exclude.

        Returns
        -------
        Studyset
            A new Studyset with analyses from the specified studies removed.
        """
        if isinstance(study_ids, str):
            study_ids = [study_ids]
        exclude = set(study_ids)
        full_ids = [fid for fid in self.ids if fid.rsplit("-", 1)[0] not in exclude]
        return self.filter_ids(full_ids)

    def filter_metadata(self, field, op, value):
        """Return a Studyset filtered by one metadata field."""
        metadata_df = self.metadata
        if field not in metadata_df.columns:
            raise ValueError(f"Unknown metadata field: {field}")

        op_map = {
            "==": operator.eq,
            "eq": operator.eq,
            "!=": operator.ne,
            "ne": operator.ne,
            ">": operator.gt,
            "gt": operator.gt,
            ">=": operator.ge,
            "ge": operator.ge,
            "<": operator.lt,
            "lt": operator.lt,
            "<=": operator.le,
            "le": operator.le,
            "in": lambda series, rhs: series.isin(rhs),
            "contains": lambda series, rhs: series.astype(str).str.contains(rhs, na=False),
        }
        if op not in op_map:
            raise ValueError(f"Unsupported metadata operator: {op}")

        series = metadata_df[field]
        keep = op_map[op](series, value)
        if not isinstance(keep, pd.Series):
            keep = pd.Series(keep, index=metadata_df.index)
        keep = keep.fillna(False)
        return self.filter_ids(metadata_df.loc[keep, "id"].tolist())

    def slice(self, ids=None, *, analyses=None, filter_level="analysis"):
        """Create a new Studyset keeping only the requested IDs.

        Parameters
        ----------
        ids : :obj:`str` or :obj:`list` of :obj:`str`, optional
            Identifiers to keep. Can also be passed via the deprecated
            ``analyses`` keyword for backwards compatibility.
        analyses : :obj:`str` or :obj:`list` of :obj:`str`, optional
            Deprecated alias for *ids*.  Will be removed in a future release.
        filter_level : ``"analysis"`` or ``"study"``, optional
            When ``"analysis"`` (default), *ids* are treated as analysis-level
            identifiers (full ``<study_id>-<analysis_id>`` strings or short
            analysis IDs).  When ``"study"``, *ids* are treated as study-level
            identifiers and every analysis belonging to those studies is kept.

        Returns
        -------
        Studyset
            A new Studyset containing only the matching analyses.
        """
        if ids is None and analyses is not None:
            ids = analyses
        elif ids is None and analyses is None:
            raise TypeError("slice() requires at least one of 'ids' or 'analyses'")

        if filter_level == "study":
            return self.filter_study_ids(ids)
        elif filter_level == "analysis":
            return self.filter_ids(ids)
        else:
            raise ValueError(f"filter_level must be 'analysis' or 'study', got {filter_level!r}")

    def merge(self, right):
        """Merge a separate Studyset into the current one.

        Parameters
        ----------
        right : Studyset
            The other Studyset to merge with this one.

        Returns
        -------
        Studyset
            A new Studyset containing merged studies from both input Studysets.
            For studies with the same ID, their analyses and metadata are combined,
            with data from self (left) taking precedence in case of conflicts.
        """
        if not isinstance(right, Studyset):
            raise ValueError("Can only merge with another Studyset")

        # Create new source dictionary starting with left (self) studyset
        merged_source = self.to_dict()
        merged_source["id"] = f"{self.id}_{right.id}"
        merged_source["name"] = f"Merged: {self.name} + {right.name}"

        # Create lookup of existing studies by ID
        left_studies = {study["id"]: study for study in merged_source["studies"]}

        # Process studies from right studyset
        right_dict = right.to_dict()
        for right_study in right_dict["studies"]:
            study_id = right_study["id"]

            if study_id in left_studies:
                # Merge study data
                left_study = left_studies[study_id]
                left_study["metadata"] = left_study.get("metadata", {}) or {}
                right_metadata = right_study.get("metadata", {}) or {}

                # Keep metadata from left unless missing
                left_study["metadata"].update(
                    {k: v for k, v in right_metadata.items() if k not in left_study["metadata"]}
                )

                # Keep basic info from left unless empty
                for field in ["name", "authors", "publication"]:
                    if not left_study[field]:
                        left_study[field] = right_study[field]

                # Combine analyses, avoiding duplicates by ID
                left_analyses = {a["id"]: a for a in left_study["analyses"]}
                for right_analysis in right_study["analyses"]:
                    if right_analysis["id"] not in left_analyses:
                        left_study["analyses"].append(right_analysis)
            else:
                # Add new study
                merged_source["studies"].append(right_study)

        # Create new merged studyset
        merged = self.__class__(
            source=merged_source,
            target=self.space,
            mask=self.masker,
            basepath=self.basepath,
            harmonize_coordinates=self._execution_profile.coordinate_space_policy == "harmonize",
        )

        # Merge annotations, preferring left's annotations for conflicts
        existing_annot_ids = {a.id for a in self.annotations}
        for right_annot in right.annotations:
            if right_annot.id not in existing_annot_ids:
                merged.annotations = right_annot.to_dict()

        return merged

    def get_analyses_by_coordinate(self, xyz, r=None, n=None):
        """Extract a list of Analyses with at least one Point near the requested coordinates.

        Parameters
        ----------
        xyz : array_like
            1 x 3 array of coordinates in mm space to search from
        r : float, optional
            Search radius in millimeters.
            Mutually exclusive with n.
        n : int, optional
            Number of closest analyses to return.
            Mutually exclusive with r.

        Returns
        -------
        list[str]
            A list of Analysis IDs with at least one point within the search criteria.

        Notes
        -----
        Either r or n must be provided, but not both.
        """
        if (r is None and n is None) or (r is not None and n is not None):
            raise ValueError("Exactly one of r or n must be provided.")

        xyz = np.asarray(xyz).ravel()
        if xyz.shape != (3,):
            raise ValueError("xyz must be a 1 x 3 array-like object.")

        all_points, analysis_ids = self._collect_analysis_points()
        if not all_points:  # Return empty list if no coordinates found
            return []

        all_points = np.array(all_points)

        # Calculate Euclidean distances to all points
        distances = np.sqrt(np.sum((all_points - xyz) ** 2, axis=1))

        if r is not None:
            # Find analyses with points within radius r
            within_radius = distances <= r
            found_analyses = set(np.array(analysis_ids)[within_radius])
        else:
            # Find n closest analyses
            closest_n_idx = np.argsort(distances)[:n]
            found_analyses = set(np.array(analysis_ids)[closest_n_idx])

        return list(found_analyses)

    def get_analyses_by_mask(self, img):
        """Extract a list of Analyses with at least one Point in the specified mask.

        Parameters
        ----------
        img : img_like
            Mask across which to search for coordinates.

        Returns
        -------
        list[str]
            A list of Analysis IDs with at least one point in the mask.
        """
        # Load mask
        mask = load_img(img)

        all_points, analysis_ids = self._collect_analysis_points()
        if not all_points:  # Return empty list if no coordinates found
            return []

        # Convert to voxel coordinates
        all_points = np.array(all_points)
        ijk = mm2vox(all_points, mask.affine)

        # Get mask coordinates
        mask_data = _mask_img_to_bool(mask)
        mask_coords = np.vstack(np.where(mask_data)).T

        # Check for presence of coordinates in mask
        in_mask = np.any(np.all(ijk[:, None] == mask_coords[None, :], axis=-1), axis=-1)

        # Get unique analysis IDs where points are in mask
        found_analyses = set(np.array(analysis_ids)[in_mask])

        return list(found_analyses)

    def get_analyses_by_annotations(self, key, value=None):
        """Extract a list of Analyses with a given label/annotation."""
        annotations = {}
        for _, analysis in self._iter_analyses():
            a_annot = analysis.annotations
            if key in a_annot and (value is None or a_annot[key] == value):
                annotations[analysis.id] = {key: a_annot[key]}
        return annotations

    def get_analyses_by_metadata(self, key, value=None):
        """Extract a list of Analyses with a metadata field/value."""
        metadata = {}
        for _, analysis in self._iter_analyses():
            a_metadata = analysis.metadata
            if key in a_metadata and (value is None or a_metadata[key] == value):
                metadata[analysis.id] = {key: a_metadata[key]}
        return metadata

    def get_points(self, analyses):
        """Collect Points associated with specified Analyses."""
        return {
            analysis.id: analysis.points for analysis in self._iter_selected_analyses(analyses)
        }

    def get_annotations(self, analyses):
        """Collect Annotations associated with specified Analyses."""
        return {
            analysis.id: analysis.annotations
            for analysis in self._iter_selected_analyses(analyses)
        }

    def get_texts(self, analyses=None, text_type=None, ids=None):
        """Get texts for selected analyses or collect nested texts by short analysis IDs."""
        target_ids = ids if ids is not None else analyses
        is_tabular = (
            text_type is not None
            or ids is not None
            or analyses is None
            or (
                isinstance(target_ids, (str, list, tuple, np.ndarray))
                and any("-" in str(val) for val in np.atleast_1d(target_ids))
            )
        )
        if is_tabular:
            return self._generic_column_getter("texts", ids=target_ids, column=text_type)

        return {analysis.id: analysis.texts for analysis in self._iter_selected_analyses(analyses)}

    def get_images(self, analyses=None, imtype=None, ids=None):
        """Get image paths or collect nested Image objects by short analysis IDs."""
        target_ids = ids if ids is not None else analyses
        is_tabular = (
            imtype is not None
            or ids is not None
            or analyses is None
            or (
                isinstance(target_ids, (str, list, tuple, np.ndarray))
                and any("-" in str(val) for val in np.atleast_1d(target_ids))
            )
        )
        if is_tabular:
            ignore_columns = ["space"]
            ignore_columns += [c for c in self.images.columns if c.endswith("__relative")]
            return self._generic_column_getter(
                "images",
                ids=target_ids,
                column=imtype,
                ignore_columns=ignore_columns,
            )

        return {
            analysis.id: analysis.images for analysis in self._iter_selected_analyses(analyses)
        }

    def get_metadata(self, analyses=None, field=None, ids=None):
        """Get metadata values or collect nested metadata by short analysis IDs.

        Parameters
        ----------
        analyses : list of str, optional
            List of short Analysis IDs to get nested metadata for.
        field : str, optional
            Metadata field to retrieve from the tabular metadata table.
        ids : list of str, optional
            Full Dataset-style IDs to query in the tabular metadata table.

        Returns
        -------
        dict[str, dict] or list
            Nested metadata mapping or tabular metadata values, depending on the arguments.
        """
        target_ids = ids if ids is not None else analyses
        is_tabular = (
            field is not None
            or ids is not None
            or analyses is None
            or (
                isinstance(target_ids, (str, list, tuple, np.ndarray))
                and any("-" in str(val) for val in np.atleast_1d(target_ids))
            )
        )
        if is_tabular:
            return self._generic_column_getter("metadata", ids=target_ids, column=field)

        return {
            analysis.id: analysis.get_metadata()
            for analysis in self._iter_selected_analyses(analyses)
        }


class Study:
    """A collection of Analyses from the same paper.

    .. versionadded:: 0.0.14

    Attributes
    ----------
    id : str
        A unique identifier for the Study.
    name : str
        A human readable name of the Study, typically the title of the paper.
    authors : str
        A string of the authors of the paper.
    publication : str
        A string of the publication information for the paper, typically a journal name.
    metadata : dict
        A dictionary of metadata associated with the Study.
    analyses : :obj:`list` of :obj:`nimare.nimads.Analysis` objects
        The Analysis objects comprising the Study.
        An analysis represents a contrast with statistical results.
    """

    _TRACKED_ATTRS = frozenset({"metadata", "analyses"})
    _DICT_ATTRS = frozenset({"metadata"})

    def __init__(self, source):
        self._on_mutate = None
        self.id = source["id"]
        self.name = source.get("name", "")
        self.description = source.get("description", "")
        self.doi = source.get("doi", "")
        self.pmid = source.get("pmid", "")
        self.authors = source.get("authors", "")
        self.publication = source.get("publication", "")
        self.year = source.get("year", None)
        self.metadata = source.get("metadata", {}) or {}
        self.analyses = [Analysis(a, study=self) for a in source.get("analyses", [])]

    def __setattr__(self, name, value):  # noqa: D105
        cb = getattr(self, "_on_mutate", None)
        if cb is not None and name in self._DICT_ATTRS:
            if isinstance(value, dict) and not isinstance(value, _NotifyDict):
                value = _NotifyDict(value, cb)
        super().__setattr__(name, value)
        if cb is not None and name in self._TRACKED_ATTRS:
            cb()

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Study: {self.id}>")

    def __str__(self):
        """My Simple representation."""
        return str(" ".join([self.name, f"analyses: {len(self.analyses)}"]))

    def __getstate__(self):
        """Drop live mutation callbacks before pickling."""
        state = self.__dict__.copy()
        state["_on_mutate"] = None
        return state

    def get_analyses(self):
        """Collect Analyses from the Study.

        Notes
        -----
        What filters, if any, should we support in this method?
        """
        ...

    def to_dict(self):
        """Return a dictionary representation of the Study."""
        return {
            "id": self.id,
            "name": self.name,
            "authors": self.authors,
            "publication": self.publication,
            "metadata": self.metadata,
            "analyses": [a.to_dict() for a in self.analyses],
        }


class Analysis:
    """A single statistical contrast from a Study.

    .. versionadded:: 0.0.14

    Attributes
    ----------
    id : str
        A unique identifier for the Analysis.
    name : str
        A human readable name of the Analysis.
    conditions : list of Condition objects
        The Conditions in the Analysis.
    annotations : list of Annotation objects
        Any Annotations available for the Analysis.
        Each Annotation should come from the same Annotator.
    images : dict of Image objects
        A dictionary of type: Image pairs.
    points : list of Point objects
        Any significant Points from the Analysis.
    metadata: dict
        A dictionary of metadata associated with the Analysis.

    Notes
    -----
    Should the images attribute be a list instead, if the Images contain type information?
    """

    _TRACKED_ATTRS = frozenset(
        {"annotations", "metadata", "texts", "points", "images", "conditions"}
    )
    _DICT_ATTRS = frozenset({"annotations", "metadata", "texts"})

    def __init__(self, source, study=None):
        self._on_mutate = None
        self.id = source["id"]
        self.name = source["name"]
        conditions = source.get("conditions", []) or [{"name": "default", "description": ""}]
        weights = source.get("weights", []) or [1.0] * len(conditions)
        if len(weights) < len(conditions):
            weights = list(weights) + [1.0] * (len(conditions) - len(weights))
        self.conditions = [Condition(c, w) for c, w in zip(conditions, weights)]
        self.images = [Image(i) for i in source.get("images", [])]
        self.points = [Point(p) for p in source.get("points", [])]
        self.metadata = source.get("metadata", {}) or {}
        annotations = source.get("annotations", {}) or {}
        self.annotations = annotations if isinstance(annotations, dict) else {}
        texts = source.get("texts", {}) or {}
        self.texts = texts if isinstance(texts, dict) else {}
        self._study = weakref.proxy(study) if study else None

    def __setattr__(self, name, value):  # noqa: D105
        cb = getattr(self, "_on_mutate", None)
        if cb is not None and name in self._DICT_ATTRS:
            if isinstance(value, dict) and not isinstance(value, _NotifyDict):
                value = _NotifyDict(value, cb)
        super().__setattr__(name, value)
        if cb is not None and name in self._TRACKED_ATTRS:
            cb()

    def __getstate__(self):
        """Drop live mutation callbacks before pickling."""
        state = self.__dict__.copy()
        state["_on_mutate"] = None
        return state

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Analysis: {self.id}>")

    def __str__(self):
        """My Simple representation."""
        return str(
            " ".join([self.name, f"images: {len(self.images)}", f"points: {len(self.points)}"])
        )

    def get_metadata(self) -> "dict[str, any]":
        """Get combined metadata from both analysis and parent study.

        Returns
        -------
        dict[str, any]
            Combined metadata dictionary with analysis metadata taking precedence
            over study metadata for any overlapping keys.
        """
        if self._study is None:
            return self.metadata.copy()

        combined_metadata = self._study.metadata.copy()
        combined_metadata.update(self.metadata)
        return combined_metadata

    def to_dict(self):
        """Convert the Analysis to a dictionary."""
        analysis_dict = {
            "id": self.id,
            "name": self.name,
            "conditions": [
                {k: v for k, v in c.to_dict().items() if k in ["name", "description"]}
                for c in self.conditions
            ],
            "images": [i.to_dict() for i in self.images],
            "points": [p.to_dict() for p in self.points],
            "weights": [c.to_dict()["weight"] for c in self.conditions],
            "metadata": self.metadata,
        }
        if self.annotations:
            analysis_dict["annotations"] = self.annotations
        if self.texts:
            analysis_dict["texts"] = self.texts
        return analysis_dict


class Condition:
    """A condition within an Analysis.

    .. versionadded:: 0.0.14

    Attributes
    ----------
    name: str
        A human readable name of the Condition. Good examples are from cognitive atlas.
    description
        A human readable description of the Condition.
    weight
        The weight of the Condition in the Analysis.

    Notes
    -----
    Condition-level Annotations, like condition-wise trial counts, are stored in the parent
    Analysis's Annotations, preferably with names that make it clear that they correspond to a
    specific Condition.
    """

    def __init__(self, condition, weight):
        self.name = condition["name"]
        self.description = condition["description"]
        self.weight = weight

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Condition: {self.id}>")

    def to_dict(self):
        """Convert the Condition to a dictionary."""
        return {"name": self.name, "description": self.description, "weight": self.weight}


class Annotation:
    """A collection of labels and associated weights from the same Annotator.

    .. versionadded:: 0.0.14

    Attributes
    ----------
    term_weights : :obj:`pandas.DataFrame`
        A pandas DataFrame containing the annotation group's labels and weights.
        This is the main attribute of interest for NeuroStore.
        A dictionary could also work.

    Notes
    -----
    Where would p(term|topic) and p(voxel|topic) arrays/DataFrames go? Having one Annotation per
    Analysis (for each Annotator), and storing these arrays in the Annotation, would make for
    *a lot* of duplication.
    The same goes for metadata/provenance, but that will generally be much lighter on memory than
    the arrays.

    Could be a dictionary with analysis objects as keys?
    (need to define __hash__ and __eq__ for Analysis)
    Or could use Analysis.id as key.
    """

    def __init__(self, source, studyset):
        self.name = source["name"]
        self.id = source["id"]
        self._analysis_ref = {
            a.id: weakref.proxy(a) for study in studyset.studies for a in study.analyses
        }
        self.notes = [Note(self._analysis_ref[n["analysis"]], n["note"]) for n in source["notes"]]
        for note in self.notes:
            self._analysis_ref[note.analysis.id].annotations[self.id] = note.note

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Annotation: {self.id}>")

    def to_dict(self):
        """Convert the Annotation to a dictionary."""
        return {"name": self.name, "id": self.id, "notes": [note.to_dict() for note in self.notes]}


class Note:
    """A Note within an annotation.

    .. versionadded:: 0.0.14

    Attributes
    ----------
    analysis : Analysis object
        the analysis the note is associated with
    note : dict
        the attributes pertaining to the analysis
    """

    def __init__(self, analysis, note):
        self.analysis = analysis
        self.note = note

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Note: {self.id}>")

    def to_dict(self):
        """Convert the Note to a dictionary."""
        return {"analysis": self.analysis.id, "note": self.note}


class Image:
    """A single statistical map from an Analysis.

    .. versionadded:: 0.0.14

    Attributes
    ----------
    filename
    type?

    Notes
    -----
    Should we support remote paths, with some kind of fetching method?
    """

    def __init__(self, source):
        self.url = source["url"]
        self.filename = source["filename"]
        self.space = source["space"]
        self.value_type = source["value_type"]

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Image: {self.id}>")

    def to_dict(self):
        """Convert the Image to a dictionary."""
        return {
            "url": self.url,
            "filename": self.filename,
            "space": self.space,
            "value_type": self.value_type,
        }


class Point:
    """A single peak coordinate from an Analysis.

    .. versionadded:: 0.0.14

    Attributes
    ----------
    x : float
    y : float
    z : float
    space
    kind
    image
    point_values
    """

    def __init__(self, source):
        coordinates = source.get("coordinates", [None, None, None])
        self.space = source.get("space")
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        self.kind = source.get("kind")
        self.label_id = source.get("label_id")
        self.image = source.get("image")
        self.values = deepcopy(source.get("values", []) or [])

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Point: {self.id}>")

    def to_dict(self):
        """Convert the Point to a dictionary."""
        point_dict = {"space": self.space, "coordinates": [self.x, self.y, self.z]}
        if self.kind is not None:
            point_dict["kind"] = self.kind
        if self.label_id is not None:
            point_dict["label_id"] = self.label_id
        if self.image is not None:
            point_dict["image"] = self.image
        if self.values:
            point_dict["values"] = deepcopy(self.values)
        return point_dict
