"""NIMADS-related classes for NiMARE."""

import json
import weakref
from copy import deepcopy

import numpy as np
import pandas as pd
from nilearn.image import load_img

from nimare.exceptions import InvalidStudysetError
from nimare.io import convert_nimads_to_dataset
from nimare.utils import _mask_img_to_bool, get_masker, mm2vox


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

    def __init__(self, source, target=None, mask=None, annotations=None):
        # load source as json
        if isinstance(source, str):
            with open(source, "r+") as f:
                source = json.load(f)

        _validate_studyset_source(source)

        self.id = source["id"]
        self.name = source.get("name", "") or ""
        self.studies = [Study(s) for s in source["studies"]]
        self._annotations = []
        self._nimare_space = None
        self._nimare_masker = None
        self._nimare_basepath = None
        self._nimare_table_cache = None
        self._set_execution_context(space=target, masker=mask)
        for annotation in source.get("annotations", []):
            self.annotations = annotation
        if annotations:
            self.annotations = annotations

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Studyset: {self.id}>")

    def __str__(self):
        """Give useful information about the Studyset."""
        return str(" ".join(["Studyset:", self.name, "::", f"studies: {len(self.studies)}"]))

    @property
    def annotations(self):
        """Return existing Annotations."""
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
        if isinstance(annotations, (list, tuple)):
            loaded_annotations = [
                self._coerce_annotation(annotation) for annotation in annotations
            ]
        else:
            loaded_annotations = [self._coerce_annotation(annotations)]

        self._annotations.extend(loaded_annotations)
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

    def touch(self):
        """Invalidate Studyset-derived caches after in-place mutation."""
        self._nimare_table_cache = None

    def view(self, materialize_ids=True):
        """Return a Dataset-like tabular view of the Studyset."""
        from nimare.studyset import StudysetView

        return StudysetView(self, materialize_ids=materialize_ids)

    @property
    def space(self):
        """Execution-space label used for Dataset-like Studyset views."""
        view = self.view(materialize_ids=False)
        return view.space

    @property
    def masker(self):
        """Masker used for Dataset-like Studyset views."""
        view = self.view(materialize_ids=False)
        return view.masker

    @property
    def basepath(self):
        """Base path used for image resolution in Dataset-like Studyset views."""
        return self._get_execution_context()["basepath"]

    @property
    def ids(self):
        """numpy.ndarray: 1D array of full analysis identifiers."""
        return self.view().ids

    @property
    def coordinates(self):
        """pandas.DataFrame: Dataset-like coordinates table."""
        return self.view().coordinates

    @property
    def images(self):
        """pandas.DataFrame: Dataset-like images table."""
        return self.view().images

    @property
    def metadata(self):
        """pandas.DataFrame: Dataset-like metadata table."""
        return self.view().metadata

    @property
    def texts(self):
        """pandas.DataFrame: Dataset-like texts table."""
        return self.view().texts

    @property
    def annotations_df(self):
        """pandas.DataFrame: Flattened analysis-level annotations table."""
        return self.view().annotations

    @annotations_df.setter
    def annotations_df(self, annotations_df):
        self.set_annotations_df(annotations_df)

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

    def get(self, dict_, drop_invalid=True):
        """Retrieve files and/or metadata from the current Studyset."""
        return self.view().get(dict_, drop_invalid=drop_invalid)

    def get_labels(self, ids=None):
        """Extract labels present in the Studyset's analysis-level annotations."""
        return self.view().get_labels(ids=ids)

    def get_studies_by_label(self, labels=None, label_threshold=0.001):
        """Extract full analysis IDs whose annotation values exceed the threshold."""
        return self.view().get_studies_by_label(
            labels=labels,
            label_threshold=label_threshold,
        )

    def get_studies_by_mask(self, mask):
        """Extract full analysis IDs with at least one focus inside ``mask``."""
        return self.view().get_studies_by_mask(mask)

    def get_studies_by_coordinate(self, xyz, r=20):
        """Extract full analysis IDs with at least one focus near the requested coordinates."""
        return self.view().get_studies_by_coordinate(xyz, r=r)

    def _set_execution_context(self, *, space=None, masker=None, basepath=None):
        """Update cached execution context used by StudysetView."""
        if space is not None:
            self._nimare_space = space
        if masker is not None:
            self._nimare_masker = get_masker(masker)
        if basepath is not None:
            self._nimare_basepath = basepath

    def _get_execution_context(self):
        """Return cached execution context used by Studyset-backed estimators."""
        return {
            "space": self._nimare_space,
            "masker": self._nimare_masker,
            "basepath": self._nimare_basepath,
            "table_cache": self._nimare_table_cache,
        }

    def _set_table_cache(self, table_cache):
        """Store Dataset-compatible cached tables for reuse by StudysetView."""
        self._nimare_table_cache = table_cache

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
        from nimare.studyset import _snapshot_dataset_tables

        self._set_execution_context(
            space=dataset.space,
            masker=dataset.masker,
            basepath=dataset.basepath,
        )
        self._set_table_cache(_snapshot_dataset_tables(dataset, copy_tables=True))

    @classmethod
    def from_nimads(cls, filename):
        """Create a Studyset from a NIMADS JSON file."""
        with open(filename, "r+") as fn:
            nimads = json.load(fn)

        return cls(nimads)

    @classmethod
    def from_dataset(cls, dataset):
        """Create a Studyset from a NiMARE Dataset."""
        from nimare.io import convert_dataset_to_nimads_dict

        nimads = convert_dataset_to_nimads_dict(dataset)
        studyset = cls(nimads)
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

        return studyset

    def to_nimads(self, filename):
        """Write the Studyset to a NIMADS JSON file."""
        with open(filename, "w+") as fn:
            json.dump(self.to_dict(), fn)

    def to_dict(self):
        """Return a dictionary representation of the Studyset."""
        studyset_dict = {
            "id": self.id,
            "name": self.name,
            "studies": [s.to_dict() for s in self.studies],
        }
        if self.annotations:
            studyset_dict["annotations"] = [
                annotation.to_dict() for annotation in self.annotations
            ]
        return studyset_dict

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

        # Update current instance with loaded data
        self.id = loaded_data.id
        self.name = loaded_data.name
        self.studies = loaded_data.studies
        self._annotations = loaded_data._annotations
        self._nimare_table_cache = None
        self.touch()
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

    def copy(self):
        """Create a copy of the Studyset."""
        return deepcopy(self)

    def slice(self, analyses):
        """Create a new Studyset with only requested analyses.

        Parameters
        ----------
        analyses : :obj:`list` of :obj:`str` or :obj:`str`
            Requested analysis IDs.
        """
        if isinstance(analyses, str):
            analyses = [analyses]

        requested_ids = {str(analysis) for analysis in analyses}
        studyset_dict = self.to_dict()
        annotations = [annot.to_dict() for annot in self.annotations]
        studyset_dict.pop("annotations", None)
        retained_analysis_ids = set()

        for study in studyset_dict["studies"]:
            kept_analyses = []
            for analysis in study["analyses"]:
                if analysis["id"] in requested_ids:
                    kept_analyses.append(analysis)
                    retained_analysis_ids.add(analysis["id"])
            study["analyses"] = kept_analyses

        studyset = self.__class__(source=studyset_dict)
        context = self._get_execution_context()
        studyset._set_execution_context(
            space=context["space"],
            masker=context["masker"],
            basepath=context["basepath"],
        )

        for annot in annotations:
            annot["notes"] = [n for n in annot["notes"] if n["analysis"] in retained_analysis_ids]
            studyset.annotations = annot

        return studyset

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

                # Keep metadata from left unless missing
                left_study["metadata"].update(
                    {
                        k: v
                        for k, v in right_study["metadata"].items()
                        if k not in left_study["metadata"]
                    }
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
        merged = self.__class__(source=merged_source)

        # Merge annotations, preferring left's annotations for conflicts
        existing_annot_ids = {a.id for a in self.annotations}
        for right_annot in right.annotations:
            if right_annot.id not in existing_annot_ids:
                merged.annotations = right_annot.to_dict()

        return merged

    def get_analyses_by_coordinates(self, xyz, r=None, n=None):
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

    def get_texts(self, analyses):
        """Collect texts associated with specified Analyses."""
        raise NotImplementedError("Getting texts is not yet supported.")

    def get_images(self, analyses):
        """Collect image files associated with specified Analyses."""
        return {
            analysis.id: analysis.images for analysis in self._iter_selected_analyses(analyses)
        }

    def get_metadata(self, analyses):
        """Collect metadata associated with specified Analyses.

        Parameters
        ----------
        analyses : list of str
            List of Analysis IDs to get metadata for.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping Analysis IDs to their combined metadata (including study metadata).
        """
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

    def __init__(self, source):
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

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Study: {self.id}>")

    def __str__(self):
        """My Simple representation."""
        return str(" ".join([self.name, f"analyses: {len(self.analyses)}"]))

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

    def __init__(self, source, study=None):
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
        self.space = source["space"]
        self.x = source["coordinates"][0]
        self.y = source["coordinates"][1]
        self.z = source["coordinates"][2]

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Point: {self.id}>")

    def to_dict(self):
        """Convert the Point to a dictionary."""
        return {"space": self.space, "coordinates": [self.x, self.y, self.z]}
