"""NIMADS-related classes for NiMARE."""

import json
import weakref
from copy import deepcopy

import numpy as np
from nilearn._utils import load_niimg

from nimare.io import convert_nimads_to_dataset
from nimare.utils import mm2vox


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

    def __init__(self, source, target_space=None, mask=None, annotations=None):
        # load source as json
        if isinstance(source, str):
            with open(source, "r+") as f:
                source = json.load(f)

        self.id = source["id"]
        self.name = source["name"] or ""
        self.studies = [Study(s) for s in source["studies"]]
        self._annotations = []
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

    @annotations.setter
    def annotations(self, annotation):
        if isinstance(annotation, dict):
            loaded_annotation = Annotation(annotation, self)
        elif isinstance(annotation, str):
            with open(annotation, "r+") as f:
                loaded_annotation = Annotation(json.load(f), self)
        elif isinstance(annotation, Annotation):
            loaded_annotation = annotation
        self._annotations.append(loaded_annotation)

    @annotations.deleter
    def annotations(self, annotation_id=None):
        if annotation_id:
            self._annotations = [a for a in self._annotations if a.id != annotation_id]
        else:
            self._annotations = []

    @classmethod
    def from_nimads(cls, filename):
        """Create a Studyset from a NIMADS JSON file."""
        with open(filename, "r+") as fn:
            nimads = json.load(fn)

        return cls(nimads)

    def combine_analyses(self):
        """Combine analyses in Studyset."""
        studyset = self.copy()
        for study in studyset.studies:
            if len(study.analyses) > 1:
                source_lst = [analysis.to_dict() for analysis in study.analyses]
                ids, names, conditions, images, points, weights, metadata = [
                    [source[key] for source in source_lst] for key in source_lst[0]
                ]

                new_source = {
                    "id": "_".join(ids),
                    "name": "; ".join(names),
                    "conditions": [cond for c_list in conditions for cond in c_list],
                    "images": [image for i_list in images for image in i_list],
                    "points": [point for p_list in points for point in p_list],
                    "weights": [weight for w_list in weights for weight in w_list],
                    "metadata": {k: v for m_dict in metadata for k, v in m_dict.items()},
                }
                study.analyses = [Analysis(new_source)]

        return studyset

    def to_nimads(self, filename):
        """Write the Studyset to a NIMADS JSON file."""
        with open(filename, "w+") as fn:
            json.dump(self.to_dict(), fn)

    def to_dict(self):
        """Return a dictionary representation of the Studyset."""
        return {
            "id": self.id,
            "name": self.name,
            "studies": [s.to_dict() for s in self.studies],
        }

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
        """Create a new Studyset with only requested Analyses."""
        studyset_dict = self.to_dict()
        annotations = [annot.to_dict() for annot in self.annotations]

        for study in studyset_dict["studies"]:
            study["analyses"] = [a for a in study["analyses"] if a["id"] in analyses]

        studyset = self.__class__(source=studyset_dict)

        for annot in annotations:
            annot["notes"] = [n for n in annot["notes"] if n["analysis"] in analyses]
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

        # Extract all points from all analyses
        all_points = []
        analysis_ids = []
        for study in self.studies:
            for analysis in study.analyses:
                for point in analysis.points:
                    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                        all_points.append([point.x, point.y, point.z])
                        analysis_ids.append(analysis.id)

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
        mask = load_niimg(img)

        # Extract all points from all analyses
        all_points = []
        analysis_ids = []
        for study in self.studies:
            for analysis in study.analyses:
                for point in analysis.points:
                    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                        all_points.append([point.x, point.y, point.z])
                        analysis_ids.append(analysis.id)

        if not all_points:  # Return empty list if no coordinates found
            return []

        # Convert to voxel coordinates
        all_points = np.array(all_points)
        ijk = mm2vox(all_points, mask.affine)

        # Get mask coordinates
        mask_data = mask.get_fdata()
        mask_coords = np.vstack(np.where(mask_data)).T

        # Check for presence of coordinates in mask
        in_mask = np.any(np.all(ijk[:, None] == mask_coords[None, :], axis=-1), axis=-1)

        # Get unique analysis IDs where points are in mask
        found_analyses = set(np.array(analysis_ids)[in_mask])

        return list(found_analyses)

    def get_analyses_by_annotations(self, key, value=None):
        """Extract a list of Analyses with a given label/annotation."""
        annotations = {}
        for study in self.studies:
            for analysis in study.analyses:
                a_annot = analysis.annotations
                if key in a_annot and (value is None or a_annot[key] == value):
                    annotations[analysis.id] = {key: a_annot[key]}
        return annotations

    def get_analyses_by_metadata(self, key, value=None):
        """Extract a list of Analyses with a metadata field/value."""
        metadata = {}
        for study in self.studies:
            for analysis in study.analyses:
                a_metadata = analysis.metadata
                if key in a_metadata and (value is None or a_metadata[key] == value):
                    metadata[analysis.id] = {key: a_metadata[key]}
        return metadata

    def get_points(self, analyses):
        """Collect Points associated with specified Analyses."""
        points = {}
        for study in self.studies:
            for analysis in study.analyses:
                if analysis.id in analyses:
                    points[analysis.id] = analysis.points
        return points

    def get_annotations(self, analyses):
        """Collect Annotations associated with specified Analyses."""
        annotations = {}
        for study in self.studies:
            for analysis in study.analyses:
                if analysis.id in analyses:
                    annotations[analysis.id] = analysis.annotations

        return annotations

    def get_texts(self, analyses):
        """Collect texts associated with specified Analyses."""
        raise NotImplementedError("Getting texts is not yet supported.")

    def get_images(self, analyses):
        """Collect image files associated with specified Analyses."""
        images = {}
        for study in self.studies:
            for analysis in study.analyses:
                if analysis.id in analyses:
                    images[analysis.id] = analysis.images
        return images

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
        metadata = {}
        for study in self.studies:
            for analysis in study.analyses:
                if analysis.id in analyses:
                    metadata[analysis.id] = analysis.get_metadata()
        return metadata


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
        self.name = source["name"] or ""
        self.authors = source["authors"] or ""
        self.publication = source["publication"] or ""
        self.metadata = source.get("metadata", {}) or {}
        self.analyses = [Analysis(a, study=self) for a in source["analyses"]]

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
        self.conditions = [
            Condition(c, w) for c, w in zip(source["conditions"], source["weights"])
        ]
        self.images = [Image(i) for i in source["images"]]
        self.points = [Point(p) for p in source["points"]]
        self.metadata = source.get("metadata", {}) or {}
        self.annotations = {}
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
        return {
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
