"""NIMADS-related classes for NiMARE."""

import json
import weakref
from copy import deepcopy

from nimare.io import convert_nimads_to_dataset


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
        if annotations:
            self.annotations = annotations
        else:
            self._annotations = []

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
                ids, names, conditions, images, points, weights = [
                    [source[key] for source in source_lst] for key in source_lst[0]
                ]

                new_source = {
                    "id": "_".join(ids),
                    "name": "; ".join(names),
                    "conditions": [cond for c_list in conditions for cond in c_list],
                    "images": [image for i_list in images for image in i_list],
                    "points": [point for p_list in points for point in p_list],
                    "weights": [weight for w_list in weights for weight in w_list],
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
        """Load a Studyset from a pickled file."""
        raise NotImplementedError("Loading from pickled files is not yet supported.")

    def save(self, filename):
        """Write the Studyset to a pickled file."""
        raise NotImplementedError("Saving to pickled files is not yet supported.")

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
            studyset.annotation = annot

        return studyset

    def merge(self, right):
        """Merge a separate Studyset into the current one."""
        raise NotImplementedError("Merging Studysets is not yet supported.")

    def update_image_path(self, new_path):
        """Point to a new location for image files on the local filesystem."""
        raise NotImplementedError("Updating image paths is not yet supported.")

    def get_analyses_by_coordinates(self, xyz, r=None, n=None):
        """Extract a list of Analyses with at least one Point near the requested coordinates."""
        raise NotImplementedError("Getting analyses by coordinates is not yet supported.")

    def get_analyses_by_mask(self, img):
        """Extract a list of Analyses with at least one Point in the specified mask."""
        raise NotImplementedError("Getting analyses by mask is not yet supported.")

    def get_analyses_by_annotations(self):
        """Extract a list of Analyses with a given label/annotation."""
        raise NotImplementedError("Getting analyses by annotations is not yet supported.")

    def get_analyses_by_texts(self):
        """Extract a list of Analyses with a given text."""
        raise NotImplementedError("Getting analyses by texts is not yet supported.")

    def get_analyses_by_images(self):
        """Extract a list of Analyses with a given image."""
        raise NotImplementedError("Getting analyses by images is not yet supported.")

    def get_analyses_by_metadata(self):
        """Extract a list of Analyses with a metadata field/value."""
        raise NotImplementedError("Getting analyses by metadata is not yet supported.")

    def get_points(self, analyses):
        """Collect Points associated with specified Analyses."""
        raise NotImplementedError("Getting points is not yet supported.")

    def get_annotations(self, analyses):
        """Collect Annotations associated with specified Analyses."""
        raise NotImplementedError("Getting annotations is not yet supported.")

    def get_texts(self, analyses):
        """Collect texts associated with specified Analyses."""
        raise NotImplementedError("Getting texts is not yet supported.")

    def get_images(self, analyses):
        """Collect image files associated with specified Analyses."""
        raise NotImplementedError("Getting images is not yet supported.")

    def get_metadata(self, analyses):
        """Collect metadata associated with specified Analyses."""
        raise NotImplementedError("Getting metadata is not yet supported.")


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
        self.analyses = [Analysis(a) for a in source["analyses"]]

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

    Notes
    -----
    Should the images attribute be a list instead, if the Images contain type information?
    """

    def __init__(self, source):
        self.id = source["id"]
        self.name = source["name"]
        self.conditions = [
            Condition(c, w) for c, w in zip(source["conditions"], source["weights"])
        ]
        self.images = [Image(i) for i in source["images"]]
        self.points = [Point(p) for p in source["points"]]
        self.annotations = {}

    def __repr__(self):
        """My Simple representation."""
        return repr(f"<Analysis: {self.id}>")

    def __str__(self):
        """My Simple representation."""
        return str(
            " ".join([self.name, f"images: {len(self.images)}", f"points: {len(self.points)}"])
        )

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
