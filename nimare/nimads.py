"""NIMADS-related classes for NiMARE."""
import json


class Studyset:
    """A collection of studies for meta-analysis.

    This is the primary target for Estimators and Transformers in NiMARE.

    Attributes
    ----------
    studies : list of Study objects
        The Study objects comprising the Studyset.
    """

    def __init__(self, source, target_space=None, mask=None):
        self.id = source["id"]
        self.name = source["name"] or ""
        self.studies = [Study(s) for s in source["studies"]]
        self._annotations = []

    def __repr__(self):
        """My Simple representation."""
        return repr("Studyset: " + self.id)

    def __str__(self):
        """Give useful information about the Studyset."""
        return str(" ".join(["Studyset:", self.name, "::", f"studies: {len(self.studies)}"]))

    @property
    def annotations(self):
        """Return existing Annotations."""
        return self._annotations

    @annotations.setter
    def annotations(self, annotation):
        # some logic to compare ids
        # ss_analysis_ids = set([a.id for s in self.studies for a in s.analyses])
        # annot_analysis_ids = set([n['analysis'] for n in annotation.notes])
        self._annotations.append(annotation)

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

    def to_nimads(self, filename):
        """Write the Studyset to a NIMADS JSON file."""
        ...

    def load(self, filename):
        """Load a Studyset from a pickled file."""
        ...

    def save(self, filename):
        """Write the Studyset to a pickled file."""
        ...

    def copy(self):
        """Create a copy of the Studyset."""
        ...

    def slice(self, analyses):
        """Create a new Studyset with only requested Analyses."""
        ...

    def merge(self, right):
        """Merge a separate Studyset into the current one."""
        ...

    def update_image_path(self, new_path):
        """Point to a new location for image files on the local filesystem."""
        ...

    def get_analyses_by_coordinates(self, xyz, r=None, n=None):
        """Extract a list of Analyses with at least one Point near the requested coordinates."""
        ...

    def get_analyses_by_mask(self, img):
        """Extract a list of Analyses with at least one Point in the specified mask."""
        ...

    def get_analyses_by_annotations(self):
        """Extract a list of Analyses with a given label/annotation."""
        ...

    def get_analyses_by_texts(self):
        """Extract a list of Analyses with a given text."""
        ...

    def get_analyses_by_images(self):
        """Extract a list of Analyses with a given image."""
        ...

    def get_analyses_by_metadata(self):
        """Extract a list of Analyses with a metadata field/value."""
        ...

    def get_points(self, analyses):
        """Collect Points associated with specified Analyses."""
        ...

    def get_annotations(self, analyses):
        """Collect Annotations associated with specified Analyses."""
        ...

    def get_texts(self, analyses):
        """Collect texts associated with specified Analyses."""
        ...

    def get_images(self, analyses):
        """Collect image files associated with specified Analyses."""
        ...

    def get_metadata(self, analyses):
        """Collect metadata associated with specified Analyses."""
        ...


class Study:
    """A collection of Analyses from the same paper.

    Attributes
    ----------
    id : str
        A unique identifier for the Study.
    analyses : list of Analysis objects
        The Analysis objects comprising the Study.
    """

    def __init__(self, source):
        self.id = source["id"]
        self.name = source["name"] or ""
        self.authors = source["authors"] or ""
        self.publication = source["publication"] or ""
        self.metadata = source.get("metadata", {})
        self.analyses = [Analysis(a) for a in source["analyses"]]

    def __repr__(self):
        """My Simple representation."""
        return repr(self.id)

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


class Analysis:
    """A single statistical analyses from a Study.

    Attributes
    ----------
    id : str
        A unique identifier for the Analysis.
    conditions : list of Condition objects
        The Conditions in the Analysis.
    annotations : list of Annotation objects
        Any Annotations available for the Analysis.
        Each Annotation should come from the same Annotator.
    texts : dict
        A dictionary of source: text pairs.
    images : dict of Image objects
        A dictionary of type: Image pairs.
    points : list of Point objects
        Any significant Points from the Analysis.

    Notes
    -----
    Should the images attribute be a list instead, if the Images contain type information?

    Should the conditions be linked to the annotations, images, and points at all?
    """

    def __init__(self, source):
        self.id = source["id"]
        self.name = source["name"]
        self.conditions = [
            Condition(c, w) for c, w in zip(source["conditions"], source["weights"])
        ]
        self.images = [Image(i) for i in source["images"]]
        self.points = [Point(p) for p in source["points"]]

    def __repr__(self):
        """My Simple representation."""
        return repr(self.id)

    def __str__(self):
        """My Simple representation."""
        return str(
            " ".join([self.name, f"images: {len(self.images)}", f"points: {len(self.points)}"])
        )


class Condition:
    """A condition within an Analysis.

    Attributes
    ----------
    name
    description
    weight

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


class Annotation:
    """A collection of labels and associated weights from the same Annotator.

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

    def __init__(self, source):
        self.name = source["name"]
        self.id = source["id"]
        self.notes = source["notes"]


class Note:
    """A Note within an annotation.

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


class Image:
    """A single statistical map from an Analysis.

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


class Point:
    """A single peak coordinate from an Analysis.

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
