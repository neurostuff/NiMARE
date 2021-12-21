"""NIMADS-related classes for NiMARE."""


class Studyset:
    """A collection of studies for meta-analysis.

    This is the primary target for Estimators and Transformers in NiMARE.
    """

    def __init__(self):
        ...

    def to_nimads(self, filename):
        """Write the Studyset to a NIMADS JSON file."""
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
        """Point to a new location for image files on the filesystem."""
        ...

    def get_analyses_by_coordinates(self):
        """Extract a list of Analyses with at least one focus near the requested coordinates."""
        ...

    def get_analyses_by_mask(self):
        """Extract a list of Analyses with at least one focus in the specified mask."""
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

    def get_coordinates(self, analyses):
        """Collect coordinates associated with specified Analyses."""
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
    """A collection of Analyses from the same paper."""

    def __init__(self):
        ...

    def get_analyses(self):
        """Collect Analyses from the Study."""
        ...


class Analysis:
    ...


class Condition:
    ...


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
    """

    def __init__(self):
        ...


class Image:
    """A single statistical map from an Analysis.

    Attributes
    ----------
    filename
    """

    def __init__(self):
        ...


class Point:
    """A single peak coordinate from an Analysis.

    Attributes
    ----------
    coordinates : 3-tuple
    point_values
    """

    def __init__(self):
        ...
