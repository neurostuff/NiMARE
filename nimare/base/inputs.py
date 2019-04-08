"""
Input classes for nimare data.
"""


class Analyzable(object):
    def to_array(self):
        pass


class Mappable(Analyzable):
    def to_vol(self):
        pass


class ConnMatrix(Analyzable):
    """Container for connectome data (i.e., connectivity matrices).
    """
    def __init__(self, mat):
        pass

    def to_array(self):
        pass


class Image(Mappable):
    """Container for volumetric brain images.
    """
    def __init__(self, nimg):
        pass

    def to_array(self, masker):
        pass

    def to_vol(self):
        pass


class CoordinateSet(Mappable):
    """Container for peak information, with optional additional metadata (e.g.,
    intensity values).
    """
    def __init__(self, foci):
        pass

    def to_array(self, method, masker):
        pass

    def to_vol(self, method, masker):
        pass


class Surface(Mappable):
    """Container for surface brain data (i.e., from gifti files).
    """
    def __init__(self, gimg):
        pass

    def to_array(self, masker):
        pass

    def to_vol(self, masker):
        pass
