# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import json


# Move these to "Inputs" module
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


def ActivationSet(Mappable):
    """Container for peak information, with optional additional metadata (e.g.,
    intensity values).
    """
    def __init__(self, foci):
        pass

    def to_array(self, method, masker):
        pass

    def to_vol(self, method, masker):
        pass


def Surface(Mappable):
    """Container for surface brain data (i.e., from gifti files).
    """
    def __init__(self, gimg):
        pass

    def to_array(self, masker):
        pass

    def to_vol(self, masker):
        pass
