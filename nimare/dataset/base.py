# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base classes for datasets.
"""
from ..base import ConnMatrix, Image, ActivationSet, Surface


class Contrast(object):
    """Container for contrasts (aka experiments or comparisons) nested within
    studies.
    Should store an arbitrary number of ConnMatrices, Images, and Surfaces,
    along with at most one ActivationSet.
    """
    def __init__(self, images=None, conn_matrices=None, activations=None,
                 surfaces=None):
        # Add validation method instead of doing this here. That way it can be applied to other objects.
        # Should check things like name, min/max,
        self.images = {'z': None,
                       'p': None,
                       'beta': None,
                       'se': None}
        self.surfaces = {'z': None,
                         'p': None,
                         'beta': None,
                         'se': None}
        self.connectomes = {}

        if not isinstance(images, list):
            images = [images]

        for image in images:
            if not isinstance(image, Image) and image is not None:
                raise ValueError('All images inputs must be nimare Images.')
            elif image.type in self.images.keys():
                self.images[image.type] = image

        if not isinstance(activations, ActivationSet) and activations is not None:
            raise ValueError('Input activations must be ActivationSet or None')
        else:
            self.coordinates = activations

        if not isinstance(conn_matrices, list):
            conn_matrices = [conn_matrices]

        for conn_mat in conn_matrices:
            if not isinstance(conn_mat, ConnMatrix) and conn_mat is not None:
                raise ValueError('All conn_matrices inputs must be nimare ConnMatrices.')
            elif isinstance(conn_mat, ConnMatrix):
                self.connectomes[conn_mat.type] = conn_mat

        for surf in surfaces:
            if not isinstance(surf, Surface) and surf is not None:
                raise ValueError('All conn_matrices inputs must be nimare ConnMatrices.')
            elif surf.type in self.surfaces.keys():
                self.surfaces[surf.type] = surf
