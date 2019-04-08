"""
Base classes for datasets.
"""
from abc import ABCMeta, abstractmethod

from six import with_metaclass

from .inputs import ConnMatrix, Image, CoordinateSet, Surface
from ..utils import listify


class Study(object):
    ''' Represents a single published or unpublished study--including one or
    more contrasts, and study-level metadata. '''

    def __init__(self, contrasts=None, **kwargs):
        self.contrasts = contrasts or []

    def add_contrasts(self, contrasts):
        self.contrasts.extend(listify(contrasts))

    @classmethod
    def merge(cls, studies):
        ''' Harmonizes/merges Contrasts extracted from different sources, based
        on common indexes (e.g., DOIs / table numbers, etc.).
        '''
        pass


class Contrast(object):
    """Container for contrasts (aka experiments or comparisons) nested within
    studies.
    Should store an arbitrary number of ConnMatrices, Images, and Surfaces,
    along with at most one CoordinateSet.
    """

    def __init__(self, images=None, conn_matrices=None, coordinates=None,
                 surfaces=None):
        # Add validation method instead of doing this here. That way it can be
        # applied to other objects. Should check things like name, min/max,

        self.images = []
        self.coordinates = []
        self.connectomes = []
        self.surfaces = []

        if images:
            self.add_images(images)

        if not isinstance(coordinates, CoordinateSet) and coordinates is not None:
            raise ValueError('Input coordinates must be CoordinateSet or None')
        else:
            self.coordinates = coordinates

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

    def add_images(self, images):
        ''' Add one or more images to the current list. '''
        for image in listify(images):
            if not isinstance(image, Image) and image is not None:
                raise ValueError('All images inputs must be nimare Images.')
            elif image.type in self.images.keys():
                self.images[image.type] = image

    @classmethod
    def merge(cls, contrasts):
        ''' Harmonizes/merges Contrasts extracted from different sources, based
        on common indexes (e.g., DOIs / table numbers, etc.).
        '''
        pass


class DataSource(with_metaclass(ABCMeta)):
    ''' Base class for DataSource hierarchy. '''

    @abstractmethod
    def get_data(self, level='contrast', tags=None, dois=None, **kwargs):
        pass
