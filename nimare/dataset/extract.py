"""
Classes and functions for data retrieval.
"""

from abc import ABCMeta, abstractmethod
from six import with_metaclass


__all__ = ['NeuroVaultDataSource', 'NeurosynthDataSource',
           'BrainSpellDataSource']


class DataSource(with_metaclass(ABCMeta)):
    ''' Base class for DataSource hierarchy. '''

    @abstractmethod
    def get_data(self, level='contrast', tags=None, dois=None, **kwargs):
        pass


class NeuroVaultDataSource(DataSource):
    ''' Interface with NeuroVault data. '''

    def get_data(self, **kwargs):
        pass

    def _get_collections(self):
        pass

    def _get_images(self):
        pass


class NeurosynthDataSource(DataSource):
    ''' Interface with Neurosynth data. '''
    pass

    def get_data(self, **kwargs):
        pass


class BrainSpellDataSource(DataSource):
    ''' Interface with BrainSpell data. '''
    pass

    def get_data(self, **kwargs):
        pass
