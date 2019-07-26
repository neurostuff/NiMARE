"""
Base classes for datasets.
"""
import logging
from abc import ABCMeta, abstractmethod

from six import with_metaclass

from .base import NiMAREBase, MetaResult

LGR = logging.getLogger(__name__)


class Transformer(NiMAREBase):
    """Transformers take in Datasets and return Datasets

    Initialize with hyperparameters.
    """
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, dataset):
        """Add stuff to transformer.
        """
        if not hasattr(dataset, 'slice'):
            raise ValueError('Argument "dataset" must be a valid Dataset '
                             'object, not a {0}'.format(type(dataset)))


class Estimator(NiMAREBase):
    """Estimators take in Datasets and return MetaResults
    """

    # Inputs that must be available in input Dataset. Keys are names of
    # attributes to set; values are strings indicating location in Dataset.
    _inputs = {}

    def _validate_input(self, dataset):
        if not hasattr(dataset, 'slice'):
            raise ValueError('Argument "dataset" must be a valid Dataset '
                             'object, not a {0}'.format(type(dataset)))
        for k, v in self._inputs.items():
            data = dataset.get(v[0], **v[1])
            if not data:
                raise ValueError("Estimator {0} requires input dataset to "
                                 "contain {1} with filters {2}, but none "
                                 "were found.".format(self.__class__.__name__,
                                                      v[0], v[1]))
            setattr(self, k, data)

    def fit(self, dataset):
        self._validate_input(dataset)
        maps = self._fit(dataset)
        self.results = MetaResult(self, dataset.mask, maps)

    @abstractmethod
    def _fit(self, dataset):
        """Apply estimation to dataset and output results. Must return a
        dictionary of results, where keys are names of images and values are
        ndarrays.
        """
        pass
