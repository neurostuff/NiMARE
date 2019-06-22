"""
Base classes for datasets.
"""
import logging
from abc import ABCMeta, abstractmethod

from six import with_metaclass

from .base import NiMAREBase
from ..dataset.dataset import Dataset

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
        if not isinstance(dataset, Dataset):
            raise ValueError('Argument "dataset" must be a valid Dataset '
                             'object, not a {0}'.format(type(dataset)))


class Estimator(NiMAREBase):
    """Estimators take in Datasets and return MetaResults
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, dataset):
        """Apply estimation to dataset and output results.
        """
        if not isinstance(dataset, Dataset):
            raise ValueError('Argument "dataset" must be a valid Dataset '
                             'object, not a {0}'.format(type(dataset)))


class Result(with_metaclass(ABCMeta)):
    def __init__(self):
        pass
