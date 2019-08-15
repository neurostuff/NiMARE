"""
Base classes for datasets.
"""
import logging
from abc import abstractmethod

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
    _required_inputs = {}

    def __init__(*args, **kwargs):
        pass

    def _validate_input(self, dataset):
        if not hasattr(dataset, 'slice'):
            raise ValueError('Argument "dataset" must be a valid Dataset '
                             'object, not a {0}'.format(type(dataset)))

        if self._required_inputs:
            data = dataset.get(self._required_inputs)
            self.inputs_ = {}
            for k, v in data.items():
                if not v:
                    raise ValueError(
                        "Estimator {0} requires input dataset to contain {1}, but "
                        "none were found.".format(self.__class__.__name__, k))
                self.inputs_[k] = v

    def _preprocess_input(self, dataset):
        '''
        Perform any additional preprocessing steps on data in self.input_
        '''
        pass

    def fit(self, dataset):
        """
        Fit Estimator to Dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset object to analyze.

        Returns
        -------
        :obj:`nimare.base.MetaResult`
            Results of Estimator fitting.
        """
        self._validate_input(dataset)
        self._preprocess_input(dataset)
        maps = self._fit(dataset)

        if hasattr(self, 'masker') and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        self.results = MetaResult(self, masker, maps)
        return self.results

    @abstractmethod
    def _fit(self, dataset):
        """Apply estimation to dataset and output results. Must return a
        dictionary of results, where keys are names of images and values are
        ndarrays.
        """
        pass
