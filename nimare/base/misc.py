"""
Miscellaneous base classes.
"""
from abc import abstractmethod, ABCMeta

from .base import Estimator


class Parcellator(Estimator):
    """
    Base class for meta-analytic parcellation methods.
    """
    pass


class Corrector(metaclass=ABCMeta):
    '''
    Base class for multiple comparison correction methods.
    '''
    _correction_method = None

    def transform(self, result, **kwargs):
        est = result.estimator
        method = self._correction_method
        if (self._correction_method is not None and hasattr(est, method)):
            getattr(est, method)(result, **kwargs)
        else:
            self._transform(result, **kwargs)

    @abstractmethod
    def _transform(self, result, **kwargs):
        pass
