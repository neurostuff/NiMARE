"""
Multiple comparisons correction
"""
import warnings
import inspect
from abc import ABCMeta, abstractmethod, abstractproperty

from statsmodels.sandbox.stats.multicomp import multipletests

from ..base import MetaResult

__all__ = [
    'FDRCorrector'
]

warnings.simplefilter('default')


class Corrector(metaclass=ABCMeta):
    '''
    Base class for multiple comparison correction methods.
    '''

    # The name of the method that must be implemented in an Estimator class
    # in order to override the default correction method.
    _correction_method = None

    # Maps that must be available in the MetaResult instance
    _required_maps = ('p',)

    def __init__(self):
        pass

    @abstractproperty
    def _filename_suffix(self):
        pass

    def _validate_input(self, result):
        if not isinstance(result, MetaResult):
            raise ValueError("First argument to transform() must be an "
                             "instance of class MetaResult, not {}."
                             .format(type(result)))
        for rm in self._required_maps:
            if not result.maps.get(rm):
                raise ValueError("{0} requires {1} maps to be present in the "
                                 "MetaResult, but none were found."
                                 .format(type(self), rm))

    def transform(self, result):
        self._validate_input(result)
        est = result.estimator
        method = self._correction_method
        result = result.copy()
        if (method is not None and hasattr(est, method)):
            # Feed all init arguments to the estimator's method
            kwargs = inspect.getargspec(self.__init__)[1:]
            kwargs = {k: getattr(self, k) for k in kwargs}
            getattr(est, method)(result, **kwargs)
        else:
            self._transform(result)

    @abstractmethod
    def _transform(self, result, **kwargs):
        pass


class FDRCorrector(Corrector):
    """
    Perform false discovery rate correction on a meta-analysis.

    Parameters
    ----------
    q : `obj`:float
        The FDR correction rate to use.
    """

    _correction_method = '_fdr_correct'
    _required_maps = ('p',)

    def __init__(self, q=0.05):
        self. q = q

    @property
    def _filename_suffix(self):
        return '_corrected-FDR_q-{}'.format(self.q)

    def _transform(self, result):
        p = result.maps['p']
        _, p_corr_map, _, _ = multipletests(p, alpha=self.q, method='fdr_bh',
                                            is_sorted=False,
                                            returnsorted=False)
