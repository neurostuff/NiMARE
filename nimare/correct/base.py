from abc import abstractmethod, ABCMeta


class Corrector(metaclass=ABCMeta):
    ''' Base class for multiple comparison correction. '''
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


class FDRCorrector(Corrector):

    # The name of the method that must be implemented in a MetaEstimator class
    # in order to override the default correction method.
    _correction_method = '_fdr_correct'

    def _transform(self, result, **kwargs):
        # Do standard FDR correction here, then return a copy of the
        # MetaResult that contains the new FDR-corrected image(s)
        return result
