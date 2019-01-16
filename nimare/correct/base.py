

class Corrector(object):
    ''' Base class for multiple comparison correction. '''
    _correction_method = None
    
    def transform(self, result, **kwargs):
        est = result.estimator
        method = self._correction_method
        if (self._correction_method is not None and hasattr(est, meth)):
            getattr(est, method)(result, **kwargs)
        else:
            self._transform(result, **kwargss)


class FDRCorrector(Corrector):

    # The name of the method that must be implemented in a MetaEstimator class
    # in order to override the default correction method.
    _correction_method = '_fdr_correct'

    def _transform(self, result, **kwargs):
        # Do standard FDR correction here, then return a copy of the
        # MetaResult that contains the new FDR-corrected image(s)
        return result
