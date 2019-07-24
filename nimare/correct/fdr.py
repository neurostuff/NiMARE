from ..base.misc import Corrector


class FDRCorrector(Corrector):
    """
    Perform false discovery rate correction on a meta-analysis.
    Uses the meta-analytic algorithms native method when available and
    defaults to a generic form of FDR-correction when it is not.

    Warnings
    --------
    This method is not yet implemented.
    """

    # The name of the method that must be implemented in a MetaEstimator class
    # in order to override the default correction method.
    _correction_method = '_fdr_correct'

    def _transform(self, result, **kwargs):
        # Do standard FDR correction here, then return a copy of the
        # MetaResult that contains the new FDR-corrected image(s)
        return result
