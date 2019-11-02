"""
Multiple comparisons correction methods
"""
import statsmodels.stats.multitest as mc

from .base import Corrector


class FWECorrector(Corrector):
    """
    Perform family-wise error rate correction on a meta-analysis.

    Parameters
    ----------
    method : `obj`:str
        The FWE correction to use. Either 'bonferroni' or 'permutation'.
    **kwargs
        Keyword arguments to be used by the FWE correction implementation.
    """

    _correction_method = '_fwe_correct'

    def __init__(self, method='bonferroni', **kwargs):
        self.method = method
        self.parameters = kwargs

    @property
    def _name_suffix(self):
        return '_corr-FWE_method-{}'.format(self.method)

    def _transform(self, result):
        p = result.maps['p']
        _, p_corr, _, _ = mc.multipletests(p, alpha=0.05, method=self.method,
                                           is_sorted=False)
        corr_maps = {'p': p_corr}
        self._generate_secondary_maps(result, corr_maps)
        return corr_maps


class FDRCorrector(Corrector):
    """
    Perform false discovery rate correction on a meta-analysis.

    Parameters
    ----------
    alpha : `obj`:float
        The FDR correction rate to use.
    method : `obj`:str
        The FDR correction to use. Either 'indep' (for independent or
        positively correlated values) or 'negcorr' (for general or negatively
        correlated tests).
    """

    _correction_method = '_fdr_correct'

    def __init__(self, alpha=0.05, method='indep', **kwargs):
        self.alpha = alpha
        self.method = method
        self.parameters = kwargs

    @property
    def _name_suffix(self):
        return '_corr-FDR_method-{}'.format(self.method)

    def _transform(self, result):
        p = result.maps['p']
        _, p_corr = mc.fdrcorrection(p, alpha=self.alpha, method=self.method,
                                     is_sorted=False)
        corr_maps = {'p': p_corr}
        self._generate_secondary_maps(result, corr_maps)
        return corr_maps
