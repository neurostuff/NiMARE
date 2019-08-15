"""
Model-based coordinate-based meta-analysis estimators
"""
from .base import CBMAEstimator
from ...due import due
from ... import references


@due.dcite(references.BHICP, description='Introduces the BHICP model.')
class BHICP(CBMAEstimator):
    """
    Bayesian hierarchical cluster process model [1]_.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Kang, Jian, et al. "Meta analysis of functional neuroimaging data
        via Bayesian spatial point processes." Journal of the American
        Statistical Association 106.493 (2011): 124-134.
        https://doi.org/10.1198/jasa.2011.ap09735
    """
    def __init__(self):
        pass

    def _fit(self, dataset):
        pass


@due.dcite(references.HPGRF, description='Introduces the HPGRF model.')
class HPGRF(CBMAEstimator):
    """
    Hierarchical Poisson/Gamma random field model [1]_.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Kang, Jian, et al. "A Bayesian hierarchical spatial point process
        model for multi-type neuroimaging meta-analysis." The annals of applied
        statistics 8.3 (2014): 1800.
    """
    def __init__(self):
        pass

    def _fit(self, dataset):
        pass


@due.dcite(references.SBLFR, description='Introduces the SBLFR model.')
class SBLFR(CBMAEstimator):
    """
    Spatial Bayesian latent factor regression model [1]_.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Montagna, Silvia, et al. "Spatial Bayesian latent factor regression
        modeling of coordinate‐based meta‐analysis data." Biometrics 74.1
        (2018): 342-353. https://doi.org/10.1111/biom.12713
    """
    def __init__(self):
        pass

    def _fit(self, dataset):
        pass


@due.dcite(references.SBR, description='Introduces the SBR model.')
class SBR(CBMAEstimator):
    """
    Spatial binary regression model [1]_.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Yue, Yu Ryan, Martin A. Lindquist, and Ji Meng Loh. "Meta-analysis
        of functional neuroimaging data using Bayesian nonparametric binary
        regression." The Annals of Applied Statistics 6.2 (2012): 697-718.
        https://doi.org/10.1214/11-AOAS523
    """
    def __init__(self):
        pass

    def _fit(self, dataset):
        pass
