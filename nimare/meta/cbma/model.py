"""
Model-based coordinate-based meta-analysis estimators
"""
from ...base import CBMAEstimator
from ...due import due, Doi, BibTeX


@due.dcite(Doi('10.1198/jasa.2011.ap09735'),
           description='Introduces the BHICP model.')
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
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, voxel_thresh=0.001, q=0.05, corr='FWE', n_iters=10000,
            n_cores=-1):
        """
        """
        pass


@due.dcite(BibTeX("""
           @article{kang2014bayesian,
             title={A Bayesian hierarchical spatial point process model for
                    multi-type neuroimaging meta-analysis},
             author={Kang, Jian and Nichols, Thomas E and Wager, Tor D and
                     Johnson, Timothy D},
             journal={The annals of applied statistics},
             volume={8},
             number={3},
             pages={1800},
             year={2014},
             publisher={NIH Public Access}
             }
           """),
           description='Introduces the HPGRF model.')
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
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, voxel_thresh=0.001, q=0.05, corr='FWE', n_iters=10000,
            n_cores=-1):
        """
        """
        pass


@due.dcite(Doi('10.1111/biom.12713'),
           description='Introduces the SBLFR model.')
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
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.ids = ids

    def fit(self, voxel_thresh=0.001, q=0.05, corr='FWE', n_iters=10000,
            n_cores=-1):
        """
        """
        pass


@due.dcite(Doi('10.1214/11-AOAS523'),
           description='Introduces the SBR model.')
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
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, voxel_thresh=0.001, q=0.05, corr='FWE', n_iters=10000,
            n_cores=-1):
        """
        """
        pass
