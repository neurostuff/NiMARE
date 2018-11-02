"""
Hierarchical Poisson/Gamma random field model
"""
from .base import CBMAEstimator
from ...due import due, BibTeX


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
    Hierarchical Poisson/Gamma random field model
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, sample):
        pass
