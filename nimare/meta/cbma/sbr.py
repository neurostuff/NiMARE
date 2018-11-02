"""
Spatial binary regression model
"""
from .base import CBMAEstimator
from ...due import due, Doi


@due.dcite(Doi('10.1214/11-AOAS523'),
           description='Introduces the SBR model.')
class SBR(CBMAEstimator):
    """
    Spatial binary regression model
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, sample):
        pass
