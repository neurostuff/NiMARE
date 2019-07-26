"""
Miscellaneous base classes.
"""
from abc import abstractmethod, ABCMeta

from .estimators import Estimator


class Parcellator(Estimator):
    """
    Base class for meta-analytic parcellation methods.
    """
    pass
