"""
Multiple comparisons correction
"""
import warnings

from .base import FDRCorrector


__all__ = [
    'FDRCorrector'
]

warnings.simplefilter('default')

warnings.warn(
    '{} is untested. Please do not use it.'.format(__name__),
    ImportWarning
)
