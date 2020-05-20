"""
Multiple comparisons correction
"""
import warnings

from .correct import FDRCorrector, FWECorrector


__all__ = [
    'FDRCorrector',
    'FWECorrector'
]

warnings.simplefilter('default')
