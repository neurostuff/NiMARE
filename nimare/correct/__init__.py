"""
Multiple comparisons correction
"""
import warnings


from .correct import FDRCorrector

__all__ = [
    'FDRCorrector'
]

warnings.simplefilter('default')
