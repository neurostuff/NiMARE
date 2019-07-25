"""
Multiple comparisons correction
"""
import warnings

from .fdr import FDRCorrector


__all__ = [
    'FDRCorrector'
]

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
