"""
Meta-analysis diagnostics
"""
import warnings

from .base import Diagnostic


__all__ = [
    'Diagnostic'
]

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
