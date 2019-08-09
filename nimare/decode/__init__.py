"""
Functional decoding tools
"""
import warnings

from . import continuous
from . import discrete
from . import encode

__all__ = ['continuous', 'discrete', 'encode']

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
