"""
Meta-analytic parcellation tools
"""
import warnings

from . import cbp
from . import mapbot
from . import mamp

__all__ = ['cbp', 'mapbot', 'mamp']

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
