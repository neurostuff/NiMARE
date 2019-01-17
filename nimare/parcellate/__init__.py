"""
Meta-analytic parcellation tools.
"""
import warnings

from .cbp import CoordCBP, ImCBP
from .mapbot import MAPBOT
from .mamp import MAMP

warnings.simplefilter('default')

warnings.warn(
    '{} is untested. Please do not use it.'.format(__name__),
    ImportWarning
)
