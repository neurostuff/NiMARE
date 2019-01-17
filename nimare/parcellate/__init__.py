"""
Meta-analytic parcellation tools.
"""
import warnings

from .cbp import CoordCBP, ImCBP
from .mapbot import MAPBOT
from .mamp import MAMP

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
