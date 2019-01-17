"""
Automated annotation tools.
"""
import warnings

from .text import generate_counts
from .topic import BoltzmannModel
from .ontology import extract_cogat

del BoltzmannModel, extract_cogat

__all__ = ['generate_counts']

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
