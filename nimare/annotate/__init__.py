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
    '{} is untested. Please do not use it.'.format(__name__),
    ImportWarning
)
