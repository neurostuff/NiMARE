"""
Automated annotation tools
"""
import warnings

from . import topic
from . import vector
from . import ontology
from . import text

__all__ = ['text', 'topic', 'vector', 'ontology']

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
