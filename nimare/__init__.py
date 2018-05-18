"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
from .dataset.dataset import Dataset, Database
from .meta import cbma
from .decode import GCLDADiscreteDecoder
from .version import __version__

del cbma, Dataset, Database, GCLDADiscreteDecoder
