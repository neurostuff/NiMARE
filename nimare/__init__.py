"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
from .dataset import Dataset
from .meta import cbma
from .decode import GCLDADiscreteDecoder
from .version import __version__

del cbma, Dataset, GCLDADiscreteDecoder
