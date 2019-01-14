"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
from .dataset import Dataset, Database
from .meta import cbma
from .decode import gclda_decode_roi
from .parcellate import CoordCBP
from .version import __version__

del cbma, Dataset, Database, gclda_decode_roi, CoordCBP
