"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
from .due import (due, Doi)
from .dataset import Dataset, Database
from .meta import cbma
from .decode import GCLDADiscreteDecoder
from .version import __version__

del cbma, Dataset, Database, GCLDADiscreteDecoder


# Citation for the algorithm.
due.cite(Doi('10.7490/f1000research.1115905.1'),
         description='Neuroinformatics 2018 poster introducing NiMARE.',
         path='nimare', cite_module=True)

# Citation for package version
