"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
from .due import (due, Doi)
from .dataset import Dataset
from .meta import cbma
from .version import __version__

del cbma, Dataset


# General citation for the package
due.cite(Doi('10.7490/f1000research.1115905.1'),
         description='Neuroinformatics 2018 poster introducing NiMARE.',
         path='nimare', cite_module=True)

# Citation for package version
