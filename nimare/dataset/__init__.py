"""
Top-level namespace for datasets.
"""
from .base import Contrast
from .dataset import Database, Dataset
from .extract import (NeuroVaultDataSource, NeurosynthDataSource,
                      BrainSpellDataSource)

del Database, Dataset
#__all__ = ['Database', 'Dataset',
#           'NeuroVaultDataSource', 'NeurosynthDataSource',
#           'BrainSpellDataSource']
