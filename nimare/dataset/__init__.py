"""
Top-level namespace for datasets.
"""
from .data import Database, Dataset
from .extract import (NeuroVaultDataSource, NeurosynthDataSource,
                      BrainSpellDataSource)

__all__ = ['Database', 'Dataset',
           'NeuroVaultDataSource', 'NeurosynthDataSource',
           'BrainSpellDataSource']
