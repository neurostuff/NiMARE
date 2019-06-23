"""
Automated annotation tools for existing ontologies.
"""
from .cogpo import extract_cogpo
from .cogat import extract_cogat, expand_counts, CogAtLemmatizer

__all__ = ['extract_cogpo', 'extract_cogat', 'expand_counts',
           'CogAtLemmatizer']
