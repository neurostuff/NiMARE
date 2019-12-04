"""
Automated annotation tools for existing ontologies.
"""
from .cogpo import extract_cogpo
from .cogat import download_cogat, extract_cogat, expand_counts, CogAtLemmatizer

__all__ = [
    'extract_cogpo',
    'download_cogat', 'extract_cogat', 'expand_counts', 'CogAtLemmatizer'
]
