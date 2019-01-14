"""
Automated annotation tools.
"""
from .text import generate_counts
from .topic import BoltzmannModel
from .ontology import extract_cogat

del BoltzmannModel, extract_cogat

__all__ = ['generate_counts']
