"""
Base classes used throughout NiMARE
"""

from .base import MetaResult
from .estimators import Estimator, Transformer

__all__ = [
    'Estimator',
    'Transformer',
    'MetaResult'
]
