"""
Base classes used throughout NiMARE
"""

from .base import MetaResult
from .decode import Decoder
from .annotate import AnnotationModel
from .misc import Parcellator

__all__ = ['Decoder',
           'MetaResult',
           'AnnotationModel',
           'Parcellator'
           ]
