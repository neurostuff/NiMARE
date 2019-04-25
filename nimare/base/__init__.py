"""
Top-level namespace for nimare base.
"""

from .inputs import (Analyzable, Mappable, ConnMatrix, Image, CoordinateSet,
                     Surface)
from .decode import Decoder
from .meta import (KernelTransformer, CBMAEstimator, IBMAEstimator)
from .base import MetaResult
from .annotate import (AnnotationModel)
from .misc import (Parcellator)


__all__ = [
    'Decoder',
    'KernelTransformer',
    'CBMAEstimator',
    'IBMAEstimator',
    'MetaResult',
    'AnnotationModel',
    'Parcellator'
    ]
