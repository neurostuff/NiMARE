"""
Top-level namespace for nimare base.
"""

from .inputs import (Analyzable, Mappable, ConnMatrix, Image, CoordinateSet,
                     Surface)
from .decode import Decoder
from .meta import (KernelTransformer, MetaEstimator, CBMAEstimator,
                   IBMAEstimator, MetaResult)
from .annotate import (AnnotationModel)
from .misc import (Parcellator)
from .data import Study, Contrast
