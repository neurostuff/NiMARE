"""
Top-level namespace for nimare base.
"""

from .inputs import (Analyzable, Mappable, ConnMatrix, Image, CoordinateSet,
                     Surface)
from .decode import Decoder
from .meta import (MetaResult, MetaEstimator, CBMAEstimator, KernelEstimator,
                   IBMAEstimator)
from .data import Study, Contrast, DataSource
