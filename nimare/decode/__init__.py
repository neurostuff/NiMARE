"""
Top-level namespace for functional decoding.
"""

from .continuous import GCLDAContinuousDecoder, CorrelationDecoder
from .discrete import GCLDADiscreteDecoder, BrainMapDecoder, NeurosynthDecoder
from .encode import GCLDAEncoder
