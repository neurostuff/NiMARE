"""
Top-level namespace for functional decoding.
"""

from .continuous import CorrelationDecoder, gclda_decode_continuous
from .discrete import GCLDADiscreteDecoder, BrainMapDecoder, NeurosynthDecoder
from .encode import GCLDAEncoder
