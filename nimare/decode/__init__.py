"""
Functional decoding tools.
"""
import warnings

from .continuous import corr_decode, corr_dist_decode, gclda_decode_map
from .discrete import gclda_decode_roi, brainmap_decode, neurosynth_decode
from .encode import encode_gclda

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
