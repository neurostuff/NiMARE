"""
Functional decoding tools.
"""
import warnings

from .continuous import corr_decode, corr_dist_decode, gclda_decode_map
from .discrete import gclda_decode_roi, brainmap_decode, neurosynth_decode
from .encode import encode_gclda

warnings.simplefilter('default')

warnings.warn(
    '{} is untested. Please do not use it.'.format(__name__),
    ImportWarning
)
