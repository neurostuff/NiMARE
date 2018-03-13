"""
Methods for decoding unthresholded brain maps into text.
"""
from .base import Decoder
from ...due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Describes decoding methods using GC-LDA.')
def gclda_decode_roi(model, roi, topic_priors=None, prior_weight=1):
    pass


@due.dcite(Doi('10.1038/nmeth.1635'),
           description='Introduces Neurosynth.')
class CorrelationDecoder(Decoder):
    def __init__(self, dataset):
        pass
