"""
Base classes for decoding/encoding.
"""
from ..base import Estimator


class Decoder(Estimator):
    """
    Base class for decoders. Decoders act as Estimator, in that they take in
    Datasets and return Results.
    """
    def __init__(self, dataset):
        pass
