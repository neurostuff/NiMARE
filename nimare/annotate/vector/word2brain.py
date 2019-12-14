"""
GloVe model-based annotation.
"""
from ..base import AnnotationModel
from ...due import due
from ... import references


@due.dcite(references.WORD2BRAIN)
class Word2BrainModel(AnnotationModel):
    """
    Generate a Word2Brain vector model [1]_.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Nunes, Abraham. "word2brain." bioRxiv (2018): 299024.
        https://doi.org/10.1101/299024
    """
    def __init__(self, text_df, coordinates_df):
        pass
