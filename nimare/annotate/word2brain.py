"""
GloVe model-based annotation.
"""
from ..base import NiMAREBase
from ..due import due
from .. import references


@due.dcite(references.WORD2BRAIN)
class Word2BrainModel(NiMAREBase):
    """
    Generate a Word2Brain vector model.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    * Nunes, Abraham. "word2brain." bioRxiv (2018): 299024.
      https://doi.org/10.1101/299024
    """
    def __init__(self, text_df, coordinates_df):
        pass
