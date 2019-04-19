"""
GloVe model-based annotation.
"""
from ...base import AnnotationModel
from ...due import due, Doi


@due.dcite(Doi('10.1101/299024'),
           description='Introduces GloVe model-based annotation.')
class Word2BrainModel(AnnotationModel):
    """
    Generate a Word2Brain vector model.

    Warnings
    --------
    This method is not yet implemented.
    """
    def __init__(self, text_df, coordinates_df):
        pass
