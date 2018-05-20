"""
Topic modeling with a Global Vectors for Word Representation (GloVe) model.
"""
from .base import TopicModel
from ...due import due, Doi


@due.dcite(Doi('10.1101/299024'),
           description='Introduces GloVe model-based annotation.')
class GloveModel(TopicModel):
    """
    Generate a GloVe topic model.
    """
    def __init__(self, text_df, coordinates_df):
        pass
