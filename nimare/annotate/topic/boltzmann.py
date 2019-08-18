"""
Topic modeling with deep Boltzmann machines.
"""
from ..base import AnnotationModel
from ...due import due
from ... import references


@due.dcite(references.BOLTZMANNMODEL)
class BoltzmannModel(AnnotationModel):
    """
    Generate a deep Boltzmann machine topic model [1]_.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Monti, Ricardo, et al. "Text-mining the NeuroSynth corpus using deep
        Boltzmann machines." 2016 International Workshop on Pattern Recognition
        in NeuroImaging (PRNI). IEEE, 2016.
        https://doi.org/10.1109/PRNI.2016.7552329
    """
    def __init__(self, text_df, coordinates_df):
        pass

    def fit(self):
        pass
