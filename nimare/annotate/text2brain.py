"""
Generate a Text2Brain vector model.
"""
from ..base import NiMAREBase
from ..due import due
from .. import references


@due.dcite(references.TEXT2BRAIN)
class Text2BrainModel(NiMAREBase):
    """
    Generate a Text2Brain vector model.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    * Dockès, Jérôme, et al. "Text to brain: predicting the spatial
      distribution of neuroimaging observations from text reports."
      International Conference on Medical Image Computing and
      Computer-Assisted Intervention. Springer, Cham, 2018.
      https://doi.org/10.1007/978-3-030-00931-1_67
    """
    def __init__(self, text_df, coordinates_df):
        pass
