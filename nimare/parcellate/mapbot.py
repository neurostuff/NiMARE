"""
Automated annotation of Cognitive Paradigm Ontology labels.
"""
from ..due import due, Doi


@due.dcite(Doi('10.1016/j.neuroimage.2017.06.032'),
           description='Introduces the MAPBOT algorithm.')
class MAPBOT(Parcellator):
    """
    MAPBOT: Uses text to parcellate
    """
    def __init__(self, dataset, ids):
        self.mask = dataset.mask
        self.ids = ids

    def fit(self, target_mask):
        """
        """
        pass
