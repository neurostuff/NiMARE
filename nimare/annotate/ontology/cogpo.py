"""
Automated annotation of Cognitive Paradigm Ontology labels.
"""
from ...due import due, Doi


@due.dcite(Doi('10.1007/s12021-011-9126-x'),
           description='Introduces the Cognitive Paradigm Ontology.')
def extract_cogpo():
    """
    Predict CogPO labels with ATHENA.

    Warnings
    --------
    This function is not yet implemented.
    """
    pass
