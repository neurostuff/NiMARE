"""
Automated annotation of Cognitive Atlas labels.
"""
from ...due import due, Doi


@due.dcite(Doi('10.3389/fninf.2011.00017'),
           description='Introduces the Cognitive Atlas.')
def extract_cogat():
    """
    Extract CogAt terms and perform hierarchical expansion.
    """
    pass


def expansion():
    """
    Perform hierarchical expansion of CogAt labels.
    """
    pass
