"""
Automated annotation of Cognitive Paradigm Ontology labels.
"""
from ...due import due
from ... import references


@due.dcite(references.COGNITIVE_PARADIGM_ONTOLOGY,
           description='Introduces the Cognitive Paradigm Ontology.')
@due.dcite(references.ATHENA, description='Introduces ATHENA classifiers.')
def extract_cogpo():
    """
    Predict Cognitive Paradigm Ontology [1]_ labels with ATHENA classifiers
    [2]_.

    Warnings
    --------
    This function is not yet implemented.

    References
    ----------
    .. [1] Turner, Jessica A., and Angela R. Laird. "The cognitive paradigm
        ontology: design and application." Neuroinformatics 10.1 (2012): 57-66.
        https://doi.org/10.1007/s12021-011-9126-x
    .. [2] Riedel, Michael Cody, et al. "Automated, efficient, and accelerated
        knowledge modeling of the cognitive neuroimaging literature using the
        ATHENA toolkit." Frontiers in neuroscience 13 (2019): 494.
        https://doi.org/10.3389/fnins.2019.00494
    """
    pass
