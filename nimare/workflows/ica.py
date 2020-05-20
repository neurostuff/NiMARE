from ..due import due
from .. import references


@due.dcite(references.META_ICA,
           description='Introduces metaICA using BrainMap database.')
@due.dcite(references.META_ICA2,
           description='Compares results of BrainMap metaICA with resting state ICA.')
def meta_ica_workflow():
    """
    Perform a meta-ICA analysis on a database.

    Warnings
    --------
    This method is not yet implemented.
    """
    pass
