from ..due import due, Doi


@due.dcite(Doi('10.1162/jocn_a_00077'),
           description='Introduces metaICA using BrainMap database.')
@due.dcite(Doi('10.1162/jocn_a_00077'),
           description='Compares results of BrainMap metaICA with resting state ICA.')
def meta_ica_workflow():
    """
    Perform a meta-ICA analysis on a database.

    Warnings
    --------
    This method is not yet implemented.
    """
    pass
