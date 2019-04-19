"""
Generate a Text2Brain vector model.
"""
from ...base import AnnotationModel
from ...due import due, BibTeX


@due.dcite(BibTeX(r"""
    @article{2018arXiv180601139D,
    author = {{Dock{\`e}s}, J. and {Wassermann}, D. and {Poldrack}, R. and
              {Suchanek}, F. and {Thirion}, B. and {Varoquaux}, G.},
    title = "{Text to brain: predicting the spatial distribution of
              neuroimaging observations from text reports}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1806.01139},
    primaryClass = "stat.ME",
    keywords = {Statistics - Methodology, Computer Science - Information
                Retrieval, Computer Science - Learning},
    year = 2018,
    month = jun,
    adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180601139D},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    """), description='Introduced text2brain models for annotation.')
class Text2BrainModel(AnnotationModel):
    """
    Generate a Text2Brain vector model.

    Warnings
    --------
    This method is not yet implemented.
    """
    def __init__(self, text_df, coordinates_df):
        pass
