"""
Topic modeling with latent Dirichlet allocation via MALLET.
"""
from .base import TopicModel
from ...due import due, BibTeX, Doi


@due.dcite(BibTeX("""
    @article{blei2003latent,
      title={Latent dirichlet allocation},
      author={Blei, David M and Ng, Andrew Y and Jordan, Michael I},
      journal={Journal of machine Learning research},
      volume={3},
      number={Jan},
      pages={993--1022},
      year={2003}}
    """), description='Introduces LDA.')
@due.dcite(Doi('10.1371/journal.pcbi.1002707'),
           description='First use of LDA for automated annotation of '
           'neuroimaging literature.')
class LDAModel(TopicModel):
    """
    Generate an LDA topic model.
    """
    def __init__(self, text_df):
        pass
