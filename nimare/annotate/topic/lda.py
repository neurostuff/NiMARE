"""
Topic modeling with latent Dirichlet allocation via MALLET.
"""
import logging

from ...base import AnnotationModel
from ...due import due, BibTeX, Doi

LGR = logging.getLogger(__name__)


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
@due.dcite(BibTeX("""
    @article{mallettoolbox,
      title={MALLET: A Machine Learning for Language Toolkit.},
      author={McCallum, Andrew K},
      year={2002}}
    """), description='Citation for MALLET toolbox')
@due.dcite(Doi('10.1371/journal.pcbi.1002707'),
           description='First use of LDA for automated annotation of '
           'neuroimaging literature.')
class LDAModel(AnnotationModel):
    """
    Perform topic modeling using Latent Dirichlet Allocation with the
    Java toolbox MALLET.

    Parameters
    ----------
    text_df : :obj:`pandas.DataFrame`
        A pandas DataFrame with two columns ('id' and 'text') containing
        article text_df.
    n_topics : :obj:`int`, optional
        Number of topics to generate. Default=50.
    n_words : :obj:`int`, optional
        Number of top words to return for each topic. Default=31, based on
        Poldrack et al. (2012). Not used.
    n_iters : :obj:`int`, optional
        Number of iterations to run in training topic model. Default=1000.
    alpha : :obj:`float`, optional
        The Dirichlet prior on the per-document topic distributions.
        Default: 50 / n_topics, based on Poldrack et al. (2012).
    beta : :obj:`float`, optional
        The Dirichlet prior on the per-topic word distribution. Default: 0.001,
        based on Poldrack et al. (2012).
    """
    def __init__(self, text_df, n_topics=50, n_iters=1000, alpha='auto',
                 beta=0.001):
        pass
