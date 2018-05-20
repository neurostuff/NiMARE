"""
Topic modeling with generalized correspondence latent Dirichlet allocation.
"""
from .base import TopicModel
from ...due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Introduces GC-LDA decoding.')
class GCLDAModel(TopicModel):
    """
    Generate a GCLDA topic model.

    Parameters
    ----------
    count_df : :obj:`pandas.DataFrame`
        A DataFrame with feature counts for the model. The index is 'id',
        used for identifying studies. Other columns are features (e.g.,
        unigrams and bigrams from Neurosynth), where each value is the number
        of times the feature is found in a given article.
    coordinates_df : :obj:`pandas.DataFrame`
        A DataFrame with a list of foci in the dataset. The index is 'id',
        used for identifying studies. Additional columns include 'i', 'j' and
        'k' (the matrix indices of the foci in standard space).
    n_topics : :obj:`int`, optional
        Number of topics to generate in model. The default is 100.
    n_regions : :obj:`int`, optional
        Number of subregions per topic (>=1). The default is 2.
    alpha : :obj:`float`, optional
        Prior count on topics for each document. The default is 0.1.
    beta : :obj:`float`, optional
        Prior count on word-types for each topic. The default is 0.01.
    gamma : :obj:`float`, optional
        Prior count added to y-counts when sampling z assignments. The
        default is 0.01.
    delta : :obj:`float`, optional
        Prior count on subregions for each topic. The default is 1.0.
    dobs : :obj:`int`, optional
        Spatial region 'default observations' (# observations weighting
        Sigma estimates in direction of default 'roi_size' value). The
        default is 25.
    roi_size : :obj:`float`, optional
        Default spatial 'region of interest' size (default value of
        diagonals in covariance matrix for spatial distribution, which the
        distributions are biased towards). The default is 50.0.
    symmetric : :obj:`bool`, optional
        Whether or not to use symmetry constraint on subregions. Symmetry
        requires n_regions = 2. The default is False.
    seed_init : :obj:`int`, optional
        Initial value of random seed. The default is 1.
    """
    def __init__(self, count_df, coordinates_df, n_topics=100, n_regions=2,
                 symmetric=False, alpha=.1, beta=.01, gamma=.01, delta=1.0,
                 dobs=25, roi_size=50.0, seed_init=1):
        pass
