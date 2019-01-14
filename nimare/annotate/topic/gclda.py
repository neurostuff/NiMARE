"""
Topic modeling with generalized correspondence latent Dirichlet allocation.
"""
import logging

from ...base import AnnotationModel
from ...due import due, Doi

LGR = logging.getLogger(__name__)


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Introduces GC-LDA decoding.')
class GCLDAModel(AnnotationModel):
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
    name : :obj:`str`, optional
        Name of model.
    """
    def __init__(self, count_df, coordinates_df, mask='Mni152_2mm',
                 n_topics=100, n_regions=2, symmetric=True, alpha=.1,
                 beta=.01, gamma=.01, delta=1.0, dobs=25, roi_size=50.0,
                 seed_init=1, name='gclda'):
        pass

    def update(self, loglikely_freq=1, verbose=2):
        """
        Run a complete update cycle (sample z, sample y&r, update regions).

        Parameters
        ----------
        loglikely_freq : :obj:`int`, optional
            The frequency with which log-likelihood is updated. Default value
            is 1 (log-likelihood is updated every iteration).
        verbose : {0, 1, 2}, optional
            Determines how much info is printed to console. 0 = none,
            1 = a little, 2 = a lot. Default value is 2.
        """
        pass

    def run(n_iters=10, loglikely_freq=10, verbose=1):
        """
        Run multiple iterations.
        """
        pass

    @due.dcite(Doi('10.1145/1577069.1755845'),
               description='Describes method for computing log-likelihood '
                           'used in model.')
    def compute_log_likelihood(self, model=None, update_vectors=True):
        """
        Compute Log-likelihood of a model object given current model.
        Computes the log-likelihood of data in any model object (either train
        or test) given the posterior predictive distributions over peaks and
        word-types for the model. Note that this is not computing the joint
        log-likelihood of model parameters and data.

        Parameters
        ----------
        model : :obj:`gclda.Model`, optional
            The model for which log-likelihoods will be calculated.
            If not provided, log-likelihood will be calculated for the current
            model (self).
        update_vectors : :obj:`bool`, optional
            Whether to update model's log-likelihood vectors or not.

        Returns
        -------
        x_loglikely : :obj:`float`
            Total log-likelihood of all peak tokens.
        w_loglikely : :obj:`float`
            Total log-likelihood of all word tokens.
        tot_loglikely : :obj:`float`
            Total log-likelihood of peak + word tokens.

        References
        ----------
        [1] Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009).
        Distributed algorithms for topic models. Journal of Machine Learning
        Research, 10(Aug), 1801-1828.
        """
        pass

    def get_spatial_probs(self):
        """
        Get conditional probability of selecting each voxel in the brain mask
        given each topic.

        Returns
        -------
        p_voxel_g_topic : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A voxel-by-topic array of conditional probabilities: p(voxel|topic).
            For cell ij, the value is the probability of voxel i being selected
            given topic j has already been selected.
        p_topic_g_voxel : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A voxel-by-topic array of conditional probabilities: p(topic|voxel).
            For cell ij, the value is the probability of topic j being selected
            given voxel i is active.
        """
        pass

    def save_model_params(self, out_dir, n_top_words=15):
        """
        Run all export-methods: calls all save-methods to export parameters to
        files.

        Parameters
        ----------
        out_dir : :obj:`str`
            The name of the output directory.
        n_top_words : :obj:`int`, optional
            The number of words associated with each topic to report in topic
            word probabilities file.
        """
        pass
