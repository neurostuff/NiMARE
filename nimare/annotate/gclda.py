"""
Topic modeling with generalized correspondence latent Dirichlet allocation.
"""
import logging
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from .. import references
from ..base import NiMAREBase
from ..due import due
from ..utils import get_template

LGR = logging.getLogger(__name__)


@due.dcite(references.GCLDAMODEL)
class GCLDAModel(NiMAREBase):
    """
    Generate a generalized correspondence latent Dirichlet allocation
    (GCLDA) topic model.

    Parameters
    ----------
    count_df : :obj:`pandas.DataFrame`
        A DataFrame with feature counts for the model. The index is 'id',
        used for identifying studies. Other columns are features (e.g.,
        unigrams and bigrams from Neurosynth), where each value is the number
        of times the feature is found in a given article.
    coordinates_df : :obj:`pandas.DataFrame`
        A DataFrame with a list of foci in the dataset. The index is 'id',
        used for identifying studies. Additional columns include 'x', 'y' and
        'z' (foci in standard space).
    n_topics : :obj:`int`, optional
        Number of topics to generate in model. As a good rule of thumb, the
        number of topics should be less than the number of studies in the
        dataset. Otherwise, there can be errors during model training.
        The default is 100.
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

    Attributes
    ----------
    p_topic_g_voxel_ : (V x T) :obj:`numpy.ndarray`
        Probability of each topic (T) give a voxel (V)
    p_voxel_g_topic_ : (V x T) :obj:`numpy.ndarray`
        Probability of each voxel (V) given a topic (T)
    p_topic_g_word_ : (W x T) :obj:`numpy.ndarray`
        Probability of each topic (T) given a word (W)
    p_word_g_topic_ : (W x T) :obj:`numpy.ndarray`
        Probability of each word (W) given a topic (T)

    References
    ----------
    * Rubin, Timothy N., et al. "Decoding brain activity using a
      large-scale probabilistic functional-anatomical atlas of human
      cognition." PLoS computational biology 13.10 (2017): e1005649.
      https://doi.org/10.1371/journal.pcbi.1005649

    See Also
    --------
    nimare.decode.continuous.gclda_decode_map : GCLDA map decoding
    nimare.decode.discrete.gclda_decode_roi : GCLDA ROI decoding
    nimare.decode.encode.encode_gclda : GCLDA text-to-map encoding
    """

    def __init__(
        self,
        count_df,
        coordinates_df,
        mask="mni152_2mm",
        n_topics=100,
        n_regions=2,
        symmetric=True,
        alpha=0.1,
        beta=0.01,
        gamma=0.01,
        delta=1.0,
        dobs=25,
        roi_size=50.0,
        seed_init=1,
    ):
        LGR.info("Constructing/Initializing GCLDA Model")
        count_df = count_df.copy()
        coordinates_df = coordinates_df.copy()

        # Check IDs from DataFrames
        count_df.index = count_df.index.astype(str)
        count_df["id"] = count_df.index
        count_ids = count_df.index.tolist()
        if "id" not in coordinates_df.columns:
            coordinates_df["id"] = coordinates_df.index
        coordinates_df["id"] = coordinates_df["id"].astype(str)
        coord_ids = sorted(list(set(coordinates_df["id"].tolist())))
        ids = sorted(list(set(count_ids).intersection(coord_ids)))
        if len(count_ids) != len(coord_ids) != len(ids):
            union_ids = sorted(list(set(count_ids + coord_ids)))
            LGR.info(
                "IDs mismatch detected: retaining {0} of {1} unique "
                "IDs".format(len(ids), len(union_ids))
            )

        # Reduce inputs based on shared IDs
        count_df = count_df.loc[count_df["id"].isin(ids)]
        coordinates_df = coordinates_df.loc[coordinates_df["id"].isin(ids)]

        # --- Checking to make sure parameters are valid
        if (symmetric is True) and (n_regions != 2):
            # symmetric model only valid if R = 2
            raise ValueError(
                "Cannot run a symmetric model unless # subregions " "(n_regions) == 2 !"
            )

        # Initialize sampling parameters
        self.iter = 0  # Tracks the global sampling iteration of the model
        self.seed = 0  # Tracks current random seed to use (gets incremented
        # after initialization and each sampling update)

        # Set up model hyperparameters
        # Pseudo-count hyperparams need to be floats so that when sampling
        # distributions are computed the count matrices/vectors are converted
        # to floats
        self.params = {
            "n_topics": n_topics,  # Number of topics (T)
            "n_regions": n_regions,  # Number of subregions (R)
            "alpha": alpha,  # Prior count on topics for each doc
            "beta": beta,  # Prior count on word-types for each topic
            "gamma": gamma,  # Prior count added to y-counts when sampling z assignments
            "delta": delta,  # Prior count on subregions for each topic
            "roi_size": roi_size,  # Default ROI (default covariance spatial
            # region we regularize towards) (not in paper)
            "dobs": dobs,  # Sample constant (# observations weighting
            # sigma in direction of default covariance)
            # (not in paper)
            "symmetric": symmetric,  # Use constrained symmetry on subregions?
            # (only for n_regions = 2)
            "seed_init": seed_init,  # Random seed for initializing model
        }

        # Add dictionaries for other model info
        self.data = {}
        self.topics = {}

        # Prepare data
        if isinstance(mask, str) and not op.isfile(mask):
            self.mask = get_template(mask, mask="brain")
        elif isinstance(mask, str) and op.isfile(mask):
            self.mask = nib.load(mask)
        elif isinstance(mask, nib.Nifti1Image):
            self.mask = mask
        else:
            raise Exception('Input "mask" could not be figured out.')

        # Extract document and word indices from count_df
        docidx_mapper = {id_: i for (i, id_) in enumerate(ids)}
        self.ids = ids

        # Create docidx column
        count_df["docidx"] = count_df["id"].map(docidx_mapper)
        count_df = count_df.dropna(subset=["docidx"])
        count_df = count_df.drop("id", 1)

        # Remove words not found anywhere in the corpus
        count_df = count_df.loc[:, (count_df != 0).any(axis=0)]

        # Get updated vocabulary
        # List of word-strings (wtoken_word_idx values are indices into this list)
        vocabulary = count_df.columns.tolist()
        vocabulary.remove("docidx")
        self.vocabulary = vocabulary
        widx_mapper = {word: i for (i, word) in enumerate(self.vocabulary)}

        # Melt dataframe and create widx column
        widx_df = pd.melt(count_df, id_vars=["docidx"], var_name="word", value_name="count")
        widx_df["widx"] = widx_df["word"].map(widx_mapper)

        # Replicate rows based on count
        widx_df = widx_df.loc[np.repeat(widx_df.index.values, widx_df["count"])]
        widx_df = widx_df[["docidx", "widx"]].astype(int)
        widx_df.sort_values(by=["docidx", "widx"], inplace=True)

        # List of document-indices for word-tokens
        self.data["wtoken_doc_idx"] = widx_df["docidx"].tolist()
        # List of word-indices for word-tokens
        self.data["wtoken_word_idx"] = widx_df["widx"].tolist()

        # Import all peak-indices into lists
        coordinates_df["docidx"] = coordinates_df["id"].astype(str).map(docidx_mapper)
        coordinates_df = coordinates_df.dropna(subset=["docidx"])
        coordinates_df = coordinates_df[["docidx", "x", "y", "z"]]
        coordinates_df["docidx"] = coordinates_df["docidx"].astype(int)
        # List of document-indices for peak-tokens x
        self.data["ptoken_doc_idx"] = coordinates_df["docidx"].tolist()
        self.data["peak_vals"] = coordinates_df[["x", "y", "z"]].values

        # Seed random number generator
        np.random.seed(self.params["seed_init"])  # pylint: disable=no-member

        # Preallocate vectors of assignment indices
        self.topics["wtoken_topic_idx"] = np.zeros(
            len(self.data["wtoken_word_idx"]), dtype=int
        )  # word->topic assignments

        # Randomly initialize peak->topic assignments (y) ~ unif(1...n_topics)
        self.topics["peak_topic_idx"] = np.random.randint(
            self.params["n_topics"],  # pylint: disable=no-member
            size=(len(self.data["ptoken_doc_idx"])),
        )

        self.topics["peak_region_idx"] = np.zeros(
            len(self.data["ptoken_doc_idx"]), dtype=int
        )  # peak->region assignments

        # Preallocate count matrices
        # Peaks: D x T: Number of peak-tokens assigned to each topic per document
        self.topics["n_peak_tokens_doc_by_topic"] = np.zeros(
            (len(self.ids), self.params["n_topics"]), dtype=int
        )

        # Peaks: R x T: Number of peak-tokens assigned to each subregion per topic
        self.topics["n_peak_tokens_region_by_topic"] = np.zeros(
            (self.params["n_regions"], self.params["n_topics"]), dtype=int
        )

        # Words: W x T: Number of word-tokens assigned to each topic per word-type
        self.topics["n_word_tokens_word_by_topic"] = np.zeros(
            (len(self.vocabulary), self.params["n_topics"]), dtype=int
        )

        # Words: D x T: Number of word-tokens assigned to each topic per document
        self.topics["n_word_tokens_doc_by_topic"] = np.zeros(
            (len(self.ids), self.params["n_topics"]), dtype=int
        )

        # Words: 1 x T: Total number of word-tokens assigned to each topic (across all docs)
        self.topics["total_n_word_tokens_by_topic"] = np.zeros(
            (1, self.params["n_topics"]), dtype=int
        )

        # Preallocate Gaussians for all subregions
        # Regions_Mu & Regions_Sigma: Gaussian mean and covariance for all
        # subregions of all topics
        # Formed using lists (over topics) of lists (over subregions) of numpy
        # arrays
        #   regions_mu = (n_topics, n_regions, 1, n_peak_dims)
        #   regions_sigma = (n_topics, n_regions, n_peak_dims, n_peak_dims)
        # (\mu^{(t)}_r)
        self.topics["regions_mu"] = np.zeros(
            (self.params["n_topics"], self.params["n_regions"], 1, self.data["peak_vals"].shape[1])
        )
        # (\sigma^{(t)}_r)
        self.topics["regions_sigma"] = np.zeros(
            (
                self.params["n_topics"],
                self.params["n_regions"],
                self.data["peak_vals"].shape[1],
                self.data["peak_vals"].shape[1],
            )
        )

        # Initialize lists for tracking log-likelihood of data over sampling iterations
        self.loglikely_iter = []  # Tracks iteration we compute each loglikelihood at
        self.loglikely_x = []  # Tracks log-likelihood of peak tokens
        self.loglikely_w = []  # Tracks log-likelihood of word tokens
        self.loglikely_tot = []  # Tracks log-likelihood of peak + word tokens

        # Initialize peak->subregion assignments (r)
        if not self.params["symmetric"]:
            # if symmetric model use deterministic assignment :
            #     if peak_val[0] > 0, r = 1, else r = 0
            self.topics["peak_region_idx"][:] = np.random.randint(
                self.params["n_regions"],  # pylint: disable=no-member
                size=(len(self.data["ptoken_doc_idx"])),
            )
        else:
            # if asymmetric model, randomly sample r ~ unif(1...n_regions)
            self.topics["peak_region_idx"][:] = (self.data["peak_vals"][:, 0] > 0).astype(int)

        # Update model vectors and count matrices to reflect y and r assignments
        for i_ptoken in range(len(self.data["ptoken_doc_idx"])):
            # document -idx (d)
            doc = self.data["ptoken_doc_idx"][i_ptoken]
            topic = self.topics["peak_topic_idx"][i_ptoken]  # peak-token -> topic assignment (y_i)
            region = self.topics["peak_region_idx"][
                i_ptoken
            ]  # peak-token -> subregion assignment (c_i)
            self.topics["n_peak_tokens_doc_by_topic"][
                doc, topic
            ] += 1  # Increment document-by-topic counts
            self.topics["n_peak_tokens_region_by_topic"][
                region, topic
            ] += 1  # Increment region-by-topic

        # Randomly Initialize Word->Topic Assignments (z) for each word
        # token w_i: sample z_i proportional to p(topic|doc_i)
        for i_wtoken in range(len(self.data["wtoken_word_idx"])):
            # w_i word-type
            word = self.data["wtoken_word_idx"][i_wtoken]

            # w_i doc-index
            doc = self.data["wtoken_doc_idx"][i_wtoken]

            # Estimate p(t|d) for current doc
            p_topic_g_doc = self.topics["n_peak_tokens_doc_by_topic"][doc] + self.params["gamma"]

            # Sample a topic from p(t|d) for the z-assignment
            # Compute a cdf of the sampling distribution for z
            probs = np.cumsum(p_topic_g_doc)
            # Which elements of cdf are less than random sample?
            sample_locs = np.where(probs < np.random.rand() * probs[-1])[
                0
            ]  # pylint: disable=no-member
            # How many elements of cdf are less than sample
            # z = # elements of cdf less than rand-sample
            topic = len(sample_locs)

            # Update model assignment vectors and count-matrices to reflect z
            self.topics["wtoken_topic_idx"][
                i_wtoken
            ] = topic  # Word-token -> topic assignment (z_i)
            self.topics["n_word_tokens_word_by_topic"][word, topic] += 1
            self.topics["total_n_word_tokens_by_topic"][0, topic] += 1
            self.topics["n_word_tokens_doc_by_topic"][doc, topic] += 1

    def fit(self, n_iters=10000, loglikely_freq=10, verbose=1):
        """
        Run multiple iterations.

        Parameters
        ----------
        n_iters : :obj:`int`, optional
            Number of iterations to run. Default is 10000.
        loglikely_freq : :obj:`int`, optional
            The frequency with which log-likelihood is updated. Default value
            is 1 (log-likelihood is updated every iteration).
        verbose : {0, 1, 2}, optional
            Determines how much info is printed to console. 0 = none,
            1 = a little, 2 = a lot. Default value is 2.
        """
        if self.iter == 0:
            # Get Initial Spatial Parameter Estimates
            self._update_regions()

            # Get Log-Likelihood of data for Initialized model and save to
            # variables tracking loglikely
            self.compute_log_likelihood()

        for i in range(self.iter, n_iters):
            self._update(loglikely_freq=loglikely_freq, verbose=verbose)

        # TODO: Handle this more elegantly
        (p_topic_g_voxel, p_voxel_g_topic, p_topic_g_word, p_word_g_topic) = self.get_probs()
        self.p_topic_g_voxel_ = p_topic_g_voxel
        self.p_voxel_g_topic_ = p_voxel_g_topic
        self.p_topic_g_word_ = p_topic_g_word
        self.p_word_g_topic_ = p_word_g_topic

    def _update(self, loglikely_freq=1, verbose=2):
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
        self.iter += 1  # Update total iteration count

        if verbose == 2:
            LGR.info("Iter {0:04d}: Sampling z".format(self.iter))
        self.seed += 1
        self._update_word_topic_assignments(self.seed)  # Update z-assignments

        if verbose == 2:
            LGR.info("Iter {0:04d}: Sampling y|r".format(self.iter))
        self.seed += 1
        self._update_peak_assignments(self.seed)  # Update y-assignments

        if verbose == 2:
            LGR.info("Iter {0:04d}: Updating spatial params".format(self.iter))
        self._update_regions()  # Update gaussian estimates for all subregions

        # Only update loglikelihood every 'loglikely_freq' iterations
        # (Computing log-likelihood isn't necessary and slows things down a bit)
        if self.iter % loglikely_freq == 0:
            if verbose == 2:
                LGR.info("Iter {0:04d}: Computing log-likelihood".format(self.iter))
            # Compute log-likelihood of model in current state
            self.compute_log_likelihood()
            if verbose > 0:
                LGR.info(
                    "Iter {0:04d} Log-likely: x = {1:10.1f}, w = {2:10.1f}, "
                    "tot = {3:10.1f}".format(
                        self.iter,
                        self.loglikely_x[-1],
                        self.loglikely_w[-1],
                        self.loglikely_tot[-1],
                    )
                )

    def _update_word_topic_assignments(self, randseed):
        """
        Update wtoken_topic_idx (z) indicator variables assigning words->topics.

        Parameters
        ----------
        randseed : :obj:`int`
            Random seed for this iteration.
        """
        # --- Seed random number generator
        np.random.seed(randseed)  # pylint: disable=no-member

        # Loop over all word tokens
        for i_wtoken in range(len(self.data["wtoken_word_idx"])):
            # Get indices for current token
            word = self.data["wtoken_word_idx"][i_wtoken]  # w_i word-type
            doc = self.data["wtoken_doc_idx"][i_wtoken]  # w_i doc-index
            # current topic assignment for word token w_i
            topic = self.topics["wtoken_topic_idx"][i_wtoken]

            # Decrement count-matrices to remove current wtoken_topic_idx
            self.topics["n_word_tokens_word_by_topic"][word, topic] -= 1
            self.topics["total_n_word_tokens_by_topic"][0, topic] -= 1
            self.topics["n_word_tokens_doc_by_topic"][doc, topic] -= 1

            # Get sampling distribution:
            #    p(z_i|z,d,w) ~ p(w|t) * p(t|d)
            #                 ~ p_w_t * p_topic_g_doc
            p_word_g_topic = (
                self.topics["n_word_tokens_word_by_topic"][word, :] + self.params["beta"]
            ) / (
                self.topics["total_n_word_tokens_by_topic"]
                + self.params["beta"] * len(self.vocabulary)
            )
            p_topic_g_doc = (
                self.topics["n_peak_tokens_doc_by_topic"][doc, :] + self.params["gamma"]
            )
            probs = p_word_g_topic * p_topic_g_doc  # The unnormalized sampling distribution

            # Sample a z_i assignment for the current word-token from the sampling distribution
            probs = np.squeeze(probs) / np.sum(probs)  # Normalize the sampling distribution
            # Numpy returns a [1 x T] vector with a '1' in the index of sampled topic
            vec = np.random.multinomial(1, probs)  # pylint: disable=no-member
            topic = np.where(vec)[0][0]  # Extract selected topic from vector

            # Update the indices and the count matrices using the sampled z assignment
            self.topics["wtoken_topic_idx"][i_wtoken] = topic  # Update w_i topic-assignment
            self.topics["n_word_tokens_word_by_topic"][word, topic] += 1
            self.topics["total_n_word_tokens_by_topic"][0, topic] += 1
            self.topics["n_word_tokens_doc_by_topic"][doc, topic] += 1

    def _update_peak_assignments(self, randseed):
        """
        Update y / r indicator variables assigning peaks->topics/subregions.

        Parameters
        ----------
        randseed : :obj:`int`
            Random seed for this iteration.
        """
        # Seed random number generator
        np.random.seed(randseed)  # pylint: disable=no-member

        # Retrieve p(x|r,y) for all subregions
        peak_probs = self._get_peak_probs(self)

        # Iterate over all peaks x, and sample a new y and r assignment for each
        for i_ptoken in range(len(self.data["ptoken_doc_idx"])):
            doc = self.data["ptoken_doc_idx"][i_ptoken]
            topic = self.topics["peak_topic_idx"][i_ptoken]
            region = self.topics["peak_region_idx"][i_ptoken]

            # Decrement count in Subregion x Topic count matrix
            self.topics["n_peak_tokens_region_by_topic"][region, topic] -= 1

            # Decrement count in Document x Topic count matrix
            self.topics["n_peak_tokens_doc_by_topic"][doc, topic] -= 1

            # Retrieve the probability of generating current x from all
            # subregions: [R x T] array of probs
            p_x_subregions = (peak_probs[i_ptoken, :, :]).transpose()

            # Compute the probabilities of all subregions given doc
            #     p(r|d) ~ p(r|t) * p(t|d)
            # Counts of subregions per topic + prior: p(r|t)
            p_region_g_topic = self.topics["n_peak_tokens_region_by_topic"] + self.params["delta"]

            # Normalize the columns such that each topic's distribution over
            # subregions sums to 1
            p_region_g_topic = p_region_g_topic / np.sum(p_region_g_topic, axis=0)

            # Counts of topics per document + prior: p(t|d)
            p_topic_g_doc = (
                self.topics["n_peak_tokens_doc_by_topic"][doc, :] + self.params["alpha"]
            )

            # Reshape from (ntopics,) to (nregions, ntopics) with duplicated rows
            p_topic_g_doc = np.array([p_topic_g_doc] * self.params["n_regions"])

            # Compute p(subregion | document): p(r|d) ~ p(r|t) * p(t|d)
            # [R x T] array of probs
            p_region_g_doc = p_topic_g_doc * p_region_g_topic

            # Compute the multinomial probability: p(z|y)
            # Need the current vector of all z and y assignments for current doc
            # The multinomial from which z is sampled is proportional to number
            # of y assigned to each topic, plus constant \gamma
            doc_y_counts = self.topics["n_peak_tokens_doc_by_topic"][doc, :] + self.params["gamma"]
            doc_z_counts = self.topics["n_word_tokens_doc_by_topic"][doc, :]
            p_peak_g_topic = self._compute_prop_multinomial_from_zy_vectors(
                doc_z_counts, doc_y_counts
            )

            # Reshape from (ntopics,) to (nregions, ntopics) with duplicated rows
            p_peak_g_topic = np.array([p_peak_g_topic] * self.params["n_regions"])

            # Get the full sampling distribution:
            # [R x T] array containing the proportional probability of all y/r combinations
            probs_pdf = p_x_subregions * p_region_g_doc * p_peak_g_topic

            # Convert from a [R x T] matrix into a [R*T x 1] array we can sample from
            probs_pdf = probs_pdf.transpose().ravel()

            # Normalize the sampling distribution
            probs_pdf = probs_pdf / np.sum(probs_pdf)

            # Sample a single element (corresponding to a y_i and c_i assignment
            # for the peak token) from the sampling distribution
            # Returns a [1 x R*T] vector with a '1' in location that was sampled
            vec = np.random.multinomial(1, probs_pdf)  # pylint: disable=no-member
            sample_idx = np.where(vec)[0][0]  # Extract linear index value from vector

            # Transform the linear index of the sampled element into the
            # subregion/topic (r/y) assignment indices
            # Subregion sampled (r)
            region = np.remainder(
                sample_idx, self.params["n_regions"]
            )  # pylint: disable=no-member
            topic = int(np.floor(sample_idx / self.params["n_regions"]))  # Topic sampled (y)

            # Update the indices and the count matrices using the sampled y/r assignments
            # Increment count in Subregion x Topic count matrix
            self.topics["n_peak_tokens_region_by_topic"][region, topic] += 1
            # Increment count in Document x Topic count matrix
            self.topics["n_peak_tokens_doc_by_topic"][doc, topic] += 1
            self.topics["peak_topic_idx"][i_ptoken] = topic  # Update y->topic assignment
            self.topics["peak_region_idx"][i_ptoken] = region  # Update y->subregion assignment

    def _update_regions(self):
        """
        Update spatial distribution parameters (Gaussians params for all
        subregions).
        Updates regions_mu and regions_sigma, indicating location and
        distribution of each subregion.
        """
        # Generate default ROI based on default_width
        default_roi = self.params["roi_size"] * np.eye(self.data["peak_vals"].shape[1])

        if not self.params["symmetric"]:
            # For each region, compute a mean and a regularized covariance matrix
            for i_topic in range(self.params["n_topics"]):
                for j_region in range(self.params["n_regions"]):
                    # Get all peaks assigned to current topic & subregion
                    idx = (self.topics["peak_topic_idx"] == i_topic) & (
                        self.topics["peak_region_idx"] == j_region
                    )
                    vals = self.data["peak_vals"][idx]
                    n_obs = self.topics["n_peak_tokens_region_by_topic"][j_region, i_topic]

                    # Estimate mean
                    # If there are no observations, we set mean equal to zeros,
                    # otherwise take MLE
                    if n_obs == 0:
                        mu = np.zeros([self.data["peak_vals"].shape[1]])
                    else:
                        mu = np.mean(vals, axis=0)

                    # Estimate covariance
                    # if there are 1 or fewer observations, we set sigma_hat
                    # equal to default ROI, otherwise take MLE
                    if n_obs <= 1:
                        c_hat = default_roi
                    else:
                        c_hat = np.cov(np.transpose(vals))

                    # Regularize the covariance, using the ratio of observations
                    # to dobs (default constant # observations)
                    d_c = n_obs / (n_obs + self.params["dobs"])
                    sigma = d_c * c_hat + (1 - d_c) * default_roi

                    # Store estimates in model object
                    self.topics["regions_mu"][i_topic, j_region, ...] = mu
                    self.topics["regions_sigma"][i_topic, j_region, ...] = sigma
        else:
            # With symmetric subregions, we jointly compute all estimates for
            # subregions 1 & 2, constraining the means to be symmetric w.r.t.
            # the origin along x-dimension
            for i_topic in range(self.params["n_topics"]):
                # Get all peaks assigned to current topic & subregion 1
                idx1 = (self.topics["peak_topic_idx"] == i_topic) & (
                    self.topics["peak_region_idx"] == 0
                )
                vals1 = self.data["peak_vals"][idx1]
                n_obs1 = self.topics["n_peak_tokens_region_by_topic"][0, i_topic]

                # Get all peaks assigned to current topic & subregion 2
                idx2 = (self.topics["peak_topic_idx"] == i_topic) & (
                    self.topics["peak_region_idx"] == 1
                )
                vals2 = self.data["peak_vals"][idx2]
                n_obs2 = self.topics["n_peak_tokens_region_by_topic"][1, i_topic]

                # Get all peaks assigned to current topic & either subregion
                allvals = self.data["peak_vals"][idx1 | idx2]

                # Estimate means
                # If there are no observations, we set mean equal to zeros,
                # otherwise take MLE

                # Estimate independent mean for subregion 1
                if n_obs1 == 0:
                    m = np.zeros([self.data["peak_vals"].shape[1]])
                else:
                    m = np.mean(vals1, axis=0)

                # Estimate independent mean for subregion 2
                if n_obs2 == 0:
                    n = np.zeros([self.data["peak_vals"].shape[1]])
                else:
                    n = np.mean(vals2, axis=0)

                # Estimate the weighted means of all dims, where for dim1 we
                # compute the mean w.r.t. absolute distance from the origin
                weighted_mean_dim1 = (-m[0] * n_obs1 + n[0] * n_obs2) / (n_obs1 + n_obs2)
                weighted_mean_otherdims = np.mean(allvals[:, 1:], axis=0)

                # Store weighted mean estimates
                mu1 = np.zeros([1, self.data["peak_vals"].shape[1]])
                mu2 = np.zeros([1, self.data["peak_vals"].shape[1]])
                mu1[0, 0] = -weighted_mean_dim1
                mu1[0, 1:] = weighted_mean_otherdims
                mu2[0, 0] = weighted_mean_dim1
                mu2[0, 1:] = weighted_mean_otherdims

                # Store estimates in model object
                self.topics["regions_mu"][i_topic, 0, ...] = mu1
                self.topics["regions_mu"][i_topic, 1, ...] = mu2

                # Estimate Covariances
                # Covariances are estimated independently
                # Cov for subregion 1
                if n_obs1 <= 1:
                    c_hat1 = default_roi
                else:
                    c_hat1 = np.cov(np.transpose(vals1))

                # Cov for subregion 2
                if n_obs2 <= 1:
                    c_hat2 = default_roi
                else:
                    c_hat2 = np.cov(np.transpose(vals2))

                # Regularize the covariances, using the ratio of observations
                # to sample_constant
                d_c_1 = (n_obs1) / (n_obs1 + self.params["dobs"])
                d_c_2 = (n_obs2) / (n_obs2 + self.params["dobs"])
                sigma1 = d_c_1 * c_hat1 + (1 - d_c_1) * default_roi
                sigma2 = d_c_2 * c_hat2 + (1 - d_c_2) * default_roi

                # Store estimates in model object
                self.topics["regions_sigma"][i_topic, 0, ...] = sigma1
                self.topics["regions_sigma"][i_topic, 1, ...] = sigma2

    @due.dcite(
        references.LOG_LIKELIHOOD,
        description="Describes method for computing log-likelihood " "used in model.",
    )
    def compute_log_likelihood(self, model=None, update_vectors=True):
        """
        Compute log-likelihood of a model object given current model.

        Computes the log-likelihood of data in any model object (either train
        or test) given the posterior predictive distributions over peaks and
        word-types for the model, using the method described in
        Newman et al. (2009) [1]_. Note that this is not computing the joint
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
        .. [1] Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009).
            Distributed algorithms for topic models. Journal of Machine
            Learning Research, 10(Aug), 1801-1828.
        """
        if model is None:
            model = self
        elif update_vectors:
            LGR.info("External model detected: Disabling update_vectors")
            update_vectors = False

        # Pre-compute all probabilities from count matrices that are needed
        # for loglikelihood computations
        # Compute docprobs for y = ND x NT: p( y_i=t | d )
        doccounts = self.topics["n_peak_tokens_doc_by_topic"] + self.params["alpha"]
        doccounts_sum = np.sum(doccounts, axis=1)
        docprobs_y = np.transpose(np.transpose(doccounts) / doccounts_sum)

        # Compute docprobs for z = ND x NT: p( z_i=t | y^(d) )
        doccounts = self.topics["n_peak_tokens_doc_by_topic"] + self.params["gamma"]
        doccounts_sum = np.sum(doccounts, axis=1)
        docprobs_z = np.transpose(np.transpose(doccounts) / doccounts_sum)

        # Compute regionprobs = NR x NT: p( r | t )
        regioncounts = (self.topics["n_peak_tokens_region_by_topic"]) + self.params["delta"]
        regioncounts_sum = np.sum(regioncounts, axis=0)
        regionprobs = regioncounts / regioncounts_sum

        # Compute wordprobs = NW x NT: p( w | t )
        wordcounts = self.topics["n_word_tokens_word_by_topic"] + self.params["beta"]
        wordcounts_sum = np.sum(wordcounts, axis=0)
        wordprobs = wordcounts / wordcounts_sum

        # Get the matrix giving p(x_i|r,t) for all x:
        #    NY x NT x NR matrix of probabilities of all peaks given all
        #    topic/subregion spatial distributions
        peak_probs = self._get_peak_probs(model)

        # Compute observed peaks (x) Loglikelihood:
        # p(x|model, doc) = p(topic|doc) * p(subregion|topic) * p(x|subregion)
        #                 = p_topic_g_doc * p_region_g_topic * p_x_r
        # Initialize variable tracking total loglikelihood of all x tokens
        x_loglikely = 0

        # Go over all observed peaks and add p(x|model) to running total
        for i_ptoken in range(len(self.data["ptoken_doc_idx"])):
            doc = self.data["ptoken_doc_idx"][i_ptoken] - 1  # convert didx from 1-idx to 0-idx
            p_x = 0  # Running total for p(x|d) across subregions:
            # Compute p(x_i|d) for each subregion separately and then
            # sum across the subregions
            for j_region in range(self.params["n_regions"]):
                # p(t|d) - p(topic|doc)
                p_topic_g_doc = docprobs_y[doc]

                # p(r|t) - p(subregion|topic)
                p_region_g_topic = regionprobs[j_region]

                # p(r|d) - p(subregion|document) = p(topic|doc)*p(subregion|topic)
                p_region_g_doc = p_topic_g_doc * p_region_g_topic

                # p(x|r) - p(x|subregion)
                p_x_r = peak_probs[i_ptoken, :, j_region]

                # p(x|subregion,doc) = sum_topics ( p(subregion|doc) * p(x|subregion) )
                p_x_rd = np.dot(p_region_g_doc, p_x_r)
                p_x += p_x_rd  # Add probability for current subregion to total
                # probability for token across subregions
            # Add probability for current token to running total for all x tokens
            x_loglikely += np.log(p_x)  # pylint: disable=no-member

        # Compute observed words (w) Loglikelihoods:
        # p(w|model, doc) = p(topic|doc) * p(word|topic)
        #                 = p_topic_g_doc * p_w_t
        w_loglikely = 0  # Initialize variable tracking total loglikelihood of all w tokens

        # Compute a matrix of posterior predictives over words:
        # = ND x NW p(w|d) = sum_t ( p(t|d) * p(w|t) )
        p_wtoken_g_doc = np.dot(docprobs_z, np.transpose(wordprobs))

        # Go over all observed word tokens and add p(w|model) to running total
        for i_wtoken in range(len(self.data["wtoken_word_idx"])):
            # convert wtoken_word_idx from 1-idx to 0-idx
            word_token = self.data["wtoken_word_idx"][i_wtoken] - 1
            # convert wtoken_doc_idx from 1-idx to 0-idx
            doc = self.data["wtoken_doc_idx"][i_wtoken] - 1
            # Probability of sampling current w token from d
            p_wtoken = p_wtoken_g_doc[doc, word_token]
            # Add log-probability of current token to running total for all w tokens
            w_loglikely += np.log(p_wtoken)  # pylint: disable=no-member
        tot_loglikely = x_loglikely + w_loglikely

        # Update model log-likelihood history vector (if update_vectors == True)
        if update_vectors:
            self.loglikely_iter.append(self.iter)
            self.loglikely_x.append(x_loglikely)
            self.loglikely_w.append(w_loglikely)
            self.loglikely_tot.append(tot_loglikely)

        # Return loglikely values (used when computing log-likelihood for a
        # model-object containing hold-out data)
        return (x_loglikely, w_loglikely, tot_loglikely)

    def _get_peak_probs(self, model):
        """
        Compute a matrix giving p(x|r,t), using all x values in a model
        object, and each topic's spatial parameters.

        Returns
        -------
        peak_probs : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            nPeaks x nTopics x nRegions matrix of probabilities, giving
            probability of sampling each peak (x) from all subregions.
        """
        peak_probs = np.zeros(
            (len(model.data["ptoken_doc_idx"]), self.params["n_topics"], self.params["n_regions"]),
            dtype=float,
        )
        for i_topic in range(self.params["n_topics"]):
            for j_region in range(self.params["n_regions"]):
                pdf = multivariate_normal.pdf(
                    model.data["peak_vals"],
                    mean=self.topics["regions_mu"][i_topic, j_region, 0, :],
                    cov=self.topics["regions_sigma"][i_topic, j_region, ...],
                )
                peak_probs[:, i_topic, j_region] = pdf
        return peak_probs

    def _compute_prop_multinomial_from_zy_vectors(self, z, y):
        """
        Compute proportional multinomial probabilities of current x vector
        given current y vector, for all proposed y_i values.
        Note that this only returns values proportional to the relative
        probabilities of all proposals for y_i.

        Parameters
        ----------
        z : :obj:`numpy.ndarray` of :obj:`numpy.int64`
            A 1-by-T vector of current z counts for document d.
        y : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A 1-by-T vector of current y counts (plus gamma) for document d.

        Returns
        -------
        p : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A 1-by-T vector giving the proportional probability of z, given
            that topic t was incremented.
        """
        # Compute the proportional probabilities in log-space
        logp = z * np.log((y + 1) / y)  # pylint: disable=no-member
        p = np.exp(logp - np.max(logp))  # Add a constant before exponentiating
        # to avoid any underflow issues
        return p

    def get_probs(self):
        """
        Get conditional probability of selecting each voxel in the brain mask
        given each topic.

        Returns
        -------
        p_topic_g_voxel : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A voxel-by-topic array of conditional probabilities: p(topic|voxel).
            For cell ij, the value is the probability of topic j being selected
            given voxel i is active.
        p_voxel_g_topic : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A voxel-by-topic array of conditional probabilities: p(voxel|topic).
            For cell ij, the value is the probability of voxel i being selected
            given topic j has already been selected.
        p_topic_g_word : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A word-by-topic array of conditional probabilities: p(topic|word).
            For cell ij, the value is the probability of topic i being selected
            given word j is present.
        p_word_g_topic : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A word-by-topic array of conditional probabilities: p(word|topic).
            For cell ij, the value is the probability of word j being selected
            given topic i has already been selected.
        """
        affine = self.mask.affine
        mask_ijk = np.vstack(np.where(self.mask.get_fdata())).T
        mask_xyz = nib.affines.apply_affine(affine, mask_ijk)

        spatial_dists = np.zeros((mask_xyz.shape[0], self.params["n_topics"]), float)
        for i_topic in range(self.params["n_topics"]):
            for j_region in range(self.params["n_regions"]):
                pdf = multivariate_normal.pdf(
                    mask_xyz,
                    mean=self.topics["regions_mu"][i_topic, j_region, 0, :],
                    cov=self.topics["regions_sigma"][i_topic, j_region, ...],
                )
                spatial_dists[:, i_topic] += pdf
        p_topic_g_voxel = spatial_dists / np.sum(spatial_dists, axis=1)[:, None]
        p_topic_g_voxel = np.nan_to_num(p_topic_g_voxel, 0)  # might be unnecessary

        p_voxel_g_topic = spatial_dists / np.sum(spatial_dists, axis=0)[None, :]
        p_voxel_g_topic = np.nan_to_num(p_voxel_g_topic, 0)  # might be unnecessary

        n_word_tokens_per_topic = np.sum(self.topics["n_word_tokens_word_by_topic"], axis=0)
        p_word_g_topic = (
            self.topics["n_word_tokens_word_by_topic"] / n_word_tokens_per_topic[None, :]
        )
        p_word_g_topic = np.nan_to_num(p_word_g_topic, 0)

        n_topics_per_word_token = np.sum(self.topics["n_word_tokens_word_by_topic"], axis=1)
        p_topic_g_word = (
            self.topics["n_word_tokens_word_by_topic"] / n_topics_per_word_token[:, None]
        )
        p_topic_g_word = np.nan_to_num(p_topic_g_word, 0)

        return p_topic_g_voxel, p_voxel_g_topic, p_topic_g_word, p_word_g_topic
