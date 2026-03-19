"""Topic modeling with generalized correspondence latent Dirichlet allocation."""

import logging
import os.path as op

import nibabel as nib
import numpy as np
from numba import njit, prange
from nilearn.image import load_img

from nimare.base import NiMAREBase
from nimare.utils import _mask_img_to_bool, get_template

LGR = logging.getLogger(__name__)


@njit(cache=True)
def _sample_from_unnormalized(weights):
    """Sample an index from non-negative unnormalized weights."""
    total = 0.0
    for i_weight in range(weights.shape[0]):
        total += weights[i_weight]

    threshold = np.random.random() * total
    cumulative = 0.0
    for i_weight in range(weights.shape[0]):
        cumulative += weights[i_weight]
        if threshold < cumulative:
            return i_weight

    return weights.shape[0] - 1


@njit(cache=True, parallel=True)
def _jit_spatial_pdf(points, mean, precision, log_norm):
    """Evaluate one Gaussian PDF over many points."""
    n_points, n_dim = points.shape
    probs = np.empty(n_points, dtype=np.float64)

    for i_point in prange(n_points):
        quad = 0.0
        for i_dim in range(n_dim):
            centered_i = points[i_point, i_dim] - mean[i_dim]
            inner = 0.0
            for j_dim in range(n_dim):
                inner += precision[i_dim, j_dim] * (points[i_point, j_dim] - mean[j_dim])
            quad += centered_i * inner
        probs[i_point] = np.exp(log_norm - 0.5 * quad)

    return probs


@njit(cache=True, parallel=True)
def _jit_get_peak_probs(points, regions_mu, regions_precision, regions_log_norm):
    """Compute peak probabilities for all topic-region pairs in parallel."""
    n_points, n_dim = points.shape
    n_topics, n_regions, _ = regions_mu.shape
    peak_probs = np.empty((n_points, n_topics, n_regions), dtype=np.float64)

    for i_point in prange(n_points):
        for i_topic in range(n_topics):
            for j_region in range(n_regions):
                quad = 0.0
                for i_dim in range(n_dim):
                    centered_i = points[i_point, i_dim] - regions_mu[i_topic, j_region, i_dim]
                    inner = 0.0
                    for j_dim in range(n_dim):
                        inner += regions_precision[i_topic, j_region, i_dim, j_dim] * (
                            points[i_point, j_dim] - regions_mu[i_topic, j_region, j_dim]
                        )
                    quad += centered_i * inner
                peak_probs[i_point, i_topic, j_region] = np.exp(
                    regions_log_norm[i_topic, j_region] - 0.5 * quad
                )

    return peak_probs


@njit(cache=True, parallel=True)
def _jit_get_spatial_dists(points, regions_mu, regions_precision, regions_log_norm):
    """Accumulate topic spatial distributions over all regions in parallel."""
    n_points, n_dim = points.shape
    n_topics, n_regions, _ = regions_mu.shape
    spatial_dists = np.zeros((n_points, n_topics), dtype=np.float64)

    for i_point in prange(n_points):
        for i_topic in range(n_topics):
            topic_prob = 0.0
            for j_region in range(n_regions):
                quad = 0.0
                for i_dim in range(n_dim):
                    centered_i = points[i_point, i_dim] - regions_mu[i_topic, j_region, i_dim]
                    inner = 0.0
                    for j_dim in range(n_dim):
                        inner += regions_precision[i_topic, j_region, i_dim, j_dim] * (
                            points[i_point, j_dim] - regions_mu[i_topic, j_region, j_dim]
                        )
                    quad += centered_i * inner
                topic_prob += np.exp(regions_log_norm[i_topic, j_region] - 0.5 * quad)
            spatial_dists[i_point, i_topic] = topic_prob

    return spatial_dists


@njit(cache=True)
def _jit_accumulate_region_stats(coords, peak_topic_idx, peak_region_idx, n_topics, n_regions):
    """Accumulate sums and cross-products for each topic-region pair."""
    n_dim = coords.shape[1]
    region_sums = np.zeros((n_regions, n_topics, n_dim), dtype=np.float64)
    region_cross = np.zeros((n_regions, n_topics, n_dim, n_dim), dtype=np.float64)

    for i_coord in range(coords.shape[0]):
        topic = peak_topic_idx[i_coord]
        region = peak_region_idx[i_coord]
        for i_dim in range(n_dim):
            val_i = coords[i_coord, i_dim]
            region_sums[region, topic, i_dim] += val_i
            for j_dim in range(n_dim):
                region_cross[region, topic, i_dim, j_dim] += val_i * coords[i_coord, j_dim]

    return region_sums, region_cross


@njit(cache=True)
def _jit_initialize_word_topic_assignments(
    randseed,
    wtoken_doc_idx,
    wtoken_word_idx,
    peak_doc_by_topic,
    gamma,
    wtoken_topic_idx,
    word_by_topic,
    total_word_by_topic,
    word_doc_by_topic,
):
    """Initialize word-topic assignments from p(topic|doc)."""
    np.random.seed(randseed)
    n_topics = peak_doc_by_topic.shape[1]
    probs = np.empty(n_topics, dtype=np.float64)

    for i_wtoken in range(wtoken_word_idx.shape[0]):
        doc = wtoken_doc_idx[i_wtoken]
        word = wtoken_word_idx[i_wtoken]

        for i_topic in range(n_topics):
            probs[i_topic] = peak_doc_by_topic[doc, i_topic] + gamma

        topic = _sample_from_unnormalized(probs)
        wtoken_topic_idx[i_wtoken] = topic
        word_by_topic[word, topic] += 1
        total_word_by_topic[topic] += 1
        word_doc_by_topic[doc, topic] += 1


@njit(cache=True)
def _jit_update_word_topic_assignments(
    randseed,
    wtoken_word_idx,
    wtoken_doc_idx,
    wtoken_topic_idx,
    word_by_topic,
    total_word_by_topic,
    word_doc_by_topic,
    peak_doc_by_topic,
    beta,
    beta_vocabulary,
    gamma,
):
    """Update word-topic assignments in place."""
    np.random.seed(randseed)
    n_topics = peak_doc_by_topic.shape[1]
    probs = np.empty(n_topics, dtype=np.float64)

    for i_wtoken in range(wtoken_word_idx.shape[0]):
        word = wtoken_word_idx[i_wtoken]
        doc = wtoken_doc_idx[i_wtoken]
        topic = wtoken_topic_idx[i_wtoken]

        word_by_topic[word, topic] -= 1
        total_word_by_topic[topic] -= 1
        word_doc_by_topic[doc, topic] -= 1

        for i_topic in range(n_topics):
            probs[i_topic] = (
                (word_by_topic[word, i_topic] + beta)
                / (total_word_by_topic[i_topic] + beta_vocabulary)
            ) * (peak_doc_by_topic[doc, i_topic] + gamma)

        topic = _sample_from_unnormalized(probs)
        wtoken_topic_idx[i_wtoken] = topic
        word_by_topic[word, topic] += 1
        total_word_by_topic[topic] += 1
        word_doc_by_topic[doc, topic] += 1


@njit(cache=True)
def _jit_update_peak_assignments(
    randseed,
    ptoken_doc_idx,
    peak_probs,
    peak_topic_idx,
    peak_region_idx,
    region_by_topic,
    doc_by_topic,
    word_doc_by_topic,
    delta,
    alpha,
    gamma,
):
    """Update peak topic/subregion assignments in place."""
    np.random.seed(randseed)
    n_regions, n_topics = region_by_topic.shape
    region_totals = np.empty(n_topics, dtype=np.float64)
    peak_topic_probs = np.empty(n_topics, dtype=np.float64)
    probs_pdf = np.empty(n_regions * n_topics, dtype=np.float64)
    region_total_prior = delta * n_regions

    for i_topic in range(n_topics):
        topic_total = 0.0
        for j_region in range(n_regions):
            topic_total += region_by_topic[j_region, i_topic]
        region_totals[i_topic] = topic_total

    for i_ptoken in range(ptoken_doc_idx.shape[0]):
        doc = ptoken_doc_idx[i_ptoken]
        topic = peak_topic_idx[i_ptoken]
        region = peak_region_idx[i_ptoken]

        region_by_topic[region, topic] -= 1
        doc_by_topic[doc, topic] -= 1
        region_totals[topic] -= 1

        max_logp = -np.inf
        for i_topic in range(n_topics):
            doc_topic_peak_counts = doc_by_topic[doc, i_topic] + gamma
            logp = word_doc_by_topic[doc, i_topic] * np.log1p(1.0 / doc_topic_peak_counts)
            peak_topic_probs[i_topic] = logp
            if logp > max_logp:
                max_logp = logp

        for i_topic in range(n_topics):
            peak_topic_probs[i_topic] = np.exp(peak_topic_probs[i_topic] - max_logp)

        flat_idx = 0
        for j_region in range(n_regions):
            for i_topic in range(n_topics):
                probs_pdf[flat_idx] = (
                    peak_probs[i_ptoken, i_topic, j_region]
                    * ((region_by_topic[j_region, i_topic] + delta) / (region_totals[i_topic] + region_total_prior))
                    * (doc_by_topic[doc, i_topic] + alpha)
                    * peak_topic_probs[i_topic]
                )
                flat_idx += 1

        sampled_idx = _sample_from_unnormalized(probs_pdf)
        region = sampled_idx // n_topics
        topic = sampled_idx % n_topics

        region_by_topic[region, topic] += 1
        doc_by_topic[doc, topic] += 1
        region_totals[topic] += 1
        peak_topic_idx[i_ptoken] = topic
        peak_region_idx[i_ptoken] = region


class GCLDAModel(NiMAREBase):
    """Generate a generalized correspondence latent Dirichlet allocation (GCLDA) topic model.

    This model was originally described in :footcite:t:`rubin2017decoding`.

    .. versionchanged:: 0.0.8

        * [ENH] Support symmetric GC-LDA topics with more than two subregions.

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
    .. footbibliography::

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
            LGR.warning(
                f"IDs mismatch detected: retaining {len(ids)} of {len(union_ids)} unique IDs"
            )
        self.ids = ids

        # Reduce inputs based on shared IDs
        count_df = count_df.loc[count_df["id"].isin(ids)]
        coordinates_df = coordinates_df.loc[coordinates_df["id"].isin(ids)]

        # --- Checking to make sure parameters are valid
        if (symmetric is True) and (n_regions % 2 != 0):
            # symmetric model only valid if R = 2
            raise ValueError("Cannot run a symmetric model unless n_regions is even.")

        # Initialize sampling parameters
        # The global sampling iteration of the model
        self.iter = 0
        # Current random seed (is incremented after initialization and each sampling update)
        self.seed = 0

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
            # Default ROI (default covariance spatial region we regularize towards) (not in paper)
            "roi_size": roi_size,
            # Sample constant (# observations weighting sigma in direction of default covariance)
            # (not in paper)
            "dobs": dobs,
            # Use constrained symmetry on subregions? (only for n_regions = 2)
            "symmetric": symmetric,
            "seed_init": seed_init,  # Random seed for initializing model
        }

        # Add dictionaries for other model info
        self.data = {}
        self.topics = {}

        # Prepare data
        if isinstance(mask, str) and not op.isfile(mask):
            self.mask = get_template(mask, mask="brain")
        else:
            self.mask = load_img(mask)
        mask_ijk = np.vstack(np.where(_mask_img_to_bool(self.mask))).T
        self.data["mask_xyz"] = nib.affines.apply_affine(self.mask.affine, mask_ijk)

        # Extract document and word indices from count_df
        docidx_mapper = {id_: i for (i, id_) in enumerate(ids)}

        # Create docidx column
        count_df["docidx"] = count_df["id"].map(docidx_mapper)
        count_df = count_df.drop(columns=["id"])

        # Remove words not found anywhere in the corpus
        n_terms = len(count_df.columns) - 1  # number of columns minus one for docidx
        count_df = count_df.loc[:, (count_df != 0).any(axis=0)]
        n_terms_in_corpus = len(count_df.columns) - 1
        if n_terms_in_corpus != n_terms:
            LGR.warning(
                "Some terms in count_df do not appear in corpus. "
                f"Retaining {n_terms_in_corpus / n_terms} terms."
            )

        # Get updated vocabulary
        # List of word-strings (wtoken_word_idx values are indices into this list)
        vocabulary = count_df.columns.tolist()
        vocabulary.remove("docidx")
        self.vocabulary = vocabulary
        word_counts = count_df[self.vocabulary].to_numpy(dtype=np.int64, copy=False)
        word_docs = count_df["docidx"].to_numpy(dtype=np.int64, copy=False)
        nz_docs, nz_words = np.nonzero(word_counts)
        nz_counts = word_counts[nz_docs, nz_words].astype(np.int64, copy=False)

        # Expand the dense document-term matrix into token-level indices in NumPy
        # order, which already matches the old docidx-then-widx sorting.
        self.data["wtoken_doc_idx"] = np.repeat(word_docs[nz_docs], nz_counts)
        self.data["wtoken_word_idx"] = np.repeat(nz_words, nz_counts)

        # Import all peak-indices into lists
        coordinates_df["docidx"] = coordinates_df["id"].astype(str).map(docidx_mapper)
        coordinates_df = coordinates_df[["docidx", "x", "y", "z"]]
        coordinates_df["docidx"] = coordinates_df["docidx"].astype(int)

        # List of document-indices for peak-tokens x
        self.data["ptoken_doc_idx"] = coordinates_df["docidx"].to_numpy(dtype=np.int64, copy=False)
        self.data["ptoken_coords"] = coordinates_df[["x", "y", "z"]].to_numpy(
            dtype=float, copy=False
        )

        # Seed random number generator
        np.random.seed(self.params["seed_init"])

        # Preallocate vectors of assignment indices
        # word->topic assignments
        self.topics["wtoken_topic_idx"] = np.zeros(len(self.data["wtoken_word_idx"]), dtype=int)

        # Randomly initialize peak->topic assignments (y) ~ unif(1...n_topics)
        self.topics["peak_topic_idx"] = np.random.randint(
            self.params["n_topics"],
            size=(len(self.data["ptoken_doc_idx"])),
        )

        # peak->region assignments
        self.topics["peak_region_idx"] = np.zeros(len(self.data["ptoken_doc_idx"]), dtype=int)

        # Preallocate count matrices
        # Peaks: D x T: Number of peak-tokens assigned to each topic per document
        self.topics["n_peak_tokens_doc_by_topic"] = np.zeros(
            (len(self.ids), self.params["n_topics"]),
            dtype=int,
        )

        # Peaks: R x T: Number of peak-tokens assigned to each subregion per topic
        self.topics["n_peak_tokens_region_by_topic"] = np.zeros(
            (self.params["n_regions"], self.params["n_topics"]),
            dtype=int,
        )

        # Words: W x T: Number of word-tokens assigned to each topic per word-type
        self.topics["n_word_tokens_word_by_topic"] = np.zeros(
            (len(self.vocabulary), self.params["n_topics"]),
            dtype=int,
        )

        # Words: D x T: Number of word-tokens assigned to each topic per document
        self.topics["n_word_tokens_doc_by_topic"] = np.zeros(
            (len(self.ids), self.params["n_topics"]),
            dtype=int,
        )

        # Words: 1 x T: Total number of word-tokens assigned to each topic (across all docs)
        self.topics["total_n_word_tokens_by_topic"] = np.zeros(
            self.params["n_topics"],
            dtype=int,
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
            (
                self.params["n_topics"],
                self.params["n_regions"],
                1,
                self.data["ptoken_coords"].shape[1],  # generally 3
            ),
        )
        # (\sigma^{(t)}_r)
        self.topics["regions_sigma"] = np.zeros(
            (
                self.params["n_topics"],
                self.params["n_regions"],
                self.data["ptoken_coords"].shape[1],  # generally 3
                self.data["ptoken_coords"].shape[1],  # generally 3
            )
        )
        self.topics["regions_precision"] = np.zeros_like(self.topics["regions_sigma"])
        self.topics["regions_log_norm"] = np.zeros(
            (self.params["n_topics"], self.params["n_regions"])
        )

        # Initialize lists for tracking log-likelihood of data over sampling iterations
        self.loglikelihood = {
            "iter": [],  # Tracks iteration associated with the log-likelihood values
            "x": [],  # Tracks log-likelihood of peak tokens
            "w": [],  # Tracks log-likelihood of word tokens
            "total": [],  # Tracks log-likelihood of peak + word tokens
        }

        # Initialize peak->subregion assignments (r)
        if self.params["symmetric"]:
            # if symmetric model use deterministic assignment :
            #     if peak_val[0] > 0, r = 1, else r = 0
            # Namely, check whether x-coordinate is greater than zero.
            n_pairs = int(self.params["n_regions"] / 2)
            initial_assignments = np.random.randint(
                n_pairs,
                size=(len(self.data["ptoken_doc_idx"])),
            )
            signs = (self.data["ptoken_coords"][:, 0] > 0).astype(int)
            self.topics["peak_region_idx"][:] = (initial_assignments * 2) + signs
        else:
            # if asymmetric model, randomly sample r ~ unif(1...n_regions)
            self.topics["peak_region_idx"][:] = np.random.randint(
                self.params["n_regions"],
                size=(len(self.data["ptoken_doc_idx"])),
            )

        # Update model vectors and count matrices to reflect y and r assignments
        for i_ptoken, peak_doc in enumerate(self.data["ptoken_doc_idx"]):
            # peak-token -> topic assignment (y_i)
            peak_topic = self.topics["peak_topic_idx"][i_ptoken]
            # peak-token -> subregion assignment (c_i)
            peak_region = self.topics["peak_region_idx"][i_ptoken]
            # Increment document-by-topic counts
            self.topics["n_peak_tokens_doc_by_topic"][peak_doc, peak_topic] += 1
            # Increment region-by-topic
            self.topics["n_peak_tokens_region_by_topic"][peak_region, peak_topic] += 1

        # Randomly Initialize Word->Topic Assignments (z) for each word
        # token w_i: sample z_i proportional to p(topic|doc_i)
        peak_doc_by_topic = self.topics["n_peak_tokens_doc_by_topic"]
        word_topic_idx = self.topics["wtoken_topic_idx"]
        word_by_topic = self.topics["n_word_tokens_word_by_topic"]
        total_word_by_topic = self.topics["total_n_word_tokens_by_topic"]
        word_doc_by_topic = self.topics["n_word_tokens_doc_by_topic"]
        gamma = self.params["gamma"]
        _jit_initialize_word_topic_assignments(
            self.params["seed_init"],
            self.data["wtoken_doc_idx"],
            self.data["wtoken_word_idx"],
            peak_doc_by_topic,
            gamma,
            word_topic_idx,
            word_by_topic,
            total_word_by_topic,
            word_doc_by_topic,
        )

    def fit(self, n_iters=5000, loglikely_freq=10):
        """Run multiple iterations.

        .. versionchanged:: 0.0.8

            [ENH] Remove ``verbose`` parameter.

        Parameters
        ----------
        n_iters : :obj:`int`, default=5000
            Number of iterations to run. Default is 5000.
        loglikely_freq : :obj:`int`, optional
            The frequency with which log-likelihood is updated. Default value
            is 1 (log-likelihood is updated every iteration).
        """
        if self.iter == 0:
            # Get Initial Spatial Parameter Estimates
            self._update_regions()

            # Get Log-Likelihood of data for Initialized model and save to
            # variables tracking loglikely
            self.compute_log_likelihood()

        for i in range(self.iter, n_iters):
            self._update(loglikely_freq=loglikely_freq)

        # TODO: Handle this more elegantly
        (
            p_topic_g_voxel,
            p_voxel_g_topic,
            p_topic_g_word,
            p_word_g_topic,
        ) = self.get_probability_distributions()
        self.p_topic_g_voxel_ = p_topic_g_voxel
        self.p_voxel_g_topic_ = p_voxel_g_topic
        self.p_topic_g_word_ = p_topic_g_word
        self.p_word_g_topic_ = p_word_g_topic

    def _update(self, loglikely_freq=1):
        """Run a complete update cycle (sample z, sample y&r, update regions).

        .. versionchanged:: 0.0.8

            [ENH] Remove ``verbose`` parameter.

        Parameters
        ----------
        loglikely_freq : :obj:`int`, optional
            The frequency with which log-likelihood is updated. Default value
            is 1 (log-likelihood is updated every iteration).
        """
        self.iter += 1  # Update total iteration count

        LGR.debug(f"Iter {self.iter:04d}: Sampling z")
        self.seed += 1
        self._update_word_topic_assignments(self.seed)  # Update z-assignments

        LGR.debug(f"Iter {self.iter:04d}: Sampling y|r")
        self.seed += 1
        self._update_peak_assignments(self.seed)  # Update y-assignments

        LGR.debug(f"Iter {self.iter:04d}: Updating spatial params")
        self._update_regions()  # Update gaussian estimates for all subregions

        # Only update log-likelihood every 'loglikely_freq' iterations
        # (Computing log-likelihood isn't necessary and slows things down a bit)
        if self.iter % loglikely_freq == 0:
            LGR.debug(f"Iter {self.iter:04d}: Computing log-likelihood")

            # Compute log-likelihood of model in current state
            self.compute_log_likelihood()
            LGR.info(
                f"Iter {self.iter:04d} Log-likely: x = {self.loglikelihood['x'][-1]:10.1f}, "
                f"w = {self.loglikelihood['w'][-1]:10.1f}, "
                f"tot = {self.loglikelihood['total'][-1]:10.1f}"
            )

    def _spatial_pdf(self, points, topic_idx, region_idx):
        """Evaluate the cached Gaussian PDF for one topic-region pair."""
        return _jit_spatial_pdf(
            points,
            self.topics["regions_mu"][topic_idx, region_idx, 0, :],
            self.topics["regions_precision"][topic_idx, region_idx, ...],
            self.topics["regions_log_norm"][topic_idx, region_idx],
        )

    def _cache_region_pdf_params(self, topic_idx, region_idx, sigma):
        """Cache Gaussian parameters used repeatedly during sampling and decoding."""
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise np.linalg.LinAlgError("Region covariance must be positive definite.")
        self.topics["regions_precision"][topic_idx, region_idx, ...] = np.linalg.inv(sigma)
        self.topics["regions_log_norm"][topic_idx, region_idx] = -0.5 * (
            sigma.shape[0] * np.log(2 * np.pi) + logdet
        )

    def _compute_covariance_from_stats(self, sum_vector, cross_matrix, n_obs):
        """Compute sample covariance from sufficient statistics."""
        centered_cross = cross_matrix - np.outer(sum_vector, sum_vector) / n_obs
        return centered_cross / (n_obs - 1)

    def _update_word_topic_assignments(self, randseed):
        """Update wtoken_topic_idx (z) indicator variables assigning words->topics.

        Parameters
        ----------
        randseed : :obj:`int`
            Random seed for this iteration.
        """
        # --- Seed random number generator
        np.random.seed(randseed)

        word_topic_idx = self.topics["wtoken_topic_idx"]
        word_by_topic = self.topics["n_word_tokens_word_by_topic"]
        total_word_by_topic = self.topics["total_n_word_tokens_by_topic"]
        word_doc_by_topic = self.topics["n_word_tokens_doc_by_topic"]
        peak_doc_by_topic = self.topics["n_peak_tokens_doc_by_topic"]
        beta = self.params["beta"]
        beta_vocabulary = beta * len(self.vocabulary)
        gamma = self.params["gamma"]
        _jit_update_word_topic_assignments(
            randseed,
            self.data["wtoken_word_idx"],
            self.data["wtoken_doc_idx"],
            word_topic_idx,
            word_by_topic,
            total_word_by_topic,
            word_doc_by_topic,
            peak_doc_by_topic,
            beta,
            beta_vocabulary,
            gamma,
        )

    def _update_peak_assignments(self, randseed):
        """Update y / r indicator variables assigning peaks->topics/subregions.

        Parameters
        ----------
        randseed : :obj:`int`
            Random seed for this iteration.
        """
        # Seed random number generator
        np.random.seed(randseed)

        # Retrieve p(x|r,y) for all subregions
        peak_probs = self._get_peak_probs(self)
        region_by_topic = self.topics["n_peak_tokens_region_by_topic"]
        doc_by_topic = self.topics["n_peak_tokens_doc_by_topic"]
        word_doc_by_topic = self.topics["n_word_tokens_doc_by_topic"]
        peak_topic_idx = self.topics["peak_topic_idx"]
        peak_region_idx = self.topics["peak_region_idx"]
        delta = self.params["delta"]
        alpha = self.params["alpha"]
        gamma = self.params["gamma"]
        _jit_update_peak_assignments(
            randseed,
            self.data["ptoken_doc_idx"],
            peak_probs,
            peak_topic_idx,
            peak_region_idx,
            region_by_topic,
            doc_by_topic,
            word_doc_by_topic,
            delta,
            alpha,
            gamma,
        )

    def _update_regions(self):
        """Update spatial distribution parameters (Gaussians params for all subregions).

        Updates regions_mu and regions_sigma, indicating location and
        distribution of each subregion.
        """
        # Generate default ROI based on default_width
        default_roi = self.params["roi_size"] * np.eye(self.data["ptoken_coords"].shape[1])
        n_topics = self.params["n_topics"]
        n_regions = self.params["n_regions"]
        n_dim = self.data["ptoken_coords"].shape[1]
        region_counts = self.topics["n_peak_tokens_region_by_topic"]
        region_sums, region_cross = _jit_accumulate_region_stats(
            self.data["ptoken_coords"],
            self.topics["peak_topic_idx"],
            self.topics["peak_region_idx"],
            n_topics,
            n_regions,
        )
        zero_vec = np.zeros(n_dim)

        if self.params["symmetric"]:
            n_pairs = int(n_regions / 2)

            # With symmetric subregions, we jointly compute all estimates for subregions 1 & 2,
            # constraining the means to be symmetric w.r.t. the origin along x-dimension
            for i_topic in range(n_topics):
                for j_pair in range(n_pairs):
                    region1, region2 = j_pair * 2, (j_pair * 2) + 1

                    n_obs1 = region_counts[region1, i_topic]
                    n_obs2 = region_counts[region2, i_topic]
                    sum1 = region_sums[region1, i_topic, :]
                    sum2 = region_sums[region2, i_topic, :]
                    cross1 = region_cross[region1, i_topic, :, :]
                    cross2 = region_cross[region2, i_topic, :, :]
                    total_obs = n_obs1 + n_obs2

                    # Estimate means
                    # If there are no observations, we set mean equal to zeros, otherwise take MLE

                    # Estimate independent mean (centroid of peaks) for subregion 1
                    if n_obs1 == 0:
                        reg1_center_xyz = zero_vec
                    else:
                        reg1_center_xyz = sum1 / n_obs1

                    # Estimate independent mean (centroid of peaks) for subregion 2
                    if n_obs2 == 0:
                        reg2_center_xyz = zero_vec
                    else:
                        reg2_center_xyz = sum2 / n_obs2

                    # Estimate the weighted means of all dims, where for dim1 we
                    # compute the mean w.r.t. absolute distance from the origin
                    if total_obs == 0:
                        weighted_mean_dim1 = 0.0
                        weighted_mean_otherdims = zero_vec[1:]
                    else:
                        weighted_mean_dim1 = (
                            (-reg1_center_xyz[0] * n_obs1) + (reg2_center_xyz[0] * n_obs2)
                        ) / total_obs
                        weighted_mean_otherdims = (sum1[1:] + sum2[1:]) / total_obs

                    # Store weighted mean estimates
                    mu1 = np.zeros((1, n_dim))
                    mu2 = np.zeros((1, n_dim))
                    mu1[0, 0] = -weighted_mean_dim1
                    mu1[0, 1:] = weighted_mean_otherdims
                    mu2[0, 0] = weighted_mean_dim1
                    mu2[0, 1:] = weighted_mean_otherdims

                    # Store estimates in model object
                    self.topics["regions_mu"][i_topic, region1, ...] = mu1
                    self.topics["regions_mu"][i_topic, region2, ...] = mu2

                    # Estimate Covariances
                    # Covariances are estimated independently
                    # Covariance for subregion 1
                    if n_obs1 <= 1:
                        c_hat1 = default_roi
                    else:
                        c_hat1 = self._compute_covariance_from_stats(sum1, cross1, n_obs1)

                    # Covariance for subregion 2
                    if n_obs2 <= 1:
                        c_hat2 = default_roi
                    else:
                        c_hat2 = self._compute_covariance_from_stats(sum2, cross2, n_obs2)

                    # Regularize the covariances, using the ratio of observations to
                    # sample_constant
                    d_c_1 = (n_obs1) / (n_obs1 + self.params["dobs"])
                    d_c_2 = (n_obs2) / (n_obs2 + self.params["dobs"])
                    sigma1 = (d_c_1 * c_hat1) + ((1 - d_c_1) * default_roi)
                    sigma2 = (d_c_2 * c_hat2) + ((1 - d_c_2) * default_roi)

                    # Store estimates in model object
                    self.topics["regions_sigma"][i_topic, region1, ...] = sigma1
                    self.topics["regions_sigma"][i_topic, region2, ...] = sigma2
                    self._cache_region_pdf_params(i_topic, region1, sigma1)
                    self._cache_region_pdf_params(i_topic, region2, sigma2)
        else:
            # For each region, compute a mean and a regularized covariance matrix
            for i_topic in range(n_topics):
                for j_region in range(n_regions):
                    n_obs = region_counts[j_region, i_topic]
                    sum_vector = region_sums[j_region, i_topic, :]
                    cross_matrix = region_cross[j_region, i_topic, :, :]

                    # Estimate mean
                    # If there are no observations, we set mean equal to zeros, otherwise take MLE
                    if n_obs == 0:
                        mu = zero_vec
                    else:
                        mu = sum_vector / n_obs

                    # Estimate covariance
                    # if there are 1 or fewer observations, we set sigma_hat equal to default ROI,
                    # otherwise take MLE
                    if n_obs <= 1:
                        c_hat = default_roi
                    else:
                        c_hat = self._compute_covariance_from_stats(sum_vector, cross_matrix, n_obs)

                    # Regularize the covariance, using the ratio of observations
                    # to dobs (default constant # observations)
                    d_c = n_obs / (n_obs + self.params["dobs"])
                    sigma = (d_c * c_hat) + ((1 - d_c) * default_roi)

                    # Store estimates in model object
                    self.topics["regions_mu"][i_topic, j_region, ...] = mu
                    self.topics["regions_sigma"][i_topic, j_region, ...] = sigma
                    self._cache_region_pdf_params(i_topic, j_region, sigma)

    def compute_log_likelihood(self, model=None, update_vectors=True):
        """Compute log-likelihood of a model object given current model.

        Computes the log-likelihood of data in any model object (either train or test) given the
        posterior predictive distributions over peaks and word-types for the model,
        using the method described in :footcite:t:`newman2009distributed`.
        Note that this is not computing the joint log-likelihood of model parameters and data.

        Parameters
        ----------
        model : :obj:`~nimare.annotate.gclda.GCLDAModel`, optional
            The model for which log-likelihoods will be calculated.
            If not provided, log-likelihood will be calculated for the current model (self).
            Default is None.
        update_vectors : :obj:`bool`, optional
            Whether to update model's log-likelihood vectors or not.
            Default is True.

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
        .. footbibliography::
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
            x_loglikely += np.log(p_x)

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
            w_loglikely += np.log(p_wtoken)
        tot_loglikely = x_loglikely + w_loglikely

        # Update model log-likelihood history vector (if update_vectors == True)
        if update_vectors:
            self.loglikelihood["iter"].append(self.iter)
            self.loglikelihood["x"].append(x_loglikely)
            self.loglikelihood["w"].append(w_loglikely)
            self.loglikelihood["total"].append(tot_loglikely)

        # Return loglikely values (used when computing log-likelihood for a
        # model-object containing hold-out data)
        return (x_loglikely, w_loglikely, tot_loglikely)

    def _get_peak_probs(self, model):
        """Compute a matrix giving p(x|r,t).

        This uses all x values in a model object, and each topic's spatial parameters.

        Returns
        -------
        peak_probs : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            nPeaks x nTopics x nRegions matrix of probabilities, giving
            probability of sampling each peak (x) from all subregions.
        """
        peak_probs = _jit_get_peak_probs(
            model.data["ptoken_coords"],
            self.topics["regions_mu"][:, :, 0, :],
            self.topics["regions_precision"],
            self.topics["regions_log_norm"],
        )
        return peak_probs

    def get_probability_distributions(self):
        """Get conditional probability of selecting each voxel in the brain mask given each topic.

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
        mask_xyz = self.data["mask_xyz"]

        spatial_dists = _jit_get_spatial_dists(
            mask_xyz,
            self.topics["regions_mu"][:, :, 0, :],
            self.topics["regions_precision"],
            self.topics["regions_log_norm"],
        )
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
