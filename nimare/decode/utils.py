"""Utility functions for decoding/encoding."""
import numpy as np


def weight_priors(topic_priors, prior_weight):
    """
    Combine topic priors with prior weight.

    Parameters
    ----------
    topic_priors : array-like
        The prior weights for topics (n_topics-long array). Scale may be
        arbitrary, as the array will be normalized.
    prior_weight : :obj:`float`
        Scalar by which to weight priors.

    Returns
    -------
    weighted_priors : :obj:`numpy.ndarray`
        Updated prior weights for topics.
    """
    if not isinstance(prior_weight, (float, int)):
        raise IOError("Input prior_weight must be a float in range (0, 1)")
    elif not 0.0 <= prior_weight <= 1:
        raise ValueError("Input prior_weight must be in range (0, 1)")

    # Enforce compatible types
    topic_priors = topic_priors.astype(float)
    prior_weight = float(prior_weight)

    # Normalize priors
    topic_priors /= np.sum(topic_priors)

    # Weight priors
    topic_priors *= prior_weight

    # Create uniform distribution to combine with priors
    uniform = np.ones(topic_priors.shape)
    uniform /= np.sum(uniform)
    uniform *= 1 - prior_weight

    # Weight priors with uniform base
    weighted_priors = topic_priors + uniform
    return weighted_priors
