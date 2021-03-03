"""Methods for encoding text into brain maps."""
import numpy as np
from nilearn.masking import unmask
from sklearn.feature_extraction.text import CountVectorizer

from .. import references
from ..due import due
from .utils import weight_priors


@due.dcite(references.TEXT2BRAIN, description="Introduced text2brain models for annotation.")
def text2brain():
    """
    Perform text-to-image encoding with the text2brain model.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    * Dockès, Jérôme, et al. "Text to brain: predicting the spatial
      distribution of neuroimaging observations from text reports."
      International Conference on Medical Image Computing and
      Computer-Assisted Intervention. Springer, Cham, 2018.
      https://doi.org/10.1007/978-3-030-00931-1_67
    """
    pass


@due.dcite(references.GCLDA_DECODING, description="Citation for GCLDA encoding.")
def gclda_encode(model, text, out_file=None, topic_priors=None, prior_weight=1.0):
    r"""Perform text-to-image encoding according to the method described in Rubin et al. (2017).

    Parameters
    ----------
    model : :obj:`nimare.annotate.topic.GCLDAModel`
        Model object needed for decoding.
    text : :obj:`str` or :obj:`list`
        Text to encode into an image.
    out_file : :obj:`str`, optional
        If not None, writes the encoded image to a file.
    topic_priors : :obj:`numpy.ndarray` of :obj:`float`, optional
        A 1d array of size (n_topics) with values for topic weighting.
        If None, no weighting is done. Default is None.
    prior_weight : :obj:`float`, optional
        The weight by which the prior will affect the encoding.
        Default is 1.

    Returns
    -------
    img : :obj:`nibabel.nifti1.Nifti1Image`
        The encoded image.
    topic_weights : :obj:`numpy.ndarray` of :obj:`float`
        The weights of the topics used in encoding.

    Notes
    -----
    ======================    ==============================================================
    Notation                  Meaning
    ======================    ==============================================================
    :math:`v`                 Voxel
    :math:`t`                 Topic
    :math:`w`                 Word type
    :math:`h`                 Input text
    :math:`p(v|t)`            Probability of voxel given topic (``p_voxel_g_topic_``)
    :math:`\\tau_{t}`          Topic weight vector (``topic_weights``)
    :math:`p(w|t)`            Probability of word type given topic (``p_word_g_topic``)
    :math:`\omega`            1d array from input image (``input_values``)
    ======================    ==============================================================

    1.  Compute :math:`p(v|t)`
        (``p_voxel_g_topic``).
            - From :func:`gclda.model.Model.get_spatial_probs()`
    2.  Compute :math:`p(t|w)`
        (``p_topic_g_word``).
    3.  Vectorize input text according to model vocabulary.
    4.  Reduce :math:`p(t|w)` to only include word types in input text.
    5.  Compute :math:`p(t|h)` (``p_topic_g_text``) by multiplying
        :math:`p(t|w)` by word counts for input text.
    6.  Sum topic weights (:math:`\\tau_{t}`) across
        words.
            - :math:`\\tau_{t} = \sum_{i}{p(t|h_{i})}`
    7.  Compute voxel
        weights.
            - :math:`p(v|h) \propto p(v|t) \cdot \\tau_{t}`
    8.  The resulting array (``voxel_weights``) reflects arbitrarily scaled
        voxel weights for the input text.
    9.  Unmask and reshape ``voxel_weights`` into brain image.

    See Also
    --------
    :class:`nimare.annotate.gclda.GCLDAModel`
    :func:`nimare.decode.continuous.gclda_decode_map`
    :func:`nimare.decode.discrete.gclda_decode_roi`

    References
    ----------
    * Rubin, Timothy N., et al. "Decoding brain activity using a
      large-scale probabilistic functional-anatomical atlas of human
      cognition." PLoS computational biology 13.10 (2017): e1005649.
      https://doi.org/10.1371/journal.pcbi.1005649
    """
    if isinstance(text, list):
        text = " ".join(text)

    # Assume that words in vocabulary are underscore-separated.
    # Convert to space-separation for vectorization of input string.
    vocabulary = [term.replace("_", " ") for term in model.vocabulary]
    max_len = max([len(term.split(" ")) for term in vocabulary])
    vectorizer = CountVectorizer(vocabulary=model.vocabulary, ngram_range=(1, max_len))
    word_counts = np.squeeze(vectorizer.fit_transform([text]).toarray())
    keep_idx = np.where(word_counts > 0)[0]
    text_counts = word_counts[keep_idx]

    # n_topics_per_word_token = np.sum(model.n_word_tokens_word_by_topic, axis=1)
    # p_topic_g_word = model.n_word_tokens_word_by_topic / n_topics_per_word_token[:, None]
    # p_topic_g_word = np.nan_to_num(p_topic_g_word, 0)
    p_topic_g_text = model.p_topic_g_word_[keep_idx]  # p(T|W) for words in text only
    prod = p_topic_g_text * text_counts[:, None]  # Multiply p(T|W) by words in text
    topic_weights = np.sum(prod, axis=0)  # Sum across words
    if topic_priors is not None:
        weighted_priors = weight_priors(topic_priors, prior_weight)
        topic_weights *= weighted_priors

    voxel_weights = np.dot(model.p_voxel_g_topic_, topic_weights)
    img = unmask(voxel_weights, model.mask)

    if out_file is not None:
        img.to_filename(out_file)
    return img, topic_weights
