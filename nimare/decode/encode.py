"""
Methods for encoding text into brain maps.
"""
from ...due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Citation for GCLDA encoding.')
def encode_gclda(model, text, out_file=None, topic_priors=None,
                 prior_weight=1.):
    """
    Perform text-to-image encoding.

    Parameters
    ----------
    model : :obj:`gclda.model.Model`
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
    img : :obj:`nibabel.nifti.Nifti1Image`
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
    :math:`p(v|t)`            Probability of topic given voxel (``p_topic_g_voxel``)
    :math:`\\tau_{t}`          Topic weight vector (``topic_weights``)
    :math:`p(w|t)`            Probability of word type given topic (``p_word_g_topic``)
    :math:`\omega`            1d array from input image (``input_values``)
    ======================    ==============================================================

    1.  Compute :math:`p(v|t)`
        (``p_voxel_g_topic``).
            - From :obj:`gclda.model.Model.get_spatial_probs()`
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
    """
    pass
