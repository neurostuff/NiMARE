"""
Methods for decoding subsets of voxels (e.g., ROIs) or experiments (e.g., from
meta-analytic clustering on a database) into text.
"""
from ..due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Citation for GCLDA decoding.')
def gclda_decode_roi(model, roi, topic_priors=None, prior_weight=1.):
    """
    Perform image-to-text decoding for discrete image inputs (e.g., regions
    of interest, significant clusters).

    Parameters
    ----------
    model : :obj:`gclda.model.Model`
        Model object needed for decoding.
    roi : :obj:`nibabel.nifti.Nifti1Image` or :obj:`str`
        Binary image to decode into text. If string, path to a file with
        the binary image.
    topic_priors : :obj:`numpy.ndarray` of :obj:`float`, optional
        A 1d array of size (n_topics) with values for topic weighting.
        If None, no weighting is done. Default is None.
    prior_weight : :obj:`float`, optional
        The weight by which the prior will affect the decoding.
        Default is 1.

    Returns
    -------
    decoded_df : :obj:`pandas.DataFrame`
        A DataFrame with the word-tokens and their associated weights.
    topic_weights : :obj:`numpy.ndarray` of :obj:`float`
        The weights of the topics used in decoding.

    Notes
    -----
    ======================    ==============================================================
    Notation                  Meaning
    ======================    ==============================================================
    :math:`v`                 Voxel
    :math:`t`                 Topic
    :math:`w`                 Word type
    :math:`r`                 Region of interest (ROI)
    :math:`p(v|t)`            Probability of topic given voxel (``p_topic_g_voxel``)
    :math:`\\tau_{t}`          Topic weight vector (``topic_weights``)
    :math:`p(w|t)`            Probability of word type given topic (``p_word_g_topic``)
    ======================    ==============================================================

    1.  Compute
        :math:`p(v|t)`.
            - From :obj:`gclda.model.Model.get_spatial_probs()`
    2.  Compute topic weight vector (:math:`\\tau_{t}`) by adding across voxels
        within ROI.
            - :math:`\\tau_{t} = \sum_{i} {p(t|v_{i})}`
    3.  Multiply :math:`\\tau_{t}` by
        :math:`p(w|t)`.
            - :math:`p(w|r) \propto \\tau_{t} \cdot p(w|t)`
    4.  The resulting vector (``word_weights``) reflects arbitrarily scaled
        term weights for the ROI.
    """
    pass


@due.dcite(Doi('10.1007/s00429-013-0698-0'),
           description='Citation for BrainMap-style decoding.')
def brainmap_decode(coordinates, annotations, ids, ids2=None, features=None,
                    frequency_threshold=0.001, u=0.05, correction='fdr_bh'):
    """
    Perform image-to-text decoding for discrete image inputs (e.g., regions
    of interest, significant clusters) according to the BrainMap method.
    """
    pass


@due.dcite(Doi('10.1038/nmeth.1635'), description='Introduces Neurosynth.')
def neurosynth_decode(coordinates, annotations, ids, ids2=None, features=None,
                      frequency_threshold=0.001, prior=0.5, u=0.05,
                      correction='fdr_bh'):
    """
    Perform discrete functional decoding according to Neurosynth's
    meta-analytic method. This does not employ correlations between
    unthresholded maps, which are the method of choice for decoding within
    Neurosynth and Neurovault.
    Metadata (i.e., feature labels) for studies within the selected sample
    (`ids`) are compared to the unselected studies remaining in the database
    (`dataset`).
    """
    pass
