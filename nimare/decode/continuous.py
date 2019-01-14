"""
Methods for decoding unthresholded brain maps into text.
"""
from ..base import Decoder
from ..due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Describes decoding methods using GC-LDA.')
def gclda_decode_continuous(model, roi, topic_priors=None, prior_weight=1):
    """
    Perform image-to-text decoding for continuous inputs (e.g.,
    unthresholded statistical maps).

    Parameters
    ----------
    model : :obj:`gclda.model.Model`
        Model object needed for decoding.
    image : :obj:`nibabel.nifti.Nifti1Image` or :obj:`str`
        Whole-brain image to decode into text. Must be in same space as
        model and dataset. Model's template available in
        `model.dataset.mask_img`.
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
    :math:`i`                 Input image
    :math:`p(v|t)`            Probability of topic given voxel (``p_topic_g_voxel``)
    :math:`\\tau_{t}`          Topic weight vector (``topic_weights``)
    :math:`p(w|t)`            Probability of word type given topic (``p_word_g_topic``)
    :math:`\omega`            1d array from input image (``input_values``)
    ======================    ==============================================================

    1.  Compute :math:`p(t|v)`
        (``p_topic_g_voxel``).
            - From :obj:`gclda.model.Model.get_spatial_probs()`
    2.  Squeeze input image to 1d array :math:`\omega` (``input_values``).
    3.  Compute topic weight vector (:math:`\\tau_{t}`) by multiplying
        :math:`p(t|v)` by input image.
            - :math:`\\tau_{t} = p(t|v) \cdot \omega`
    4.  Multiply :math:`\\tau_{t}` by
        :math:`p(w|t)`.
            - :math:`p(w|i) \propto \\tau_{t} \cdot p(w|t)`
    5.  The resulting vector (``word_weights``) reflects arbitrarily scaled
        term weights for the input image.
    """
    pass


@due.dcite(Doi('10.1038/nmeth.1635'),
           description='Introduces Neurosynth.')
def corr_decode(img, dataset, features=None, frequency_threshold=0.001,
                meta_estimator=None, target_image='specificity_z'):
    """
    Parameters
    ----------
    img : :obj:`nibabel.Nifti1.Nifti1Image`
        Input image to decode. Must have same affine/dimensions as dataset
        mask.
    dataset
        A dataset with coordinates.
    features : :obj:`list`, optional
        List of features in dataset annotations to use for decoding.
        Default is None, which uses all features available.
    frequency_threshold : :obj:`float`, optional
        Threshold to apply to dataset annotations. Values greater than or
        equal to the threshold as assigned as label+, while values below
        the threshold are considered label-. Default is 0.001.
    meta_estimator : initialized :obj:`nimare.meta.cbma.base.CBMAEstimator`, optional
        Defaults to MKDAChi2.
    target_image : :obj:`str`, optional
        Image from ``meta_estimator``'s results to use for decoding.
        Dependent on estimator.
    """
    pass


def corr_dist_decode(img, dataset, features=None, frequency_threshold=0.001,
                     target_image='z'):
    """
    Builds feature-specific distributions of correlations with input image
    for image-based meta-analytic functional decoding.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1.Nifti1Image`
        Input image to decode. Must have same affine/dimensions as dataset
        mask.
    dataset
        A dataset with images.
    features : :obj:`list`, optional
        List of features in dataset annotations to use for decoding.
        Default is None, which uses all features available.
    frequency_threshold : :obj:`float`, optional
        Threshold to apply to dataset annotations. Values greater than or
        equal to the threshold as assigned as label+, while values below
        the threshold are considered label-. Default is 0.001.
    target_image : {'z', 'con'}, optional
        Image type from database to use for decoding.

    Returns
    -------
    out_df : :obj:`pandas.DataFrame`
        DataFrame with a row for each feature used for decoding and two
        columns: mean and std. Values describe the distributions of
        correlation coefficients (in terms of Fisher-transformed z-values).
    """
    pass
