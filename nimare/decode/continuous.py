"""
Methods for decoding unthresholded brain maps into text.
"""
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask

from .utils import weight_priors
from ..meta.cbma import MKDAChi2
from ..due import due
from .. import references


@due.dcite(references.GCLDA_DECODING,
           description='Describes decoding methods using GC-LDA.')
def gclda_decode_map(model, image, topic_priors=None, prior_weight=1):
    r"""
    Perform image-to-text decoding for continuous inputs (e.g.,
    unthresholded statistical maps), according to the method described in [1]_.

    Parameters
    ----------
    model : :obj:`nimare.annotate.topic.GCLDAModel`
        Model object needed for decoding.
    image : :obj:`nibabel.nifti1.Nifti1Image` or :obj:`str`
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

    References
    ----------
    .. [1] Rubin, Timothy N., et al. "Decoding brain activity using a
        large-scale probabilistic functional-anatomical atlas of human
        cognition." PLoS computational biology 13.10 (2017): e1005649.
        https://doi.org/10.1371/journal.pcbi.1005649
    """
    if isinstance(image, str):
        image = nib.load(image)
    elif not isinstance(image, nib.Nifti1Image):
        raise IOError('Input image must be either a nifti image '
                      '(nibabel.Nifti1Image) or a path to one.')

    # Load image file and get voxel values
    input_values = apply_mask(image, model.mask)
    topic_weights = np.squeeze(np.dot(model.p_topic_g_voxel.T,
                                      input_values[:, None]))
    if topic_priors is not None:
        weighted_priors = weight_priors(topic_priors, prior_weight)
        topic_weights *= weighted_priors

    # Multiply topic_weights by topic-by-word matrix (p_word_g_topic).
    # n_word_tokens_per_topic = np.sum(model.n_word_tokens_word_by_topic, axis=0)
    # p_word_g_topic = model.n_word_tokens_word_by_topic / n_word_tokens_per_topic[None, :]
    # p_word_g_topic = np.nan_to_num(p_word_g_topic, 0)
    word_weights = np.dot(model.p_word_g_topic, topic_weights)

    decoded_df = pd.DataFrame(index=model.vocabulary,
                              columns=['Weight'], data=word_weights)
    decoded_df.index.name = 'Term'
    return decoded_df, topic_weights


@due.dcite(references.NEUROSYNTH, description='Introduces Neurosynth.')
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

    Returns
    -------
    out_df : :obj:`pandas.DataFrame`
        A DataFrame with two columns: 'feature' (label) and 'r' (correlation
        coefficient). There will be one row for each feature.
    """
    # Check that input image is compatible with dataset
    assert np.array_equal(img.affine, dataset.mask.affine)

    # Load input data
    input_data = apply_mask(img, dataset.mask)

    if meta_estimator is None:
        meta_estimator = MKDAChi2(dataset)

    if features is None:
        features = dataset.annotations.columns.values

    out_df = pd.DataFrame(index=features, columns=['r'],
                          data=np.zeros(len(features)))
    out_df.index.name = 'feature'

    for feature in features:
        # TODO: Search for !feature to get ids2, if possible. Will compare
        # between label+ and label- without analyzing unlabeled studies.
        ids = dataset.get(features=[feature],
                          frequency_threshold=frequency_threshold)
        meta_estimator.fit(ids, corr='FDR')
        feature_data = apply_mask(meta_estimator.results[target_image],
                                  dataset.mask)
        corr = np.corrcoef(feature_data, input_data)[0, 1]
        out_df.loc[feature, 'r'] = corr

    return out_df


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
    # Check that input image is compatible with dataset
    assert np.array_equal(img.affine, dataset.mask.affine)

    # Load input data
    input_data = apply_mask(img, dataset.mask)

    if features is None:
        features = dataset.annotations.columns.values

    out_df = pd.DataFrame(index=features, columns=['mean', 'std'],
                          data=np.zeros(len(features), 2))
    out_df.index.name = 'feature'

    for feature in features:
        test_imgs = dataset.get_images(features=[feature],
                                       frequency_threshold=frequency_threshold,
                                       image_types=[target_image])
        feature_z_dist = np.zeros(len(test_imgs))
        for i, test_img in enumerate(test_imgs):
            feature_data = apply_mask(test_img, dataset.mask)
            corr = np.corrcoef(feature_data, input_data)[0, 1]
            feature_z_dist[i] = np.arctanh(corr)  # transform to z for normality
        out_df.loc[feature, 'mean'] = np.mean(feature_z_dist)
        out_df.loc[feature, 'std'] = np.std(feature_z_dist)

    return out_df
