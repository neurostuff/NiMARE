"""Methods for decoding unthresholded brain maps into text."""
import inspect
import logging

import numpy as np
import pandas as pd
from nilearn._utils import load_niimg
from nilearn.masking import apply_mask

from .. import references
from ..base import Decoder
from ..due import due
from ..meta.cbma.base import CBMAEstimator
from ..meta.cbma.mkda import MKDAChi2
from ..stats import pearson
from ..utils import check_type
from .utils import weight_priors

LGR = logging.getLogger(__name__)


@due.dcite(references.GCLDA_DECODING, description="Describes decoding methods using GC-LDA.")
def gclda_decode_map(model, image, topic_priors=None, prior_weight=1):
    r"""Perform image-to-text decoding for continuous inputs using method from Rubin et al. (2017).

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
            - From :func:`gclda.model.Model.get_spatial_probs()`
    2.  Squeeze input image to 1d array :math:`\omega` (``input_values``).
    3.  Compute topic weight vector (:math:`\\tau_{t}`) by multiplying
        :math:`p(t|v)` by input image.
            - :math:`\\tau_{t} = p(t|v) \cdot \omega`
    4.  Multiply :math:`\\tau_{t}` by
        :math:`p(w|t)`.
            - :math:`p(w|i) \propto \\tau_{t} \cdot p(w|t)`
    5.  The resulting vector (``word_weights``) reflects arbitrarily scaled
        term weights for the input image.

    See Also
    --------
    :class:`nimare.annotate.gclda.GCLDAModel`
    :func:`nimare.decode.discrete.gclda_decode_roi`
    :func:`nimare.decode.encode.gclda_encode`

    References
    ----------
    * Rubin, Timothy N., et al. "Decoding brain activity using a
      large-scale probabilistic functional-anatomical atlas of human
      cognition." PLoS computational biology 13.10 (2017): e1005649.
      https://doi.org/10.1371/journal.pcbi.1005649
    """
    image = load_niimg(image)

    # Load image file and get voxel values
    input_values = apply_mask(image, model.mask)
    topic_weights = np.squeeze(np.dot(model.p_topic_g_voxel_.T, input_values[:, None]))
    if topic_priors is not None:
        weighted_priors = weight_priors(topic_priors, prior_weight)
        topic_weights *= weighted_priors

    # Multiply topic_weights by topic-by-word matrix (p_word_g_topic).
    # n_word_tokens_per_topic = np.sum(model.n_word_tokens_word_by_topic, axis=0)
    # p_word_g_topic = model.n_word_tokens_word_by_topic / n_word_tokens_per_topic[None, :]
    # p_word_g_topic = np.nan_to_num(p_word_g_topic, 0)
    word_weights = np.dot(model.p_word_g_topic_, topic_weights)

    decoded_df = pd.DataFrame(index=model.vocabulary, columns=["Weight"], data=word_weights)
    decoded_df.index.name = "Term"
    return decoded_df, topic_weights


@due.dcite(references.NEUROSYNTH, description="Introduces Neurosynth.")
class CorrelationDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps.

    Parameters
    ----------
    feature_group : :obj:`str`
        Feature group
    features : :obj:`list`
        Features
    frequency_threshold : :obj:`float`
        Frequency threshold
    meta_estimator : :class:`nimare.base.CBMAEstimator`, optional
        Meta-analysis estimator. Default is :class:`nimare.meta.mkda.MKDAChi2`.
    target_image : :obj:`str`
        Name of meta-analysis results image to use for decoding.

    Warning
    -------
    Coefficients from correlating two maps have very large degrees of freedom,
    so almost all results will be statistically significant. Do not attempt to
    evaluate results based on significance.
    """

    def __init__(
        self,
        feature_group=None,
        features=None,
        frequency_threshold=0.001,
        meta_estimator=None,
        target_image="z_desc-specificity",
    ):

        if meta_estimator is None:
            meta_estimator = MKDAChi2(low_memory=True, kernel__low_memory=True)
        else:
            meta_estimator = check_type(meta_estimator, CBMAEstimator)

        self.feature_group = feature_group
        self.features = features
        self.frequency_threshold = frequency_threshold
        self.meta_estimator = meta_estimator
        self.target_image = target_image
        self.results = None

    def _fit(self, dataset):
        """Generate feature-specific meta-analytic maps for dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset for which to run meta-analyses to generate maps.

        Attributes
        ----------
        masker : :class:`nilearn.input_data.NiftiMasker` or similar
            Masker from dataset
        features_ : :obj:`list`
            Reduced list of features
        images_ : array_like
            Masked meta-analytic maps
        """
        self.masker = dataset.masker

        # Pre-generate MA maps to speed things up
        kernel_transformer = self.meta_estimator.kernel_transformer
        dataset = kernel_transformer.transform(dataset, return_type="dataset")

        for i, feature in enumerate(self.features_):
            feature_ids = dataset.get_studies_by_label(
                labels=[feature], label_threshold=self.frequency_threshold
            )
            feature_dset = dataset.slice(feature_ids)
            # This seems like a somewhat inelegant solution
            # Check if the meta method is a pairwise estimator
            if "dataset2" in inspect.getfullargspec(self.meta_estimator.fit).args:
                nonfeature_ids = sorted(list(set(dataset.ids) - set(feature_ids)))
                nonfeature_dset = dataset.slice(nonfeature_ids)
                self.meta_estimator.fit(feature_dset, nonfeature_dset)
            else:
                self.meta_estimator.fit(feature_dset)

            feature_data = self.meta_estimator.results.get_map(
                self.target_image, return_type="array"
            )
            if i == 0:
                images_ = np.zeros((len(self.features_), len(feature_data)))
            images_[i, :] = feature_data
        self.images_ = images_

    def transform(self, img):
        """Correlate target image with each feature-specific meta-analytic map.

        Parameters
        ----------
        img : :obj:`nibabel.nifti1.Nifti1Image`
            Image to decode. Must be in same space as ``dataset``.

        Returns
        -------
        out_df : :obj:`pandas.DataFrame`
            DataFrame with one row for each feature, an index named "feature", and one column: "r".
        """
        img_vec = self.masker.transform(img)
        corrs = pearson(img_vec, self.images_)
        out_df = pd.DataFrame(index=self.features_, columns=["r"], data=corrs)
        out_df.index.name = "feature"
        self.results = out_df
        return out_df


class CorrelationDistributionDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with study-wise images.

    Parameters
    ----------
    feature_group : :obj:`str`, optional
        Feature group. Default is None, which uses all available features.
    features : :obj:`list`, optional
        Features. Default is None, which uses all available features.
    frequency_threshold : :obj:`float`, optional
        Frequency threshold. Default is 0.001.
    target_image : {'z', 'con'}, optional
        Name of meta-analysis results image to use for decoding. Default is 'z'.

    Warning
    -------
    Coefficients from correlating two maps have very large degrees of freedom,
    so almost all results will be statistically significant. Do not attempt to
    evaluate results based on significance.
    """

    def __init__(
        self, feature_group=None, features=None, frequency_threshold=0.001, target_image="z"
    ):
        self.feature_group = feature_group
        self.features = features
        self.frequency_threshold = frequency_threshold
        self.target_image = target_image
        self.results = None

    def _fit(self, dataset):
        """Collect sets of maps from the Dataset corresponding to each requested feature.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset for which to run meta-analyses to generate maps.

        Attributes
        ----------
        masker : :class:`nilearn.input_data.NiftiMasker` or similar
            Masker from dataset
        features_ : :obj:`list`
            Reduced list of features
        images_ : array_like
            Masked meta-analytic maps
        """
        self.masker = dataset.masker

        images_ = {}
        for feature in self.features_:
            feature_ids = dataset.get_studies_by_label(
                labels=[feature], label_threshold=self.frequency_threshold
            )
            test_imgs = dataset.get_images(ids=feature_ids, imtype=self.target_image)
            test_imgs = list(filter(None, test_imgs))
            if len(test_imgs):
                feature_arr = self.masker.transform(test_imgs)
                images_[feature] = feature_arr
            else:
                LGR.info('Skipping feature "{}". No images found.'.format(feature))
        # reduce features again
        self.features_ = [f for f in self.features_ if f in images_.keys()]
        self.images_ = images_

    def transform(self, img):
        """Correlate target image with each map associated with each feature.

        Parameters
        ----------
        img : :obj:`nibabel.nifti1.Nifti1Image`
            Image to decode. Must be in same space as ``dataset``.

        Returns
        -------
        out_df : :obj:`pandas.DataFrame`
            DataFrame with one row for each feature, an index named "feature", and two columns:
            "mean" and "std".
        """
        img_vec = self.masker.transform(img)
        out_df = pd.DataFrame(
            index=self.features_, columns=["mean", "std"], data=np.zeros((len(self.features_), 2))
        )
        out_df.index.name = "feature"
        for feature, feature_arr in self.images_.items():
            corrs = pearson(img_vec, feature_arr)
            corrs_z = np.arctanh(corrs)
            out_df.loc[feature, "mean"] = np.mean(corrs_z)
            out_df.loc[feature, "std"] = np.std(corrs_z)
        self.results = out_df
        return out_df
