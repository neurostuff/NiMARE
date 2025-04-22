"""Methods for decoding unthresholded brain maps into text."""

import inspect
import logging
import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn._utils import load_niimg
from nilearn.masking import apply_mask
from tqdm.auto import tqdm

from nimare.decode.base import Decoder
from nimare.decode.utils import weight_priors
from nimare.meta.cbma.base import CBMAEstimator
from nimare.meta.cbma.mkda import MKDAChi2
from nimare.results import MetaResult
from nimare.stats import pearson
from nimare.utils import _check_ncores, _check_type, _safe_transform, get_masker

LGR = logging.getLogger(__name__)


def gclda_decode_map(model, image, topic_priors=None, prior_weight=1):
    r"""Perform image-to-text decoding for continuous inputs using method from Rubin et al. (2017).

    The method used in this function was originally described in :footcite:t:`rubin2017decoding`.

    Parameters
    ----------
    model : :obj:`~nimare.annotate.gclda.GCLDAModel`
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

    1.  Compute :math:`p(t|v)` (``p_topic_g_voxel``).

        - From :func:`gclda.model.Model.get_spatial_probs()`

    2.  Squeeze input image to 1d array :math:`\omega` (``input_values``).
    3.  Compute topic weight vector (:math:`\\tau_{t}`) by multiplying :math:`p(t|v)` by input
        image.

            - :math:`\\tau_{t} = p(t|v) \cdot \omega`

    4.  Multiply :math:`\\tau_{t}` by :math:`p(w|t)`.

            - :math:`p(w|i) \propto \\tau_{t} \cdot p(w|t)`

    5.  The resulting vector (``word_weights``) reflects arbitrarily scaled term weights for the
        input image.

    See Also
    --------
    :class:`~nimare.annotate.gclda.GCLDAModel`
    :func:`~nimare.decode.discrete.gclda_decode_roi`
    :func:`~nimare.decode.encode.gclda_encode`

    References
    ----------
    .. footbibliography::
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


class CorrelationDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps.

    .. versionchanged:: 0.1.0

        * New method: `load_imgs`. Load pre-generated meta-analytic maps for decoding.

        * New attribute: `results_`. MetaResult object containing masker, meta-analytic maps,
          and tables. This attribute replaces `masker`, `features_`, and `images_`.

    .. versionchanged:: 0.0.13

        * New parameter: `n_cores`. Number of cores to use for parallelization.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays.

    Parameters
    ----------
    feature_group : :obj:`str`, optional
        Feature group
    features : :obj:`list`, optional
        Features
    frequency_threshold : :obj:`float`, optional
        Frequency threshold
    meta_estimator : :class:`~nimare.base.CBMAEstimator`, optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.mkda.MKDAChi2`.
    target_image : :obj:`str`, optional
        Name of meta-analysis results image to use for decoding.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Warnings
    --------
    Coefficients from correlating two maps have very large degrees of freedom,
    so almost all results will be statistically significant. Do not attempt to
    evaluate results based on significance.
    """

    _required_inputs = {
        "coordinates": ("coordinates", None),
        "annotations": ("annotations", None),
    }

    def __init__(
        self,
        feature_group=None,
        features=None,
        frequency_threshold=0.001,
        meta_estimator=None,
        target_image="z_desc-association",
        n_cores=1,
    ):
        meta_estimator = (
            MKDAChi2() if meta_estimator is None else _check_type(meta_estimator, CBMAEstimator)
        )

        self.feature_group = feature_group
        self.features = features
        self.frequency_threshold = frequency_threshold
        self.meta_estimator = meta_estimator
        self.target_image = target_image
        self.n_cores = _check_ncores(n_cores)

    def _fit(self, dataset):
        """Generate feature-specific meta-analytic maps for dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset for which to run meta-analyses to generate maps.

        Attributes
        ----------
        results_ : :obj:`~nimare.results.MetaResult`
            MetaResult with meta-analytic maps and masker added.
        """
        n_features = len(self.features_)
        maps = {
            r: v
            for r, v in tqdm(
                Parallel(return_as="generator", n_jobs=self.n_cores)(
                    delayed(self._run_fit)(feature, dataset) for feature in self.features_
                ),
                total=n_features,
            )
        }

        self.results_ = MetaResult(self, mask=dataset.masker, maps=maps)

    def _run_fit(self, feature, dataset):
        feature_ids = dataset.get_studies_by_label(
            labels=[feature],
            label_threshold=self.frequency_threshold,
        )
        # Limit selected studies to studies with valid data
        feature_ids = sorted(list(set(feature_ids).intersection(self.inputs_["id"])))

        # Create the reduced Dataset
        feature_dset = dataset.slice(feature_ids)

        # Check if the meta method is a pairwise estimator
        # This seems like a somewhat inelegant solution
        if "dataset2" in inspect.getfullargspec(self.meta_estimator.fit).args:
            nonfeature_ids = sorted(list(set(self.inputs_["id"]) - set(feature_ids)))
            nonfeature_dset = dataset.slice(nonfeature_ids)
            meta_results = self.meta_estimator.fit(feature_dset, nonfeature_dset)
        else:
            meta_results = self.meta_estimator.fit(feature_dset)

        feature_data = meta_results.get_map(
            self.target_image,
            return_type="array",
        )

        return feature, feature_data

    def load_imgs(self, features_imgs, mask=None):
        """Load pregenerated maps from disk.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        features_imgs : :obj:`dict`, or str
            Dictionary with feature names as keys and paths to images as values.
            If a string is provided, it is assumed to be a path to a folder with NIfTI images,
            where the file's name (without the extension .nii.gz) will be considered as the
            feature name by the decoder.
        mask : str, :class:`nibabel.nifti1.Nifti1Image`, or any nilearn Masker
            Mask to apply to pre-generated maps.

        Attributes
        ----------
        results_ : :obj:`~nimare.results.MetaResult`
            MetaResult with meta-analytic maps and masker added.
        """
        if isinstance(features_imgs, dict):
            feature_imgs_dict = features_imgs
        elif isinstance(features_imgs, str):
            img_paths = sorted(glob(os.path.join(features_imgs, "*.nii*")))
            img_names = [os.path.basename(img).split(os.extsep)[0] for img in img_paths]
            feature_imgs_dict = dict(zip(img_names, img_paths))
        else:
            raise ValueError(
                f'"feature_imgs" must be a dictionary or a string, not a {type(features_imgs)}.'
            )

        # Replace attributes of initialized class self with Nones, so that default values are not
        # confused with the parameters used before to generate the maps that are read from disk.
        for attr in self.__dict__:
            setattr(self, attr, None)

        if mask is not None:
            mask = get_masker(mask)
        else:
            raise ValueError("A mask must be provided.")
        self.masker = mask

        # Load pre-generated maps
        features, images = ([], [])
        for feature, img_path in feature_imgs_dict.items():
            img = nib.load(img_path)
            features.append(feature)
            images.append(np.squeeze(self.masker.transform(img)))

        maps = {feature: image for feature, image in zip(features, images)}
        self.results_ = MetaResult(self, mask=self.masker, maps=maps)

    def transform(self, img):
        """Correlate target image with each feature-specific meta-analytic map.

        Parameters
        ----------
        img : :obj:`~nibabel.nifti1.Nifti1Image`
            Image to decode. Must be in same space as ``dataset``.

        Returns
        -------
        out_df : :obj:`pandas.DataFrame`
            DataFrame with one row for each feature, an index named "feature", and one column: "r".
        """
        if not hasattr(self, "results_"):
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' or 'load_imgs' before using 'transform'."
            )

        # Make sure we return a copy of the MetaResult
        results = self.results_.copy()
        features = list(results.maps.keys())
        images = np.array(list(results.maps.values()))

        img_vec = results.masker.transform(img)
        corrs = pearson(img_vec, images)
        out_df = pd.DataFrame(index=features, columns=["r"], data=corrs)
        out_df.index.name = "feature"

        # Update self.results_ to include the new table
        results.tables["correlation"] = out_df
        self.results_ = results

        return out_df


class CorrelationDistributionDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with study-wise images.

    .. versionchanged:: 0.1.0

        * New attribute: `results_`. MetaResult object containing masker, meta-analytic maps,
          and tables. This attribute replaces `masker`, `features_`, and `images_`.

    .. versionchanged:: 0.0.13

        * New parameter: `n_cores`. Number of cores to use for parallelization.

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
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Warnings
    --------
    Coefficients from correlating two maps have very large degrees of freedom,
    so almost all results will be statistically significant. Do not attempt to
    evaluate results based on significance.
    """

    _required_inputs = {
        "annotations": ("annotations", None),
    }

    def __init__(
        self,
        feature_group=None,
        features=None,
        frequency_threshold=0.001,
        target_image="z",
        n_cores=1,
    ):
        self.feature_group = feature_group
        self.features = features
        self.frequency_threshold = frequency_threshold
        self._required_inputs["images"] = ("image", target_image)
        self.n_cores = _check_ncores(n_cores)

    def _fit(self, dataset):
        """Collect sets of maps from the Dataset corresponding to each requested feature.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset for which to run meta-analyses to generate maps.

        Attributes
        ----------
        results : :obj:`~nimare.results.MetaResult`
            MetaResult with meta-analytic maps and masker added.
        """
        n_features = len(self.features_)
        maps = {
            r: v
            for r, v in tqdm(
                Parallel(return_as="generator", n_jobs=self.n_cores)(
                    delayed(self._run_fit)(feature, dataset) for feature in self.features_
                ),
                total=n_features,
            )
        }

        self.results_ = MetaResult(self, mask=dataset.masker, maps=maps)

    def _run_fit(self, feature, dataset):
        feature_ids = dataset.get_studies_by_label(
            labels=[feature], label_threshold=self.frequency_threshold
        )
        selected_ids = sorted(list(set(feature_ids).intersection(self.inputs_["id"])))
        selected_id_idx = [
            i_id for i_id, id_ in enumerate(self.inputs_["id"]) if id_ in selected_ids
        ]
        test_imgs = [
            img for i_img, img in enumerate(self.inputs_["images"]) if i_img in selected_id_idx
        ]
        if len(test_imgs):
            feature_arr = _safe_transform(
                test_imgs,
                dataset.masker,
                memfile=None,
            )
            return feature, feature_arr
        else:
            LGR.info(f"Skipping feature '{feature}'. No images found.")

    def transform(self, img):
        """Correlate target image with each map associated with each feature.

        Parameters
        ----------
        img : :obj:`~nibabel.nifti1.Nifti1Image`
            Image to decode. Must be in same space as ``dataset``.

        Returns
        -------
        out_df : :obj:`pandas.DataFrame`
            DataFrame with one row for each feature, an index named "feature", and two columns:
            "mean" and "std".
        """
        if not hasattr(self, "results_"):
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' before using 'transform'."
            )

        # Make sure we return a copy of the MetaResult
        results = self.results_.copy()
        features = list(results.maps.keys())

        img_vec = results.masker.transform(img)
        out_df = pd.DataFrame(
            index=features, columns=["mean", "std"], data=np.zeros((len(features), 2))
        )
        out_df.index.name = "feature"
        for feature, feature_arr in results.maps.items():
            corrs = pearson(img_vec, feature_arr)
            corrs_z = np.arctanh(corrs)
            out_df.loc[feature, "mean"] = np.mean(corrs_z)
            out_df.loc[feature, "std"] = np.std(corrs_z)

        # Update self.results_ to include the new table
        results.tables["correlation"] = out_df
        self.results_ = results

        return out_df
