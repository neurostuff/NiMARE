"""Methods for decoding subsets of voxels or experiments into text."""
import numpy as np
import pandas as pd
from nilearn._utils import load_niimg
from scipy import special
from scipy.stats import binom
from statsmodels.sandbox.stats.multicomp import multipletests

from .. import references
from ..base import Decoder
from ..due import due
from ..stats import one_way, two_way
from ..transforms import p_to_z
from .utils import weight_priors


@due.dcite(references.GCLDA_DECODING, description="Citation for GCLDA decoding.")
def gclda_decode_roi(model, roi, topic_priors=None, prior_weight=1.0):
    r"""Perform image-to-text decoding for discrete inputs using method from Rubin et al. (2017).

    Parameters
    ----------
    model : :obj:`nimare.annotate.topic.GCLDAModel`
        Model object needed for decoding.
    roi : :obj:`nibabel.nifti1.Nifti1Image` or :obj:`str`
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
            - From :func:`gclda.model.Model.get_spatial_probs()`
    2.  Compute topic weight vector (:math:`\\tau_{t}`) by adding across voxels
        within ROI.
            - :math:`\\tau_{t} = \sum_{i} {p(t|v_{i})}`
    3.  Multiply :math:`\\tau_{t}` by
        :math:`p(w|t)`.
            - :math:`p(w|r) \propto \\tau_{t} \cdot p(w|t)`
    4.  The resulting vector (``word_weights``) reflects arbitrarily scaled
        term weights for the ROI.

    See Also
    --------
    :class:`nimare.annotate.gclda.GCLDAModel`
    :func:`nimare.decode.continuous.gclda_decode_map`
    :func:`nimare.decode.encode.gclda_encode`

    References
    ----------
    * Rubin, Timothy N., et al. "Decoding brain activity using a
      large-scale probabilistic functional-anatomical atlas of human
      cognition." PLoS computational biology 13.10 (2017): e1005649.
      https://doi.org/10.1371/journal.pcbi.1005649
    """
    roi = load_niimg(roi)

    dset_aff = model.mask.affine
    if not np.array_equal(roi.affine, dset_aff):
        raise ValueError(
            "Input roi must have same affine as mask img:"
            "\n{0}\n{1}".format(np.array2string(roi.affine), np.array2string(dset_aff))
        )

    # Load ROI file and get ROI voxels overlapping with brain mask
    mask_vec = model.mask.get_fdata().ravel().astype(bool)
    roi_vec = roi.get_fdata().astype(bool).ravel()
    roi_vec = roi_vec[mask_vec]
    roi_idx = np.where(roi_vec)[0]
    p_topic_g_roi = model.p_topic_g_voxel_[roi_idx, :]  # p(T|V) for voxels in ROI only
    topic_weights = np.sum(p_topic_g_roi, axis=0)  # Sum across words
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


@due.dcite(references.BRAINMAP_DECODING, description="Citation for BrainMap-style decoding.")
class BrainMapDecoder(Decoder):
    """Perform image-to-text decoding for discrete inputs according to the BrainMap method.

    Parameters
    ----------
    feature_group : :obj:`str`, optional
        Feature group name used to select labels from a specific source.
        Feature groups are stored as prefixes to feature name columns in
        Dataset.annotations, with the format ``[source]_[valuetype]__``.
        Input may or may not include the trailing underscore.
        Default is None, which uses all feature groups available.
    features : :obj:`list`, optional
        List of features in dataset annotations to use for decoding.
        If feature_group is provided, then features should not include the
        feature group prefix.
        If feature_group is *not* provided, then features *should* include the
        prefix.
        Default is None, which uses all features available.
    frequency_threshold : :obj:`float`, optional
        Threshold to apply to dataset annotations. Values greater than or
        equal to the threshold as assigned as label+, while values below
        the threshold are considered label-. Default is 0.001.
    u : :obj:`float`, optional
        Alpha level for multiple comparisons correction. Default is 0.05.
    correction : :obj:`str` or None, optional
        Multiple comparisons correction method to apply. Corresponds to
        available options for :func:`statsmodels.stats.multitest.multipletests`.
        Default is 'fdr_bh' (Benjamini-Hochberg FDR correction).

    See Also
    --------
    :func:`nimare.decode.discrete.brainmap_decode`: The associated function for this method.

    References
    ----------
    * Amft, Maren, et al. "Definition and characterization of an extended
      social-affective default network." Brain Structure and Function 220.2
      (2015): 1031-1049. https://doi.org/10.1007/s00429-013-0698-0
    """

    def __init__(
        self,
        feature_group=None,
        features=None,
        frequency_threshold=0.001,
        u=0.05,
        correction="fdr_bh",
    ):
        self.feature_group = feature_group
        self.features = features
        self.frequency_threshold = frequency_threshold
        self.u = u
        self.correction = correction
        self.results = None

    def _fit(self, dataset):
        self.inputs_ = {"coordinates": dataset.coordinates, "annotations": dataset.annotations}

    def transform(self, ids, ids2=None):
        """Apply the decoding method to a Dataset.

        Parameters
        ----------
        ids : :obj:`list`
            Subset of studies in coordinates/annotations dataframes indicating
            target for decoding. Examples include studies reporting at least one
            peak in an ROI, or studies selected from a clustering analysis.
        ids2 : :obj:`list` or None, optional
            Second subset of studies, representing "unselected" studies. If None,
            then all studies in coordinates/annotations dataframes **not** in
            ``ids`` will be used.

        Returns
        -------
        results : :class:`pandas.DataFrame`
            Table with each label and the following values associated with each
            label: 'pForward', 'zForward', 'likelihoodForward', 'pReverse',
            'zReverse', and 'probReverse'.
        """
        results = brainmap_decode(
            self.inputs_["coordinates"],
            self.inputs_["annotations"],
            ids=ids,
            ids2=ids2,
            features=self.features_,
            frequency_threshold=self.frequency_threshold,
            u=self.u,
            correction=self.correction,
        )
        self.results = results
        return results


@due.dcite(references.BRAINMAP_DECODING, description="Citation for BrainMap-style decoding.")
def brainmap_decode(
    coordinates,
    annotations,
    ids,
    ids2=None,
    features=None,
    frequency_threshold=0.001,
    u=0.05,
    correction="fdr_bh",
):
    """Perform image-to-text decoding for discrete inputs according to the BrainMap method.

    Parameters
    ----------
    coordinates : :class:`pandas.DataFrame`
        DataFrame containing coordinates. Must include a column named 'id' and
        must have a separate row for each reported peak coordinate for each
        study (i.e., there are multiple rows per ID).
        IDs from ``coordinates`` must match those from ``annotations``.
    annotations : :class:`pandas.DataFrame`
        DataFrame containing labels. Must include a column named 'id' and each
        row must correspond to a study. Other columns may correspond to
        individual labels.
        IDs from ``annotations`` must match those from ``coordinates``.
    ids : :obj:`list`
        Subset of studies in coordinates/annotations dataframes indicating
        target for decoding. Examples include studies reporting at least one
        peak in an ROI, or studies selected from a clustering analysis.
    ids2 : :obj:`list` or None, optional
        Second subset of studies, representing "unselected" studies. If None,
        then all studies in coordinates/annotations dataframes **not** in
        ``ids`` will be used.
    features : :obj:`list`, optional
        List of features in dataset annotations to use for decoding.
        Default is None, which uses all features available.
    frequency_threshold : :obj:`float`, optional
        Threshold to apply to dataset annotations. Values greater than or
        equal to the threshold as assigned as label+, while values below
        the threshold are considered label-. Default is 0.001.
    u : :obj:`float`, optional
        Alpha level for multiple comparisons correction. Default is 0.05.
    correction : :obj:`str` or None, optional
        Multiple comparisons correction method to apply. Corresponds to
        available options for :func:`statsmodels.stats.multitest.multipletests`.
        Default is 'fdr_bh' (Benjamini-Hochberg FDR correction).

    Returns
    -------
    out_df : :class:`pandas.DataFrame`
        Table with each label and the following values associated with each
        label: 'pForward', 'zForward', 'likelihoodForward', 'pReverse',
        'zReverse', and 'probReverse'.

    See Also
    --------
    :func:`nimare.decode.discrete.BrainMapDecoder`: The associated class for this method.

    References
    ----------
    * Amft, Maren, et al. "Definition and characterization of an extended
      social-affective default network." Brain Structure and Function 220.2
      (2015): 1031-1049. https://doi.org/10.1007/s00429-013-0698-0
    """
    dataset_ids = sorted(list(set(coordinates["id"].values)))
    if ids2 is None:
        unselected = sorted(list(set(dataset_ids) - set(ids)))
    else:
        unselected = ids2[:]

    # Binarize with frequency threshold
    features_df = annotations.set_index("id", drop=True)
    features_df = features_df[features].ge(frequency_threshold)

    sel_array = features_df.loc[ids].values
    unsel_array = features_df.loc[unselected].values

    n_selected = len(ids)
    n_unselected = len(unselected)

    # the number of times any term is used (e.g., if one experiment uses
    # two terms, that counts twice). Why though?
    n_exps_across_terms = np.sum(np.sum(features_df))

    n_selected_term = np.sum(sel_array, axis=0)
    n_unselected_term = np.sum(unsel_array, axis=0)

    n_selected_noterm = n_selected - n_selected_term
    n_unselected_noterm = n_unselected - n_unselected_term

    n_term = n_selected_term + n_unselected_term
    p_term = n_term / n_exps_across_terms

    n_foci_in_database = coordinates.shape[0]
    p_selected = n_selected / n_foci_in_database

    # I hope there's a way to do this without the for loop
    n_term_foci = np.zeros(len(features))
    n_noterm_foci = np.zeros(len(features))
    for i, term in enumerate(features):
        term_ids = features_df.loc[features_df[term] == 1].index.values
        noterm_ids = features_df.loc[features_df[term] == 0].index.values
        n_term_foci[i] = coordinates["id"].isin(term_ids).sum()
        n_noterm_foci[i] = coordinates["id"].isin(noterm_ids).sum()

    p_selected_g_term = n_selected_term / n_term_foci  # probForward
    l_selected_g_term = p_selected_g_term / p_selected  # likelihoodForward
    p_selected_g_noterm = n_selected_noterm / n_noterm_foci

    p_term_g_selected = p_selected_g_term * p_term / p_selected  # probReverse
    p_term_g_selected = p_term_g_selected / np.nansum(p_term_g_selected)  # Normalize

    # Significance testing
    # Forward inference significance is determined with a binomial distribution
    p_fi = 1 - binom.cdf(k=n_selected_term, n=n_term_foci, p=p_selected)
    sign_fi = np.sign(
        n_selected_term - np.mean(n_selected_term)
    ).ravel()  # pylint: disable=no-member

    # Two-way chi-square test for specificity of activation
    cells = np.array(
        [
            [n_selected_term, n_selected_noterm],  # pylint: disable=no-member
            [n_unselected_term, n_unselected_noterm],
        ]
    ).T
    chi2_ri = two_way(cells)
    p_ri = special.chdtrc(1, chi2_ri)
    sign_ri = np.sign(p_selected_g_term - p_selected_g_noterm).ravel()  # pylint: disable=no-member

    # Ignore rare features
    p_fi[n_selected_term < 5] = 1.0
    p_ri[n_selected_term < 5] = 1.0

    # Multiple comparisons correction across features. Separately done for FI and RI.
    if correction is not None:
        _, p_corr_fi, _, _ = multipletests(p_fi, alpha=u, method=correction, returnsorted=False)
        _, p_corr_ri, _, _ = multipletests(p_ri, alpha=u, method=correction, returnsorted=False)
    else:
        p_corr_fi = p_fi
        p_corr_ri = p_ri

    # Compute z-values
    z_corr_fi = p_to_z(p_corr_fi, "two") * sign_fi
    z_corr_ri = p_to_z(p_corr_ri, "two") * sign_ri

    # Effect size
    arr = np.array(
        [
            p_corr_fi,
            z_corr_fi,
            l_selected_g_term,  # pylint: disable=no-member
            p_corr_ri,
            z_corr_ri,
            p_term_g_selected,
        ]
    ).T

    out_df = pd.DataFrame(
        data=arr,
        index=features,
        columns=[
            "pForward",
            "zForward",
            "likelihoodForward",
            "pReverse",
            "zReverse",
            "probReverse",
        ],
    )
    out_df.index.name = "Term"
    return out_df


@due.dcite(references.NEUROSYNTH, description="Introduces Neurosynth.")
class NeurosynthDecoder(Decoder):
    """Perform discrete functional decoding according to Neurosynth's meta-analytic method.

    This does not employ correlations between unthresholded maps, which are the
    method of choice for decoding within Neurosynth and Neurovault.
    Metadata (i.e., feature labels) for studies within the selected sample
    (`ids`) are compared to the unselected studies remaining in the database
    (`dataset`).

    Parameters
    ----------
    feature_group : :obj:`str`, optional
        Feature group name used to select labels from a specific source.
        Feature groups are stored as prefixes to feature name columns in
        Dataset.annotations, with the format ``[source]_[valuetype]__``.
        Input may or may not include the trailing underscore.
        Default is None, which uses all feature groups available.
    features : :obj:`list`, optional
        List of features in dataset annotations to use for decoding.
        If feature_group is provided, then features should not include the
        feature group prefix.
        If feature_group is *not* provided, then features *should* include the
        prefix.
        Default is None, which uses all features available.
    frequency_threshold : :obj:`float`, optional
        Threshold to apply to dataset annotations. Values greater than or
        equal to the threshold as assigned as label+, while values below
        the threshold are considered label-. Default is 0.001.
    prior : :obj:`float`, optional
        Uniform prior probability of each label being active in a study in
        the absence of evidence (labels or selection) from the study.
        Default is 0.5 (50%).
    u : :obj:`float`, optional
        Alpha level for multiple comparisons correction. Default is 0.05.
    correction : :obj:`str` or None, optional
        Multiple comparisons correction method to apply. Corresponds to
        available options for :func:`statsmodels.stats.multitest.multipletests`.
        Default is 'fdr_bh' (Benjamini-Hochberg FDR correction).

    See Also
    --------
    :func:`nimare.decode.discrete.neurosynth_decode`: The associated function for this method.

    References
    ----------
    * Yarkoni, Tal, et al. "Large-scale automated synthesis of human
      functional neuroimaging data." Nature methods 8.8 (2011): 665.
      https://doi.org/10.1038/nmeth.1635
    """

    def __init__(
        self,
        feature_group=None,
        features=None,
        frequency_threshold=0.001,
        prior=0.5,
        u=0.05,
        correction="fdr_bh",
    ):
        self.feature_group = feature_group
        self.features = features
        self.frequency_threshold = frequency_threshold
        self.prior = prior
        self.u = u
        self.correction = correction
        self.results = None

    def _fit(self, dataset):
        self.inputs_ = {"coordinates": dataset.coordinates, "annotations": dataset.annotations}

    def transform(self, ids, ids2=None):
        """Apply the decoding method to a Dataset.

        Parameters
        ----------
        ids : :obj:`list`
            Subset of studies in coordinates/annotations dataframes indicating
            target for decoding. Examples include studies reporting at least one
            peak in an ROI, or studies selected from a clustering analysis.
        ids2 : :obj:`list` or None, optional
            Second subset of studies, representing "unselected" studies. If None,
            then all studies in Dataset **not** in
            ``ids`` will be used.

        Returns
        -------
        results : :class:`pandas.DataFrame`
            Table with each label and the following values associated with each
            label: 'pForward', 'zForward', 'probForward', 'pReverse', 'zReverse',
            and 'probReverse'.
        """
        results = neurosynth_decode(
            self.inputs_["coordinates"],
            self.inputs_["annotations"],
            ids=ids,
            ids2=ids2,
            features=self.features_,
            frequency_threshold=self.frequency_threshold,
            prior=self.prior,
            u=self.u,
            correction=self.correction,
        )
        self.results = results
        return results


@due.dcite(references.NEUROSYNTH, description="Introduces Neurosynth.")
def neurosynth_decode(
    coordinates,
    annotations,
    ids,
    ids2=None,
    feature_group=None,
    features=None,
    frequency_threshold=0.001,
    prior=0.5,
    u=0.05,
    correction="fdr_bh",
):
    """Perform discrete functional decoding according to Neurosynth's meta-analytic method.

    This does not employ correlations between unthresholded maps, which are the
    method of choice for decoding within Neurosynth and Neurovault.
    Metadata (i.e., feature labels) for studies within the selected sample
    (`ids`) are compared to the unselected studies remaining in the database
    (`dataset`).

    Parameters
    ----------
    coordinates : :class:`pandas.DataFrame`
        DataFrame containing coordinates. Must include a column named 'id' and
        must have a separate row for each reported peak coordinate for each
        study (i.e., there are multiple rows per ID).
        IDs from ``coordinates`` must match those from ``annotations``.
    annotations : :class:`pandas.DataFrame`
        DataFrame containing labels. Must include a column named 'id' and each
        row must correspond to a study. Other columns may correspond to
        individual labels.
        IDs from ``annotations`` must match those from ``coordinates``.
    ids : :obj:`list`
        Subset of studies in coordinates/annotations dataframes indicating
        target for decoding. Examples include studies reporting at least one
        peak in an ROI, or studies selected from a clustering analysis.
    ids2 : :obj:`list` or None, optional
        Second subset of studies, representing "unselected" studies. If None,
        then all studies in coordinates/annotations dataframes **not** in
        ``ids`` will be used.
    features : :obj:`list`, optional
        List of features in dataset annotations to use for decoding.
        Default is None, which uses all features available.
    frequency_threshold : :obj:`float`, optional
        Threshold to apply to dataset annotations. Values greater than or
        equal to the threshold as assigned as label+, while values below
        the threshold are considered label-. Default is 0.001.
    prior : :obj:`float`, optional
        Uniform prior probability of each label being active in a study in
        the absence of evidence (labels or selection) from the study.
        Default is 0.5 (50%).
    u : :obj:`float`, optional
        Alpha level for multiple comparisons correction. Default is 0.05.
    correction : :obj:`str` or None, optional
        Multiple comparisons correction method to apply. Corresponds to
        available options for :func:`statsmodels.stats.multitest.multipletests`.
        Default is 'fdr_bh' (Benjamini-Hochberg FDR correction).

    Returns
    -------
    out_df : :class:`pandas.DataFrame`
        Table with each label and the following values associated with each
        label: 'pForward', 'zForward', 'probForward', 'pReverse', 'zReverse',
        and 'probReverse'.

    See Also
    --------
    :class:`nimare.decode.discrete.NeurosynthDecoder`: The associated class for this method.
    :func:`nimare.decode.continuous.corr_decode`: The correlation-based decoding
        method employed in Neurosynth and NeuroVault.

    References
    ----------
    * Yarkoni, Tal, et al. "Large-scale automated synthesis of human
      functional neuroimaging data." Nature methods 8.8 (2011): 665.
      https://doi.org/10.1038/nmeth.1635
    """
    dataset_ids = sorted(list(set(coordinates["id"].values)))
    if ids2 is None:
        unselected = sorted(list(set(dataset_ids) - set(ids)))
    else:
        unselected = ids2[:]

    # Binarize with frequency threshold
    features_df = annotations.set_index("id", drop=True)
    features_df = features_df[features].ge(frequency_threshold)

    sel_array = features_df.loc[ids].values
    unsel_array = features_df.loc[unselected].values

    n_selected = len(ids)
    n_unselected = len(unselected)

    n_selected_term = np.sum(sel_array, axis=0)
    n_unselected_term = np.sum(unsel_array, axis=0)

    n_selected_noterm = n_selected - n_selected_term
    n_unselected_noterm = n_unselected - n_unselected_term

    n_term = n_selected_term + n_unselected_term
    n_noterm = n_selected_noterm + n_unselected_noterm

    p_term = n_term / (n_term + n_noterm)

    p_selected_g_term = n_selected_term / n_term
    p_selected_g_noterm = n_selected_noterm / n_noterm

    # Recompute conditions with empirically derived prior (or inputted one)
    if prior is None:
        # if this is used, p_term_g_selected_prior = p_selected (regardless of term)
        prior = p_term

    # Significance testing
    # One-way chi-square test for consistency of term frequency across terms
    chi2_fi = one_way(n_selected_term, n_term)
    p_fi = special.chdtrc(1, chi2_fi)
    sign_fi = np.sign(
        n_selected_term - np.mean(n_selected_term)
    ).ravel()  # pylint: disable=no-member

    # Two-way chi-square test for specificity of activation
    cells = np.array(
        [
            [n_selected_term, n_selected_noterm],  # pylint: disable=no-member
            [n_unselected_term, n_unselected_noterm],
        ]
    ).T
    chi2_ri = two_way(cells)
    p_ri = special.chdtrc(1, chi2_ri)
    sign_ri = np.sign(p_selected_g_term - p_selected_g_noterm).ravel()  # pylint: disable=no-member

    # Multiple comparisons correction across terms. Separately done for FI and RI.
    if correction is not None:
        _, p_corr_fi, _, _ = multipletests(p_fi, alpha=u, method=correction, returnsorted=False)
        _, p_corr_ri, _, _ = multipletests(p_ri, alpha=u, method=correction, returnsorted=False)
    else:
        p_corr_fi = p_fi
        p_corr_ri = p_ri

    # Compute z-values
    z_corr_fi = p_to_z(p_corr_fi, "two") * sign_fi
    z_corr_ri = p_to_z(p_corr_ri, "two") * sign_ri

    # Effect size
    # est. prob. of brain state described by term finding activation in ROI
    p_selected_g_term_g_prior = prior * p_selected_g_term + (1 - prior) * p_selected_g_noterm

    # est. prob. of activation in ROI reflecting brain state described by term
    p_term_g_selected_g_prior = p_selected_g_term * prior / p_selected_g_term_g_prior

    arr = np.array(
        [
            p_corr_fi,
            z_corr_fi,
            p_selected_g_term_g_prior,  # pylint: disable=no-member
            p_corr_ri,
            z_corr_ri,
            p_term_g_selected_g_prior,
        ]
    ).T

    out_df = pd.DataFrame(
        data=arr,
        index=features,
        columns=["pForward", "zForward", "probForward", "pReverse", "zReverse", "probReverse"],
    )
    out_df.index.name = "Term"
    return out_df
