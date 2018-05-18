"""
Methods for decoding subsets of voxels (e.g., ROIs) or experiments (e.g., from
meta-analytic clustering on a database) into text.
"""
import numpy as np
import pandas as pd
from scipy.stats import binom
from statsmodels.sandbox.stats.multicomp import multipletests

from .base import Decoder
from ..utils import p_to_z
from ..stats import one_way, two_way
from ..due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Citation for GCLDA decoding.')
class GCLDADiscreteDecoder(Decoder):
    def __init__(self, model, roi_img, topic_priors, prior_weight):
        pass


@due.dcite(Doi('10.1007/s00429-013-0698-0'),
           description='Citation for BrainMap-style decoding.')
class BrainMapDecoder(Decoder):
    """

    """
    def __init__(self, coordinates, annotations):
        self.coordinates = coordinates
        self.annotations = annotations

    def fit(self, ids, ids2=None, features=None, frequency_threshold=0.001,
            u=0.05, correction='fdr_bh'):
        dataset_ids = sorted(list(set(self.coordinates['ids'].values)))
        if ids2 is None:
            unselected = sorted(list(set(dataset_ids) - set(ids)))
        else:
            unselected = ids2[:]

        if features is None:
            features = self.annotations.columns.values

        # Binarize with frequency threshold
        features_df = self.annotations[features].ge(frequency_threshold)

        terms = self.annotations.columns.values
        sel_array = self.annotations.loc[ids].values
        unsel_array = self.annotations.loc[unselected].values

        n_selected = len(ids)
        n_unselected = len(unselected)

        # the number of times any term is used (e.g., if one experiment uses
        # two terms, that counts twice). Why though?
        n_exps_across_terms = np.sum(np.sum(self.annotations))

        n_selected_term = np.sum(sel_array, axis=0)
        n_unselected_term = np.sum(unsel_array, axis=0)

        n_selected_noterm = n_selected - n_selected_term
        n_unselected_noterm = n_unselected - n_unselected_term

        n_term = n_selected_term + n_unselected_term
        p_term = n_term / n_exps_across_terms

        n_foci_in_database = self.coordinates.shape[0]
        p_selected = n_selected / n_foci_in_database

        # I hope there's a way to do this without the for loop
        n_term_foci = np.zeros(len(terms))
        n_noterm_foci = np.zeros(len(terms))
        for i, term in enumerate(terms):
            term_ids = self.annotations.loc[self.annotations[term] == 1].index.values
            noterm_ids = self.annotations.loc[self.annotations[term] == 0].index.values
            n_term_foci[i] = self.coordinates['id'].isin(term_ids).sum()
            n_noterm_foci[i] = self.coordinates['id'].isin(noterm_ids).sum()

        p_selected_g_term = n_selected_term / n_term_foci  # probForward
        l_selected_g_term = p_selected_g_term / p_selected  # likelihoodForward
        p_selected_g_noterm = n_selected_noterm / n_noterm_foci

        p_term_g_selected = p_selected_g_term * p_term / p_selected  # probReverse
        p_term_g_selected = p_term_g_selected / np.sum(p_term_g_selected)  # Normalize

        ## Significance testing
        # Forward inference significance is determined with a binomial distribution
        p_fi = 1 - binom.cdf(k=n_selected_term, n=n_term_foci, p=p_selected)
        sign_fi = np.sign(n_selected_term - np.mean(n_selected_term)).ravel()  # pylint: disable=no-member

        # Two-way chi-square test for specificity of activation
        cells = np.array([[n_selected_term, n_selected_noterm],  # pylint: disable=no-member
                          [n_unselected_term, n_unselected_noterm]]).T
        p_ri = two_way(cells)
        sign_ri = np.sign(p_selected_g_term - p_selected_g_noterm).ravel()  # pylint: disable=no-member

        # Ignore rare terms
        p_fi[n_selected_term < 5] = 1.
        p_ri[n_selected_term < 5] = 1.

        # Multiple comparisons correction across terms. Separately done for FI and RI.
        if correction is not None:
            _, p_corr_fi, _, _ = multipletests(p_fi, alpha=u, method=correction,
                                               returnsorted=False)
            _, p_corr_ri, _, _ = multipletests(p_ri, alpha=u, method=correction,
                                               returnsorted=False)
        else:
            p_corr_fi = p_fi
            p_corr_ri = p_ri

        # Compute z-values
        z_corr_fi = p_to_z(p_corr_fi, sign_fi)
        z_corr_ri = p_to_z(p_corr_ri, sign_ri)

        ## Effect size
        arr = np.array([p_corr_fi, z_corr_fi, l_selected_g_term,  # pylint: disable=no-member
                        p_corr_ri, z_corr_ri, p_term_g_selected]).T

        out_df = pd.DataFrame(data=arr, index=terms,
                              columns=['pForward', 'zForward', 'likelihoodForward',
                                       'pReverse', 'zReverse', 'probReverse'])
        out_df.index.name = 'Term'
        return out_df


@due.dcite(Doi('10.1038/nmeth.1635'),
           description='Introduces Neurosynth.')
class NeurosynthDecoder(Decoder):
    """
    Performs discrete functional decoding according to Neurosynth's
    meta-analytic method. This does not employ correlations between
    unthresholded maps, which are the method of choice for decoding within
    Neurosynth and Neurovault.
    Metadata (i.e., feature labels) for studies within the selected sample
    (`ids`) are compared to the unselected studies remaining in the database
    (`dataset`).
    """
    def __init__(self, coordinates, annotations):
        self.coordinates = coordinates
        self.annotations = annotations

    def fit(ids, ids2=None, features=None, frequency_threshold=0.001,
            prior=0.5, u=0.05, correction='fdr_bh'):
        dataset_ids = sorted(list(set(self.coordinates['ids'].values)))
        if ids2 is None:
            unselected = sorted(list(set(dataset_ids) - set(ids)))
        else:
            unselected = ids2[:]

        if features is None:
            features = self.annotations.columns.values

        # Binarize with frequency threshold
        features_df = self.annotations[features].ge(frequency_threshold)

        terms = features_df.columns.values
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

        ## Significance testing
        # One-way chi-square test for consistency of term frequency across terms
        p_fi = stats.one_way(n_selected_term, n_term)
        sign_fi = np.sign(n_selected_term - np.mean(n_selected_term)).ravel()  # pylint: disable=no-member

        # Two-way chi-square test for specificity of activation
        cells = np.array([[n_selected_term, n_selected_noterm],  # pylint: disable=no-member
                          [n_unselected_term, n_unselected_noterm]]).T
        p_ri = stats.two_way(cells)
        sign_ri = np.sign(p_selected_g_term - p_selected_g_noterm).ravel()  # pylint: disable=no-member

        # Multiple comparisons correction across terms. Separately done for FI and RI.
        if correction is not None:
            _, p_corr_fi, _, _ = multipletests(p_fi, alpha=u, method=correction,
                                               returnsorted=False)
            _, p_corr_ri, _, _ = multipletests(p_ri, alpha=u, method=correction,
                                               returnsorted=False)
        else:
            p_corr_fi = p_fi
            p_corr_ri = p_ri

        # Compute z-values
        z_corr_fi = p_to_z(p_corr_fi, sign_fi)
        z_corr_ri = p_to_z(p_corr_ri, sign_ri)

        ## Effect size
        # est. prob. of brain state described by term finding activation in ROI
        p_selected_g_term_g_prior = prior * p_selected_g_term + (1 - prior) * p_selected_g_noterm

        # est. prob. of activation in ROI reflecting brain state described by term
        p_term_g_selected_g_prior = p_selected_g_term * prior / p_selected_g_term_g_prior

        arr = np.array([p_corr_fi, z_corr_fi, p_selected_g_term_g_prior,  # pylint: disable=no-member
                        p_corr_ri, z_corr_ri, p_term_g_selected_g_prior]).T

        out_df = pd.DataFrame(data=arr, index=terms,
                              columns=['pForward', 'zForward', 'probForward',
                                       'pReverse', 'zReverse', 'probReverse'])
        out_df.index.name = 'Term'
        return out_df
