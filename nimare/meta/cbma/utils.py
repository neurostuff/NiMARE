# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for coordinate-based meta-analysis estimators
"""
import numpy as np
from scipy.stats import norm


def p_to_z(p, sign):
    """From Neurosynth.
    """
    p = p/2  # convert to two-tailed
    # prevent underflow
    p[p < 1e-240] = 1e-240
    # Convert to z and assign tail
    z = np.abs(norm.ppf(p)) * sign
    # Set very large z's to max precision
    z[np.isinf(z)] = norm.ppf(1e-240)*-1
    return z


def compute_ma(shape, ijk, kernel):
    """
    Generate modeled activation (MA) maps.

    Replaces the values around each focus in ijk with the experiment-specific kernel.
    Takes the element-wise maximum when looping through foci, which accounts for foci
    which are near to one another and may have overlapping kernels.

    Parameters
    ----------
    shape : tuple
        Shape of brain image + buffer. Typically (91, 109, 91) + (30, 30, 30).
    ijk : array-like
        Indices of foci. Each row is a coordinate, with the three columns
        corresponding to index in each of three dimensions.
    kernel : array-like
        3D array of smoothing kernel. Typically of shape (30, 30, 30).

    Returns
    -------
    ma_values : array-like
        1d array of modeled activation values.
    """
    ma_values = np.zeros(shape)
    for j_peak in range(ijk.shape[0]):
        i = ijk[j_peak, 0]
        j = ijk[j_peak, 1]
        k = ijk[j_peak, 2]
        ma_values[i:i+31, j:j+31, k:k+31] = np.maximum(ma_values[i:i+31, j:j+31, k:k+31],
                                                       kernel)

    # Reduce to original dimensions and convert to 1d.
    ma_values = ma_values[15:-15, 15:-15, 15:-15]
    ma_values = ma_values.ravel()
    return ma_values


def _get_null(hist_bins, ma_hists):
    """
    Compute ALE null distribution.
    """
    # Inverse of step size in histBins (0.0001) = 10000
    step = 1 / np.mean(np.diff(hist_bins))

    # Null distribution to convert ALE to p-values.
    ale_hist = ma_hists[0, :]
    for i_exp in range(1, ma_hists.shape[0]):
        temp_hist = np.copy(ale_hist)
        ma_hist = np.copy(ma_hists[i_exp, :])

        # Find histogram bins with nonzero values for each histogram.
        ale_idx = np.where(temp_hist > 0)[0]
        exp_idx = np.where(ma_hist > 0)[0]

        # Normalize histograms.
        temp_hist /= np.sum(temp_hist)
        ma_hist /= np.sum(ma_hist)

        # Perform weighted convolution of histograms.
        ale_hist = np.zeros(hist_bins.shape[0])
        for j_idx in exp_idx:
            # Compute probabilities of observing each ALE value in histBins
            # by randomly combining maps represented by maHist and aleHist.
            # Add observed probabilities to corresponding bins in ALE
            # histogram.
            probabilities = ma_hist[j_idx] * temp_hist[ale_idx]
            ale_scores = 1 - (1 - hist_bins[j_idx]) * (1 - hist_bins[ale_idx])
            score_idx = np.floor(ale_scores * step).astype(int)
            np.add.at(ale_hist, score_idx, probabilities)

    # Convert aleHist into null distribution. The value in each bin
    # represents the probability of finding an ALE value (stored in
    # histBins) of that value or lower.
    last_used = np.where(ale_hist > 0)[0][-1]
    null_distribution = ale_hist[:last_used+1] / np.sum(ale_hist)
    null_distribution = np.cumsum(null_distribution[::-1])[::-1]
    null_distribution /= np.max(null_distribution)
    return null_distribution


def _compute_ale(experiments, dims, shape, prior, hist_bins=None):
    """
    Generate ALE-value array and null distribution from list of experiments.
    For ALEs on the original dataset, computes the null distribution.
    For permutation ALEs and all SCALEs, just computes ALE values.
    Returns masked array of ALE values and 1XnBins null distribution.
    """
    ale_values = np.ones(dims).ravel()
    if hist_bins is not None:
        ma_hists = np.zeros((len(experiments), hist_bins.shape[0]))
    else:
        ma_hists = None

    for i, exp in enumerate(experiments):
        ma_values = compute_ma(shape, exp.ijk, exp.kernel)

        # Remember that histogram uses bin edges (not centers), so it returns
        # a 1xhist_bins-1 array
        if hist_bins is not None:
            brain_ma_values = ma_values[prior]
            n_zeros = len(np.where(brain_ma_values == 0)[0])
            reduced_ma_values = brain_ma_values[brain_ma_values > 0]
            ma_hists[i, 0] = n_zeros
            ma_hists[i, 1:] = np.histogram(a=reduced_ma_values, bins=hist_bins,
                                           density=False)[0]
        ale_values *= (1. - ma_values)

    ale_values = 1 - ale_values
    ale_values = ale_values[prior]

    if hist_bins is not None:
        null_distribution = _get_null(hist_bins, ma_hists)
    else:
        null_distribution = None

    return ale_values, null_distribution


def _make_hist(oned_arr, hist_bins):
    hist_ = np.histogram(a=oned_arr, bins=hist_bins,
                         range=(np.min(hist_bins), np.max(hist_bins)),
                         density=False)[0]
    return hist_
