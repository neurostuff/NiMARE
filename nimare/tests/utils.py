"""Utility functions for testing nimare."""
import os.path as op
from contextlib import ExitStack as does_not_raise

import nibabel as nib
import numpy as np
import pytest

from ..meta.utils import compute_kda_ma

# set significance levels used for testing.
# duplicated in test_estimator_performance
ALPHA = 0.05


def get_test_data_path():
    """Return the path to test datasets, terminated with separator.

    Test-related data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "data") + op.sep)


def _create_signal_mask(ground_truth_foci_ijks, mask):
    """Create complementary binary images to identify areas of likely significance and nonsignificance.

    Parameters
    ----------
    ground_truth_foci_ijks : array_like
        Ground truth ijk coordinates of foci.
    mask : :obj:`nibabel.Nifti1Image`
        Input mask to define shape and size of output binary masks

    Returns
    -------
    sig_map : :obj:`nibabel.Nifti1Image`
        Binary image representing regions around the
        ground truth foci expected to be significant.
    nonsig_map : :obj:`nibabel.Nifti1Image`
        Binary image representing regions not expected
        to be significant within the brain.
    """
    dims = mask.shape
    vox_dims = mask.header.get_zooms()

    # area where I'm reasonably certain there are significant results
    sig_prob_map = compute_kda_ma(
        dims, vox_dims, ground_truth_foci_ijks, r=2, value=1, sum_overlap=False
    )

    # area where I'm reasonably certain there are not significant results
    nonsig_prob_map = compute_kda_ma(
        dims, vox_dims, ground_truth_foci_ijks, r=14, value=1, sum_overlap=False
    )
    sig_map = nib.Nifti1Image((sig_prob_map == 1).astype(int), affine=mask.affine)
    nonsig_map = nib.Nifti1Image((nonsig_prob_map == 0).astype(int), affine=mask.affine)
    return sig_map, nonsig_map


def _check_p_values(
    p_array,
    masker,
    sig_idx,
    nonsig_idx,
    alpha,
    ground_truth_foci_ijks,
    n_iters=None,
    good_sensitivity=True,
    good_specificity=True,
):
    """Check if p-values are within the correct range."""
    ################################################
    # CHECK IF P-VALUES ARE WITHIN THE CORRECT RANGE
    ################################################
    if n_iters:
        assert p_array.min() >= (1.0 / n_iters)
        assert p_array.max() <= 1.0 - (1.0 / n_iters)
    else:
        assert (p_array >= 0).all() and (p_array <= 1).all()

    p_map = masker.inverse_transform(p_array).get_fdata()

    # reformat coordinate indices to index p_map
    gtf_idx = [
        [ground_truth_foci_ijks[i][j] for i in range(len(ground_truth_foci_ijks))]
        for j in range(3)
    ]

    best_chance_p_values = p_map[gtf_idx]
    assert all(best_chance_p_values < ALPHA) == good_sensitivity

    p_array_sig = p_array[sig_idx]
    p_array_nonsig = p_array[nonsig_idx]

    # assert that at least 50% of voxels surrounding the foci
    # are significant at alpha = .05
    observed_sig = p_array_sig < alpha
    observed_sig_perc = observed_sig.sum() / len(observed_sig)
    assert (observed_sig_perc >= 0.5) == good_sensitivity

    # assert that more than 95% of voxels farther away
    # from foci are nonsignificant at alpha = 0.05
    observed_nonsig = p_array_nonsig > alpha
    observed_nonsig_perc = observed_nonsig.sum() / len(observed_nonsig)
    assert np.isclose(observed_nonsig_perc, (1 - alpha), atol=0.05) == good_specificity


def _transform_res(meta, meta_res, corr):
    """Evaluate whether meta estimator and corrector work together."""
    #######################################
    # CHECK IF META/CORRECTOR WORK TOGETHER
    #######################################
    # all combinations of meta-analysis estimators and multiple comparison correctors
    # that do not work together
    corr_expectation = does_not_raise()

    with corr_expectation:
        cres = corr.transform(meta_res)

    # if multiple correction failed (expected) do not continue
    if isinstance(corr_expectation, type(pytest.raises(ValueError))):
        pytest.xfail("this meta-analysis & corrector combo fails")
    return cres
