"""Test estimator, kerneltransformer, and multiple comparisons corrector performance."""
import os
from contextlib import ExitStack as does_not_raise

import nibabel as nib
import numpy as np
import pytest

from ..correct import FDRCorrector, FWECorrector
from ..generate import create_coordinate_dataset
from ..meta import ale, kernel, mkda
from ..meta.utils import compute_kda_ma
from ..results import MetaResult
from ..utils import mm2vox

# set significance levels used for testing.
ALPHA = 0.05

if os.environ.get("CIRCLECI"):
    N_CORES = 1
else:
    N_CORES = -1


# PRECOMPUTED FIXTURES
# --------------------

##########################################
# random state
##########################################
@pytest.fixture(scope="session")
def random():
    """Set random state for the tests."""
    np.random.seed(1939)


##########################################
# simulated dataset(s)
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            {
                "foci": 5,
                "fwhm": 10.0,
                "n_studies": 40,
                "sample_size": 30,
                "n_noise_foci": 20,
                "seed": 1939,
            },
            id="normal_data",
        )
    ],
)
def simulatedata_cbma(request):
    """Set the simulated CBMA data according to parameters."""
    return request.param["fwhm"], create_coordinate_dataset(**request.param)


##########################################
# signal and non-signal masks
##########################################
@pytest.fixture(scope="session")
def signal_masks(simulatedata_cbma):
    """Define masks of signal and non-signal for performance evaluation."""
    _, (ground_truth_foci, dataset) = simulatedata_cbma
    ground_truth_foci_ijks = [
        tuple(mm2vox(focus, dataset.masker.mask_img.affine)) for focus in ground_truth_foci
    ]
    return _create_signal_mask(np.array(ground_truth_foci_ijks), dataset.masker.mask_img)


##########################################
# meta-analysis estimators
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param(ale.ALE, id="ale"),
        pytest.param(mkda.MKDADensity, id="mkda"),
        pytest.param(mkda.KDA, id="kda"),
    ],
)
def meta_est(request):
    """Define meta-analysis estimators for tests."""
    return request.param


##########################################
# meta-analysis estimator parameters
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param({"null_method": "montecarlo", "n_iters": 100}, id="montecarlo"),
        pytest.param({"null_method": "approximate"}, id="approximate"),
        pytest.param(
            {"null_method": "reduced_montecarlo", "n_iters": 100}, id="reduced_montecarlo"
        ),
    ],
)
def meta_params(request):
    """Define meta-analysis Estimator parameters for tests."""
    return request.param


##########################################
# meta-analysis kernels
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param(kernel.ALEKernel, id="ale_kernel"),
        pytest.param(kernel.MKDAKernel, id="mkda_kernel"),
        pytest.param(kernel.KDAKernel, id="kda_kernel"),
    ],
)
def kern(request):
    """Define kernel transformers for tests."""
    return request.param


##########################################
# multiple comparison correctors (testing)
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param(FWECorrector(method="bonferroni"), id="fwe_bonferroni"),
        pytest.param(
            FWECorrector(method="montecarlo", voxel_thresh=ALPHA, n_iters=100, n_cores=N_CORES),
            id="fwe_montecarlo",
        ),
        pytest.param(FDRCorrector(method="indep", alpha=ALPHA), id="fdr_indep"),
        pytest.param(FDRCorrector(method="negcorr", alpha=ALPHA), id="fdr_negcorr"),
    ],
)
def corr(request):
    """Define multiple comparisons correctors for tests."""
    return request.param


##########################################
# multiple comparison correctors (smoke)
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param(FWECorrector(method="bonferroni"), id="fwe_bonferroni"),
        pytest.param(
            FWECorrector(method="montecarlo", voxel_thresh=ALPHA, n_iters=2, n_cores=1),
            id="fwe_montecarlo",
        ),
        pytest.param(FDRCorrector(method="indep", alpha=ALPHA), id="fdr_indep"),
        pytest.param(FDRCorrector(method="negcorr", alpha=ALPHA), id="fdr_negcorr"),
    ],
)
def corr_small(request):
    """Define multiple comparisons correctors for tests."""
    return request.param


###########################################
# all meta-analysis estimator/kernel combos
###########################################
@pytest.fixture(scope="session")
def meta(simulatedata_cbma, meta_est, kern, meta_params):
    """Define estimator/kernel combinations for tests."""
    fwhm, (_, _) = simulatedata_cbma
    if kern == kernel.KDAKernel or kern == kernel.MKDAKernel:
        kern = kern(r=fwhm / 2)
    else:
        kern = kern()

    # instantiate meta-analysis estimator
    return meta_est(kern, **meta_params)


###########################################
# meta-analysis estimator results
###########################################
@pytest.fixture(scope="session")
def meta_res(simulatedata_cbma, meta, random):
    """Define estimators for tests."""
    _, (_, dataset) = simulatedata_cbma
    # CHECK IF META/KERNEL WORK TOGETHER
    ####################################
    meta_expectation = does_not_raise()

    with meta_expectation:
        res = meta.fit(dataset)
    # if creating the result failed (expected), do not continue
    if isinstance(meta_expectation, type(pytest.raises(ValueError))):
        pytest.xfail("this meta-analysis & kernel combo fails")
    # instantiate meta-analysis estimator
    return res


###########################################
# corrected results (testing)
###########################################
@pytest.fixture(scope="session")
def meta_cres(meta, meta_res, corr, random):
    """Define corrected results for tests."""
    return _transform_res(meta, meta_res, corr)


###########################################
# corrected results (smoke)
###########################################
@pytest.fixture(scope="session")
def meta_cres_small(meta, meta_res, corr_small, random):
    """Define corrected results for tests."""
    return _transform_res(meta, meta_res, corr_small)


# --------------
# TEST FUNCTIONS
# --------------


def test_meta_fit_smoke(meta_res):
    """Smoke test for meta-analytic estimator fit."""
    assert isinstance(meta_res, MetaResult)


@pytest.mark.performance
def test_meta_fit_performance(meta_res, signal_masks, simulatedata_cbma):
    """Test meta-analytic estimator fit performance."""
    _, (ground_truth_foci, _) = simulatedata_cbma
    mask = meta_res.masker.mask_img
    ground_truth_foci_ijks = [tuple(mm2vox(focus, mask.affine)) for focus in ground_truth_foci]
    sig_idx, nonsig_idx = [
        meta_res.masker.transform(img).astype(bool).squeeze() for img in signal_masks
    ]

    # all estimators generate p-values
    p_array = meta_res.get_map("p", return_type="array")

    # poor performer(s)
    if (
        isinstance(meta_res.estimator, ale.ALE)
        and isinstance(meta_res.estimator.kernel_transformer, kernel.KDAKernel)
        and meta_res.estimator.get_params().get("null_method") == "approximate"
    ):
        good_sensitivity = True
        good_specificity = False
    elif (
        isinstance(meta_res.estimator, ale.ALE)
        and isinstance(meta_res.estimator.kernel_transformer, kernel.KDAKernel)
        and "montecarlo" in meta_res.estimator.get_params().get("null_method")
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_res.estimator, ale.ALE)
        and type(meta_res.estimator.kernel_transformer) == kernel.KDAKernel
        and "montecarlo" in meta_res.estimator.get_params().get("null_method")
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_res.estimator, ale.ALE)
        and type(meta_res.estimator.kernel_transformer) == kernel.KDAKernel
        and meta_res.estimator.get_params().get("null_method") == "approximate"
    ):
        good_sensitivity = True
        good_specificity = False
    elif (
        isinstance(meta_res.estimator, mkda.MKDADensity)
        and isinstance(meta_res.estimator.kernel_transformer, kernel.ALEKernel)
        and meta_res.estimator.get_params().get("null_method") != "reduced_montecarlo"
    ):
        good_sensitivity = False
        good_specificity = True
    else:
        good_sensitivity = True
        good_specificity = True

    _check_p_values(
        p_array,
        meta_res.masker,
        sig_idx,
        nonsig_idx,
        ALPHA,
        ground_truth_foci_ijks,
        n_iters=None,
        good_sensitivity=good_sensitivity,
        good_specificity=good_specificity,
    )


def test_corr_transform_smoke(meta_cres_small):
    """Smoke test for corrector transform."""
    assert isinstance(meta_cres_small, MetaResult)


@pytest.mark.performance
def test_corr_transform_performance(meta_cres, corr, signal_masks, simulatedata_cbma):
    """Test corrector transform performance."""
    _, (ground_truth_foci, _) = simulatedata_cbma
    mask = meta_cres.masker.mask_img
    ground_truth_foci_ijks = [tuple(mm2vox(focus, mask.affine)) for focus in ground_truth_foci]
    sig_idx, nonsig_idx = [
        meta_cres.masker.transform(img).astype(bool).squeeze() for img in signal_masks
    ]

    p_array = meta_cres.maps.get("p")
    if p_array is None or corr.method == "montecarlo":
        p_array = 10 ** -meta_cres.maps.get("logp_level-voxel_corr-FWE_method-montecarlo")

    n_iters = corr.parameters.get("n_iters")

    # ALE with MKDA kernel with montecarlo correction
    # combination gives poor performance
    if (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.MKDAKernel)
        and meta_cres.estimator.get_params().get("null_method") == "approximate"
        and corr.method != "montecarlo"
    ):
        good_sensitivity = True
        good_specificity = False
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.MKDAKernel)
        and "montecarlo" in meta_cres.estimator.get_params().get("null_method")
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.MKDAKernel)
        and meta_cres.estimator.get_params().get("null_method") == "approximate"
        and corr.method == "montecarlo"
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and type(meta_cres.estimator.kernel_transformer) == kernel.KDAKernel
        and (
            "montecarlo" in meta_cres.estimator.get_params().get("null_method")
            or (
                meta_cres.estimator.get_params().get("null_method") == "approximate"
                and corr.method == "montecarlo"
            )
        )
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and type(meta_cres.estimator.kernel_transformer) == kernel.KDAKernel
        and meta_cres.estimator.get_params().get("null_method") == "approximate"
    ):
        good_sensitivity = True
        good_specificity = False
    elif (
        isinstance(meta_cres.estimator, mkda.MKDADensity)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.ALEKernel)
        and meta_cres.estimator.get_params().get("null_method") != "reduced_montecarlo"
        and corr.method != "montecarlo"
    ):
        good_sensitivity = False
        good_specificity = True
    else:
        good_sensitivity = True
        good_specificity = True

    _check_p_values(
        p_array,
        meta_cres.masker,
        sig_idx,
        nonsig_idx,
        ALPHA,
        ground_truth_foci_ijks,
        n_iters=n_iters,
        good_sensitivity=good_sensitivity,
        good_specificity=good_specificity,
    )


# -----------------
# UTILITY FUNCTIONS
# -----------------


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
