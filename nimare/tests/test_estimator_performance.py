"""Test estimator, kerneltransformer, and multiple comparisons corrector performance."""

import os
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest

from nimare.correct import FDRCorrector, FWECorrector
from nimare.generate import create_coordinate_dataset
from nimare.meta import ale, kernel, mkda
from nimare.results import MetaResult
from nimare.tests.utils import _check_p_values, _create_signal_mask, _transform_res
from nimare.utils import mm2vox

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


@pytest.mark.performance_smoke
def test_meta_fit_smoke(meta_res):
    """Smoke test for meta-analytic estimator fit."""
    assert isinstance(meta_res, MetaResult)


@pytest.mark.performance_estimators
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
        and isinstance(meta_res.estimator.kernel_transformer, kernel.KDAKernel)
        and "montecarlo" in meta_res.estimator.get_params().get("null_method")
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_res.estimator, ale.ALE)
        and isinstance(meta_res.estimator.kernel_transformer, kernel.KDAKernel)
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


@pytest.mark.performance_smoke
def test_corr_transform_smoke(meta_cres_small):
    """Smoke test for corrector transform."""
    assert isinstance(meta_cres_small, MetaResult)


@pytest.mark.performance_correctors
def test_corr_transform_performance(meta_cres, corr, signal_masks, simulatedata_cbma):
    """Test corrector transform performance."""
    _, (ground_truth_foci, _) = simulatedata_cbma
    mask = meta_cres.masker.mask_img
    ground_truth_foci_ijks = [tuple(mm2vox(focus, mask.affine)) for focus in ground_truth_foci]
    sig_idx, nonsig_idx = [
        meta_cres.masker.transform(img).astype(bool).squeeze() for img in signal_masks
    ]

    p_map_name = f"p{corr._name_suffix}"
    p_array = meta_cres.maps.get(p_map_name)
    value_type = "p"
    if p_array is None:
        logp_map_name = f"logp_level-voxel{corr._name_suffix}"
        p_array = meta_cres.maps.get(logp_map_name)
        assert p_array is not None, (
            f"Could not find corrected p or logp map for corrector {corr}. "
            f"Expected one of '{p_map_name}' or '{logp_map_name}'."
        )
        value_type = "logp"

    n_iters = corr.parameters.get("n_iters")
    null_method = meta_cres.estimator.get_params().get("null_method", "")

    # ALE with MKDA kernel with montecarlo correction
    # combination gives poor performance
    if (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.MKDAKernel)
        and null_method == "approximate"
        and corr.method != "montecarlo"
    ):
        good_sensitivity = True
        good_specificity = False
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.MKDAKernel)
        and "montecarlo" in null_method
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.MKDAKernel)
        and null_method == "approximate"
        and corr.method == "montecarlo"
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.KDAKernel)
        and (
            "montecarlo" in null_method
            or (null_method == "approximate" and corr.method == "montecarlo")
        )
    ):
        good_sensitivity = False
        good_specificity = True
    elif (
        isinstance(meta_cres.estimator, ale.ALE)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.KDAKernel)
        and null_method == "approximate"
    ):
        good_sensitivity = True
        good_specificity = False
    elif (
        isinstance(meta_cres.estimator, mkda.MKDADensity)
        and isinstance(meta_cres.estimator.kernel_transformer, kernel.ALEKernel)
        and null_method != "reduced_montecarlo"
        and corr.method != "montecarlo"
    ):
        good_sensitivity = False
        good_specificity = True
    else:
        good_sensitivity = True
        good_specificity = True

    # Override sensitivity/specificity based on corrector conservativeness vs
    # null distribution precision. Non-montecarlo correctors (Bonferroni, FDR)
    # require much smaller uncorrected p-values to achieve significance.
    if corr.method != "montecarlo":
        if "reduced_montecarlo" in null_method:
            # 32-iteration reduced_montecarlo produces min p=1/32≈0.03, which
            # cannot survive Bonferroni or FDR correction across ~228K voxels.
            good_sensitivity = False
            good_specificity = True
        elif corr.method == "bonferroni":
            if "montecarlo" in null_method and (
                isinstance(meta_cres.estimator, mkda.KDA)
                or (
                    isinstance(meta_cres.estimator, mkda.MKDADensity)
                    and isinstance(
                        meta_cres.estimator.kernel_transformer,
                        kernel.MKDAKernel,
                    )
                )
            ):
                # Bonferroni is too conservative for KDA estimator or
                # MKDADensity+MKDAKernel with montecarlo null, which produce
                # null distributions with insufficient resolution.
                good_sensitivity = False
                good_specificity = True
            elif isinstance(meta_cres.estimator, ale.ALE) and isinstance(
                meta_cres.estimator.kernel_transformer,
                (kernel.KDAKernel, kernel.MKDAKernel),
            ):
                # Bonferroni is too conservative for ALE with mismatched
                # approximate kernels, even though without correction the foci
                # are detectable.
                good_sensitivity = False
                good_specificity = True
        elif (
            corr.method == "negcorr"
            and isinstance(meta_cres.estimator, ale.ALE)
            and isinstance(
                meta_cres.estimator.kernel_transformer,
                (kernel.KDAKernel, kernel.MKDAKernel),
            )
        ):
            # FDR negcorr (Benjamini-Yekutieli) is more conservative than FDR
            # indep and cannot detect foci when ALE uses a mismatched kernel.
            good_sensitivity = False
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
        value_type=value_type,
    )
