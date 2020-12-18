import os

import pytest
from contextlib import ExitStack as does_not_raise
import numpy as np

from ..results import MetaResult
from ..generate import create_image_dataset
from ..correct import FDRCorrector, FWECorrector
from ..meta.ibma import (
    Fishers,
    Stouffers,
    WeightedLeastSquares,
    DerSimonianLaird,
    Hedges,
    SampleSizeBasedLikelihood,
    VarianceBasedLikelihood,
    PermutedOLS,
)

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
    np.random.seed(1939)


##########################################
# simulated dataset(s)
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            {
                "percent_signal_voxels": 0.10,
                "n_studies": 20,
                "n_participants": (5, 31),
                "variance": (1, 10),
                "img_dir": None,
                "seed": 1939,
                "neurosynth": False,
            },
            id="image_data",
        )
    ],
)
def simulatedata_ibma(request):
    return create_image_dataset(**request.param)


##########################################
# meta-analysis estimators
##########################################
@pytest.fixture(
    scope="session",
    params=[
        pytest.param((Fishers, {}), id="fishers"),
        pytest.param((Stouffers, {"use_sample_size": True}), id="stouffers+sample"),
        pytest.param((Stouffers, {"use_sample_size": False}), id="stouffers+nosample"),
        pytest.param((WeightedLeastSquares, {"tau2": 0}), id="weightedleastsquares+tau0"),
        pytest.param((WeightedLeastSquares, {"tau2": 1}), id="weightedleastsquares+tau1"),
        pytest.param((DerSimonianLaird, {}), id="dersimonianlaird"),
        pytest.param((Hedges, {}), id="Hedges"),
        pytest.param(
            (SampleSizeBasedLikelihood, {"method": "ml"}),
            id="samplesizebasedlikelihood+ml",
        ),
        pytest.param(
            (SampleSizeBasedLikelihood, {"method": "reml"}),
            id="samplesizebasedlikelihood+reml",
        ),
        pytest.param(
            (VarianceBasedLikelihood, {"method": "ml"}),
            id="variancebasedlikelihood+ml",
        ),
        pytest.param(
            (VarianceBasedLikelihood, {"method": "reml"}),
            id="variancebasedlikelihood+reml",
        ),
        pytest.param((PermutedOLS, {}), id="permutedols"),
    ],
)
def meta_est(request):
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
    return request.param


###########################################
# meta-analysis estimator results
###########################################
@pytest.fixture(scope="session")
def meta_result(simulatedata_ibma, meta_est, random):
    _, dataset = simulatedata_ibma
    meta, meta_kwargs = meta_est
    meta_expectation = does_not_raise()

    with meta_expectation:
        res = meta(**meta_kwargs).fit(dataset)
    # if creating the result failed (expected), do not continue
    if isinstance(meta_expectation, type(pytest.raises(ValueError))):
        pytest.xfail("this meta-analysis algorithm")

    return res


###########################################
# corrected results (testing)
###########################################
@pytest.fixture(scope="session")
def meta_cres(meta_est, meta_result, corr, random):
    return _transform_res(meta_est[0], meta_result, corr)


###########################################
# corrected results (smoke)
###########################################
@pytest.fixture(scope="session")
def meta_cres_small(meta_est, meta_result, corr_small, random):
    return _transform_res(meta_est[0], meta_result, corr_small)


# --------------
# TEST FUNCTIONS
# --------------


def test_meta_fit_smoke(meta_result):
    assert isinstance(meta_result, MetaResult)


@pytest.mark.performance
def test_meta_fit_performance(meta_result, simulatedata_ibma):
    pass


def test_corr_transform_smoke(meta_cres_small):
    assert isinstance(meta_cres_small, MetaResult)


@pytest.mark.performance
def test_corr_transform_performance(meta_cres, corr, simulatedata_ibma):
    pass


def _transform_res(meta, meta_result, corr):
    if (
        isinstance(corr, FWECorrector)
        and corr.method == "montecarlo"
        and isinstance(
            meta_result.estimator,
            (
                SampleSizeBasedLikelihood,
                WeightedLeastSquares,
                DerSimonianLaird,
                Hedges,
                VarianceBasedLikelihood,
                WeightedLeastSquares,
                Stouffers,
                Fishers,
            ),
        )
    ):
        # ValueError: Unsupported fwe correction method "montecarlo"
        corr_expectation = pytest.raises(ValueError)
    elif (
        isinstance(corr, FWECorrector)
        and corr.method == "montecarlo"
        and isinstance(meta_result.estimator, PermutedOLS)
    ):
        # Error: failed on setup with "TypeError:
        # correct_fwe_montecarlo() got an unexpected keyword argument 'voxel_thresh'"
        corr_expectation = pytest.raises(TypeError)
    elif (
        isinstance(corr, FWECorrector)
        and corr.method == "bonferroni"
        and isinstance(
            meta_result.estimator,
            (
                WeightedLeastSquares,
                DerSimonianLaird,
                Hedges,
                SampleSizeBasedLikelihood,
                VarianceBasedLikelihood,
            ),
        )
    ):
        # Error: failed on setup with
        # "IndexError: index 39 is out of bounds for axis 0 with size 1"
        corr_expectation = pytest.raises(IndexError)
    elif (
        isinstance(corr, FWECorrector)
        and corr.method == "bonferroni"
        and isinstance(meta_result.estimator, (PermutedOLS,))
    ):
        # Error: failed on setup with "ValueError: <class 'nimare.correct.FWECorrector'>
        # requires "p" maps to be present in the MetaResult, but none were found."
        corr_expectation = pytest.raises(ValueError)
    elif isinstance(corr, FDRCorrector) and isinstance(
        meta_result.estimator,
        (
            WeightedLeastSquares,
            DerSimonianLaird,
            Hedges,
            SampleSizeBasedLikelihood,
            VarianceBasedLikelihood,
        ),
    ):
        # pvals_corrected_[pvals_sortind] = pvals_corrected
        # IndexError: index 39 is out of bounds for axis 0 with size 1
        corr_expectation = pytest.raises(IndexError)
    elif isinstance(corr, FDRCorrector) and isinstance(meta_result.estimator, PermutedOLS):
        # Error: failed on setup with "ValueError: <class 'nimare.correct.FDRCorrector'>
        # requires "p" maps to be present in the MetaResult, but none were found."
        corr_expectation = pytest.raises(ValueError)
    else:
        corr_expectation = does_not_raise()

    with corr_expectation:
        cres = corr.transform(meta_result)

    # if multiple correction failed (expected) do not continue
    if isinstance(corr_expectation, type(pytest.raises(ValueError))):
        pytest.xfail("this meta-analysis & corrector combo fails")
    return cres
