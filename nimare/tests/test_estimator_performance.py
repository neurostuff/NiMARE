import os

import pytest
from contextlib import ExitStack as does_not_raise
import numpy as np

from ..correct import FDRCorrector, FWECorrector
from ..meta import ale, kernel, mkda
from ..meta.utils import compute_kda_ma
from ..transforms import mm2vox

# set significance levels used for testing.
ALPHA = 0.05
BETA = 1 - ALPHA

if os.environ.get("CIRCLECI"):
    N_CORES = 1
else:
    N_CORES = -1


@pytest.mark.parametrize(
    "meta_alg",
    [
        pytest.param(ale.ALE, id="ale"),
        pytest.param(mkda.MKDADensity, id="mkda"),
        pytest.param(mkda.KDA, id="kda"),
    ],
)
@pytest.mark.parametrize(
    "kern",
    [
        pytest.param(kernel.ALEKernel, id="ale_kernel"),
        pytest.param(kernel.MKDAKernel, id="mkda_kernel"),
        pytest.param(kernel.KDAKernel, id="kda_kernel"),
        pytest.param(kernel.Peaks2MapsKernel, id="p2m_kernel"),
    ],
)
@pytest.mark.parametrize(
    "corr",
    [
        pytest.param(FWECorrector(method="bonferroni"), id="fwe_bonferroni"),
        pytest.param(
            FWECorrector(method="montecarlo", voxel_thresh=ALPHA, n_iters=100, n_cores=N_CORES),
            id="fwe_montecarlo",
        ),
        pytest.param(FDRCorrector(method="indep", alpha=ALPHA), id="fdr_indep"),
        pytest.param(FDRCorrector(method="negcorr", alpha=ALPHA), id="fdr_negcorr"),
    ],
)
def test_estimators(simulatedata_cbma, meta_alg, kern, corr, mni_mask):

    # set up testing dataset
    fwhm, (ground_truth_foci, dataset) = simulatedata_cbma

    # instantiate KDA and MKDA with the appropriate radii (half of full wide half max)
    if kern == kernel.KDAKernel or kern == kernel.MKDAKernel:
        kern = kern(r=fwhm / 2)
    else:
        kern = kern()

    # create meta-analysis
    meta = meta_alg(kern)

    ####################################
    # CHECK IF META/KERNEL WORK TOGETHER
    ####################################
    # peaks2MapsKernel does not work with any meta-analysis estimator
    if isinstance(kern, kernel.Peaks2MapsKernel):
        # AttributeError: 'DataFrame' object has no attribute 'masker'
        meta_expectation = pytest.raises(AttributeError)
    elif isinstance(kern, kernel.KDAKernel) and isinstance(meta, ale.ALE):
        # IndexError: index 20000 is out of bounds for axis 0 with size 10010
        meta_expectation = pytest.raises(IndexError)
    else:
        meta_expectation = does_not_raise()

    with meta_expectation:
        res = meta.fit(dataset)
    # if creating the result failed (expected), do not continue
    if isinstance(meta_expectation, type(pytest.raises(ValueError))):
        pytest.xfail("this meta-analysis & kernel combo fails")
        return 0

    #######################################
    # CHECK IF META/CORRECTOR WORK TOGETHER
    #######################################
    # all combinations of meta-analysis estimators and multiple comparison correctors
    # that do not work together
    if isinstance(meta, mkda.MKDADensity) and isinstance(corr, FDRCorrector):
        # ValueError: <class 'nimare.correct.FDRCorrector'> requires "p" maps
        # to be present in the MetaResult, but none were found.
        corr_expectation = pytest.raises(ValueError)
    elif isinstance(meta, mkda.KDA) and isinstance(corr, FDRCorrector):
        # ValueError: <class 'nimare.correct.FDRCorrector'> requires "p" maps
        # to be present in the MetaResult, but none were found.
        corr_expectation = pytest.raises(ValueError)
    elif isinstance(meta, mkda.MKDADensity) and corr.method == "bonferroni":
        # ValueError: <class 'nimare.correct.FWECorrector'> requires "p" maps
        # to be present in the MetaResult, but none were found.
        corr_expectation = pytest.raises(ValueError)
    elif isinstance(meta, mkda.KDA) and corr.method == "montecarlo":
        # TypeError: correct_fwe_montecarlo() got an unexpected keyword argument 'voxel_thresh'
        corr_expectation = pytest.raises(TypeError)
    elif isinstance(meta, mkda.KDA) and corr.method == "bonferroni":
        # ValueError: <class 'nimare.correct.FWECorrector'> requires "p" maps
        # to be present in the MetaResult, but none were found.
        corr_expectation = pytest.raises(ValueError)
    else:
        corr_expectation = does_not_raise()

    with corr_expectation:
        cres = corr.transform(res)

    # if multiple correction failed (expected) do not continue
    if isinstance(corr_expectation, type(pytest.raises(ValueError))):
        pytest.xfail("this meta-analysis & corrector combo fails")
        return 0

    ################################################
    # CHECK IF P-VALUES ARE WITHIN THE CORRECT RANGE
    ################################################
    # default value to assume p-value outputs are correct
    contains_invalid_p_values = False
    if corr.method == "montecarlo":
        logp_values_img = cres.get_map(
            "logp_level-voxel_corr-FWE_method-montecarlo", return_type="image"
        )
        # transform logp values back into regular p values
        p_values_data = 10 ** -logp_values_img.get_fdata()

        if isinstance(meta, mkda.KDA) and isinstance(kern, kernel.MKDAKernel):
            # KDA with the MKDAKernel gives p-values twice as small as allowed
            assert p_values_data.min() >= (1.0 / (corr.parameters["n_iters"] * 2))
        else:
            # there should not be p-values less than 1 / n_iters
            assert p_values_data.min() >= (1.0 / corr.parameters["n_iters"])
        # max p-values should be less than 1, but several meta/kernel combinations
        # retain the max p-value of 1.
        if isinstance(meta, ale.ALE) and isinstance(kern, kernel.ALEKernel):
            contains_invalid_p_values = True
            assert p_values_data.max() == 1.0
        elif isinstance(meta, mkda.MKDADensity) and isinstance(kern, kernel.ALEKernel):
            contains_invalid_p_values = True
            assert p_values_data.max() == 1.0
        elif isinstance(meta, ale.ALE) and isinstance(kern, kernel.MKDAKernel):
            contains_invalid_p_values = True
            assert p_values_data.max() == 1.0
        elif isinstance(meta, mkda.MKDADensity) and isinstance(kern, kernel.MKDAKernel):
            contains_invalid_p_values = True
            assert p_values_data.max() == 1.0
        elif isinstance(meta, mkda.KDA) and isinstance(kern, kernel.MKDAKernel):
            contains_invalid_p_values = True
            assert p_values_data.max() == 1.0
        elif isinstance(meta, mkda.MKDADensity) and isinstance(kern, kernel.KDAKernel):
            contains_invalid_p_values = True
            assert p_values_data.max() == 1.0
        else:
            assert p_values_data.max() < 1.0
    else:
        p_values_img = cres.get_map("p", return_type="image")
        p_values_data = p_values_img.get_fdata()

    # ensure all values are between 0 and 1 (inclusively)
    assert (p_values_data >= 0).all() and (p_values_data <= 1).all()

    ################################################################
    # CHECK IF META/KERNEL/CORRECTOR COMBINATION PERFORMS ADEQUATELY
    ################################################################
    # The below combinations cannot even detect
    # significance at the ground truth foci voxel
    # locations
    if (
        (
            isinstance(meta, mkda.MKDADensity)
            and isinstance(kern, kernel.KDAKernel)
            and corr.method != "montecarlo"
        )
        or (
            isinstance(meta, mkda.MKDADensity)
            and isinstance(kern, kernel.MKDAKernel)
            and corr.method != "montecarlo"
        )
        or (
            isinstance(meta, ale.ALE)
            and isinstance(kern, kernel.MKDAKernel)
            and corr.method == "montecarlo"
        )
    ):
        good_performance = False
    else:
        good_performance = True

    ground_truth_foci_ijks = [tuple(mm2vox(focus, mni_mask.affine)) for focus in ground_truth_foci]
    # reformat coordinate indices to index p_values_data
    gtf_idx = [
        [ground_truth_foci_ijks[i][j] for i in range(len(ground_truth_foci_ijks))]
        for j in range(3)
    ]

    # the ground truth foci should be significant at minimum
    best_chance_p_values = p_values_data[gtf_idx]
    assert all(best_chance_p_values < ALPHA) == good_performance

    # double the radius of what is expected to be significant
    r = fwhm
    # create an expectation of significant/non-significant regions
    sig_regions, nonsig_regions = _create_signal_mask(mni_mask, ground_truth_foci_ijks, r=r)

    # while this combination passes the basic
    # test of significance at the ground truth foci voxels,
    # it does not meet the criteria for 50% power
    # around the ground truth foci
    if (
        isinstance(meta, mkda.MKDADensity)
        and isinstance(kern, kernel.MKDAKernel)
        and corr.method == "montecarlo"
    ) or (
        isinstance(meta, mkda.MKDADensity)
        and isinstance(kern, kernel.KDAKernel)
        and corr.method == "montecarlo"
    ):
        good_performance = False

    # assert that at least 50% of voxels surrounding the foci
    # are significant at alpha = .05
    observed_sig = p_values_data[sig_regions] < ALPHA
    observed_sig_perc = observed_sig.sum() / len(observed_sig)
    assert (observed_sig_perc >= 0.5) == good_performance

    # assert that more than 95% of voxels farther away
    # from foci are nonsignificant at alpha = 0.05
    mni_nonzero = np.nonzero(mni_mask.get_fdata())
    observed_nonsig = p_values_data[mni_nonzero][nonsig_regions[mni_nonzero]] > ALPHA
    observed_nonsig_perc = observed_nonsig.sum() / len(observed_nonsig)
    assert observed_nonsig_perc >= BETA

    if contains_invalid_p_values:
        pytest.xfail("this meta-analysis/kenrel/corrector combo contains invalid p-values")

    if not good_performance:
        pytest.xfail("this meta-analysis/kenrel/corrector combo has poor performance")

    # TODO: use output in reports
    return {
        "meta": meta,
        "kernel": kern,
        "corr": corr,
        "estimated_power": observed_sig_perc,
        "specificity": observed_nonsig_perc,
    }


def _create_signal_mask(mni_mask, ground_truth_foci_ijks, r=8):
    dims = mni_mask.shape
    vox_dims = mni_mask.header.get_zooms()
    prob_map = compute_kda_ma(
        dims,
        vox_dims,
        ground_truth_foci_ijks,
        r=r,
        value=1,
        allow_overlap=False,
    )
    sig_regions = prob_map == 1
    nonsig_regions = prob_map == 0

    return sig_regions, nonsig_regions
