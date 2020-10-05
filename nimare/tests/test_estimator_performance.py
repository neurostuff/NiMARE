import pytest
from contextlib import ExitStack as does_not_raise
import numpy as np

from ..correct import FDRCorrector, FWECorrector
from ..meta import ale, kernel, mkda
from ..meta.utils import compute_ma, get_ale_kernel
from ..transforms import mm2vox


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
        pytest.param(kernel.ALEKernel(), id="ale_kernel"),
        pytest.param(kernel.MKDAKernel(), id="mkda_kernel"),
        pytest.param(kernel.KDAKernel(), id="kda_kernel"),
        pytest.param(kernel.Peaks2MapsKernel(), id="p2m_kernel"),
    ],
)
@pytest.mark.parametrize(
    "corr",
    [
        pytest.param(FWECorrector(method="bonferroni"), id="fwe_bonferroni"),
        pytest.param(
            FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=10, n_cores=1),
            id="fwe_montecarlo",
        ),
        pytest.param(FDRCorrector(method="indep", alpha=0.05), id="fdr_indep"),
        pytest.param(FDRCorrector(method="negcorr", alpha=0.05), id="fdr_negcorr"),
    ],
)
def test_estimators(simulatedata_cbma, meta_alg, kern, corr, mni_mask):

    # set up testing dataset
    ground_truth_foci, dataset = simulatedata_cbma

    # create meta-analysis
    meta = meta_alg(kern)

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
        return 0

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
        return 0

    # ALE with the MKDAKernel does not perform well
    if isinstance(meta, ale.ALE) and isinstance(kern, kernel.MKDAKernel):
        good_performance = False
    else:
        good_performance = True

    if corr.method == "montecarlo":
        logp_values_img = cres.get_map(
            "logp_level-voxel_corr-FWE_method-montecarlo", return_type="image"
        )
        # transform logp values back into regular p values
        p_values_data = 10 ** -logp_values_img.get_fdata()
    else:
        p_values_img = cres.get_map("p", return_type="image")
        p_values_data = p_values_img.get_fdata()

    ground_truth_foci_ijks = [tuple(mm2vox(focus, mni_mask.affine)) for focus in ground_truth_foci]

    # reformat coordinate indices to index p_values_data
    gtf_idx = [
        [ground_truth_foci_ijks[i][j] for i in range(len(ground_truth_foci_ijks))]
        for j in range(3)
    ]

    # the ground truth foci should be significant at minimum
    best_chance_p_values = p_values_data[gtf_idx]
    assert all(best_chance_p_values < 0.05) == good_performance

    # create an expectation of significant/non-significant regions
    fwhm = 10
    sig_regions, nonsig_regions = _create_null_mask(mni_mask, ground_truth_foci_ijks, fwhm=fwhm)

    # assert that at least 50% of voxels surrounding the foci
    # are significant at alpha = .05
    observed_sig = p_values_data[sig_regions] < 0.05
    observed_sig_perc = observed_sig.sum() / len(observed_sig)
    assert (observed_sig_perc >= 0.5) == good_performance

    # assert that more than 95% of voxels farther away
    # from foci are nonsignificant at alpha = 0.05
    mni_nonzero = np.nonzero(mni_mask.get_fdata())
    observed_nonsig = p_values_data[mni_nonzero][nonsig_regions[mni_nonzero]] > 0.05
    observed_nonsig_perc = observed_nonsig.sum() / len(observed_nonsig)
    assert observed_nonsig_perc >= 0.95

    # TODO: use output in reports
    return {
        "meta": meta,
        "kernel": kern,
        "corr": corr,
        "sensitivity": observed_sig_perc,
        "specificity": observed_nonsig_perc,
    }


def _create_null_mask(mni_mask, ground_truth_foci_ijks, fwhm):
    threshold_percentage = 90
    _, kernel = get_ale_kernel(mni_mask, fwhm=fwhm)
    prob_map = compute_ma(mni_mask.shape, np.array(ground_truth_foci_ijks), kernel)
    threshold = np.percentile(prob_map[np.nonzero(prob_map)], threshold_percentage)
    sig_regions = prob_map > threshold
    nonsig_regions = prob_map < threshold
    return sig_regions, nonsig_regions
