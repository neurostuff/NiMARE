import pytest
from contextlib import ExitStack as does_not_raise
import numpy as np

from ..correct import FDRCorrector, FWECorrector
from ..meta import ale, kernel, mkda
from ..meta.utils import compute_ma, get_ale_kernel
from ..transforms import mm2vox


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(ale.ALE(), id="ale"),
        pytest.param(mkda.MKDADensity(kernel.MKDAKernel(r=10)), id="mkda"),
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
def test_estimators(simulatedata_cbma, meta, corr, mni_mask):
    # mkda only works with the FWECorrector with the montecarlo method
    if (isinstance(meta, mkda.MKDADensity) and isinstance(corr, FDRCorrector)) or (
        isinstance(meta, mkda.MKDADensity) and corr.method != "montecarlo"
    ):
        expectation = pytest.raises(ValueError)
    else:
        expectation = does_not_raise()

    ground_truth_foci, dataset = simulatedata_cbma

    res = meta.fit(dataset)

    with expectation:
        cres = corr.transform(res)

    if isinstance(expectation, does_not_raise):
        if corr.method == "montecarlo":
            logp_values_img = cres.get_map(
                "logp_level-voxel_corr-FWE_method-montecarlo", return_type="image"
            )
            # transform logp values back into regular p values
            p_values_data = 10 ** -logp_values_img.get_fdata()
        else:
            p_values_img = cres.get_map("p", return_type="image")
            p_values_data = p_values_img.get_fdata()

        ground_truth_foci_ijks = [
            tuple(mm2vox(focus, mni_mask.affine)) for focus in ground_truth_foci
        ]

        # the ground truth foci should be significant at minimum
        for ground_truth_focus in ground_truth_foci_ijks:
            assert p_values_data[ground_truth_focus] < 0.05

        # create an expectation of significant/non-significant regions
        fwhm = 10
        sig_regions, nonsig_regions = _create_null_mask(
            mni_mask, ground_truth_foci_ijks, fwhm=fwhm
        )

        # assert that at least 60% of voxels surrounding the foci
        # are significant at alpha = .05
        observed_sig = p_values_data[sig_regions] < 0.05
        assert observed_sig.sum() / len(observed_sig) >= 0.6

        # assert that more than 95% of voxels farther away
        # from foci are nonsignificant at alpha = 0.05
        mni_nonzero = np.nonzero(mni_mask.get_fdata())
        observed_nonsig = p_values_data[mni_nonzero][nonsig_regions[mni_nonzero]] > 0.05
        assert observed_nonsig.sum() / len(observed_nonsig) >= 0.95


def _create_null_mask(mni_mask, ground_truth_foci_ijks, fwhm):
    threshold_percentage = 90
    _, kernel = get_ale_kernel(mni_mask, fwhm=fwhm)
    prob_map = compute_ma(mni_mask.shape, np.array(ground_truth_foci_ijks), kernel)
    threshold = np.percentile(prob_map[np.nonzero(prob_map)], threshold_percentage)
    sig_regions = prob_map > threshold
    nonsig_regions = prob_map < threshold
    return sig_regions, nonsig_regions
