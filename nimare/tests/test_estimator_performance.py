import pytest

from ..correct import FDRCorrector, FWECorrector
from ..meta import ale  # , kernel, mkda
from ..transforms import mm2vox


@pytest.mark.parametrize(
    "meta",
    [
        ale.ALE(),
    ],
)
@pytest.mark.parametrize(
    "corr",
    [
        FWECorrector(method="bonferroni"),
        FDRCorrector(method="indep", alpha=0.05),
        FDRCorrector(method="negcorr", alpha=0.05),
    ],
)
def test_estimators(simulatedata_cbma, meta, corr):
    ground_truth_foci, dataset = simulatedata_cbma

    res = meta.fit(dataset)

    cres = corr.transform(res)

    p_values_img = cres.get_map("p", return_type="image")
    p_values_data = p_values_img.get_data()
    ground_truth_foci_ijks = [
        tuple(mm2vox(focus, p_values_img.affine)) for focus in ground_truth_foci
    ]
    for ground_truth_focus in ground_truth_foci_ijks:
        assert p_values_data[ground_truth_focus] < 0.05
