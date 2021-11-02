import pytest

from nimare.diagnostics import Jackknife
from nimare.meta import cbma, ibma


@pytest.mark.parametrize(
    "estimator,meta_type",
    [
        (cbma.ALE, "cbma"),
        (cbma.MKDADensity, "cbma"),
        (cbma.KDA, "cbma"),
        (ibma.Fishers, "ibma"),
        (ibma.Stouffers, "ibma"),
        (ibma.WeightedLeastSquares, "ibma"),
        (ibma.DerSimonianLaird, "ibma"),
        (ibma.Hedges, "ibma"),
        # (ibma.SampleSizeBasedLikelihood, "ibma"),
        # (ibma.VarianceBasedLikelihood, "ibma"),
        (ibma.PermutedOLS, "ibma"),
    ],
)
def test_Jackknife(testdata_ibma, testdata_cbma_full, estimator, meta_type):
    meta = estimator()
    testdata = testdata_ibma if meta_type == "ibma" else testdata_cbma_full
    res = meta.fit(testdata)

    jackknife = Jackknife(target_image="z", voxel_thresh=1.65)
    cluster_table, cluster_img = jackknife.transform(res)

    assert cluster_table.shape[0] == len(meta.inputs_["id"]) + 1
