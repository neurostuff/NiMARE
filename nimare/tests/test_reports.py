"""Test nimare.reports."""
import os.path as op

import pytest

from nimare.correct import FWECorrector
from nimare.meta.cbma import ALESubtraction
from nimare.meta.cbma.base import PairwiseCBMAEstimator
from nimare.reports.base import run_reports
from nimare.workflows import CBMAWorkflow, PairwiseCBMAWorkflow


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics,voxel_thresh",
    [
        ("ale", FWECorrector(method="montecarlo", n_iters=10), "jackknife", 1.65),
        ("kda", "fdr", "focuscounter", 1.65),
        ("mkdachi2", FWECorrector(method="montecarlo", n_iters=10), "jackknife", 0.1),
        (ALESubtraction(n_iters=10), "bonferroni", "focuscounter", 0.1),
    ],
)
def test_reports_function_smoke(
    tmp_path_factory,
    testdata_cbma_full,
    estimator,
    corrector,
    diagnostics,
    voxel_thresh,
):
    """Run smoke test for CBMA workflow."""
    tmpdir = tmp_path_factory.mktemp("test_reports_function_smoke")
    if estimator == "mkdachi2" or issubclass(type(estimator), PairwiseCBMAEstimator):
        dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
        dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])

        workflow = PairwiseCBMAWorkflow(
            estimator=estimator,
            corrector=corrector,
            diagnostics=diagnostics,
            voxel_thresh=voxel_thresh,
            output_dir=tmpdir,
        )
        results = workflow.fit(dset1, dset2)
    else:
        workflow = CBMAWorkflow(
            estimator=estimator,
            corrector=corrector,
            diagnostics=diagnostics,
            output_dir=tmpdir,
        )
        results = workflow.fit(testdata_cbma_full)

    run_reports(results, tmpdir)

    filename = "report.html"
    outpath = op.join(tmpdir, filename)
    assert op.isfile(outpath)
