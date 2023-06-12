"""Test nimare.reports."""
import os.path as op

import pytest

from nimare.correct import FWECorrector
from nimare.diagnostics import FocusCounter, Jackknife
from nimare.meta.cbma import ALESubtraction
from nimare.meta.cbma.base import PairwiseCBMAEstimator
from nimare.reports.base import run_reports
from nimare.workflows import CBMAWorkflow, PairwiseCBMAWorkflow


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics",
    [
        ("ale", FWECorrector(method="montecarlo", n_iters=10), "jackknife"),
        ("kda", "fdr", "focuscounter"),
        (
            "mkdachi2",
            FWECorrector(method="montecarlo", n_iters=10),
            Jackknife(voxel_thresh=0.1),
        ),
        (
            ALESubtraction(n_iters=10),
            "fdr",
            FocusCounter(voxel_thresh=0.01, display_second_group=True),
        ),
    ],
)
def test_reports_function_smoke(
    tmp_path_factory,
    testdata_cbma_full,
    estimator,
    corrector,
    diagnostics,
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
