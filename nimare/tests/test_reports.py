"""Test nimare.reports."""
import os.path as op

import pytest

from nimare import workflows
from nimare.correct import FWECorrector
from nimare.reports.base import run_reports


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics",
    [
        ("ale", FWECorrector(method="montecarlo", n_iters=10), "jackknife"),
        ("kda", "fdr", "focuscounter"),
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
    results = workflows.cbma_workflow(
        testdata_cbma_full,
        estimator=estimator,
        corrector=corrector,
        diagnostics=diagnostics,
        output_dir=tmpdir,
    )

    run_reports(results, tmpdir)

    filename = "report.html"
    outpath = op.join(tmpdir, filename)
    assert op.isfile(outpath)
