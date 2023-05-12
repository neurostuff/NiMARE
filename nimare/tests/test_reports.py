"""Test nimare.reports."""
import os.path as op

from nimare import workflows
from nimare.reports.base import run_reports


def test_reports_function_smoke(tmp_path_factory, testdata_cbma_full):
    """Run smoke test for CBMA workflow."""
    # tmpdir = tmp_path_factory.mktemp("test_reports_function_smoke")
    output_dir = "/Users/jperaza/Desktop/test_report"
    results = workflows.cbma_workflow(
        testdata_cbma_full,
        estimator="ale",
        corrector="fdr",
        diagnostics="focuscounter",
        output_dir=output_dir,
    )

    run_reports(results, output_dir)

    filename = "report.html"
    outpath = op.join(output_dir, filename)
    assert op.isfile(outpath)
