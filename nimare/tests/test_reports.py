"""Test nimare.reports."""

import os.path as op

import pytest

from nimare.correct import FWECorrector
from nimare.diagnostics import FocusCounter, Jackknife
from nimare.meta.cbma import ALESubtraction
from nimare.meta.ibma import FixedEffectsHedges, Stouffers
from nimare.reports.base import run_reports
from nimare.workflows import CBMAWorkflow, IBMAWorkflow, PairwiseCBMAWorkflow


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics,meta_type",
    [
        ("ale", FWECorrector(method="montecarlo", n_iters=10), "jackknife", "cbma"),
        ("kda", "fdr", "focuscounter", "cbma"),
        (
            "mkdachi2",
            FWECorrector(method="montecarlo", n_iters=10),
            Jackknife(voxel_thresh=0.1),
            "pairwise_cbma",
        ),
        (
            ALESubtraction(n_iters=10),
            "fdr",
            FocusCounter(voxel_thresh=0.01, display_second_group=True),
            "pairwise_cbma",
        ),
    ],
)
def test_reports_function_smoke(
    tmp_path_factory,
    testdata_cbma_full,
    estimator,
    corrector,
    diagnostics,
    meta_type,
):
    """Run smoke test for CBMA workflow."""
    tmpdir = tmp_path_factory.mktemp("test_reports_function_smoke")

    if meta_type == "cbma":
        workflow = CBMAWorkflow(
            estimator=estimator,
            corrector=corrector,
            diagnostics=diagnostics,
            output_dir=tmpdir,
        )
        results = workflow.fit(testdata_cbma_full)

    elif meta_type == "pairwise_cbma":
        dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
        dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])

        workflow = PairwiseCBMAWorkflow(
            estimator=estimator,
            corrector=corrector,
            diagnostics=diagnostics,
            output_dir=tmpdir,
        )
        results = workflow.fit(dset1, dset2)

    run_reports(results, tmpdir)

    filename = "report.html"
    outpath = op.join(tmpdir, filename)
    assert op.isfile(outpath)


@pytest.mark.parametrize("aggressive_mask", [True, False], ids=["aggressive", "liberal"])
def test_reports_ibma_smoke(tmp_path_factory, testdata_ibma, aggressive_mask):
    """Smoke test for IBMA reports."""
    tmpdir = tmp_path_factory.mktemp("test_reports_ibma_smoke")

    # Generate a report with z maps as inputs
    stouffers_dir = op.join(tmpdir, "stouffers")
    workflow = IBMAWorkflow(
        estimator=Stouffers(aggressive_mask=aggressive_mask),
        corrector="fdr",
        diagnostics="jackknife",
        voxel_thresh=3.2,
        output_dir=stouffers_dir,
    )
    results = workflow.fit(testdata_ibma)

    run_reports(results, stouffers_dir)

    filename = "report.html"
    outpath = op.join(stouffers_dir, filename)
    assert op.isfile(outpath)

    # Generate a report with t maps as inputs
    hedges_dir = op.join(tmpdir, "hedges")
    workflow = IBMAWorkflow(
        estimator=FixedEffectsHedges(aggressive_mask=aggressive_mask),
        corrector="fdr",
        diagnostics="jackknife",
        voxel_thresh=3.2,
        output_dir=hedges_dir,
    )
    results = workflow.fit(testdata_ibma)

    run_reports(results, hedges_dir)

    filename = "report.html"
    outpath = op.join(hedges_dir, filename)
    assert op.isfile(outpath)


def test_reports_ibma_multiple_contrasts_smoke(tmp_path_factory, testdata_ibma_multiple_contrasts):
    """Smoke test for IBMA reports for multiple contrasts."""
    tmpdir = tmp_path_factory.mktemp("test_reports_ibma_smoke")

    # Generate a report with z maps as inputs
    stouffers_dir = op.join(tmpdir, "stouffers")
    workflow = IBMAWorkflow(
        estimator=Stouffers(aggressive_mask=True),
        corrector="fdr",
        diagnostics="jackknife",
        voxel_thresh=3.2,
        output_dir=stouffers_dir,
    )
    results = workflow.fit(testdata_ibma_multiple_contrasts)

    run_reports(results, stouffers_dir)

    filename = "report.html"
    outpath = op.join(stouffers_dir, filename)
    assert op.isfile(outpath)
