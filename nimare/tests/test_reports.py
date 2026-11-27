"""Test nimare.reports."""

import os.path as op
import subprocess
import sys
from pathlib import Path

import pytest

from nimare.correct import FWECorrector
from nimare.diagnostics import FocusCounter, Jackknife
from nimare.meta.cbma import ALESubtraction
from nimare.meta.ibma import FixedEffectsHedges, Stouffers
from nimare.reports.base import Reportlet, _gen_diag_summary, run_reports
from nimare.workflows import CBMAWorkflow, IBMAWorkflow, PairwiseCBMAWorkflow


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics,meta_type",
    [
        ("ale", FWECorrector(method="montecarlo", n_iters=10), "jackknife", "cbma"),
        ("kda", "fdr", "focuscounter", "cbma"),
        (
            "mkdachi2",
            FWECorrector(method="montecarlo", n_iters=10),
            Jackknife(target_threshold=0.1),
            "pairwise_cbma",
        ),
        (
            ALESubtraction(n_iters=10),
            "fdr",
            FocusCounter(target_threshold=0.01, display_second_group=True),
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


def test_reports_alesubtraction_montecarlo_uses_pairwise_mass_map(
    tmp_path_factory,
    testdata_cbma_full,
):
    """Ensure ALESubtraction Monte Carlo maps are discovered and rendered in reports."""
    tmpdir = tmp_path_factory.mktemp("test_reports_alesubtraction_montecarlo")
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])

    workflow = PairwiseCBMAWorkflow(
        estimator=ALESubtraction(n_iters=2, n_cores=1, vfwe_only=False, voxel_thresh=0.05),
        corrector=FWECorrector(
            method="montecarlo",
            n_iters=2,
            n_cores=1,
            vfwe_only=False,
            voxel_thresh=0.05,
        ),
        diagnostics=FocusCounter(target_threshold=0.01, display_second_group=True),
        output_dir=tmpdir,
    )
    results = workflow.fit(dset1, dset2)
    run_reports(results, tmpdir)

    summary_file = op.join(tmpdir, "figures", "corrector_figure-summary.html")
    assert op.isfile(summary_file)
    with open(summary_file, encoding="utf-8") as fo:
        summary_text = fo.read()
    assert "z_desc-group1MinusGroup2Mass_level-cluster_corr-FWE_method-montecarlo" in summary_text


def test_reportlet_reads_utf8_html(tmp_path):
    """Reportlets should read generated HTML fragments as UTF-8 on all platforms."""
    expected_text = "caf\u00e9 \u2014 \u4f60\u597d"
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    html_file = figures_dir / "utf8_fragment.html"
    html_file.write_text(f"<div>{expected_text}</div>", encoding="utf-8")

    reportlet = Reportlet(
        Path(tmp_path),
        config={
            "name": "utf8_fragment",
            "bids": {"value": "utf8_fragment", "suffix": "html"},
        },
    )

    assert len(reportlet.components) == 1
    assert expected_text in reportlet.components[0][0]


@pytest.mark.parametrize(
    "kwargs",
    [{"target_threshold": 1.65}, {"voxel_thresh": 1.65}],
    ids=["target_threshold", "deprecated-alias"],
)
def test_diagnostic_summary_reports_the_threshold_that_was_used(tmp_path, kwargs, recwarn):
    """The summary must show the threshold clustering actually used.

    Workflows pass their ``voxel_thresh`` on as ``target_threshold``, so a template keyed on
    the deprecated ``voxel_thresh`` renders "None" for every workflow-generated report.
    """
    diagnostic = Jackknife(**kwargs)
    out_filename = tmp_path / "diagnostics_summary.html"

    _gen_diag_summary(diagnostic, out_filename)

    assert "1.65" in out_filename.read_text(encoding="UTF-8")


def test_importing_figures_keeps_matplotlib_backend():
    """Importing the report figures must not switch the session's matplotlib backend.

    A forced switch to Agg turned off inline figures in Jupyter. The import runs in a fresh
    interpreter because this session has already imported the module.
    """
    code = (
        "import matplotlib\n"
        "matplotlib.use('svg')\n"
        "import matplotlib.pyplot\n"
        "import nimare.reports.figures\n"
        "print(matplotlib.get_backend())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "svg"


def test_reorder_matrix_defined_only_matches_reorder_when_all_defined():
    """Without undefined values the heatmap ordering must not change."""
    import numpy as np

    from nimare.reports.figures import _reorder_matrix, _reorder_matrix_defined_only

    rng = np.random.default_rng(0)
    mat = rng.random((5, 4))
    rows = [f"s{i}" for i in range(5)]
    cols = [f"c{i}" for i in range(4)]

    expected = _reorder_matrix(mat, rows, cols)
    observed = _reorder_matrix_defined_only(mat, rows, cols)

    assert np.array_equal(expected[0], observed[0])
    assert expected[1] == observed[1]
    assert expected[2] == observed[2]


def test_reorder_matrix_defined_only_keeps_undefined_values_out_of_the_clustering():
    """Undefined values must not reach scipy's linkage, and must not be filled in."""
    import numpy as np

    from nimare.reports.figures import _reorder_matrix, _reorder_matrix_defined_only

    rng = np.random.default_rng(1)
    mat = rng.random((6, 5))
    mat[2, 1] = np.nan
    rows = [f"s{i}" for i in range(6)]
    cols = [f"c{i}" for i in range(5)]

    # The plain reordering cannot handle a NaN.
    with pytest.raises(ValueError):
        _reorder_matrix(mat, rows, cols)

    new_mat, new_rows, new_cols = _reorder_matrix_defined_only(mat, rows, cols)

    assert sorted(new_rows) == sorted(rows)
    assert sorted(new_cols) == sorted(cols)
    # The row holding the undefined value is set aside and placed last.
    assert new_rows[-1] == "s2"
    assert sorted(new_cols) == sorted(cols)
    # Values travel with their labels, and the single undefined value is still undefined.
    assert np.isnan(new_mat).sum() == 1
    for i, row in enumerate(new_rows):
        for j, col in enumerate(new_cols):
            expected = mat[rows.index(row), cols.index(col)]
            assert np.isclose(new_mat[i, j], expected, equal_nan=True)


def test_reorder_matrix_defined_only_symmetric():
    """A symmetric matrix with an undefined row/column is reordered on the rest."""
    import numpy as np

    from nimare.reports.figures import _reorder_matrix_defined_only

    rng = np.random.default_rng(2)
    base = rng.random((6, 6))
    mat = (base + base.T) / 2
    mat[3, :] = np.nan
    mat[:, 3] = np.nan
    labels = [f"s{i}" for i in range(6)]

    new_mat, new_rows, new_cols = _reorder_matrix_defined_only(
        mat, labels, labels, symmetric=True
    )

    assert new_rows == new_cols
    assert new_rows[-1] == "s3"
    assert np.isnan(new_mat[-1, :]).all()
    assert np.isnan(new_mat[:, -1]).all()
