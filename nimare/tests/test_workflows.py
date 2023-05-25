"""Test nimare.workflows."""
import os.path as op

import pytest

import nimare
from nimare import cli, workflows
from nimare.correct import FWECorrector
from nimare.diagnostics import FocusCounter, Jackknife
from nimare.meta.cbma import ALE, MKDAChi2
from nimare.meta.ibma import Fishers
from nimare.tests.utils import get_test_data_path


def test_ale_workflow_function_smoke(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_function_smoke")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"

    # The same test is run with both workflow function and CLI
    workflows.ale_sleuth_workflow(
        sleuth_file, output_dir=tmpdir, prefix=prefix, n_iters=10, n_cores=1
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_input_coordinates.txt"))


def test_ale_workflow_cli_smoke(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_cli_smoke")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"

    cli._main(
        [
            "ale",
            "--output_dir",
            str(tmpdir),
            "--prefix",
            prefix,
            "--n_iters",
            "10",
            "--n_cores",
            "1",
            sleuth_file,
        ]
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_input_coordinates.txt"))


def test_ale_workflow_function_smoke_2(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow with subtraction analysis."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_function_smoke_2")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"

    # The same test is run with both workflow function and CLI
    workflows.ale_sleuth_workflow(
        sleuth_file,
        sleuth_file2=sleuth_file,
        output_dir=tmpdir,
        prefix=prefix,
        n_iters=10,
        n_cores=1,
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_group2_input_coordinates.txt"))


def test_ale_workflow_cli_smoke_2(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow with subtraction analysis."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_cli_smoke_2")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"
    cli._main(
        [
            "ale",
            "--output_dir",
            str(tmpdir),
            "--prefix",
            prefix,
            "--n_iters",
            "10",
            "--n_cores",
            "1",
            "--file2",
            sleuth_file,
            sleuth_file,
        ]
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_group2_input_coordinates.txt"))


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics",
    [
        (ALE, FWECorrector(method="montecarlo", n_iters=10), [Jackknife]),
        ("ales", "bonferroni", Jackknife),
        ("ale", "bonferroni", [Jackknife, FocusCounter]),
        ("kda", "fdr", Jackknife),
        ("mkdadensity", "fdr", "focuscounter"),
        (MKDAChi2, "montecarlo", None),
        (Fishers, "montecarlo", "jackknife"),
    ],
)
def test_cbma_workflow_function_smoke(
    tmp_path_factory,
    testdata_cbma_full,
    estimator,
    corrector,
    diagnostics,
):
    """Run smoke test for CBMA workflow."""
    tmpdir = tmp_path_factory.mktemp("test_cbma_workflow_function_smoke")

    if estimator == MKDAChi2:
        with pytest.raises(AttributeError):
            workflows.cbma_workflow(
                testdata_cbma_full,
                estimator=estimator,
                corrector=corrector,
                diagnostics=diagnostics,
            )
    elif estimator == Fishers:
        with pytest.raises((AttributeError, ValueError)):
            workflows.cbma_workflow(
                testdata_cbma_full,
                estimator=estimator,
                corrector=corrector,
                diagnostics=diagnostics,
            )
    elif estimator == "ales":
        with pytest.raises(ValueError):
            workflows.cbma_workflow(
                testdata_cbma_full,
                estimator=estimator,
                corrector=corrector,
                diagnostics=diagnostics,
            )
    else:
        cres = workflows.cbma_workflow(
            testdata_cbma_full,
            estimator=estimator,
            corrector=corrector,
            diagnostics=diagnostics,
            output_dir=tmpdir,
        )

        assert isinstance(cres, nimare.results.MetaResult)
        assert op.isfile(op.join(tmpdir, "boilerplate.txt"))
        assert op.isfile(op.join(tmpdir, "references.bib"))

        for imgtype in cres.maps.keys():
            filename = imgtype + ".nii.gz"
            outpath = op.join(tmpdir, filename)
            # For estimator == ALE, maps are None
            if estimator != ALE:
                assert op.isfile(outpath)

        for tabletype in cres.tables.keys():
            filename = tabletype + ".tsv"
            outpath = op.join(tmpdir, filename)
            # For estimator == ALE, tables are None
            if estimator != ALE:
                assert op.isfile(outpath)
