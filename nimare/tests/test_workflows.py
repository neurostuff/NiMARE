"""Test nimare.workflows."""
import os.path as op

from nimare import cli, workflows
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
