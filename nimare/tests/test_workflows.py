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
    assert op.isfile(op.join(tmpdir, "{}_input_coordinates.txt".format(prefix)))


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
    assert op.isfile(op.join(tmpdir, "{}_input_coordinates.txt".format(prefix)))


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
    assert op.isfile(op.join(tmpdir, "{}_group2_input_coordinates.txt".format(prefix)))


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
    assert op.isfile(op.join(tmpdir, "{}_group2_input_coordinates.txt".format(prefix)))


def test_scale_workflow_function_smoke(tmp_path_factory):
    """Run smoke test of the SCALE workflow as a function."""
    tmpdir = tmp_path_factory.mktemp("test_scale_workflow_function_smoke")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"
    baseline = op.join(get_test_data_path(), "test_baseline.txt")

    # The same test is run with both workflow function and CLI
    workflows.scale_workflow(
        sleuth_file, baseline=baseline, output_dir=tmpdir, prefix=prefix, n_iters=5, n_cores=1
    )
    assert op.isfile(op.join(tmpdir, "{}_input_coordinates.txt".format(prefix)))


def test_scale_workflow_cli_smoke(tmp_path_factory):
    """Run smoke test of the SCALE workflow as a CLI."""
    tmpdir = tmp_path_factory.mktemp("test_scale_workflow_cli_smoke")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"
    baseline = op.join(get_test_data_path(), "test_baseline.txt")

    cli._main(
        [
            "scale",
            "--baseline",
            baseline,
            "--output_dir",
            str(tmpdir),
            "--prefix",
            prefix,
            "--n_iters",
            "5",
            "--n_cores",
            "1",
            sleuth_file,
        ]
    )
    assert op.isfile(op.join(tmpdir, "{}_input_coordinates.txt".format(prefix)))


def test_conperm_workflow_function_smoke(testdata_ibma, tmp_path_factory):
    """Run smoke test of the contrast permutation workflow as a function."""
    tmpdir = tmp_path_factory.mktemp("test_conperm_workflow_function_smoke")
    dset = testdata_ibma
    files = dset.get_images(imtype="beta")
    mask_image = op.join(get_test_data_path(), "test_pain_dataset", "mask.nii.gz")
    prefix = "test"

    # The same test is run with both workflow function and CLI
    workflows.conperm_workflow(
        files, mask_image=mask_image, output_dir=tmpdir, prefix=prefix, n_iters=5
    )
    assert op.isfile(op.join(tmpdir, "{}_logp.nii.gz".format(prefix)))


def test_conperm_workflow_cli_smoke(testdata_ibma, tmp_path_factory):
    """Run smoke test of the contrast permutation workflow as a CLI."""
    tmpdir = tmp_path_factory.mktemp("test_conperm_workflow_cli_smoke")
    dset = testdata_ibma
    files = dset.get_images(imtype="beta")
    mask_image = op.join(get_test_data_path(), "test_pain_dataset", "mask.nii.gz")
    prefix = "test"

    cli._main(
        [
            "conperm",
            "--output_dir",
            str(tmpdir),
            "--mask",
            mask_image,
            "--prefix",
            prefix,
            "--n_iters",
            "5",
        ]
        + files
    )
    assert op.isfile(op.join(tmpdir, "{}_logp.nii.gz".format(prefix)))
