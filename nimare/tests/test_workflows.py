"""
Test nimare.workflows.
"""
import os
import os.path as op
import shutil

import numpy as np

import nimare
from nimare import cli, dataset, io, workflows
from nimare.tests.utils import get_test_data_path


def test_ale_workflow_smoke():
    """
    Smoke test for Sleuth ALE workflow.
    """
    sleuth_file = op.join(get_test_data_path(), 'test_sleuth_file.txt')
    out_dir = op.join(os.getcwd(), 'TEST')
    prefix = 'test'

    # The same test is run with both workflow function and CLI
    workflows.ale_sleuth_workflow(
        sleuth_file, output_dir=out_dir, prefix=prefix, n_iters=10, n_cores=1)
    assert op.isfile(op.join(out_dir, '{}_input_coordinates.txt'.format(prefix)))
    shutil.rmtree(out_dir)

    args = ['ale', '--output_dir', out_dir, '--prefix', prefix,
            '--n_iters', '10', '--n_cores', '1', sleuth_file]
    cli._main(args)
    assert op.isfile(op.join(out_dir, '{}_input_coordinates.txt'.format(prefix)))
    shutil.rmtree(out_dir)


def test_ale_workflow_smoke_2():
    """
    Smoke test for Sleuth ALE workflow with subtraction analysis
    """
    sleuth_file = op.join(get_test_data_path(), 'test_sleuth_file.txt')
    out_dir = op.join(os.getcwd(), 'TEST')
    prefix = 'test'

    # The same test is run with both workflow function and CLI
    workflows.ale_sleuth_workflow(
        sleuth_file, sleuth_file2=sleuth_file,
        output_dir=out_dir, prefix=prefix,
        n_iters=10, n_cores=1)
    assert op.isfile(op.join(out_dir, '{}_group2_input_coordinates.txt'.format(prefix)))
    shutil.rmtree(out_dir)

    args = ['ale', '--output_dir', out_dir, '--prefix', prefix,
            '--n_iters', '10', '--n_cores', '1',
            '--file2', sleuth_file, sleuth_file]
    cli._main(args)
    assert op.isfile(op.join(out_dir, '{}_group2_input_coordinates.txt'.format(prefix)))
    shutil.rmtree(out_dir)


def test_scale_workflow_smoke_1():
    """
    """
    sleuth_file = op.join(get_test_data_path(), 'test_sleuth_file.txt')
    out_dir = op.join(os.getcwd(), 'TEST')
    prefix = 'test'

    # The same test is run with both workflow function and CLI
    workflows.scale_workflow(
        sleuth_file,
        output_dir=out_dir, prefix=prefix,
        n_iters=5, n_cores=1)
    assert op.isfile(op.join(out_dir, '{}_input_coordinates.txt'.format(prefix)))
    shutil.rmtree(out_dir)

    args = ['scale', '--output_dir', out_dir, '--prefix', prefix,
            '--n_iters', '5', '--n_cores', '1',
            sleuth_file]
    cli._main(args)
    assert op.isfile(op.join(out_dir, '{}_input_coordinates.txt'.format(prefix)))
    shutil.rmtree(out_dir)
