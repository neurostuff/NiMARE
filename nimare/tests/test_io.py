"""
Test nimare.dataset (Dataset IO/transformations).
"""
import os.path as op

import numpy as np

import nimare
from nimare import dataset
from nimare import io
from nimare.tests.utils import get_test_data_path


def test_convert_sleuth_to_dataset_smoke():
    """
    Smoke test for Sleuth text file conversion.
    """
    sleuth_file = op.join(get_test_data_path(), 'test_sleuth_file.txt')
    dset = io.convert_sleuth_to_dataset(sleuth_file)
    assert isinstance(dset, nimare.dataset.Dataset)


def test_convert_neurosynth_to_dataset_smoke():
    """
    Smoke test for Sleuth text file conversion.
    """
    db_file = op.join(get_test_data_path(), 'test_neurosynth_database.txt')
    features_file = op.join(get_test_data_path(), 'test_neurosynth_features.txt')
    dset = io.convert_neurosynth_to_dataset(db_file, features_file)
    assert isinstance(dset, nimare.dataset.Dataset)
