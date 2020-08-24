"""
Test nimare.io (Dataset IO/transformations).
"""
import os

import pytest

import nimare
from nimare import io
from nimare.tests.utils import get_test_data_path


def test_convert_sleuth_to_dataset_smoke():
    """
    Smoke test for Sleuth text file conversion.
    """
    sleuth_file = os.path.join(get_test_data_path(), "test_sleuth_file.txt")
    sleuth_file2 = os.path.join(get_test_data_path(), "test_sleuth_file2.txt")
    # Use one input file
    dset = io.convert_sleuth_to_dataset(sleuth_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert dset.coordinates.shape[0] == 7
    assert len(dset.ids) == 3
    # Use two input files
    dset2 = io.convert_sleuth_to_dataset([sleuth_file, sleuth_file2])
    assert isinstance(dset2, nimare.dataset.Dataset)
    assert dset2.coordinates.shape[0] == 11
    assert len(dset2.ids) == 5
    # Use invalid input
    with pytest.raises(ValueError):
        io.convert_sleuth_to_dataset(5)


def test_convert_sleuth_to_json_smoke():
    """
    Smoke test for Sleuth text file conversion.
    """
    out_file = os.path.abspath("temp.json")
    sleuth_file = os.path.join(get_test_data_path(), "test_sleuth_file.txt")
    sleuth_file2 = os.path.join(get_test_data_path(), "test_sleuth_file2.txt")
    # Use one input file
    io.convert_sleuth_to_json(sleuth_file, out_file)
    dset = nimare.dataset.Dataset(out_file)
    assert os.path.isfile(out_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert dset.coordinates.shape[0] == 7
    assert len(dset.ids) == 3
    os.remove(out_file)
    # Use two input files
    io.convert_sleuth_to_json([sleuth_file, sleuth_file2], out_file)
    dset2 = nimare.dataset.Dataset(out_file)
    assert isinstance(dset2, nimare.dataset.Dataset)
    assert dset2.coordinates.shape[0] == 11
    assert len(dset2.ids) == 5
    # Use invalid input
    with pytest.raises(ValueError):
        io.convert_sleuth_to_json(5, out_file)


def test_convert_neurosynth_to_dataset_smoke():
    """
    Smoke test for Neurosynth file conversion.
    """
    db_file = os.path.join(get_test_data_path(), "test_neurosynth_database.txt")
    features_file = os.path.join(get_test_data_path(), "test_neurosynth_features.txt")
    dset = io.convert_neurosynth_to_dataset(db_file, features_file)
    assert isinstance(dset, nimare.dataset.Dataset)


def test_convert_neurosynth_to_json_smoke():
    """
    Smoke test for Neurosynth file conversion.
    """
    out_file = os.path.abspath("temp.json")
    db_file = os.path.join(get_test_data_path(), "test_neurosynth_database.txt")
    features_file = os.path.join(get_test_data_path(), "test_neurosynth_features.txt")
    io.convert_neurosynth_to_json(db_file, out_file, annotations_file=features_file)
    dset = nimare.dataset.Dataset(out_file)
    assert os.path.isfile(out_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    os.remove(out_file)
