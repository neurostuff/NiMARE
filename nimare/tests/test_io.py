"""
Test nimare.io (Dataset IO/transformations).
"""
import os

import nimare
from nimare import io
from nimare.tests.utils import get_test_data_path


def test_convert_sleuth_to_dataset_smoke():
    """
    Smoke test for Sleuth text file conversion.
    """
    sleuth_file = os.path.join(get_test_data_path(), "test_sleuth_file.txt")
    dset = io.convert_sleuth_to_dataset(sleuth_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert dset.coordinates.shape[0] == 7
    assert len(dset.ids) == 3


def test_convert_sleuth_to_json_smoke():
    """
    Smoke test for Sleuth text file conversion.
    """
    out_file = os.path.abspath("temp.json")
    sleuth_file = os.path.join(get_test_data_path(), "test_sleuth_file.txt")
    io.convert_sleuth_to_json(sleuth_file, out_file)
    dset = nimare.dataset.Dataset(out_file)
    assert os.path.isfile(out_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert dset.coordinates.shape[0] == 7
    assert len(dset.ids) == 3
    os.remove(out_file)


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
