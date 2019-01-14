"""
Test nimare.dataset (Dataset IO/transformations).
"""
import os.path as op

import nimare
from nimare import dataset
from nimare.tests.utils import get_test_data_path


def test_database_smoke():
    """
    Smoke test for nimare.dataset.Database initialization.
    """
    db_file = op.join(get_test_data_path(), 'neurosynth_dset.json')
    dbase = dataset.Database(db_file)
    assert isinstance(dbase, nimare.dataset.Database)


def test_dataset_smoke():
    """
    Smoke test for nimare.dataset.Dataset initialization.
    """
    db_file = op.join(get_test_data_path(), 'neurosynth_dset.json')
    dbase = dataset.Database(db_file)
    dset = dbase.get_dataset()
    assert isinstance(dset, nimare.dataset.Dataset)
