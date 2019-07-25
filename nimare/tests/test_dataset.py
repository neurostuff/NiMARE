"""
Test nimare.dataset (Dataset IO/transformations).
"""
import os.path as op

import numpy as np

import nimare
from nimare import dataset
from nimare.tests.utils import get_test_data_path


def test_dataset_smoke():
    """
    Smoke test for nimare.dataset.Dataset initialization and get methods.
    """
    db_file = op.join(get_test_data_path(), 'neurosynth_dset.json')
    dset = dataset.Dataset(db_file)
    dset.update_path(get_test_data_path())
    assert isinstance(dset, nimare.dataset.Dataset)
    assert isinstance(dset.get_images(imtype='beta'), list)
    assert isinstance(dset.get_labels(), list)
    assert isinstance(dset.get_metadata(field='sample_sizes'), list)
    assert isinstance(dset.get_studies_by_label('cogat_cognitive_control'), list)
    assert isinstance(dset.get_studies_by_coordinate(
        np.array([[20, 20, 20]])), list)
