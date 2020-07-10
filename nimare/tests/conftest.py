"""
Runs before tests
"""
import os.path as op
from glob import glob

import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask

import nimare
from nimare.extract import download_nidm_pain
from nimare.transforms import mm2vox
from nimare.utils import get_template, get_masker
from nimare.tests.utils import get_test_data_path


@pytest.fixture(scope='session')
def download_data():
    """
    Download and save 21 pain studies from NeuroVault to test IBMA functions.
    """
    nidm_path = download_nidm_pain()
    pytest.dset_dir = nidm_path
    return nidm_path


@pytest.fixture(scope='session')
def testdata(download_data):
    """
    Load data from dataset into global variables.
    """
    # Load dataset
    dset_file = op.join(get_test_data_path(), 'test_pain_dataset.json')
    dset_dir = op.join(get_test_data_path(), 'test_pain_dataset')
    dset = nimare.dataset.Dataset(dset_file)
    dset.update_path(dset_dir)

    # Only retain one peak in each study in coordinates
    # Otherwise centers of mass will be obscured in kernel tests by overlapping
    # kernels
    dset.coordinates = dset.coordinates.drop_duplicates(subset=['id'])
    return dset


@pytest.fixture(scope='session')
def testdata_laird():
    """
    Load data from dataset into global variables.
    """
    testdata_laird = nimare.dataset.Dataset.load(
        op.join(get_test_data_path(), 'neurosynth_laird_studies.pkl.gz'))
    return testdata_laird
