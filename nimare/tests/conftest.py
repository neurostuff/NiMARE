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
def testdata_ibma():
    """
    Load data from dataset into global variables.
    """
    # Load dataset
    dset_file = op.join(get_test_data_path(), 'test_pain_dataset.json')
    dset_dir = op.join(get_test_data_path(), 'test_pain_dataset')
    mask_file = op.join(dset_dir, 'mask.nii.gz')
    dset = nimare.dataset.Dataset(dset_file, mask=mask_file)
    dset.update_path(dset_dir)
    return dset


@pytest.fixture(scope='session')
def testdata_cbma():
    dset_file = op.join(get_test_data_path(), 'nidm_pain_dset.json')
    dset = nimare.dataset.Dataset(dset_file)

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
