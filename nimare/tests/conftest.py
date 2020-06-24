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
    dset_file = op.join(get_test_data_path(), 'nidm_pain_dset.json')
    dset = nimare.dataset.Dataset(dset_file)
    dset.update_path(pytest.dset_dir)

    # Only retain one peak in each study in coordinates
    # Otherwise centers of mass will be obscured in kernel tests by overlapping
    # kernels
    dset.coordinates = dset.coordinates.drop_duplicates(subset=['id'])

    # Ugly searching until better methods are implemented.
    ids_z = [id_ for id_ in dset.ids if dset.get_images(id_, imtype='z') is not None]
    files_z = dset.get_images(ids_z, imtype='z')
    sample_sizes_z = dset.get_metadata(ids_z, 'sample_sizes')
    sample_sizes_z = np.array([np.mean(sample_size) for sample_size in sample_sizes_z])

    # Create reduced dataset for ibma
    dset_z = dset.slice(ids_z)

    # Now get the actual data for esma
    imgs_z = [nib.load(f) for f in files_z]
    data_z = apply_mask(imgs_z, dset.masker.mask_img)

    # Ugly searching until better methods are implemented.
    beta_ids = [id_ for id_ in dset.ids if dset.get_images(id_, imtype='beta') is not None]
    se_ids = [id_ for id_ in dset.ids if dset.get_images(id_, imtype='se') is not None]
    betase_ids = sorted(list(set(beta_ids).intersection(se_ids)))

    # Create reduced dataset for ibma
    dset_betase = dset.slice(betase_ids)

    # Now get the actual data for esma
    beta_files = dset.get_images(betase_ids, imtype='beta')
    se_files = dset.get_images(betase_ids, imtype='se')
    sample_sizes_beta = dset.get_metadata(betase_ids, 'sample_sizes')
    sample_sizes_beta = np.array([np.mean(sample_size) for sample_size in sample_sizes_beta])
    beta_imgs = [nib.load(f) for f in beta_files]
    se_imgs = [nib.load(f) for f in se_files]
    data_beta = apply_mask(beta_imgs, dset.masker.mask_img)
    data_se = apply_mask(se_imgs, dset.masker.mask_img)
    testdata = {
        'dset': dset,
        'dset_z': dset_z,
        'data_z': data_z,
        'sample_sizes_z': sample_sizes_z,
        'dset_betase': dset_betase,
        'data_beta': data_beta,
        'data_se': data_se,
        'sample_sizes_beta': sample_sizes_beta
    }
    return testdata


@pytest.fixture(scope='session')
def testdata_laird():
    """
    Load data from dataset into global variables.
    """
    testdata_laird = nimare.dataset.Dataset.load(
        op.join(get_test_data_path(), 'neurosynth_laird_studies.pkl.gz'))
    return testdata_laird
