"""
Runs before tests
"""
import os.path as op

import pytest
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask

import nimare
from nimare.tests.utils import download_nidm_pain, get_test_data_path


@pytest.fixture(scope='session', autouse=True)
def download_data(tmpdir_factory):
    """
    Download 21 pain studies from NeuroVault to test IBMA functions.
    """
    tst_dir = tmpdir_factory.mktemp('tests')
    out_dir = tst_dir.ensure('resources',
                             'data',
                             'neurovault-data',
                             'collection-1425',
                             dir=True)
    if not op.isdir(op.join(out_dir, 'pain_21.nidm')):
        download_nidm_pain(out_dir)
    pytest.dset_dir = out_dir
    return tst_dir


@pytest.fixture(scope='session', autouse=True)
def get_data(download_data):
    """
    Load data from dataset into global variables.
    """
    # Load dataset
    dset_file = op.join(get_test_data_path(), 'nidm_pain_dset.json')
    dset = nimare.dataset.Dataset(dset_file)
    dset.update_path(pytest.dset_dir)

    # Ugly searching until better methods are implemented.
    z_ids = [id_ for id_ in dset.ids if dset.get_images(id_, imtype='z') is not None]
    z_files = dset.get_images(z_ids, imtype='z')
    sample_sizes = dset.get_metadata(z_ids, 'sample_sizes')
    sample_sizes = np.array([np.mean(n) for n in sample_sizes])

    # Create reduced dataset for ibma
    pytest.dset_z = dset.slice(z_ids)

    # Now get the actual data for esma
    z_imgs = [nib.load(f) for f in z_files]
    z_data = apply_mask(z_imgs, dset.mask)
    pytest.data_z = z_data
    pytest.sample_sizes_z = sample_sizes

    # Ugly searching until better methods are implemented.
    con_ids = [id_ for id_ in dset.ids if dset.get_images(id_, imtype='con') is not None]
    se_ids = [id_ for id_ in dset.ids if dset.get_images(id_, imtype='se') is not None]
    conse_ids = sorted(list(set(con_ids).intersection(se_ids)))

    # Create reduced dataset for ibma
    pytest.dset_conse = dset.slice(conse_ids)

    # Now get the actual data for esma
    con_files = dset.get_images(conse_ids, imtype='con')
    se_files = dset.get_images(conse_ids, imtype='se')
    sample_sizes = dset.get_metadata(conse_ids, 'sample_sizes')
    sample_sizes = np.array([np.mean(n) for n in sample_sizes])
    con_imgs = [nib.load(f) for f in con_files]
    se_imgs = [nib.load(f) for f in se_files]
    con_data = apply_mask(con_imgs, dset.mask)
    se_data = apply_mask(se_imgs, dset.mask)
    pytest.data_con = con_data
    pytest.data_se = se_data
    pytest.sample_sizes_con = sample_sizes
