"""
Runs before tests
"""
import os.path as op

import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask

import nimare
from nimare.utils import get_template, mm2vox, get_masker
from nimare.tests.utils import download_nidm_pain, get_test_data_path


@pytest.fixture(scope='session', autouse=True)
def download_data():
    """
    Download and save 21 pain studies from NeuroVault to test IBMA functions.
    """
    home = op.expanduser("~")
    out_dir = op.join(home, '.nimare', 'data', 'neurovault', 'collection-1425')
    if not op.isdir(op.join(out_dir, 'pain_21.nidm')):
        download_nidm_pain(out_dir)
    pytest.dset_dir = out_dir
    return out_dir


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
    z_data = apply_mask(z_imgs, dset.masker.mask_img)
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
    con_data = apply_mask(con_imgs, dset.masker.mask_img)
    se_data = apply_mask(se_imgs, dset.masker.mask_img)
    pytest.data_con = con_data
    pytest.data_se = se_data
    pytest.sample_sizes_con = sample_sizes


# Fixtures used in the rest of the tests
class DummyDataset(object):
    def __init__(self, df, img):
        self.coordinates = df
        self.masker = get_masker(img)

    def slice(self):
        pass


@pytest.fixture(scope='session', autouse=True)
def cbma_testdata1():
    mask_img = get_template(space='mni152_2mm', mask='brain')
    df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'n', 'space'],
                      data=[[1, -28, -20, -16, 100, 'mni'],
                            [2, -28, -20, -16, 100, 'mni'],
                            [3, -28, -20, -16, 100, 'mni'],
                            [4, -28, -20, -16, 100, 'mni'],
                            [5, -28, -20, -16, 100, 'mni'],
                            [6, -28, -20, -16, 100, 'mni'],
                            [7, -28, -20, -16, 100, 'mni'],
                            [8, -28, -20, -16, 100, 'mni'],
                            [9, -28, -20, -16, 100, 'mni'],
                            [10, -28, -20, -16, 100, 'mni'],
                            [11, -28, -20, -16, 100, 'mni']])
    xyz = df[['x', 'y', 'z']].values
    ijk = pd.DataFrame(mm2vox(xyz, mask_img.affine), columns=['i', 'j', 'k'])
    df = pd.concat([df, ijk], axis=1)

    dset = DummyDataset(df, mask_img)
    pytest.cbma_testdata1 = dset


@pytest.fixture(scope='session', autouse=True)
def cbma_testdata2():
    mask_img = get_template(space='mni152_2mm', mask='brain')
    df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'n', 'space'],
                      data=[[1, -24, -20, -16, 100, 'mni'],
                            [2, -24, -20, -16, 100, 'mni'],
                            [3, -24, -20, -16, 100, 'mni'],
                            [4, -24, -20, -16, 100, 'mni'],
                            [5, -24, -20, -16, 100, 'mni'],
                            [6, -24, -20, -16, 100, 'mni'],
                            [7, -24, -20, -16, 100, 'mni'],
                            [8, -24, -20, -16, 100, 'mni'],
                            [9, -24, -20, -16, 100, 'mni'],
                            [10, -24, -20, -16, 100, 'mni'],
                            [11, -24, -20, -16, 100, 'mni']])
    xyz = df[['x', 'y', 'z']].values
    ijk = pd.DataFrame(mm2vox(xyz, mask_img.affine), columns=['i', 'j', 'k'])
    df = pd.concat([df, ijk], axis=1)

    dset = DummyDataset(df, mask_img)
    pytest.cbma_testdata2 = dset


@pytest.fixture(scope='session', autouse=True)
def cbma_testdata3():
    """
    Reduced dataset for SCALE test.
    """
    mask_img = get_template(space='mni152_2mm', mask='brain')
    mask_img = nib.Nifti1Image(np.ones((10, 10, 10), int), mask_img.affine)
    df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'n', 'space'],
                      data=[[1, -28, -20, -16, 100, 'mni'],
                            [2, -28, -20, -16, 100, 'mni'],
                            [3, -28, -20, -16, 100, 'mni']])
    xyz = df[['x', 'y', 'z']].values
    ijk = pd.DataFrame(mm2vox(xyz, mask_img.affine), columns=['i', 'j', 'k'])
    df = pd.concat([df, ijk], axis=1)

    dset = DummyDataset(df, mask_img)
    pytest.cbma_testdata3 = dset
