"""
Test nimare.meta.ibma (image-based meta-analytic algorithms).
"""
import json
import os.path as op

import pytest
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask

import nimare
from nimare.meta import ibma
from nimare.stats import t_to_z
from nimare.tests.utils import get_test_data_path, download_nidm_pain


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
    download_nidm_pain(out_dir)
    return tst_dir


def _get_file(cdict, t, data_dir):
    """
    Return the file associated with a given data type within a
    folder if it exists. Otherwise, returns an empty list.
    """
    temp = ''
    if t == 'con':
        temp = cdict['images'].get('con')
    elif t == 'se':
        temp = cdict['images'].get('se')
    elif t == 't':
        temp = cdict['images'].get('t')
    elif t == 'z':
        temp = cdict['images'].get('z')
    elif t == 't!z':
        # Get t-image only if z-image doesn't exist
        temp = cdict['images'].get('z')
        if temp is None:
            temp = cdict['images'].get('t')
        else:
            temp = None
    elif t == 'n':
        temp = cdict.get('sample_sizes', [])
        if temp:
            temp = np.mean(temp)
    else:
        raise Exception('Input type "{0}" not recognized.'.format(t))

    if isinstance(temp, str):
        temp = op.join(data_dir, temp)
    return temp


def get_files(ddict, types, data_dir=None):
    """
    Returns a list of files associated with a given data type
    from a set of subfolders within a directory. Allows for
    multiple data types and only returns a set of files from folders
    with all of the requested types.
    """
    if data_dir is None:
        data_dir = get_test_data_path()

    all_files = []
    for study in ddict.keys():
        files = []
        cdict = ddict[study]['contrasts']['1']
        for t in types:
            temp = _get_file(cdict, t, data_dir)
            if temp:
                files.append(temp)

        if len(files) == len(types):
            all_files.append(files)
    all_files = list(map(list, zip(*all_files)))
    return all_files


@pytest.fixture(scope='session', autouse=True)
def get_data(download_data):
    """
    Load data from dataset into global variables.
    """
    # Load dataset
    dset_file = op.join(get_test_data_path(), 'nidm_pain_dset.json')
    with open(dset_file, 'r') as fo:
        dset_dict = json.load(fo)
    dset = nimare.dataset.Dataset(dset_file)
    pytest.dset_dict = dset_dict
    pytest.mask_img = dset.mask

    # Regular z maps
    z_files, ns = get_files(pytest.dset_dict, ['z', 'n'], download_data)
    z_imgs = [nib.load(op.join(download_data, f)) for f in z_files]
    z_data = apply_mask(z_imgs, pytest.mask_img)

    # T maps to be converted to z
    t_files, t_ns = get_files(pytest.dset_dict, ['t!z', 'n'], download_data)
    t_imgs = [nib.load(op.join(download_data, f)) for f in t_files]
    t_data_list = [apply_mask(t_img, pytest.mask_img) for t_img in t_imgs]
    tz_data_list = [t_to_z(t_data, t_ns[i] - 1) for i, t_data
                    in enumerate(t_data_list)]
    tz_data = np.vstack(tz_data_list)

    # Combine
    z_data = np.vstack((z_data, tz_data))
    ns = np.concatenate((ns, t_ns))
    sample_sizes = np.array(ns)
    pytest.z_data = z_data
    pytest.sample_sizes_z = sample_sizes

    con_files, se_files, ns = get_files(dset_dict, ['con', 'se', 'n'], download_data)
    con_imgs = [nib.load(op.join(download_data, f)) for f in con_files]
    se_imgs = [nib.load(op.join(download_data, f)) for f in se_files]
    con_data = apply_mask(con_imgs, pytest.mask_img)
    se_data = apply_mask(se_imgs, pytest.mask_img)
    sample_sizes = np.array(ns)
    pytest.con_data = con_data
    pytest.se_data = se_data
    pytest.sample_sizes_con = sample_sizes


def test_z_perm():
    """
    Smoke test for z permutation.
    """
    result = ibma.stouffers(pytest.z_data, pytest.mask_img,
                            inference='rfx', null='empirical', n_iters=10,
                            corr='FDR')
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_stouffers_ffx():
    """
    Smoke test for Stouffer's FFX.
    """
    result = ibma.stouffers(pytest.z_data, pytest.mask_img,
                            inference='ffx', null='theoretical', n_iters=None)
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_stouffers_rfx():
    """
    Smoke test for Stouffer's RFX.
    """
    result = ibma.weighted_stouffers(pytest.z_data, pytest.sample_sizes_z,
                                     pytest.mask_img)
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_weighted_stouffers():
    """
    Smoke test for Weighted Stouffer's.
    """
    result = ibma.stouffers(pytest.z_data, pytest.mask_img,
                            inference='rfx', null='theoretical', n_iters=None)
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_fishers():
    """
    Smoke test for Fisher's.
    """
    result = ibma.fishers(pytest.z_data, pytest.mask_img)
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_con_perm():
    """
    Smoke test for contrast permutation.
    """
    result = ibma.rfx_glm(pytest.con_data, pytest.mask_img, null='empirical',
                          n_iters=10, corr='FDR')
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_rfx_glm():
    """
    Smoke test for RFX GLM.
    """
    result = ibma.rfx_glm(pytest.con_data, pytest.mask_img, null='theoretical',
                          n_iters=None)
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_ffx_glm():
    """
    Smoke test for FFX GLM.
    """
    result = ibma.ffx_glm(pytest.con_data, pytest.se_data,
                          pytest.sample_sizes_con, pytest.mask_img)
    assert isinstance(result, nimare.base.meta.MetaResult)


def test_mfx_glm():
    """
    Smoke test for MFX GLM.
    """
    result = ibma.mfx_glm(pytest.con_data, pytest.se_data,
                          pytest.sample_sizes_con, pytest.mask_img)
    assert isinstance(result, nimare.base.meta.MetaResult)
