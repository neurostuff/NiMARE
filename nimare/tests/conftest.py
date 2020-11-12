"""
Runs before tests
"""
import os
from shutil import copyfile

import pytest
import nibabel as nib

import nimare
from nimare.tests.utils import get_test_data_path
from ..utils import get_resource_path


@pytest.fixture(scope="session")
def testdata_ibma(tmp_path_factory):
    """
    Load data from dataset into global variables.
    """
    tmpdir = tmp_path_factory.mktemp("testdata_ibma")

    # Load dataset
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    dset_dir = os.path.join(get_test_data_path(), "test_pain_dataset")
    mask_file = os.path.join(dset_dir, "mask.nii.gz")
    dset = nimare.dataset.Dataset(dset_file, mask=mask_file)
    dset.update_path(dset_dir)
    # Move image contents of Dataset to temporary directory
    for c in dset.images.columns:
        if c.endswith("__relative"):
            continue
        for f in dset.images[c].values:
            if (f is None) or not os.path.isfile(f):
                continue
            new_f = f.replace(
                dset_dir.rstrip(os.path.sep), str(tmpdir.absolute()).rstrip(os.path.sep)
            )
            dirname = os.path.dirname(new_f)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            copyfile(f, new_f)
    dset.update_path(tmpdir)
    return dset


@pytest.fixture(scope="session")
def testdata_cbma():
    dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
    dset = nimare.dataset.Dataset(dset_file)

    # Only retain one peak in each study in coordinates
    # Otherwise centers of mass will be obscured in kernel tests by overlapping
    # kernels
    dset.coordinates = dset.coordinates.drop_duplicates(subset=["id"])
    return dset


@pytest.fixture(scope="session")
def testdata_cbma_full():
    """Same as above, except returns all coords, not just one per study."""
    dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
    dset = nimare.dataset.Dataset(dset_file)
    return dset


@pytest.fixture(scope="session")
def testdata_laird():
    """
    Load data from dataset into global variables.
    """
    testdata_laird = nimare.dataset.Dataset.load(
        os.path.join(get_test_data_path(), "neurosynth_laird_studies.pkl.gz")
    )
    return testdata_laird


@pytest.fixture(scope="session")
def mni_mask():
    return nib.load(
        os.path.join(get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz")
    )
