"""Generate fixtures for tests."""

import json
import os
from shutil import copyfile

import nibabel as nib
import numpy as np
import pytest
from nilearn.image import resample_img
from requests import request

import nimare
from nimare.generate import create_coordinate_dataset
from nimare.tests.utils import get_test_data_path
from nimare.utils import get_resource_path

# Only enable the following once in a while for a check for SettingWithCopyWarnings
# pd.options.mode.chained_assignment = "raise"


@pytest.fixture(scope="session")
def testdata_ibma(tmp_path_factory):
    """Load data from dataset into global variables."""
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
    """Generate coordinate-based dataset for tests."""
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    dset = nimare.dataset.Dataset(dset_file)

    # Only retain one peak in each study in coordinates
    # Otherwise centers of mass will be obscured in kernel tests by overlapping
    # kernels
    dset.coordinates = dset.coordinates.drop_duplicates(subset=["id"])
    return dset


@pytest.fixture(scope="session")
def testdata_cbma_full():
    """Generate more complete coordinate-based dataset for tests.

    Same as above, except returns all coords, not just one per study.
    """
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    dset = nimare.dataset.Dataset(dset_file)
    return dset


@pytest.fixture(scope="session")
def testdata_cbmr_simulated():
    """Simulate coordinate-based dataset for tests."""
    # simulate
    ground_truth_foci, dset = create_coordinate_dataset(
        foci=10, sample_size=(20, 40), n_studies=1000, seed=100
    )
    # set up group columns: diagnosis & drug_status
    n_rows = dset.annotations.shape[0]
    dset.annotations["diagnosis"] = [
        "schizophrenia" if i % 2 == 0 else "depression" for i in range(n_rows)
    ]
    dset.annotations["drug_status"] = ["Yes" if i % 2 == 0 else "No" for i in range(n_rows)]
    dset.annotations["drug_status"] = (
        dset.annotations["drug_status"].sample(frac=1).reset_index(drop=True)
    )  # random shuffle drug_status column
    # set up moderators: sample sizes & avg_age
    dset.annotations["sample_sizes"] = [dset.metadata.sample_sizes[i][0] for i in range(n_rows)]
    dset.annotations["avg_age"] = np.arange(n_rows)
    dset.annotations["schizophrenia_subtype"] = [
        "type1",
        "type2",
        "type3",
        "type4",
        "type5",
    ] * int(n_rows / 5)
    dset.annotations["schizophrenia_subtype"] = (
        dset.annotations["schizophrenia_subtype"].sample(frac=1).reset_index(drop=True)
    )  # random shuffle drug_status column

    return dset


@pytest.fixture(scope="session")
def testdata_laird():
    """Load data from dataset into global variables."""
    testdata_laird = nimare.dataset.Dataset(
        os.path.join(get_test_data_path(), "neurosynth_laird_studies.json")
    )
    return testdata_laird


@pytest.fixture(scope="session")
def mni_mask():
    """Load MNI mask for testing."""
    return nib.load(
        os.path.join(get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz")
    )


@pytest.fixture(scope="session")
def roi_img():
    """Load MNI mask for testing."""
    return nib.load(os.path.join(get_test_data_path(), "amygdala_roi.nii.gz"))


@pytest.fixture(scope="session")
def testdata_ibma_resample(tmp_path_factory):
    """Create dataset for image-based resampling tests."""
    tmpdir = tmp_path_factory.mktemp("testdata_ibma_resample")

    # Load dataset
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    dset_dir = os.path.join(get_test_data_path(), "test_pain_dataset")
    mask_file = os.path.join(dset_dir, "mask.nii.gz")
    dset = nimare.dataset.Dataset(dset_file, mask=mask_file)
    dset.update_path(dset_dir)

    # create reproducible random number generator for resampling
    rng = np.random.default_rng(seed=123)
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

            # create random affine to make images different shapes
            affine = np.eye(3)
            np.fill_diagonal(affine, rng.choice([1, 2, 3]))
            img = resample_img(
                nib.load(f),
                target_affine=affine,
                interpolation="linear",
                clip=True,
            )
            nib.save(img, new_f)
    dset.update_path(tmpdir)
    return dset


@pytest.fixture(scope="session")
def example_nimads_studyset():
    """Download/lookup example NiMADS studyset."""
    out_file = os.path.join(get_test_data_path(), "nimads_studyset.json")
    if not os.path.isfile(out_file):
        url = "https://neurostore.org/api/studysets/Cv2LLUqG76W9?nested=true"
        response = request("GET", url)
        with open(out_file, "wb") as f:
            f.write(response.content)
    with open(out_file, "r") as f:
        studyset = json.load(f)
    return studyset


@pytest.fixture(scope="session")
def example_nimads_annotation():
    """Download/lookup example NiMADS annotation."""
    out_file = os.path.join(get_test_data_path(), "nimads_annotation.json")
    if not os.path.isfile(out_file):
        url = "https://neurostore.org/api/annotations/76PyNqoTNEsE"
        response = request("GET", url)
        with open(out_file, "wb") as f:
            f.write(response.content)
    with open(out_file, "r") as f:
        annotation = json.load(f)
    return annotation
