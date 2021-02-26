"""Test nimare.dataset (Dataset IO/transformations)."""
import os.path as op

import nibabel as nib
import numpy as np

import nimare
from nimare import dataset
from nimare.tests.utils import get_test_data_path


def test_dataset_smoke():
    """Smoke test for nimare.dataset.Dataset initialization and get methods."""
    db_file = op.join(get_test_data_path(), "neurosynth_dset.json")
    dset = dataset.Dataset(db_file)
    dset.update_path(get_test_data_path())
    assert isinstance(dset, nimare.dataset.Dataset)
    methods = [dset.get_images, dset.get_labels, dset.get_metadata, dset.get_texts]
    for method in methods:
        assert isinstance(method(), list)
        assert isinstance(method(ids=dset.ids[:5]), list)
        assert isinstance(method(ids=dset.ids[0]), list)
    assert isinstance(dset.get_images(imtype="beta"), list)
    assert isinstance(dset.get_metadata(field="sample_sizes"), list)
    assert isinstance(dset.get_studies_by_label("cogat_cognitive_control"), list)
    assert isinstance(dset.get_studies_by_coordinate(np.array([[20, 20, 20]])), list)
    mask_data = np.zeros(dset.masker.mask_img.shape, int)
    mask_data[40, 40, 40] = 1
    mask_img = nib.Nifti1Image(mask_data, dset.masker.mask_img.affine)
    assert isinstance(dset.get_studies_by_mask(mask_img), list)


def test_empty_dset():
    """Smoke test for initialization with an empty Dataset."""
    # dictionary with no information
    minimal_dict = {"study-0": {"contrasts": {"1": {}}}}
    dataset.Dataset(minimal_dict)
