"""Test nimare.dataset (Dataset IO/transformations)."""
import os.path as op

import nibabel as nib
import numpy as np
import pytest

import nimare
from nimare import dataset
from nimare.tests.utils import get_test_data_path


def test_dataset_smoke():
    """Smoke test for nimare.dataset.Dataset initialization and get methods."""
    db_file = op.join(get_test_data_path(), "neurosynth_dset.json")
    dset = dataset.Dataset(db_file)
    dset.update_path(get_test_data_path())
    assert isinstance(dset, nimare.dataset.Dataset)
    # Test that Dataset.masker is portable
    assert not nib.is_proxy(dset.masker.mask_img_.dataobj)

    methods = [dset.get_images, dset.get_labels, dset.get_metadata, dset.get_texts]
    for method in methods:
        assert isinstance(method(), list)
        assert isinstance(method(ids=dset.ids[:5]), list)
        assert isinstance(method(ids=dset.ids[0]), list)

    assert isinstance(dset.get_images(imtype="beta"), list)
    assert isinstance(dset.get_metadata(field="sample_sizes"), list)
    assert isinstance(dset.get_studies_by_label("cogat_cognitive_control"), list)
    assert isinstance(dset.get_studies_by_coordinate(np.array([[20, 20, 20]])), list)

    # If label is not available, raise ValueError
    with pytest.raises(ValueError):
        dset.get_studies_by_label("dog")

    mask_data = np.zeros(dset.masker.mask_img.shape, int)
    mask_data[40, 40, 40] = 1
    mask_img = nib.Nifti1Image(mask_data, dset.masker.mask_img.affine)
    assert isinstance(dset.get_studies_by_mask(mask_img), list)

    dset1 = dset.slice(dset.ids[:5])
    dset2 = dset.slice(dset.ids[5:])
    assert isinstance(dset1, dataset.Dataset)
    dset_merged = dset1.merge(dset2)
    assert isinstance(dset_merged, dataset.Dataset)


def test_empty_dset():
    """Smoke test for initialization with an empty Dataset."""
    # dictionary with no information
    minimal_dict = {"study-0": {"contrasts": {"1": {}}}}
    dataset.Dataset(minimal_dict)
