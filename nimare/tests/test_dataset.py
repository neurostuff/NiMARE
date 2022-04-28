"""Test nimare.dataset (Dataset IO/transformations)."""
import os.path as op

import nibabel as nib
import numpy as np
import pytest

import nimare
from nimare import dataset
from nimare.tests.utils import get_test_data_path


def test_DatasetSearcher(testdata_laird):
    """Test the DatasetSearcher class."""
    dset = testdata_laird.copy()
    searcher = dataset.DatasetSearcher
    METHODS = [searcher.get_images, searcher.get_labels, searcher.get_metadata, searcher.get_texts]
    for method in METHODS:
        assert isinstance(method(dset), list)
        assert isinstance(method(dset, ids=dset.ids[:5]), list)
        assert isinstance(method(dset, ids=dset.ids[0]), list)

    assert isinstance(searcher.get_images(dset, imtype="beta"), list)
    assert isinstance(searcher.get_metadata(dset, field="sample_sizes"), list)
    assert isinstance(searcher.get_studies_by_label(dset, "cogat_cognitive_control"), list)
    assert isinstance(searcher.get_studies_by_coordinate(dset, np.array([[20, 20, 20]])), list)

    mask_data = np.zeros(dset.masker.mask_img.shape, int)
    mask_data[40, 40, 40] = 1
    mask_img = nib.Nifti1Image(mask_data, dset.masker.mask_img.affine)
    assert isinstance(dset.get_studies_by_mask(mask_img), list)

    # If label is not available, raise ValueError
    with pytest.raises(ValueError):
        searcher.get_studies_by_label(dset, "dog")


def test_dataset_smoke():
    """Smoke test for nimare.dataset.Dataset initialization and get methods."""
    db_file = op.join(get_test_data_path(), "neurosynth_dset.json")
    dset = dataset.Dataset(db_file)
    dset.update_path(get_test_data_path())
    assert isinstance(dset, nimare.dataset.Dataset)
    # Test that Dataset.masker is portable
    assert not nib.is_proxy(dset.masker.mask_img_.dataobj)

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
