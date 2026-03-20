"""Test nimare.dataset (Dataset IO/transformations)."""

import copy
import json
import os.path as op
import pickle
import warnings

import nibabel as nib
import numpy as np
import pytest

import nimare
from nimare import dataset
from nimare.nimads import Studyset
from nimare.tests.utils import get_test_data_path

# ---------------------------------------------------------------------------
# Helpers for the parameterized smoke test
# ---------------------------------------------------------------------------


def _make_dataset():
    """Create a Dataset from neurosynth test data."""
    db_file = op.join(get_test_data_path(), "neurosynth_dset.json")
    dset = dataset.Dataset(db_file)
    dset.update_path(get_test_data_path())
    return dset


def _make_studyset():
    """Create a Studyset from neurosynth_laird test data."""
    ss = Studyset(
        op.join(get_test_data_path(), "neurosynth_laird_studyset.json"),
        target="mni152_2mm",
    )
    return ss


@pytest.fixture(params=["dataset", "studyset"])
def dset_or_studyset(request):
    """Parameterized fixture returning either a Dataset or Studyset."""
    if request.param == "dataset":
        return _make_dataset()
    return _make_studyset()


# ---------------------------------------------------------------------------
# Parameterized tests exercising shared Dataset / Studyset API
# ---------------------------------------------------------------------------


def test_smoke_ids(dset_or_studyset):
    """Both Dataset and Studyset expose a non‑empty ``ids`` array."""
    obj = dset_or_studyset
    assert len(obj.ids) > 0


def test_smoke_masker(dset_or_studyset):
    """Both Dataset and Studyset expose a masker with a valid mask image."""
    obj = dset_or_studyset
    assert obj.masker is not None
    # NiftiMasker stores the fitted mask in mask_img_
    assert not nib.is_proxy(obj.masker.mask_img_.dataobj)


def test_smoke_get_methods(dset_or_studyset):
    """get_images / get_labels / get_metadata / get_texts return lists."""
    obj = dset_or_studyset
    methods = [obj.get_images, obj.get_labels, obj.get_metadata, obj.get_texts]
    for method in methods:
        assert isinstance(method(), list)
        assert isinstance(method(ids=obj.ids[:5]), list)
        assert isinstance(method(ids=obj.ids[0]), list)


def test_smoke_get_studies_by_label(dset_or_studyset):
    """get_studies_by_label returns a list and rejects unknown labels."""
    obj = dset_or_studyset
    labels = obj.get_labels()
    assert len(labels) > 0
    result = obj.get_studies_by_label(labels[0])
    assert isinstance(result, list)

    with pytest.raises(ValueError):
        obj.get_studies_by_label("this_label_does_not_exist_xyz")


def test_smoke_get_studies_by_coordinate(dset_or_studyset):
    """get_studies_by_coordinate returns a list."""
    obj = dset_or_studyset
    result = obj.get_studies_by_coordinate(np.array([[20, 20, 20]]))
    assert isinstance(result, list)


def test_smoke_get_studies_by_mask(dset_or_studyset):
    """get_studies_by_mask returns a list."""
    obj = dset_or_studyset
    mask_data = np.zeros(obj.masker.mask_img.shape, np.int32)
    mask_data[40, 40, 40] = 1
    mask_img = nib.Nifti1Image(mask_data, obj.masker.mask_img.affine)
    result = obj.get_studies_by_mask(mask_img)
    assert isinstance(result, list)


def test_smoke_slice_merge(dset_or_studyset):
    """Slice and merge round‑trip preserves type."""
    obj = dset_or_studyset
    obj1 = obj.slice(obj.ids[:5])
    obj2 = obj.slice(obj.ids[5:])
    assert type(obj1) is type(obj)
    merged = obj1.merge(obj2)
    assert type(merged) is type(obj)


def test_smoke_copy(dset_or_studyset):
    """Copy returns an independent instance of the same type."""
    obj = dset_or_studyset
    copied = obj.copy()
    assert type(copied) is type(obj)
    assert copied is not obj


def test_smoke_pickle_roundtrip(dset_or_studyset, tmp_path):
    """Both Dataset and Studyset survive a pickle round‑trip."""
    obj = dset_or_studyset
    if isinstance(obj, Studyset):
        _ = obj.studies
    pkl_file = tmp_path / "roundtrip.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(obj, f)
    with open(pkl_file, "rb") as f:
        restored = pickle.load(f)

    assert type(restored) is type(obj)
    np.testing.assert_array_equal(restored.ids, obj.ids)
    assert len(restored.get_labels()) == len(obj.get_labels())


# ---------------------------------------------------------------------------
# Dataset-only tests
# ---------------------------------------------------------------------------


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

    mask_data = np.zeros(dset.masker.mask_img.shape, np.int32)
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


def test_posneg_warning():
    """Smoke test for nimare.dataset.Dataset initialization with positive and negative z_stat."""
    db_file = op.join(get_test_data_path(), "neurosynth_dset.json")
    with open(db_file, "r") as f_obj:
        data = json.load(f_obj)

    data_pos_zstats = copy.deepcopy(data)
    data_neg_zstats = copy.deepcopy(data)
    for pid in data.keys():
        for expid in data[pid]["contrasts"].keys():
            exp = data[pid]["contrasts"][expid]

            if "coords" not in exp.keys():
                continue

            if "z_stat" not in exp["coords"].keys():
                continue

            n_zstats = len(exp["coords"]["z_stat"])
            rand_arr = np.random.randn(n_zstats)
            rand_pos_arr = np.abs(rand_arr)
            rand_neg_arr = np.abs(rand_arr) * -1

            data[pid]["contrasts"][expid]["coords"]["z_stat"] = rand_arr.tolist()
            data_neg_zstats[pid]["contrasts"][expid]["coords"]["z_stat"] = rand_neg_arr.tolist()
            data_pos_zstats[pid]["contrasts"][expid]["coords"]["z_stat"] = rand_pos_arr.tolist()

    # Test Warning is raised if there are positive and negative z-stat
    with pytest.warns(UserWarning, match=r"positive and negative z_stats"):
        dset_posneg = dataset.Dataset(data)

    # Test Warning is not raised if there are only positive or negative z-stat
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dset_pos = dataset.Dataset(data_pos_zstats)
        dset_neg = dataset.Dataset(data_neg_zstats)

    assert isinstance(dset_posneg, nimare.dataset.Dataset)
    assert isinstance(dset_pos, nimare.dataset.Dataset)
    assert isinstance(dset_neg, nimare.dataset.Dataset)
