"""Test nimare.io (Dataset IO/transformations)."""
import os

import pytest

import nimare
from nimare import io
from nimare.tests.utils import get_test_data_path
from nimare.utils import get_template


def test_convert_sleuth_to_dataset_smoke():
    """Smoke test for Sleuth text file conversion."""
    sleuth_file = os.path.join(get_test_data_path(), "test_sleuth_file.txt")
    sleuth_file2 = os.path.join(get_test_data_path(), "test_sleuth_file2.txt")
    sleuth_file3 = os.path.join(get_test_data_path(), "test_sleuth_file3.txt")
    sleuth_file4 = os.path.join(get_test_data_path(), "test_sleuth_file4.txt")
    sleuth_file5 = os.path.join(get_test_data_path(), "test_sleuth_file5.txt")
    # Use one input file
    dset = io.convert_sleuth_to_dataset(sleuth_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert dset.coordinates.shape[0] == 7
    assert len(dset.ids) == 3
    # Use two input files
    dset2 = io.convert_sleuth_to_dataset([sleuth_file, sleuth_file2])
    assert isinstance(dset2, nimare.dataset.Dataset)
    assert dset2.coordinates.shape[0] == 11
    assert len(dset2.ids) == 5
    # Use invalid input
    with pytest.raises(ValueError):
        io.convert_sleuth_to_dataset(5)
    # Use invalid input (one coordinate is a str instead of a number)
    with pytest.raises(ValueError):
        io.convert_sleuth_to_dataset(sleuth_file3)
    # Use invalid input (one has x & y, but not z)
    with pytest.raises(ValueError):
        io.convert_sleuth_to_dataset(sleuth_file4)
    # Use invalid input (bad space)
    with pytest.raises(ValueError):
        io.convert_sleuth_to_dataset(sleuth_file5)


def test_convert_sleuth_to_json_smoke():
    """Smoke test for Sleuth text file conversion."""
    out_file = os.path.abspath("temp.json")
    sleuth_file = os.path.join(get_test_data_path(), "test_sleuth_file.txt")
    sleuth_file2 = os.path.join(get_test_data_path(), "test_sleuth_file2.txt")
    sleuth_file3 = os.path.join(get_test_data_path(), "test_sleuth_file3.txt")
    sleuth_file4 = os.path.join(get_test_data_path(), "test_sleuth_file4.txt")
    sleuth_file5 = os.path.join(get_test_data_path(), "test_sleuth_file5.txt")
    # Use one input file
    io.convert_sleuth_to_json(sleuth_file, out_file)
    dset = nimare.dataset.Dataset(out_file)
    assert os.path.isfile(out_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert dset.coordinates.shape[0] == 7
    assert len(dset.ids) == 3
    os.remove(out_file)
    # Use two input files
    io.convert_sleuth_to_json([sleuth_file, sleuth_file2], out_file)
    dset2 = nimare.dataset.Dataset(out_file)
    assert isinstance(dset2, nimare.dataset.Dataset)
    assert dset2.coordinates.shape[0] == 11
    assert len(dset2.ids) == 5
    # Use invalid input (number instead of file)
    with pytest.raises(ValueError):
        io.convert_sleuth_to_json(5, out_file)
    # Use invalid input (one coordinate is a str instead of a number)
    with pytest.raises(ValueError):
        io.convert_sleuth_to_json(sleuth_file3, out_file)
    # Use invalid input (one has x & y, but not z)
    with pytest.raises(ValueError):
        io.convert_sleuth_to_json(sleuth_file4, out_file)
    # Use invalid input (bad space)
    with pytest.raises(ValueError):
        io.convert_sleuth_to_json(sleuth_file5, out_file)


def test_convert_neurosynth_to_dataset_smoke():
    """Smoke test for Neurosynth file conversion."""
    db_file = os.path.join(get_test_data_path(), "test_neurosynth_database.txt")
    features_file = os.path.join(get_test_data_path(), "test_neurosynth_features.txt")
    dset = io.convert_neurosynth_to_dataset(db_file, features_file)
    assert isinstance(dset, nimare.dataset.Dataset)


def test_convert_neurosynth_to_json_smoke():
    """Smoke test for Neurosynth file conversion."""
    out_file = os.path.abspath("temp.json")
    db_file = os.path.join(get_test_data_path(), "test_neurosynth_database.txt")
    features_file = os.path.join(get_test_data_path(), "test_neurosynth_features.txt")
    io.convert_neurosynth_to_json(db_file, out_file, annotations_file=features_file)
    dset = nimare.dataset.Dataset(out_file)
    assert os.path.isfile(out_file)
    assert isinstance(dset, nimare.dataset.Dataset)
    os.remove(out_file)


@pytest.mark.parametrize(
    "kwargs",
    [
        (
            {
                "collection_ids": (8836,),
                "contrasts": {"animal": "as-Animal"},
            }
        ),
        (
            {
                "collection_ids": {"informative_name": 8836},
                "contrasts": {"animal": "as-Animal"},
                "map_type_conversion": {"T map": "t"},
                "target": "mni152_2mm",
                "mask": get_template("mni152_2mm", mask="brain"),
            }
        ),
        (
            {
                "collection_ids": (6348, 6419),
                "contrasts": {"action": "action"},
                "map_type_conversion": {"univariate-beta map": "beta"},
            }
        ),
        (
            {
                "collection_ids": (778,),  # collection not found
                "contrasts": {"action": "action"},
                "map_type_conversion": {"univariate-beta map": "beta"},
            }
        ),
    ],
)
def test_convert_neurovault_to_dataset(kwargs):
    """Test conversion of neurovault collection to a dataset."""
    if 778 in kwargs["collection_ids"]:
        with pytest.raises(ValueError) as excinfo:
            dset = io.convert_neurovault_to_dataset(**kwargs)
        assert "Collection 778 not found." in str(excinfo.value)
        return
    else:
        dset = io.convert_neurovault_to_dataset(**kwargs)

    # check if names are propagated into Dataset
    if isinstance(kwargs.get("collection_ids"), dict):
        study_ids = set(kwargs["collection_ids"].keys())
    else:
        study_ids = set(map(str, kwargs["collection_ids"]))
    dset_ids = {id_.split("-")[1] for id_ in dset.ids}

    assert study_ids == dset_ids

    # check if images were downloaded and are unique
    if kwargs.get("map_type_conversion"):
        for img_type in kwargs.get("map_type_conversion").values():
            assert not dset.images[img_type].empty
            assert len(set(dset.images[img_type])) == len(dset.images[img_type])


@pytest.mark.parametrize(
    "sample_sizes,expected_sample_size",
    [
        ([1, 2, 1], 1),
        ([None, None, 1], 1),
        ([1, 1, 2, 2], 1),
    ],
)
def test_resolve_sample_sizes(sample_sizes, expected_sample_size):
    """Test modal sample size heuristic."""
    assert io._resolve_sample_size(sample_sizes) == expected_sample_size
