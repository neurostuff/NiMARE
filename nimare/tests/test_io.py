"""Test nimare.io (Dataset IO/transformations)."""

import copy
import os

import pytest

import nimare
from nimare import io
from nimare.nimads import Studyset
from nimare.tests.utils import get_test_data_path
from nimare.utils import get_template


def test_convert_nimads_to_dataset(example_nimads_studyset, example_nimads_annotation):
    """Conversion of nimads JSON to nimare dataset."""
    studyset = Studyset(example_nimads_studyset)
    first_study_with_analyses = next(study for study in studyset.studies if study.analyses)
    first_analysis = first_study_with_analyses.analyses[0]
    expected_id = f"{first_study_with_analyses.id}-{first_analysis.id}"

    dset1 = io.convert_nimads_to_dataset(studyset)
    studyset.annotations = example_nimads_annotation
    dset2 = io.convert_nimads_to_dataset(studyset)

    assert isinstance(dset1, nimare.dataset.Dataset)
    assert isinstance(dset2, nimare.dataset.Dataset)
    assert expected_id in dset1.ids
    assert "analysis_name" in dset1.metadata.columns
    assert "study_name" in dset1.metadata.columns

    md_row = dset1.metadata.loc[dset1.metadata["id"] == expected_id].iloc[0]
    assert md_row["study_name"] == (first_study_with_analyses.name or first_study_with_analyses.id)
    assert md_row["analysis_name"] == (first_analysis.name or first_analysis.id)


def test_convert_nimads_to_dataset_sample_sizes(example_nimads_studyset):
    """Conversion of nimads JSON to nimare dataset."""
    studyset = Studyset(example_nimads_studyset)
    for study in studyset.studies:
        for analysis in study.analyses:
            analysis.metadata["sample_sizes"] = [2, 20]

    dset = io.convert_nimads_to_dataset(studyset)

    assert isinstance(dset, nimare.dataset.Dataset)
    assert "sample_sizes" in dset.metadata.columns


def test_convert_nimads_to_dataset_single_sample_size(example_nimads_studyset):
    """Test conversion of nimads JSON to nimare dataset with a single sample size value."""
    studyset = Studyset(example_nimads_studyset)
    for study in studyset.studies:
        for analysis in study.analyses:
            analysis.metadata["sample_size"] = 20

    dset = io.convert_nimads_to_dataset(studyset)

    assert isinstance(dset, nimare.dataset.Dataset)
    assert "sample_sizes" in dset.metadata.columns


def test_convert_nimads_to_dataset_wonky_sample_size(sample_size_nimads_studyset):
    """Test conversion of nimads JSON to nimare dataset with wonky sample size values."""
    studyset = Studyset(sample_size_nimads_studyset)

    dset = io.convert_nimads_to_dataset(studyset)

    assert isinstance(dset, nimare.dataset.Dataset)
    assert "sample_sizes" in dset.metadata.columns


@pytest.mark.parametrize(
    "sample_sizes_val,sample_size_val,expect_col,expect_warning",
    [
        (5, None, False, True),
        ([5, "6", "7.3", "not_a_number"], None, False, True),
        (None, 10, True, False),
        (None, 10.5, True, False),
        (None, "12", True, True),
        (None, "13.5", True, True),
        (None, "not_a_number", False, True),
        (None, None, False, False),
        ([], 7, True, False),
        ([], None, False, False),
    ],
)
def test_analysis_to_dict_sample_size(
    example_nimads_studyset, sample_sizes_val, sample_size_val, expect_col, expect_warning, caplog
):
    """Test conversion of nimads JSON to nimare dataset with different sample_size(s) values."""
    studyset = Studyset(example_nimads_studyset)
    for study in studyset.studies:
        study.metadata.clear()
        if sample_sizes_val is not None:
            study.metadata["sample_sizes"] = sample_sizes_val
        if sample_size_val is not None:
            study.metadata["sample_size"] = sample_size_val
        for analysis in study.analyses:
            analysis.metadata.clear()
            if sample_sizes_val is not None:
                analysis.metadata["sample_sizes"] = sample_sizes_val
            if sample_size_val is not None:
                analysis.metadata["sample_size"] = sample_size_val

    with caplog.at_level("WARNING"):
        dset = io.convert_nimads_to_dataset(studyset)
    assert isinstance(dset, nimare.dataset.Dataset)
    if expect_col:
        assert "sample_sizes" in dset.metadata.columns
    else:
        assert "sample_sizes" not in dset.metadata.columns
    if expect_warning:
        assert any(
            "sample_size" in record.message or "sample_sizes" in record.message
            for record in caplog.records
        )
    else:
        assert not any(
            "sample_size" in record.message or "sample_sizes" in record.message
            for record in caplog.records
        )


@pytest.mark.parametrize(
    "annotation_mod,expect_success,expect_typeerror",
    [
        # No annotation at all
        (None, True, False),
        # Annotation with empty notes
        (lambda ann: ann.update({"notes": []}), True, False),
        # Annotation with extra irrelevant key
        (lambda ann: ann.update({"extra_key": 123}), True, False),
        # Annotation with missing 'notes' key (should fail)
        (lambda ann: ann.pop("notes", None), False, True),
        # Annotation with mismatched analysis id in notes (should warn/fail)
        (
            lambda ann: ann["notes"].append(
                {
                    "analysis_name": "Fake",
                    "publication": "Fake",
                    "study": ann["notes"][0]["study"],
                    "study_year": 2025,
                    "analysis": "not_in_studyset",
                    "authors": "Nobody",
                    "note": {"include": False},
                    "study_name": "Fake",
                }
            ),
            True,
            True,
        ),
    ],
)
def test_analysis_to_dict_annotation(
    example_nimads_studyset,
    example_nimads_annotation,
    annotation_mod,
    expect_success,
    expect_typeerror,
):
    """Test conversion of nimads JSON to nimare dataset with various annotation modifications."""
    studyset = Studyset(example_nimads_studyset)
    if annotation_mod is not None:
        annotation = copy.deepcopy(example_nimads_annotation)
        annotation_mod(annotation)
        if expect_typeerror:
            with pytest.raises((TypeError, ValueError, KeyError)):
                studyset.annotations = annotation
                io.convert_nimads_to_dataset(studyset)
        else:
            studyset.annotations = annotation
            dset = io.convert_nimads_to_dataset(studyset)
            assert expect_success
    else:
        # No annotation
        dset = io.convert_nimads_to_dataset(studyset)
        assert isinstance(dset, nimare.dataset.Dataset)


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
    coordinates_file = os.path.join(
        get_test_data_path(),
        "data-neurosynth_version-7_coordinates.tsv.gz",
    )
    metadata_file = os.path.join(
        get_test_data_path(),
        "data-neurosynth_version-7_metadata.tsv.gz",
    )
    features = {
        "features": os.path.join(
            get_test_data_path(),
            "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz",
        ),
        "vocabulary": os.path.join(
            get_test_data_path(), "data-neurosynth_version-7_vocab-terms_vocabulary.txt"
        ),
    }
    dset = io.convert_neurosynth_to_dataset(
        coordinates_file,
        metadata_file,
        annotations_files=features,
    )
    assert isinstance(dset, nimare.dataset.Dataset)
    assert "terms_abstract_tfidf__abilities" in dset.annotations.columns


def test_convert_neurosynth_to_json_smoke():
    """Smoke test for Neurosynth file conversion."""
    out_file = os.path.abspath("temp.json")
    coordinates_file = os.path.join(
        get_test_data_path(),
        "data-neurosynth_version-7_coordinates.tsv.gz",
    )
    metadata_file = os.path.join(
        get_test_data_path(),
        "data-neurosynth_version-7_metadata.tsv.gz",
    )
    features = {
        "features": os.path.join(
            get_test_data_path(),
            "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz",
        ),
        "vocabulary": os.path.join(
            get_test_data_path(), "data-neurosynth_version-7_vocab-terms_vocabulary.txt"
        ),
    }
    io.convert_neurosynth_to_json(
        coordinates_file,
        metadata_file,
        out_file,
        annotations_files=features,
    )
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
        (
            {
                "collection_ids": (11303,),
                "contrasts": {"rms": "rms"},
                "map_type_conversion": {"univariate-beta map": "beta"},
            }
        ),
        (
            {
                "collection_ids": (8836,),
                "contrasts": {"crab_people": "cannot hurt you because they do not exist"},
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
    elif "crab_people" in kwargs["contrasts"].keys():
        with pytest.raises(ValueError) as excinfo:
            dset = io.convert_neurovault_to_dataset(**kwargs)
        assert "No images were found for contrast crab_people" in str(excinfo.value)
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
