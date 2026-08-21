"""Test NiMADS functionality."""

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nimare import nimads
from nimare.dataset import Dataset
from nimare.tests.utils import get_test_data_path
from nimare.utils import load_json


def test_load_nimads(example_nimads_studyset, example_nimads_annotation):
    """Test loading a NiMADS studyset and selecting by an annotation label."""
    studyset = nimads.Studyset(example_nimads_studyset).with_annotation_payload(
        example_nimads_annotation
    )

    # Annotations are columns over the analyses, so selecting on one is a label
    # query rather than a walk over note objects.
    included = studyset.get_studies_by_label("include", label_threshold=0.5)[:5]
    filtered_studyset = studyset.slice(included).combine_analyses()

    assert isinstance(filtered_studyset, nimads.Studyset)
    dataset = filtered_studyset.to_dataset()
    assert isinstance(dataset, Dataset)


def test_slice_preserves_metadata_and_annotations(
    example_nimads_studyset, example_nimads_annotation
):
    """Slicing keeps metadata and annotations aligned to the analyses it kept."""
    studyset = nimads.Studyset(example_nimads_studyset).with_annotation_payload(
        example_nimads_annotation
    )
    selected = studyset.get_studies_by_label("include", label_threshold=0.5)[:2]

    sliced = studyset.slice(selected)

    assert set(sliced.ids) == set(selected)
    assert len(sliced.metadata) == len(selected)
    assert len(sliced.annotations_df) == len(selected)
    # the same values, for the analyses that survived
    full = studyset.annotations_df.set_index("id")["include"]
    kept = sliced.annotations_df.set_index("id")["include"]
    for analysis_id in selected:
        assert kept[analysis_id] == full[analysis_id]


def test_studyset_from_dataset_preserves_inferred_image_basepath(tmp_path):
    """Studyset.from_dataset should resolve images when Dataset only has inferred basepath."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_file = image_dir / "contrast_z.nii.gz"
    image_file.write_text("placeholder")

    source = {
        "study1": {
            "contrasts": {
                "contrast1": {
                    "images": {"z": str(image_file)},
                    "metadata": {"sample_sizes": [20]},
                }
            }
        }
    }

    dataset = Dataset(source)
    assert dataset.basepath is None
    assert dataset.images.loc[0, "z"] == str(image_file)
    assert dataset.images.loc[0, "z__relative"] == str(Path("contrast_z.nii.gz"))

    studyset = nimads.Studyset.from_dataset(dataset)

    assert studyset.basepath == str(image_dir)
    assert studyset.images.loc[0, "z"] == str(image_file)
    assert studyset.images.loc[0, "z__relative"] == str(Path("contrast_z.nii.gz"))


def test_studyset_from_dataset_preserves_point_values_and_coordinate_metadata():
    """Point values and per-coordinate metadata survive the Dataset bridge."""
    source = {
        "study-1": {
            "contrasts": {
                "1": {
                    "coords": {
                        "space": "MNI",
                        "x": [1, 2],
                        "y": [3, 4],
                        "z": [5, 6],
                        "z_stat": [7.0, 8.0],
                        "cluster_size": [11, 12],
                    },
                    "metadata": {"sample_sizes": [20]},
                }
            }
        }
    }

    dataset = Dataset(source)
    studyset = nimads.Studyset.from_dataset(dataset)
    analysis = studyset.studies[0].analyses[0]

    assert analysis.points[0].values == {"z_stat": 7.0, "cluster_size": 11}
    assert list(studyset.coordinates["z_stat"]) == [7.0, 8.0]
    assert [int(value) for value in studyset.coordinates["cluster_size"]] == [11, 12]


def test_studyset_init(example_nimads_studyset):
    """Test Studyset initialization."""
    # Test initialization with dict
    studyset1 = nimads.Studyset(example_nimads_studyset)
    assert studyset1.id == example_nimads_studyset["id"]
    assert studyset1.name == example_nimads_studyset["name"]
    assert len(studyset1.studies) == len(example_nimads_studyset["studies"])

    # Test initialization with JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(example_nimads_studyset, tmp)
        tmp_path = tmp.name

    try:
        studyset2 = nimads.Studyset(tmp_path)
        assert studyset2.id == example_nimads_studyset["id"]
        assert studyset2.name == example_nimads_studyset["name"]
        assert len(studyset2.studies) == len(example_nimads_studyset["studies"])
    finally:
        os.unlink(tmp_path)


def test_saved_nidm_pain_studyset_loads_directly():
    """Test loading the saved pain Studyset resource directly."""
    studyset_file = os.path.join(get_test_data_path(), "nidm_pain_studyset.json")

    studyset = nimads.Studyset(studyset_file, target="mni152_2mm")
    # update_path returns a studyset rather than mutating one
    studyset = studyset.update_path(get_test_data_path())

    assert len(studyset.studies) > 0
    assert studyset.space == "mni152_2mm"
    assert studyset.basepath == os.path.abspath(get_test_data_path())
    assert "beta" in studyset.images.columns
    assert studyset.images["beta"].notnull().any()
    assert "beta__relative" in studyset.images.columns


def test_saved_neurosynth_laird_studyset_loads_directly():
    """Test loading the saved Neurosynth Laird Studyset resource directly."""
    studyset_file = os.path.join(get_test_data_path(), "neurosynth_laird_studyset.json")

    studyset = nimads.Studyset(studyset_file, target="mni152_2mm")

    assert len(studyset.studies) > 0
    assert studyset.space == "mni152_2mm"
    assert not studyset.texts.empty
    assert "abstract" in studyset.texts.columns
    assert not studyset.annotations_df.empty


def test_studyset_string_methods(example_nimads_studyset):
    """Studyset should have readable repr and str."""
    studyset = nimads.Studyset(example_nimads_studyset)

    assert repr(studyset) == f"<Studyset: {studyset.id}>"
    assert "Studyset:" in str(studyset)
    assert "studies:" in str(studyset)


def test_studyset_save_load(example_nimads_studyset):
    """Studyset should pickle and unpickle."""
    studyset = nimads.Studyset(example_nimads_studyset)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        tmp_path = fh.name

    try:
        studyset.save(tmp_path)
        assert os.path.exists(tmp_path)

        # `load` returns a studyset rather than mutating one: a studyset is a
        # value, so there is nothing to load *into*.
        loaded = nimads.Studyset.load(tmp_path)

        assert loaded.id == studyset.id
        assert loaded.name == studyset.name
        assert len(loaded.studies) == len(studyset.studies)
        assert list(loaded.ids) == list(studyset.ids)
    finally:
        os.unlink(tmp_path)


def test_studyset_pickle_roundtrip(example_nimads_studyset):
    """Pickling should preserve the data and the store's immutability."""
    studyset = nimads.Studyset(example_nimads_studyset)
    _ = studyset.coordinates  # populate the derived caches

    restored = pickle.loads(pickle.dumps(studyset))

    assert list(restored.ids) == list(studyset.ids)
    assert restored.coordinates.shape == studyset.coordinates.shape
    # numpy does not carry the writeable flag through pickle, so the store
    # re-freezes itself on the way in.
    assert not restored.store.xyz.flags.writeable


def test_studyset_to_dict(example_nimads_studyset):
    """Test conversion to dictionary."""
    studyset = nimads.Studyset(example_nimads_studyset)
    result = studyset.to_dict()

    assert isinstance(result, dict)
    assert "id" in result
    assert "name" in result
    assert "studies" in result
    assert len(result["studies"]) == len(studyset.studies)


def test_studyset_to_nimads(example_nimads_studyset):
    """Test saving to NIMADS format."""
    studyset = nimads.Studyset(example_nimads_studyset)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        studyset.to_nimads(tmp_path)
        assert os.path.exists(tmp_path)

        # Verify the saved file can be loaded
        saved_data = load_json(tmp_path)

        assert saved_data["id"] == studyset.id
        assert saved_data["name"] == studyset.name
        assert len(saved_data["studies"]) == len(studyset.studies)
    finally:
        os.unlink(tmp_path)


def test_studyset_copy(example_nimads_studyset):
    """Test copying of Studyset."""
    studyset = nimads.Studyset(example_nimads_studyset)
    copied = studyset.copy()

    assert copied is not studyset
    assert copied.id == studyset.id
    assert copied.name == studyset.name
    assert len(copied.studies) == len(studyset.studies)


def test_studyset_merge(example_nimads_studyset):
    """Test merging of Studysets."""
    studyset1 = nimads.Studyset(example_nimads_studyset)

    # Create a modified copy for merging
    modified_data = example_nimads_studyset.copy()
    modified_data["id"] = "other_id"
    modified_data["name"] = "Other name"
    studyset2 = nimads.Studyset(modified_data)

    merged = studyset1.merge(studyset2)

    assert isinstance(merged, nimads.Studyset)
    assert merged.id == f"{studyset1.id}_{studyset2.id}"
    assert merged.name == f"Merged: {studyset1.name} + {studyset2.name}"

    # Test invalid merge
    with pytest.raises(ValueError):
        studyset1.merge("not a studyset")


def test_studyset_merge_preserves_execution_target():
    """Merging Studysets should preserve the left-hand execution context."""
    source = {
        "study1": {
            "contrasts": {
                "contrast1": {
                    "coords": {"space": "TAL", "x": [0], "y": [0], "z": [0]},
                    "metadata": {"sample_sizes": [20]},
                }
            }
        }
    }
    studyset = nimads.Studyset.from_dataset(Dataset(source, target="ale_2mm"))

    merged = studyset.merge(studyset.copy())

    assert merged.space == "ale_2mm"
    assert merged.masker is not None


def test_studyset_init_normalizes_literal_none_strings():
    """Studyset tabular views should normalize literal "None" strings like Dataset does."""
    source = {
        "id": "studyset",
        "studies": [
            {
                "id": "study1",
                "analyses": [
                    {
                        "id": "analysis1",
                        "name": "analysis1",
                        "metadata": {"foo": "None"},
                        "annotations": {"bar": "None"},
                        "texts": {"abstract": "None"},
                        "points": [],
                        "images": [],
                    }
                ],
            }
        ],
    }

    studyset = nimads.Studyset(source)

    assert studyset.metadata.loc[0, "foo"] is None
    assert studyset.annotations_df.loc[0, "bar"] is None
    assert studyset.texts.loc[0, "abstract"] is None


def test_get_analyses_by_coordinate(example_nimads_studyset):
    """Test retrieving analyses by coordinates."""
    studyset = nimads.Studyset(example_nimads_studyset)

    # Test with radius
    xyz = [0, 0, 0]
    results_r = studyset.get_analyses_by_coordinate(xyz, r=10)
    assert isinstance(results_r, list)

    # Test with n nearest
    results_n = studyset.get_analyses_by_coordinate(xyz, n=5)
    assert isinstance(results_n, list)
    assert len(results_n) <= 5

    # Test invalid parameters
    with pytest.raises(ValueError):
        studyset.get_analyses_by_coordinate(xyz)  # Neither r nor n
    with pytest.raises(ValueError):
        studyset.get_analyses_by_coordinate(xyz, r=10, n=5)  # Both r and n
    with pytest.raises(ValueError):
        studyset.get_analyses_by_coordinate([0, 0])  # Invalid coordinates


def test_get_analyses_by_mask(example_nimads_studyset, mni_mask):
    """Test retrieving analyses by mask."""
    studyset = nimads.Studyset(example_nimads_studyset)

    results = studyset.get_analyses_by_mask(mni_mask)
    assert isinstance(results, list)


def test_get_analyses_by_label(example_nimads_studyset):
    """Test retrieving analyses by label threshold."""
    studyset = nimads.Studyset(example_nimads_studyset)
    values = np.array(
        [[1.0] if i < 2 else [0.0] for i in range(len(studyset.ids))]
    )
    labelled = studyset.with_annotation("custom", ["custom_label"], values)
    expected = [str(i).rsplit("-", 1)[-1] for i in labelled.ids[:2]]

    results = labelled.get_analyses_by_label("custom_label", label_threshold=0.5)

    assert sorted(results) == sorted(expected)


def test_get_analyses_by_metadata(example_nimads_studyset):
    """Test retrieving analyses by metadata."""
    studyset = nimads.Studyset(example_nimads_studyset)

    # Add some metadata for testing
    key = "test_key"
    value = "test_value"
    for study in studyset.studies:
        for analysis in study.analyses:
            analysis.metadata[key] = value

    # Test with key only
    results1 = studyset.get_analyses_by_metadata(key)
    assert isinstance(results1, dict)

    # Test with key and value
    results2 = studyset.get_analyses_by_metadata(key, value)
    assert isinstance(results2, dict)
    assert all(list(d.values())[0] == value for d in results2.values())


def test_get_studies_by_coordinate(example_nimads_studyset):
    """Test the direct Studyset coordinate search wrapper."""
    studyset = nimads.Studyset(example_nimads_studyset)
    xyz = [[0, 0, 0]]

    results = studyset.get_studies_by_coordinate(xyz, r=10)
    expected = studyset.get_studies_by_coordinate(xyz, r=10)

    assert isinstance(results, list)
    assert set(results) == set(expected)


def test_data_retrieval_methods(example_nimads_studyset):
    """Nested and tabular retrieval should agree about what the studyset holds."""
    studyset = nimads.Studyset(example_nimads_studyset)
    analysis_ids = [str(i).rsplit("-", 1)[-1] for i in studyset.ids[:3]]

    # Nested retrieval, keyed by analysis id
    points = studyset.get_points(analysis_ids)
    assert isinstance(points, dict)
    assert set(points) <= set(analysis_ids)

    annotations = studyset.get_annotations(analysis_ids)
    assert isinstance(annotations, dict)

    # Tabular retrieval, one entry per selected analysis
    assert isinstance(studyset.get_images(), list)
    assert isinstance(studyset.get_metadata(), list)
    assert isinstance(studyset.get_texts(), list)

    # And the accessors agree with the frames
    accessor_counts = {
        analysis.id: len(analysis.points) for analysis in studyset.analyses
    }
    frame_counts = studyset.coordinates.groupby("contrast_id").size().to_dict()
    for contrast_id, count in frame_counts.items():
        assert accessor_counts[contrast_id] == count


