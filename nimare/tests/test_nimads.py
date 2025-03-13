"""Test NiMADS functionality."""

import json
import os
import tempfile

import pytest

from nimare import nimads
from nimare.dataset import Dataset


def test_load_nimads(example_nimads_studyset, example_nimads_annotation):
    """Test loading a NiMADS studyset."""
    studyset = nimads.Studyset(example_nimads_studyset)
    studyset.annotations = example_nimads_annotation
    # filter the studyset to only include analyses with include=True
    annotation = studyset.annotations[0]
    analysis_ids = [n.analysis.id for n in annotation.notes if n.note["include"]]
    analysis_ids = analysis_ids[:5]
    filtered_studyset = studyset.slice(analyses=analysis_ids)
    # Combine analyses after filtering
    filtered_studyset = filtered_studyset.combine_analyses()

    assert isinstance(filtered_studyset, nimads.Studyset)
    dataset = filtered_studyset.to_dataset()
    assert isinstance(dataset, Dataset)


def test_slice_preserves_metadata_and_annotations(
    example_nimads_studyset, example_nimads_annotation
):
    """Test that slicing preserves both metadata and annotations.

    This test verifies that both metadata attached to analyses and annotation
    notes are correctly preserved when slicing a studyset.
    """
    studyset = nimads.Studyset(example_nimads_studyset)
    studyset.annotations = example_nimads_annotation

    # Get analysis IDs from the first annotation
    annotation = studyset.annotations[0]
    analysis_ids = [n.analysis.id for n in annotation.notes if n.note["include"]]
    selected_ids = analysis_ids[:2]  # Take first two analyses

    # Add metadata to the analyses we'll keep
    metadata_map = {}
    for study in studyset.studies:
        for analysis in study.analyses:
            if analysis.id in selected_ids:
                analysis.metadata = {
                    "sample_size": 30,
                    "contrast_type": "activation",
                    "significance_threshold": 0.001,
                }
                metadata_map[analysis.id] = analysis.metadata

    # Slice studyset
    sliced_studyset = studyset.slice(analyses=selected_ids)

    # Verify analyses and their metadata are preserved
    for study in sliced_studyset.studies:
        for analysis in study.analyses:
            assert analysis.id in selected_ids
            assert analysis.metadata == metadata_map[analysis.id]

    # Verify annotations are preserved for remaining analyses
    sliced_annotation = sliced_studyset.annotations[0]
    sliced_analysis_ids = [n.analysis.id for n in sliced_annotation.notes]
    sliced_annotation_notes = {n.analysis.id: n.note for n in sliced_annotation.notes}

    # Check that notes exist only for remaining analyses
    assert set(sliced_analysis_ids) == set(selected_ids)

    # Check that annotation contents are preserved
    for analysis_id in selected_ids:
        original_note = next(n.note for n in annotation.notes if n.analysis.id == analysis_id)
        assert sliced_annotation_notes[analysis_id] == original_note


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


def test_studyset_string_methods(example_nimads_studyset):
    """Test string representation methods."""
    studyset = nimads.Studyset(example_nimads_studyset)

    # Test __repr__
    assert repr(studyset) == f"'<Studyset: {studyset.id}>'"

    # Test __str__
    expected_str = f"Studyset: {studyset.name} :: studies: {len(studyset.studies)}"
    assert str(studyset) == expected_str


def test_studyset_save_load(example_nimads_studyset):
    """Test saving and loading Studyset."""
    studyset = nimads.Studyset(example_nimads_studyset)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Test save
        studyset.save(tmp_path)
        assert os.path.exists(tmp_path)

        # Test load
        new_studyset = nimads.Studyset({"id": "temp", "name": "", "studies": []})
        new_studyset.load(tmp_path)

        assert new_studyset.id == studyset.id
        assert new_studyset.name == studyset.name
        assert len(new_studyset.studies) == len(studyset.studies)
    finally:
        os.unlink(tmp_path)


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
        with open(tmp_path, "r") as f:
            saved_data = json.load(f)

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


def test_get_analyses_by_coordinates(example_nimads_studyset):
    """Test retrieving analyses by coordinates."""
    studyset = nimads.Studyset(example_nimads_studyset)

    # Test with radius
    xyz = [0, 0, 0]
    results_r = studyset.get_analyses_by_coordinates(xyz, r=10)
    assert isinstance(results_r, list)

    # Test with n nearest
    results_n = studyset.get_analyses_by_coordinates(xyz, n=5)
    assert isinstance(results_n, list)
    assert len(results_n) <= 5

    # Test invalid parameters
    with pytest.raises(ValueError):
        studyset.get_analyses_by_coordinates(xyz)  # Neither r nor n
    with pytest.raises(ValueError):
        studyset.get_analyses_by_coordinates(xyz, r=10, n=5)  # Both r and n
    with pytest.raises(ValueError):
        studyset.get_analyses_by_coordinates([0, 0])  # Invalid coordinates


def test_get_analyses_by_mask(example_nimads_studyset, mni_mask):
    """Test retrieving analyses by mask."""
    studyset = nimads.Studyset(example_nimads_studyset)

    results = studyset.get_analyses_by_mask(mni_mask)
    assert isinstance(results, list)


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


def test_data_retrieval_methods(example_nimads_studyset):
    """Test methods that retrieve data for specified analyses."""
    studyset = nimads.Studyset(example_nimads_studyset)

    # Get some analysis IDs to test with
    analysis_ids = []
    for study in studyset.studies:
        for analysis in study.analyses:
            analysis_ids.append(analysis.id)
            if len(analysis_ids) >= 2:  # Just test with first two analyses
                break
        if len(analysis_ids) >= 2:
            break

    # Test get_points
    points = studyset.get_points(analysis_ids)
    assert isinstance(points, dict)

    # Test get_images
    images = studyset.get_images(analysis_ids)
    assert isinstance(images, dict)

    # Test get_metadata
    metadata = studyset.get_metadata(analysis_ids)
    assert isinstance(metadata, dict)

    # Test get_annotations
    annotations = studyset.get_annotations(analysis_ids)
    assert isinstance(annotations, dict)
