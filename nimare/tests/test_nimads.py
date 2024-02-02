"""Test NiMADS functionality."""

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
