"""Test NiMADS functionality."""
from nimare import nimads


def test_load_nimads(example_nimads_studyset, example_nimads_annotation):
    """Test loading a NiMADS studyset."""
    studyset = nimads.Studyset(example_nimads_studyset)
    studyset.annotations = example_nimads_annotation
    # filter the studyset to only include analyses with include=True
    annotation = studyset.annotations[0]
    analysis_ids = [n.analysis.id for n in annotation.notes if n.note["include"]]
    analysis_ids = analysis_ids[0:5]
    filtered_studyset = studyset.slice(analyses=analysis_ids)
    assert isinstance(filtered_studyset, nimads.Studyset)
