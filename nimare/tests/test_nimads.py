"""Test NiMADS functionality."""
from nimare import nimads


def test_load_nimads(example_nimads_studyset, example_nimads_annotation):
    """Test loading a NiMADS studyset."""
    studyset = nimads.Studyset(example_nimads_studyset)
    studyset.annotations = example_nimads_annotation
