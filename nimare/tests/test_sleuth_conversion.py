"""Test NIMADS to Sleuth conversion functionality."""

import tempfile
from pathlib import Path

import pytest

from nimare.exceptions import InvalidStudysetError
from nimare.io import (
    convert_dataset_to_nimads_dict,
    convert_dataset_to_studyset,
    convert_nimads_to_sleuth,
    convert_sleuth_to_dataset,
    convert_sleuth_to_nimads_dict,
)


def test_annotation_splitting_boolean(example_nimads_studyset, example_nimads_annotation):
    """Test annotation-based file splitting for boolean values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "sleuth_output"

        # Convert with boolean annotation
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset,
            output_dir=output_dir,
            annotation=example_nimads_annotation,
        )

        # Check for boolean split files (e.g., "include_true.txt")
        true_file = output_dir / "include_true.txt"

        # Should exist based on the annotation data
        assert true_file.exists()


def test_coordinate_precision(example_nimads_studyset):
    """Test configurable decimal precision for coordinates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Convert with specific precision
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset, output_dir=output_dir, decimal_precision=3
        )

        # Check output file
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1

        with open(sleuth_files[0], "r") as f:
            content = f.read()

        # Check that coordinates have 3 decimal places
        lines = content.split("\n")
        coord_lines = [line for line in lines if line and not line.startswith("//")]
        if coord_lines:
            coords = coord_lines[0].split("\t")
            # Check that each coordinate has 3 decimal places
            for coord in coords:
                if "." in coord:
                    decimal_places = len(coord.split(".")[1])
                    assert decimal_places <= 3


def test_validation_errors(example_nimads_studyset):
    """Test custom exception handling for invalid inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Test invalid decimal precision
        with pytest.raises(InvalidStudysetError):
            convert_nimads_to_sleuth(
                studyset=example_nimads_studyset, output_dir=output_dir, decimal_precision=-1
            )


def test_idempotency_sleuth_to_nimads():
    """Test Sleuth -> Dataset -> NIMADS path produces a valid Studyset."""
    sleuth_file = "nimare/tests/data/test_sleuth_file.txt"
    dset = convert_sleuth_to_dataset(sleuth_file)
    nimads_dict = convert_dataset_to_nimads_dict(dset)
    assert isinstance(nimads_dict, dict)
    assert "studies" in nimads_dict and len(nimads_dict["studies"]) > 0


def test_round_trip_sleuth_nimads_sleuth_content_equivalence():
    """Round-trip preserves coordinate content per block, ignoring order."""
    sleuth_file = Path("nimare/tests/data/test_sleuth_file.txt")

    def parse_blocks(path: Path):
        txt = path.read_text().splitlines()
        blocks = []
        current = []
        for line in txt:
            if not line.strip():
                if current:
                    blocks.append(current)
                    current = []
                continue
            if line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                except Exception:
                    continue
                current.append((round(x, 5), round(y, 5), round(z, 5)))
        if current:
            blocks.append(current)
        # Represent each block as frozenset of coords (order-insensitive)
        return [frozenset(b) for b in blocks if b]

    b_orig = parse_blocks(sleuth_file)
    with tempfile.TemporaryDirectory() as td1:
        out1 = Path(td1)

        # First pass: sleuth -> dataset -> nimads -> sleuth
        dset1 = convert_sleuth_to_dataset(str(sleuth_file))
        nimads1 = convert_dataset_to_nimads_dict(dset1, studyset_id="rt1")
        convert_nimads_to_sleuth(nimads1, out1)
        s1 = next(out1.glob("*.txt"))

        b1 = parse_blocks(s1)
        assert sorted(b1) == sorted(b_orig)


def test_metadata_embedding(example_nimads_studyset):
    """Test that metadata is embedded in Sleuth files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Convert to Sleuth
        convert_nimads_to_sleuth(studyset=example_nimads_studyset, output_dir=output_dir)

        # Check output file
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1

        with open(sleuth_files[0], "r") as f:
            content = f.read()

        # Check that metadata is included as comments
        assert "Subjects=" in content


def test_export_metadata_parameter(example_nimads_studyset):
    """Test that export_metadata parameter controls which metadata is included."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Convert to Sleuth with specific metadata fields
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset,
            output_dir=output_dir,
            export_metadata=["authors", "year"],
        )

        # Check output file
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1

        with open(sleuth_files[0], "r") as f:
            content = f.read()

        # Check that specified metadata is included
        assert "Authors=" in content
        assert "Year=" in content

        # Check that default metadata (doi, pmid) is not included when not specified
        # (unless it's in the study data)


def test_pathlib_string_support(example_nimads_studyset):
    """Test that both pathlib and string paths are supported."""
    # Test with string path
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir_str = str(Path(tmpdir) / "output_str")
        convert_nimads_to_sleuth(studyset=example_nimads_studyset, output_dir=output_dir_str)
        sleuth_files = list(Path(output_dir_str).glob("*.txt"))
        assert len(sleuth_files) == 1

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir_path = Path(tmpdir) / "output_path"
        convert_nimads_to_sleuth(studyset=example_nimads_studyset, output_dir=output_dir_path)
        sleuth_files = list(output_dir_path.glob("*.txt"))
        assert len(sleuth_files) == 1


def test_schema_validation_rejects_malformed():
    """Schema validation raises on malformed studyset."""
    # Missing required 'studies' field
    malformed = {"id": "bad", "name": "Bad"}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with pytest.raises(InvalidStudysetError):
            convert_nimads_to_sleuth(malformed, out)


def test_annotation_columns_and_values_filtering():
    """Split using specific annotation column and filter target values."""
    # Build minimal studyset + annotation
    studyset = {
        "id": "s1",
        "name": "Example",
        "studies": [
            {
                "id": "st1",
                "name": "Study",
                "authors": "Doe J.",
                "publication": "Journal",
                "metadata": {},
                "analyses": [
                    {
                        "id": "a1",
                        "name": "A",
                        "conditions": [{"name": "c1", "description": "desc"}],
                        "weights": [1.0],
                        "images": [],
                        "points": [{"space": "MNI", "coordinates": [1.0, 2.0, 3.0]}],
                        "metadata": {"sample_size": 10},
                    },
                    {
                        "id": "a2",
                        "name": "B",
                        "conditions": [{"name": "c1", "description": "desc"}],
                        "weights": [1.0],
                        "images": [],
                        "points": [{"space": "MNI", "coordinates": [4.0, 5.0, 6.0]}],
                        "metadata": {"sample_size": 12},
                    },
                ],
            }
        ],
    }

    annotation = {
        "id": "ann1",
        "name": "labels",
        "notes": [
            {"analysis": "a1", "note": {"include": True, "group": "X"}},
            {"analysis": "a2", "note": {"include": False, "group": "Y"}},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        # Only split on 'group', and only keep value 'X'
        convert_nimads_to_sleuth(
            studyset,
            out,
            annotation=annotation,
            annotation_columns=["group"],
            annotation_values={"group": ["X"]},
        )
        files = sorted([p.name for p in out.glob("*.txt")])
        assert files == ["group_X.txt"]


def test_spatial_normalization(example_nimads_studyset):
    """Test spatial normalization functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Convert with target space
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset, output_dir=output_dir, target="MNI"
        )

        # Check output file
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1

        with open(sleuth_files[0], "r") as f:
            content = f.read()

        # Check that space header is correct
        assert "Reference=MNI" in content


def test_empty_studyset():
    """Test conversion with empty studyset."""
    from nimare import nimads

    empty_studyset = nimads.Studyset({"id": "empty", "name": "Empty Studyset", "studies": []})

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Convert empty studyset
        convert_nimads_to_sleuth(studyset=empty_studyset, output_dir=output_dir)

        # Should create output directory but no files
        assert output_dir.exists()
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1  # Still creates a file, just empty


def test_studyset_with_no_points(example_nimads_studyset):
    """Test conversion with studyset that has no coordinate points."""
    # Create a copy of the studyset and remove all points
    from nimare import nimads

    studyset_copy = nimads.Studyset(example_nimads_studyset).copy()
    for study in studyset_copy.studies:
        for analysis in study.analyses:
            analysis.points = []

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Convert studyset with no points
        convert_nimads_to_sleuth(studyset=studyset_copy, output_dir=output_dir)

        # Should create output directory and file
        assert output_dir.exists()
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1


def test_convert_sleuth_to_nimads_equivalence():
    """Ensure convert_sleuth_to_nimads matches two-step conversion via Dataset."""
    sleuth_file = "nimare/tests/data/test_sleuth_file.txt"
    nimads_direct = convert_sleuth_to_nimads_dict(sleuth_file, studyset_id="csn")
    dset = convert_sleuth_to_dataset(sleuth_file)
    nimads_from_dset = convert_dataset_to_nimads_dict(dset, studyset_id="csn")
    assert nimads_direct == nimads_from_dset


def test_convert_nimads_to_sleuth_target_normalization(example_nimads_studyset):
    """Verify various `target` inputs normalize to the expected Sleuth Reference header."""
    import tempfile
    from pathlib import Path

    cases = {
        "ale_2mm": "MNI",
        "mni152_2mm": "MNI",
        "MNI": "MNI",
        "mni152": "MNI",
        "TAL": "TAL",
        "Talairach": "TAL",
        "Talaraich": "TAL",
    }

    for inp, expected in cases.items():
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            convert_nimads_to_sleuth(studyset=example_nimads_studyset, output_dir=out, target=inp)
            files = list(out.glob("*.txt"))
            assert len(files) == 1
            txt = files[0].read_text()
            assert f"//Reference={expected}" in txt


def test_convert_sleuth_to_nimads_target_variants_equivalent():
    """Ensure Sleuth->NIMADS treats variant target strings equivalently to canonical ones."""
    sleuth_file = "nimare/tests/data/test_sleuth_file.txt"

    # Talairach variants should be treated as the same dataset target (mni152_2mm)
    variants = ["TAL", "Talairach", "Talaraich", "tal"]
    canonical = "mni152_2mm"
    results = [
        convert_sleuth_to_nimads_dict(sleuth_file, target=v, studyset_id="csn") for v in variants
    ]
    canonical_result = convert_sleuth_to_nimads_dict(
        sleuth_file, target=canonical, studyset_id="csn"
    )
    for r in results:
        assert r == canonical_result

    # ALE variants should match the ALE canonical target
    ale_variants = ["ale_2mm", "ALE", "ale"]
    canonical_ale = "ale_2mm"
    results_ale = [
        convert_sleuth_to_nimads_dict(sleuth_file, target=v, studyset_id="csn")
        for v in ale_variants
    ]
    canonical_ale_result = convert_sleuth_to_nimads_dict(
        sleuth_file, target=canonical_ale, studyset_id="csn"
    )
    for r in results_ale:
        assert r == canonical_ale_result


def test_convert_dataset_to_studyset_returns_object():
    """Convert Dataset to a nimads.Studyset object via convenience wrapper."""
    sleuth_file = "nimare/tests/data/test_sleuth_file.txt"
    dset = convert_sleuth_to_dataset(sleuth_file)
    studyset = convert_dataset_to_studyset(dset, studyset_id="cdts", studyset_name="From Dataset")
    from nimare.nimads import Studyset

    assert isinstance(studyset, Studyset)
    assert studyset.id == "cdts"
    assert studyset.name == "From Dataset"
