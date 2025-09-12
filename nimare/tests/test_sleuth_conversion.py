"""Test NIMADS to Sleuth conversion functionality."""
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nimare.io import (
    convert_nimads_to_sleuth,
    convert_sleuth_to_nimads,
    convert_sleuth_to_dataset,
    convert_dataset_to_nimads,
)
from nimare.exceptions import InvalidStudysetError


## Removed: test_basic_conversion (redundant with pathlib/string support test)


def test_annotation_splitting_boolean(example_nimads_studyset, example_nimads_annotation):
    """Test annotation-based file splitting for boolean values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "sleuth_output"
        
        # Convert with boolean annotation
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset,
            output_dir=output_dir,
            annotation=example_nimads_annotation
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
            studyset=example_nimads_studyset,
            output_dir=output_dir,
            decimal_precision=3
        )
        
        # Check output file
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1
        
        with open(sleuth_files[0], 'r') as f:
            content = f.read()
            
        # Check that coordinates have 3 decimal places
        lines = content.split('\n')
        coord_lines = [line for line in lines if line and not line.startswith('//')]
        if coord_lines:
            coords = coord_lines[0].split('\t')
            # Check that each coordinate has 3 decimal places
            for coord in coords:
                if '.' in coord:
                    decimal_places = len(coord.split('.')[1])
                    assert decimal_places <= 3


def test_validation_errors(example_nimads_studyset):
    """Test custom exception handling for invalid inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Test invalid decimal precision
        with pytest.raises(InvalidStudysetError):
            convert_nimads_to_sleuth(
                studyset=example_nimads_studyset,
                output_dir=output_dir,
                decimal_precision=-1
            )


def test_idempotency_sleuth_to_nimads():
    """Test Sleuth -> Dataset -> NIMADS path produces a valid Studyset."""
    sleuth_file = 'nimare/tests/data/test_sleuth_file.txt'
    dset = convert_sleuth_to_dataset(sleuth_file)
    nimads_dict = convert_dataset_to_nimads(dset)
    assert isinstance(nimads_dict, dict)
    assert 'studies' in nimads_dict and len(nimads_dict['studies']) > 0


def test_round_trip_sleuth_nimads_sleuth_content_equivalence():
    """Round-trip preserves coordinate content per block, ignoring order."""
    sleuth_file = Path('nimare/tests/data/test_sleuth_file.txt')

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
            if line.startswith('//'):
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

    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        out1 = Path(td1)
        out2 = Path(td2)

        # First pass: sleuth -> dataset -> nimads -> sleuth
        dset1 = convert_sleuth_to_dataset(str(sleuth_file))
        nimads1 = convert_dataset_to_nimads(dset1, studyset_id='rt1')
        convert_nimads_to_sleuth(nimads1, out1)
        s1 = next(out1.glob('*.txt'))

        # Second pass: sleuth1 -> dataset -> nimads -> sleuth2
        dset2 = convert_sleuth_to_dataset(str(s1))
        nimads2 = convert_dataset_to_nimads(dset2, studyset_id='rt2')
        convert_nimads_to_sleuth(nimads2, out2)
        s2 = next(out2.glob('*.txt'))

        b1 = parse_blocks(s1)
        b2 = parse_blocks(s2)
        assert sorted(b1) == sorted(b2)


def test_metadata_embedding(example_nimads_studyset):
    """Test that metadata is embedded in Sleuth files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Convert to Sleuth
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset,
            output_dir=output_dir
        )
        
        # Check output file
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1
        
        with open(sleuth_files[0], 'r') as f:
            content = f.read()
            
        # Check that metadata is included as comments
        assert 'Subjects =' in content


def test_pathlib_string_support(example_nimads_studyset):
    """Test that both pathlib and string paths are supported."""
    # Test with string path
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir_str = str(Path(tmpdir) / "output_str")
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset,
            output_dir=output_dir_str
        )
        sleuth_files = list(Path(output_dir_str).glob("*.txt"))
        assert len(sleuth_files) == 1
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir_path = Path(tmpdir) / "output_path"
        convert_nimads_to_sleuth(
            studyset=example_nimads_studyset,
            output_dir=output_dir_path
        )
        sleuth_files = list(output_dir_path.glob("*.txt"))
        assert len(sleuth_files) == 1


## Removed: test_annotation_splitting_categorical (redundant with boolean and filtering tests)


def test_schema_validation_rejects_malformed():
    """Schema validation raises on malformed studyset."""
    # Missing required 'studies' field
    malformed = {"id": "bad", "name": "Bad"}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with pytest.raises(InvalidStudysetError):
            convert_nimads_to_sleuth(malformed, out)


def test_affine_transformation_applied():
    """Apply custom affine transform to coordinates."""
    # Minimal valid nimads-like studyset dict
    studyset = {
        "id": "s1",
        "name": "Example",
        "studies": [
            {
                "id": "st1",
                "name": "Study",
                "authors": "Doe J.; Smith A.",
                "publication": "Journal",
                "metadata": {},
                "analyses": [
                    {
                        "id": "a1",
                        "name": "Analysis",
                        "conditions": [{"name": "c1", "description": "desc"}],
                        "weights": [1.0],
                        "images": [],
                        "points": [
                            {"space": "MNI", "coordinates": [1.0, 2.0, 3.0]},
                        ],
                        "metadata": {"sample_size": 10},
                    }
                ],
            }
        ],
    }

    # Translation by +1 on all axes
    affine = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        convert_nimads_to_sleuth(studyset, out, decimal_precision=2, target_space="MNI", transform_affine=affine)
        files = list(out.glob("*.txt"))
        assert len(files) == 1
        txt = files[0].read_text()
        # Find first non-comment line (coordinates)
        lines = [l for l in txt.splitlines() if l and not l.startswith("//")]
        assert lines, "No coordinates found in output"
        # Expect 1+ tab-separated floats; first row should be 2,3,4
        parts = lines[0].split("\t")
        assert parts[:3] == ["2.00", "3.00", "4.00"]


def test_idempotent_same_input_output():
    """Running conversion twice yields identical Sleuth content (SHA-256)."""
    studyset = {
        "id": "s1",
        "name": "Example",
        "studies": [
            {
                "id": "st1",
                "name": "Study",
                "authors": "Doe J.",
                "publication": "Journal",
                "metadata": {"year": 2001},
                "analyses": [
                    {
                        "id": "a1",
                        "name": "A",
                        "conditions": [{"name": "c1", "description": "desc"}],
                        "weights": [1.0],
                        "images": [],
                        "points": [
                            {"space": "MNI", "coordinates": [1.0, 2.0, 3.0]},
                            {"space": "MNI", "coordinates": [4.0, 5.0, 6.0]},
                        ],
                        "metadata": {"sample_sizes": [25]},
                    }
                ],
            }
        ],
    }

    def sha256(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        out1 = Path(tmp1)
        out2 = Path(tmp2)
        convert_nimads_to_sleuth(studyset, out1, decimal_precision=3, target_space="MNI")
        convert_nimads_to_sleuth(studyset, out2, decimal_precision=3, target_space="MNI")
        f1 = next(out1.glob("*.txt"))
        f2 = next(out2.glob("*.txt"))
        assert sha256(f1) == sha256(f2)


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
            studyset=example_nimads_studyset,
            output_dir=output_dir,
            target_space='MNI'
        )
        
        # Check output file
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1
        
        with open(sleuth_files[0], 'r') as f:
            content = f.read()
            
        # Check that space header is correct
        assert 'Reference = MNI' in content


def test_empty_studyset():
    """Test conversion with empty studyset."""
    from nimare import nimads
    empty_studyset = nimads.Studyset({"id": "empty", "name": "Empty Studyset", "studies": []})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Convert empty studyset
        convert_nimads_to_sleuth(
            studyset=empty_studyset,
            output_dir=output_dir
        )
        
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
        convert_nimads_to_sleuth(
            studyset=studyset_copy,
            output_dir=output_dir
        )
        
        # Should create output directory and file
        assert output_dir.exists()
        sleuth_files = list(output_dir.glob("*.txt"))
        assert len(sleuth_files) == 1


if __name__ == "__main__":
    pytest.main([__file__])
