"""Parity checks for direct Sleuth conversion without a Dataset intermediary."""

import numpy as np
import pytest

from nimare import io
from nimare.studyset import Studyset


@pytest.mark.parametrize("space", ["MNI", "TAL", "Talairach"])
@pytest.mark.parametrize("target", [None, "ale_2mm", "mni152_2mm"])
@pytest.mark.parametrize("as_studyset", [False, True])
@pytest.mark.parametrize("multiple_files", [False, True])
def test_direct_sleuth_preserves_legacy_output(
    tmp_path, monkeypatch, space, target, as_studyset, multiple_files
):
    """Preserve payloads and execution context without constructing a Dataset."""
    path = tmp_path / "sleuth.txt"
    path.write_text(
        f"//Reference={space}\n//Zulu, 2020: task\n//Subjects=12\n1 2 3\n4 5 6\n"
        "//Alpha, 2021: control\n//Subjects=20\n7 8 9\n"
    )
    source = str(path)
    if multiple_files:
        other = tmp_path / "other.txt"
        other.write_text(f"//Reference={space}\n//Zulu, 2020: another\n//Subjects=18\n10 11 12\n")
        source = [str(path), str(other)]
    legacy = io.convert_sleuth_to_dataset(source, target=target)
    expected = (
        Studyset.from_dataset(legacy).to_dict()
        if as_studyset
        else io.convert_dataset_to_nimads_dict(legacy, studyset_id="audit", studyset_name="Audit")
    )

    def reject_dataset(*args, **kwargs):
        raise AssertionError("Direct conversion must not construct Dataset")

    monkeypatch.setattr(io, "Dataset", reject_dataset)
    if as_studyset:
        result = io.convert_sleuth_to_studyset(source, target=target)
        assert result.space == legacy.space
        if legacy.masker is None:
            assert result.masker is None
        else:
            np.testing.assert_array_equal(
                result.masker.mask_img_.get_fdata(), legacy.masker.mask_img_.get_fdata()
            )
            np.testing.assert_array_equal(
                result.masker.mask_img_.affine, legacy.masker.mask_img_.affine
            )
        actual = result.to_dict()
    else:
        actual = io.convert_sleuth_to_nimads_dict(
            source, target=target, studyset_id="audit", studyset_name="Audit"
        )
    assert actual == expected


@pytest.mark.parametrize("input_kind", ["path", "tuple", "list"])
def test_direct_sleuth_dict_accepts_paths(tmp_path, input_kind):
    """Keep the dictionary converter's supported path and sequence inputs."""
    path = tmp_path / "sleuth.txt"
    path.write_text("//Reference=MNI\n//Study: task\n//Subjects=12\n1 2 3\n")
    source = {"path": path, "tuple": (path,), "list": [path]}[input_kind]
    assert io.convert_sleuth_to_nimads_dict(source) == io.convert_sleuth_to_nimads_dict(str(path))


@pytest.mark.parametrize("as_studyset", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "//Reference=unknown\n//Study: task\n//Subjects=12\n1 2 3\n",
        "//Reference=MNI\n//Study: task\n//Subjects=12\n1 2\n",
    ],
)
def test_direct_sleuth_preserves_invalid_input_errors(tmp_path, text, as_studyset):
    """Retain parser errors for malformed coordinates and unknown spaces."""
    path = tmp_path / "invalid.txt"
    path.write_text(text)
    with pytest.raises(Exception) as legacy_error:
        io.convert_sleuth_to_dataset(str(path))
    convert = io.convert_sleuth_to_studyset if as_studyset else io.convert_sleuth_to_nimads_dict
    with pytest.raises(type(legacy_error.value)) as direct_error:
        convert(str(path))
    assert str(direct_error.value) == str(legacy_error.value)


def test_direct_sleuth_omits_missing_coordinates(tmp_path):
    """Match the legacy treatment of missing coordinate values."""
    path = tmp_path / "missing.txt"
    path.write_text("//Reference=MNI\n//Study: task\n//Subjects=12\nnan 2 3\n4 5 6\n")
    legacy = io.convert_sleuth_to_dataset(str(path), target=None)
    expected = io.convert_dataset_to_nimads_dict(legacy, studyset_id=None)
    assert io.convert_sleuth_to_nimads_dict(path) == expected
