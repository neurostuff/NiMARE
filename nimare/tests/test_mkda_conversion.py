"""Tests for MKDA coordinate table import."""

import pytest

from nimare.io import convert_mkda_to_studyset
from nimare.nimads import Studyset


def mkda_file(tmp_path, header, rows):
    """Write a small MKDA table, including its leading column count."""
    path = tmp_path / "coordinates.txt"
    path.write_text(
        str(len(header))
        + "\t" * (len(header) - 1)
        + "\n"
        + "\t".join(header)
        + "\n"
        + "\n".join(rows)
        + "\n"
    )
    return path


def test_convert_mkda_groups_foci_and_keeps_selected_fields(tmp_path):
    """A real MKDA header groups rows by study and contrast."""
    header = ["x", "y", "z", "Study", "Contrast", "N", "Original_Space", "Year", "Task", "Unused"]
    rows = [
        '44\t-58\t54\t"Chaminade_&_Decety,_2002"\t1\t9\tMNI\t2002\timitation\tdrop',
        '32\t-50\t52\t"Chaminade_&_Decety,_2002"\t1\t9\tMNI\t2002\timitation\tdrop',
        '1\t2\t3\t"Chaminade_&_Decety,_2002"\t2\t9\tMNI\t2002\tobservation\tdrop',
        "-50\t-58\t16\tDavid_et_al._2006\t1\t13\tMNI\t2006\tagency\tdrop",
    ]
    path = mkda_file(tmp_path, header, rows)

    studyset = convert_mkda_to_studyset(
        path, metadata_columns=["Year"], annotation_columns=["Task"]
    )

    assert isinstance(studyset, Studyset)
    assert len(studyset.studies) == 2
    assert len(studyset.analyses) == 3
    assert len(studyset.coordinates) == 4
    document = studyset.to_dict()
    studies = {study["id"]: study for study in document["studies"]}
    analyses = {
        analysis["id"]: analysis for analysis in studies["Chaminade_&_Decety,_2002"]["analyses"]
    }
    assert len(analyses["1"]["points"]) == 2
    assert analyses["1"]["metadata"]["sample_sizes"] == [9]
    assert analyses["1"]["metadata"]["Year"] == 2002
    assert (
        studyset.annotations_df.loc[
            studyset.annotations_df["id"] == "Chaminade_&_Decety,_2002-1", "Task"
        ].item()
        == "imitation"
    )
    assert "Unused" not in analyses["1"]["metadata"]
    assert "Unused" not in studyset.annotations_df.columns


def test_convert_mkda_subjects_and_t88_space(tmp_path):
    """Subjects supplies sample size; T88 is retained as Talairach space."""
    path = mkda_file(
        tmp_path,
        ["x", "y", "z", "study", "Contrast", "Subjects", "CoordSys"],
        ["1\t2\t3\tStudy A\tC1\t12\tT88"],
    )

    studyset = convert_mkda_to_studyset(path)
    analysis = studyset.to_dict()["studies"][0]["analyses"][0]
    assert analysis["metadata"]["sample_sizes"] == [12]
    assert analysis["points"][0]["space"] == "TAL"


@pytest.mark.parametrize(
    "source,expected",
    [("MNI152", "MNI"), ("ALE", "MNI"), ("Talairach-Tournoux", "TAL")],
)
def test_convert_mkda_accepts_common_space_labels(tmp_path, source, expected):
    """Common source labels use NiMARE's existing MNI and Talairach categories."""
    path = mkda_file(
        tmp_path,
        ["x", "y", "z", "Study", "Contrast", "Original_Space"],
        [f"1\t2\t3\tA\tB\t{source}"],
    )
    studyset = convert_mkda_to_studyset(path)
    assert studyset.to_dict()["studies"][0]["analyses"][0]["points"][0]["space"] == expected


def test_convert_mkda_full_ids_remain_unique_with_hyphens(tmp_path):
    """Different study and contrast pairs cannot produce the same full id."""
    path = mkda_file(
        tmp_path,
        ["x", "y", "z", "Study", "Contrast", "Original_Space"],
        [
            "1\t2\t3\tA-B\tC\tMNI",
            "4\t5\t6\tA\tB-C\tMNI",
            "7\t8\t9\tA\tB%2DC\tMNI",
        ],
    )
    studyset = convert_mkda_to_studyset(path)
    assert set(studyset.ids) == {"A-B-C", "A-B%2DC", "A-B%252DC"}
    assert {analysis.name for analysis in studyset.analyses} == {"C", "B-C", "B%2DC"}


def test_convert_mkda_requires_explicit_space_without_space_column(tmp_path):
    """Import does not silently assume a coordinate space."""
    path = mkda_file(
        tmp_path,
        ["x", "y", "z", "Study", "Contrast"],
        ["1\t2\t3\tA\tB"],
    )
    with pytest.raises(ValueError, match="specify space explicitly"):
        convert_mkda_to_studyset(path)

    studyset = convert_mkda_to_studyset(path, space="MNI")
    assert studyset.to_dict()["studies"][0]["analyses"][0]["points"][0]["space"] == "MNI"


@pytest.mark.parametrize(
    "header,rows,match",
    [
        (
            ["x", "y", "z", "Study", "Contrast", "Original_Space"],
            ["a\t2\t3\tS\tC\tMNI"],
            "coordinates must be numeric",
        ),
        (
            ["x", "y", "z", "Study", "Contrast", "Original_Space"],
            ["nan\t2\t3\tS\tC\tMNI"],
            "coordinates must be finite",
        ),
        (
            ["x", "y", "z", "Study", "Contrast", "Original_Space"],
            ["1\t2\t3\tS\tC\tXYZ"],
            "unsupported coordinate space",
        ),
        (
            ["x", "y", "z", "Study", "Contrast", "Original_Space"],
            ["1\t2\t3\tS\tC"],
            "expected 6 columns",
        ),
        (
            ["x", "y", "z", "Study", "Contrast", "N", "Original_Space"],
            ["1\t2\t3\tS\tC\t0\tMNI"],
            "positive integer",
        ),
        (
            ["x", "y", "z", "Study", "Contrast", "N", "Subjects", "Original_Space"],
            ["1\t2\t3\tS\tC\t9\t10\tMNI"],
            "N and Subjects disagree",
        ),
    ],
)
def test_convert_mkda_rejects_invalid_rows(tmp_path, header, rows, match):
    """Invalid focus rows fail with an actionable error."""
    path = mkda_file(tmp_path, header, rows)
    with pytest.raises(ValueError, match=match):
        convert_mkda_to_studyset(path)


@pytest.mark.parametrize(
    "field,kwargs",
    [("Year", {"metadata_columns": ["Year"]}), ("Task", {"annotation_columns": ["Task"]})],
)
def test_convert_mkda_rejects_conflicting_analysis_fields(tmp_path, field, kwargs):
    """Analysis attributes cannot depend on which focus row was read last."""
    path = mkda_file(
        tmp_path,
        ["x", "y", "z", "Study", "Contrast", "Original_Space", "Year", "Task"],
        ["1\t2\t3\tS\tC\tMNI\t2001\tmemory", "4\t5\t6\tS\tC\tMNI\t2002\tvisual"],
    )
    with pytest.raises(ValueError, match=f"inconsistent '{field}'"):
        convert_mkda_to_studyset(path, **kwargs)
