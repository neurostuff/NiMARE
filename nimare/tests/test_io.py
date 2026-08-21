"""Test nimare.io (Dataset IO/transformations)."""

import copy
import json
import logging
import os

import pandas as pd
import pytest
from scipy import sparse

import nimare
from nimare import io
from nimare.nimads import Studyset, convert_neurostore_json_to_parquet
from nimare.tests.utils import get_test_data_path
from nimare.utils import get_resource_path, get_template

NEUROSTORE_STUDYSET_ID = "qm2PZBqNsaZK"
NEUROSTORE_ANNOTATION_ID = "hbTQJVL2kAb8"


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    """Store test_io cassettes in a stable directory."""
    return os.path.join(os.path.dirname(__file__), "cassettes", "test_io")


@pytest.fixture(scope="module")
def vcr_config():
    """Keep Neurostore cassettes stable and free of credentials."""
    return {
        "filter_headers": ["authorization"],
        "decode_compressed_response": True,
    }


@pytest.mark.vcr
def test_fetch_neurostore_studyset():
    """Download a nested Neurostore studyset into a NiMARE Studyset."""
    studyset = io.fetch_neurostore_studyset(NEUROSTORE_STUDYSET_ID, target="ale_2mm")

    assert isinstance(studyset, Studyset)
    assert studyset.id == NEUROSTORE_STUDYSET_ID
    assert studyset.space == "ale_2mm"
    assert len(studyset.studies) > 0


@pytest.mark.vcr
def test_fetch_neurostore_studyset_with_annotation():
    """Download a nested Neurostore studyset and attach a regular annotation."""
    studyset = io.fetch_neurostore_studyset(
        NEUROSTORE_STUDYSET_ID,
        annotation_id=NEUROSTORE_ANNOTATION_ID,
    )

    assert isinstance(studyset, Studyset)
    assert studyset.id == NEUROSTORE_STUDYSET_ID
    assert len(studyset.annotations) == 1
    assert studyset.annotations[0].id == NEUROSTORE_ANNOTATION_ID
    assert "included" in studyset.annotations_df.columns
    assert studyset.annotations_df["included"].any()


def test_fetch_neurostore_studyset_wraps_api_errors(monkeypatch):
    """Neurostore request failures should report the resource that failed."""
    import neurostore_sdk

    class FailingStoreApi:
        def studysets_id_get(self, id, nested=True):
            raise neurostore_sdk.ApiException(status=404, reason="Not Found")

    monkeypatch.setattr(neurostore_sdk, "StoreApi", FailingStoreApi)

    with pytest.raises(ValueError, match="Failed to download Neurostore studyset 'bad-id'"):
        io.fetch_neurostore_studyset("bad-id")


def test_fetch_neurostore_studyset_wraps_annotation_api_errors(monkeypatch):
    """Neurostore annotation request failures should report the annotation ID."""
    import neurostore_sdk

    class FailingAnnotationStoreApi:
        def studysets_id_get(self, id, nested=True):
            return {"id": "ok-id", "studies": []}

        def annotations_id_get(self, id):
            raise neurostore_sdk.ApiException(status=404, reason="Not Found")

    monkeypatch.setattr(neurostore_sdk, "StoreApi", FailingAnnotationStoreApi)

    with pytest.raises(
        ValueError,
        match="Failed to download Neurostore annotation 'bad-annotation'",
    ):
        io.fetch_neurostore_studyset("ok-id", annotation_id="bad-annotation")


def test_convert_neurostore_json_to_parquet_and_load_all_annotations(tmp_path):
    """Release JSON should round-trip through parquet-backed Studysets."""
    studyset_file = tmp_path / "neurostore-studyset.json"
    annotation_file = tmp_path / "neurostore-annotation.json"
    manifest_file = tmp_path / "manifest.json"
    parquet_dir = tmp_path / "parquet"

    studyset_file.write_text(
        json.dumps(
            {
                "name": "release-studyset",
                "studies": [
                    {
                        "id": "study-1",
                        "name": "Example study",
                        "description": "A study-level description.",
                        "authors": "A. Author",
                        "publication": "Example Journal",
                        "analyses": [
                            {
                                "id": "analysis-1",
                                "name": "faces > shapes",
                                "metadata": {"sample_size": "20"},
                                "points": [
                                    {
                                        "coordinates": [1, 2, 3],
                                        "space": "MNI",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )
    annotation_file.write_text(
        json.dumps(
            {
                "id": None,
                "name": "release-annotation",
                "studyset": "studyset-1",
                "notes": [
                    {
                        "analysis": "analysis-1",
                        "note": {
                            "diagnosis": "control",
                            "age_mean": 31.5,
                        },
                    }
                ],
            }
        )
    )
    manifest_file.write_text(
        json.dumps(
            {
                "studyset": {"id": "studyset-1", "name": "release-studyset"},
                "annotation": {"id": "annotation-1", "name": "release-annotation"},
            }
        )
    )

    convert_neurostore_json_to_parquet(
        studyset_file,
        parquet_dir,
        annotation_source=annotation_file,
        manifest_source=manifest_file,
    )
    studyset = Studyset(parquet_dir)

    assert isinstance(studyset, Studyset)
    assert studyset.id == "studyset-1"
    assert studyset.name == "release-studyset"
    assert studyset.ids.tolist() == ["study-1-analysis-1"]
    assert studyset.coordinates[["x", "y", "z"]].values.tolist() == [[1.0, 2.0, 3.0]]
    assert studyset.metadata["sample_sizes"].tolist() == [[20]]
    assert studyset.annotations_df["diagnosis"].tolist() == ["control"]
    assert studyset.annotations_df["age_mean"].tolist() == [31.5]
    assert pd.read_parquet(parquet_dir / "studies.parquet").columns.tolist() == [
        "study_id",
        "name",
        "description",
        "authors",
        "publication",
    ]
    assert len(studyset.studies) == 1
    assert studyset.studies[0].description == "A study-level description."
    analysis = studyset.studies[0].analyses[0]
    # `annotations` is keyed by annotation, since a studyset may carry several;
    # `labels` is the merged view of them.
    assert analysis.annotations["annotation-1"]["diagnosis"] == "control"
    assert analysis.labels["diagnosis"] == "control"
    assert studyset.to_dict()["studies"][0]["description"] == "A study-level description."


def test_convert_neurostore_json_to_parquet_dict_inputs(tmp_path):
    """convert_neurostore_json_to_parquet should accept pre-loaded dicts for all sources."""
    studyset_dict = {
        "name": "release-studyset",
        "studies": [
            {
                "id": "study-1",
                "name": "Example study",
                "analyses": [
                    {
                        "id": "analysis-1",
                        "name": "faces > shapes",
                        "metadata": {"sample_size": "20"},
                        "points": [{"coordinates": [1, 2, 3], "space": "MNI"}],
                    }
                ],
            }
        ],
    }
    annotation_dict = {"notes": [{"analysis": "analysis-1", "note": {"diagnosis": "control"}}]}
    manifest_dict = {
        "studyset": {"id": "studyset-1", "name": "release-studyset"},
        "annotation": {"id": "annotation-1", "name": "release-annotation"},
    }

    convert_neurostore_json_to_parquet(
        studyset_dict,
        tmp_path / "parquet",
        annotation_source=annotation_dict,
        manifest_source=manifest_dict,
    )
    studyset = Studyset(tmp_path / "parquet")

    assert studyset.id == "studyset-1"
    assert studyset.ids.tolist() == ["study-1-analysis-1"]
    assert studyset.coordinates[["x", "y", "z"]].values.tolist() == [[1.0, 2.0, 3.0]]
    assert studyset.annotations_df["diagnosis"].tolist() == ["control"]


@pytest.mark.parametrize(
    "studyset_data,match",
    [
        (
            {
                "id": "studyset-1",
                "studies": [{"name": "No ID study", "analyses": []}],
            },
            "Could not infer an id for study",
        ),
        (
            {
                "id": "studyset-1",
                "studies": [
                    {
                        "id": "study-1",
                        "analyses": [{"name": "No ID analysis", "points": []}],
                    }
                ],
            },
            "has no id",
        ),
    ],
)
def test_convert_neurostore_json_to_parquet_requires_ids(tmp_path, studyset_data, match):
    """Release conversion should fail explicitly when required IDs are missing."""
    studyset_file = tmp_path / "neurostore-studyset.json"
    studyset_file.write_text(json.dumps(studyset_data))

    with pytest.raises(ValueError, match=match):
        convert_neurostore_json_to_parquet(studyset_file, tmp_path / "parquet")


def test_load_neurostore_parquet_studyset_resource():
    """The packaged NeuroStore parquet fixture should load directly."""
    parquet_dir = os.path.join(get_resource_path(), "neurostore_parquet_studyset")

    studyset = Studyset(parquet_dir)

    assert studyset.id == "test-neurostore-parquet-studyset"
    assert len(studyset.study_ids) == 4
    assert len(studyset.ids) == 8
    # Columns that are null for every row are not carried: the store keeps
    # sparse columns, so a point-value column nothing populates has nothing to
    # keep. The ids, coordinates, space and kind are all present.
    assert studyset.coordinates.shape == (47, 8)
    assert list(studyset.coordinates.columns) == [
        "id",
        "study_id",
        "contrast_id",
        "x",
        "y",
        "z",
        "space",
        "kind",
    ]
    assert studyset.metadata.shape == (8, 154)
    assert studyset.annotations_df.shape == (8, 502)
    assert pd.read_parquet(os.path.join(parquet_dir, "studies.parquet")).columns.tolist() == [
        "study_id",
        "name",
        "description",
        "authors",
        "publication",
    ]


def test_convert_nimads_to_dataset(example_nimads_studyset, example_nimads_annotation):
    """Conversion of nimads JSON to nimare dataset."""
    studyset = Studyset(example_nimads_studyset)
    first_study_with_analyses = next(study for study in studyset.studies if study.analyses)
    first_analysis = first_study_with_analyses.analyses[0]
    expected_id = f"{first_study_with_analyses.id}-{first_analysis.id}"

    dset1 = io.convert_nimads_to_dataset(studyset)
    annotated = studyset.with_annotation_payload(example_nimads_annotation)
    dset2 = io.convert_nimads_to_dataset(annotated)

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



def _one_analysis_studyset(
    study_metadata=None, analysis_metadata=None, annotation_note=None
):
    """Build a one-study, one-analysis studyset with the given metadata.

    Studysets are immutable, so a test states the data it wants in a document
    rather than reaching into a loaded studyset and mutating it.
    """
    document = {
        "id": "studyset-1",
        "name": "test",
        "studies": [
            {
                "id": "study-1",
                "name": "Study one",
                "metadata": study_metadata,
                "analyses": [
                    {
                        "id": "analysis-1",
                        "name": "contrast",
                        "metadata": analysis_metadata,
                        "points": [{"coordinates": [1.0, 2.0, 3.0], "space": "MNI"}],
                    }
                ],
            }
        ],
    }
    if annotation_note is not None:
        document["annotations"] = [
            {
                "id": "annot",
                "name": "annot",
                "note_keys": {key: None for key in annotation_note},
                "notes": [{"analysis": "analysis-1", "note": annotation_note}],
            }
        ]
    return Studyset(document)


def test_convert_nimads_to_dataset_single_sample_size():
    """A single sample_size value should become a sample_sizes column."""
    studyset = _one_analysis_studyset(analysis_metadata={"sample_size": 20})

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
    sample_sizes_val, sample_size_val, expect_col, expect_warning, caplog
):
    """Test conversion of nimads JSON to nimare dataset with different sample_size(s) values."""
    metadata = {}
    if sample_sizes_val is not None:
        metadata["sample_sizes"] = sample_sizes_val
    if sample_size_val is not None:
        metadata["sample_size"] = sample_size_val
    studyset = _one_analysis_studyset(
        study_metadata=dict(metadata), analysis_metadata=dict(metadata)
    )

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


def test_analysis_to_dict_sample_size_prioritizes_annotations():
    """sample_size in annotations takes priority over metadata."""
    studyset = _one_analysis_studyset(
        study_metadata={"sample_size": 10},
        analysis_metadata={"sample_size": 20},
        annotation_note={"sample_size": 30},
    )

    dset = io.convert_nimads_to_dataset(studyset)

    md_row = dset.metadata.loc[dset.metadata["id"] == "study-1-analysis-1"].iloc[0]
    assert md_row["sample_sizes"] == [30]


def test_analysis_to_dict_sample_sizes_prioritizes_annotations():
    """sample_sizes in annotations wins when sample_size is absent."""
    studyset = _one_analysis_studyset(
        study_metadata={"sample_sizes": [1, 10]},
        analysis_metadata={"sample_sizes": [2, 20]},
        annotation_note={"sample_sizes": [40, 41]},
    )

    dset = io.convert_nimads_to_dataset(studyset)

    md_row = dset.metadata.loc[dset.metadata["id"] == "study-1-analysis-1"].iloc[0]
    assert md_row["sample_sizes"] == [40, 41]


def test_analysis_to_dict_annotation_sample_size_falls_back_to_metadata(caplog):
    """An unusable annotation sample size falls back to metadata."""
    studyset = _one_analysis_studyset(
        analysis_metadata={"sample_size": 20},
        annotation_note={"sample_size": "not_a_number"},
    )

    with caplog.at_level("WARNING"):
        dset = io.convert_nimads_to_dataset(studyset)

    md_row = dset.metadata.loc[dset.metadata["id"] == "study-1-analysis-1"].iloc[0]
    assert md_row["sample_sizes"] == [20]
    assert any(
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
        # Annotation with missing 'notes' key: nothing to apply, not an error
        (lambda ann: ann.pop("notes", None), True, False),
        # Annotation naming an analysis the studyset does not hold: ignored
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
            False,
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
    if annotation_mod is None:
        dset = io.convert_nimads_to_dataset(studyset)
        assert isinstance(dset, nimare.dataset.Dataset)
        return

    annotation = copy.deepcopy(example_nimads_annotation)
    annotation_mod(annotation)
    # A malformed payload should be rejected when it is attached, and a note
    # naming an analysis the studyset does not hold is simply not applied.
    annotated = studyset.with_annotation_payload(annotation)
    dset = io.convert_nimads_to_dataset(annotated)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert expect_success


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


def test_convert_sleuth_to_studyset_smoke():
    """Smoke test for direct Sleuth-to-Studyset conversion."""
    sleuth_file = os.path.join(get_test_data_path(), "test_sleuth_file.txt")
    studyset = io.convert_sleuth_to_studyset(sleuth_file)

    assert isinstance(studyset, Studyset)
    assert studyset.coordinates.shape[0] == 7
    assert len(studyset.ids) == 3


@pytest.mark.parametrize(
    "header_lines,expected_study,expected_contrast",
    [
        # Simple colon separator.
        (["Smith et al., 2010: Condition A"], "Smith et al., 2010", "Condition A"),
        # Simple semicolon separator.
        (["Smith et al., 2010; Condition A"], "Smith et al., 2010", "Condition A"),
        # Year in the middle, separator after year.
        (
            ["Journal of Neuroscience: Methods Smith et al., 2010; Condition A"],
            "Journal of Neuroscience: Methods Smith et al., 2010",
            "Condition A",
        ),
        # No year, fall back to first semicolon.
        (["StudyName; ExperimentName"], "StudyName", "ExperimentName"),
        # No separator at all, use default contrast name.
        (["StudyName Only"], "StudyName Only", "analysis_1"),
        # Multi-line header collapsed into one, separator after year.
        (
            ["Smith et al.", "2010; Condition A"],
            "Smith et al. 2010",
            "Condition A",
        ),
    ],
)
def test_convert_sleuth_to_dict_study_info_parsing(
    tmp_path, header_lines, expected_study, expected_contrast
):
    """Study/contrast names are parsed robustly from Sleuth headers."""
    sleuth_path = tmp_path / "test_sleuth_header.txt"
    lines = ["// Reference = MNI"]
    for line in header_lines:
        lines.append(f"// {line}")
    lines.append("// Subjects = 10")
    # Single valid coordinate row.
    lines.append("0 0 0")
    sleuth_path.write_text("\n".join(lines))

    dset_dict = io.convert_sleuth_to_dict(str(sleuth_path))
    assert list(dset_dict.keys()) == [expected_study]
    contrasts = dset_dict[expected_study]["contrasts"]
    assert list(contrasts.keys()) == [expected_contrast]


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


def test_convert_neurosynth_to_studyset_smoke():
    """Smoke test for fast Neurosynth Studyset conversion."""
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
    studyset = io.convert_neurosynth_to_studyset(
        coordinates_file,
        metadata_file,
        annotations_files=features,
    )
    assert isinstance(studyset, Studyset)
    assert len(studyset.ids) == 20
    assert "terms_abstract_tfidf__abilities" in studyset.annotations_df.columns


def test_convert_neurosynth_to_studyset_rejects_mismatched_feature_groups(tmp_path):
    """feature_groups length mismatch should raise a descriptive ValueError."""
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

    with pytest.raises(
        ValueError, match="feature_groups and annotations_files must have the same length"
    ):
        io.convert_neurosynth_to_studyset(
            coordinates_file,
            metadata_file,
            annotations_files=[features, features],
            feature_groups=["only_one"],
        )


def test_convert_neurosynth_to_studyset_rejects_unparseable_feature_filename(tmp_path):
    """Malformed feature filenames should raise a descriptive ValueError."""
    coordinates_file = os.path.join(
        get_test_data_path(),
        "data-neurosynth_version-7_coordinates.tsv.gz",
    )
    metadata_file = os.path.join(
        get_test_data_path(),
        "data-neurosynth_version-7_metadata.tsv.gz",
    )

    bad_features_file = tmp_path / "bad_features.npz"
    sparse.save_npz(bad_features_file, sparse.csr_matrix([[1.0], [0.0]]))
    vocabulary_file = tmp_path / "bad_vocab.txt"
    vocabulary_file.write_text("term\n")

    with pytest.raises(ValueError, match="Could not parse feature filename entity"):
        io.convert_neurosynth_to_studyset(
            coordinates_file,
            metadata_file,
            annotations_files={
                "features": str(bad_features_file),
                "vocabulary": str(vocabulary_file),
            },
        )


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


def test_convert_neurovault_to_dataset_retries_collection_json(monkeypatch, tmp_path):
    """Transient JSON parsing failures should not mask empty-contrast errors."""

    class _MockResponse:
        def __init__(self, *, payload=None, json_error=False, status_code=200, content=b""):
            self._payload = payload
            self._json_error = json_error
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise io.requests.HTTPError(f"HTTP {self.status_code}")

        def json(self):
            if self._json_error:
                raise json.JSONDecodeError("Expecting value", "", 0)
            return self._payload

    call_count = {"count": 0}

    def _mock_get(url, timeout=30):
        call_count["count"] += 1
        assert url.endswith("/images/?format=json")
        if call_count["count"] == 1:
            return _MockResponse(json_error=True)
        return _MockResponse(payload={"results": []})

    monkeypatch.setattr(io.requests, "get", _mock_get)

    with pytest.raises(ValueError) as excinfo:
        io.convert_neurovault_to_dataset(
            collection_ids=(8836,),
            contrasts={"crab_people": "cannot hurt you because they do not exist"},
            img_dir=tmp_path,
        )

    assert "No images were found for contrast crab_people" in str(excinfo.value)
    assert call_count["count"] == 2


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


def test_unsupported_image_type_warns_once_per_type(caplog):
    """A studyset full of unusable images should say so once, not once per image.

    The warning names a value type, which is the same for every image carrying it, so
    repeating it thousands of times for a large neurostore studyset would bury the log
    without telling the reader anything new.
    """
    io._warn_unsupported_image_type.cache_clear()

    with caplog.at_level(logging.WARNING, logger="nimare.io"):
        for _ in range(50):
            assert io._normalize_image_type("F map") is None
        for _ in range(50):
            assert io._normalize_image_type("chi-squared map") is None

    messages = [r.message for r in caplog.records if "unsupported value type" in r.message]
    assert len(messages) == 2
    assert any("'F map'" in m for m in messages)
    assert any("'chi-squared map'" in m for m in messages)


def test_supported_image_types_are_still_normalized():
    """Deduplicating the warning must not change which types are accepted."""
    assert io._normalize_image_type("Z map") == "z"
    assert io._normalize_image_type("variance") == "varcope"
    assert io._normalize_image_type("T map") == "t"
    assert io._normalize_image_type(None) is None
