"""Test nimare.extract."""

import os
import sys
import types
from glob import glob
from io import BytesIO
from unittest.mock import patch

import nimare
from nimare.dataset import Dataset
from nimare.generate import create_coordinate_studyset
from nimare.nimads import Studyset
from nimare.tests.utils import get_test_data_path


def mock_urlopen(url):
    """Mock URL opener that returns appropriate mock data based on file type."""
    if "coordinates.tsv.gz" in url:
        mock_data = b"x\ty\tz\n1\t2\t3\n4\t5\t6"
    elif "metadata.tsv.gz" in url:
        mock_data = b"id\ttitle\n1\tStudy 1\n2\tStudy 2"
    elif "features.npz" in url:
        mock_data = b"mock npz content"
    elif "vocabulary.txt" in url:
        mock_data = b"term1\nterm2\nterm3"
    else:
        mock_data = b"Mock file content"
    return BytesIO(mock_data)


def _local_neurosynth_manifest():
    base = get_test_data_path()
    return [
        {
            "coordinates": os.path.join(base, "data-neurosynth_version-7_coordinates.tsv.gz"),
            "metadata": os.path.join(base, "data-neurosynth_version-7_metadata.tsv.gz"),
            "features": [
                {
                    "features": os.path.join(
                        base,
                        "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_"
                        "features.npz",
                    ),
                    "vocabulary": os.path.join(
                        base,
                        "data-neurosynth_version-7_vocab-terms_vocabulary.txt",
                    ),
                }
            ],
        }
    ]


@patch("nimare.extract.extract.urlopen", side_effect=mock_urlopen)
def test_fetch_neurosynth(mock_url, tmp_path_factory):
    """Smoke test for extract.fetch_neurosynth.

    Taken from the Neurosynth Python package.
    """
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth")
    data_files = nimare.extract.fetch_neurosynth(
        data_dir=tmpdir,
        version="7",
        overwrite=False,
        source="abstract",
        vocab="terms",
        return_type="files",
    )
    # Check data_files structure
    assert isinstance(data_files, list)
    assert len(data_files) == 1

    # Verify expected files in data_files
    files_dict = data_files[0]
    assert "coordinates" in files_dict
    assert "metadata" in files_dict
    assert "features" in files_dict
    assert len(files_dict["features"]) == 1


@patch("nimare.extract.extract.urlopen", side_effect=mock_urlopen)
def test_fetch_neuroquery(mock_url, tmp_path_factory):
    """Smoke test for extract.fetch_neuroquery."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neuroquery")
    data_files = nimare.extract.fetch_neuroquery(
        data_dir=tmpdir,
        version="1",
        overwrite=False,
        source="abstract",
        vocab="neuroquery7547",
        type="count",
        return_type="files",
    )
    files = glob(os.path.join(tmpdir, "neuroquery", "*"))
    assert len(files) == 4

    # One set of files found
    assert isinstance(data_files, list)
    assert len(data_files) == 1

    # Verify mock was called with expected URLs
    assert mock_url.call_count > 0  # Should be called for each file download
    for call in mock_url.call_args_list:
        url = call[0][0]
        assert "neuroquery/neuroquery_data/blob" in url
        assert "?raw=true" in url


def test_fetch_neurosynth_returns_studyset_by_default(monkeypatch, tmp_path_factory):
    """fetch_neurosynth should return Studysets by default."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth_returns_studyset_by_default")
    monkeypatch.setattr(
        nimare.extract.extract,
        "_fetch_database",
        lambda *args, **kwargs: _local_neurosynth_manifest(),
    )

    outputs = nimare.extract.fetch_neurosynth(
        data_dir=tmpdir,
        version="7",
        source="abstract",
        vocab="terms",
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], Studyset)


def test_fetch_neurosynth_can_return_dataset(monkeypatch, tmp_path_factory):
    """fetch_neurosynth should support the legacy Dataset return type."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth_can_return_dataset")
    monkeypatch.setattr(
        nimare.extract.extract,
        "_fetch_database",
        lambda *args, **kwargs: _local_neurosynth_manifest(),
    )

    outputs = nimare.extract.fetch_neurosynth(
        data_dir=tmpdir,
        version="7",
        source="abstract",
        vocab="terms",
        return_type="dataset",
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], Dataset)


def test_fetch_neuroquery_returns_studyset_by_default(monkeypatch, tmp_path_factory):
    """fetch_neuroquery should return Studysets by default."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neuroquery_returns_studyset_by_default")
    monkeypatch.setattr(
        nimare.extract.extract,
        "_fetch_database",
        lambda *args, **kwargs: _local_neurosynth_manifest(),
    )

    outputs = nimare.extract.fetch_neuroquery(
        data_dir=tmpdir,
        version="1",
        source="combined",
        vocab="neuroquery6308",
        type="tfidf",
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], Studyset)


def test_download_abstracts_accepts_studyset(monkeypatch):
    """download_abstracts should update Studyset texts in place."""

    class _DummyEntrez:
        email = None

        @staticmethod
        def efetch(db, id, rettype, retmode):
            return id

    class _DummyMedline:
        @staticmethod
        def parse(handle):
            for pmid in handle:
                yield {"PMID": pmid, "AB": f"Abstract for {pmid}"}

    bio_module = types.ModuleType("Bio")
    bio_module.Entrez = _DummyEntrez
    bio_module.Medline = _DummyMedline
    monkeypatch.setitem(sys.modules, "Bio", bio_module)

    _, studyset = create_coordinate_studyset(foci=2, sample_size=20, n_studies=3, seed=2)
    studyset = nimare.extract.download_abstracts(studyset, "example@example.edu")

    assert "abstract" in studyset.texts.columns
    assert studyset.texts["abstract"].notnull().all()
