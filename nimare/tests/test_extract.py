"""Test nimare.extract."""

import os
from glob import glob
from io import BytesIO
from unittest.mock import patch

import nimare


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


@patch("nimare.extract.extract.urlopen", side_effect=mock_urlopen)
def test_fetch_neurosynth(tmp_path_factory):
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
