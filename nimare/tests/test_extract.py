"""Test nimare.extract."""

import os
from glob import glob
from io import BytesIO
from unittest.mock import patch

import nimare


def mock_urlopen(url):
    """Mock URL opener that returns fake data."""
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
    files = glob(os.path.join(tmpdir, "neurosynth", "*"))
    assert len(files) == 4

    # One set of files found
    assert isinstance(data_files, list)
    assert len(data_files) == 1


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
