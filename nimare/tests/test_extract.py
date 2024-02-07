"""Test nimare.extract."""

import os
from glob import glob

import nimare


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


def test_fetch_neuroquery(tmp_path_factory):
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
