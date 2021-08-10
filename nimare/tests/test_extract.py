"""Test nimare.extract."""
import os
from glob import glob

import nimare


def test_fetch_neurosynth(tmp_path_factory):
    """Smoke test for extract.fetch_neurosynth.

    Taken from the Neurosynth Python package.
    """
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth")
    nimare.extract.fetch_neurosynth(
        path=tmpdir,
        version="7",
        overwrite=False,
        source="abstract",
        vocab="terms",
    )
    files = glob(os.path.join(tmpdir, "*"))
    assert len(files) == 4


def test_fetch_neuroquery(tmp_path_factory):
    """Smoke test for extract.fetch_neuroquery."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neuroquery")
    nimare.extract.fetch_neuroquery(
        path=tmpdir,
        version="1",
        overwrite=False,
        source="abstract",
        vocab="neuroquery7547",
        type="count",
    )
    files = glob(os.path.join(tmpdir, "*"))
    assert len(files) == 4
