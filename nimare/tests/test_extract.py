"""Test nimare.extract."""
import os
from glob import glob

import nimare


def test_fetch_neurosynth(tmp_path_factory):
    """Smoke test for extract.fetch_neurosynth.

    Taken from the Neurosynth Python package.
    """
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth")
    url = (
        "https://raw.githubusercontent.com/neurosynth/neurosynth/master/neurosynth/tests/"
        "data/test_data.tar.gz"
    )
    nimare.extract.fetch_neurosynth(tmpdir, url=url, unpack=True)
    files = glob(os.path.join(tmpdir, "*.txt"))
    assert len(files) == 2
