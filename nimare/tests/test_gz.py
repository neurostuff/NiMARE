import json
import gzip
from pathlib import Path

import pytest

from nimare.dataset import Dataset
from nimare.nimads import Studyset


def write_json_gz(path, data):
    """Helper: write json or gz file."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


def test_gzip_loading(tmp_path):
    """Test that Dataset and Studyset can be loaded from .json.gz files."""

    # ---- Mock Dataset ----
    dataset_mock = {
        "study-01": {
            "contrasts": {
                "1": {
                    "coords": {
                        "x": [0],
                        "y": [0],
                        "z": [0],
                        "space": "mni152_2mm",
                    },
                    "metadata": {"sample_size": 20},
                }
            }
        }
    }

    # ---- Mock Studyset ----
    studyset_mock = {
        "id": "test_studyset",
        "studies": [
            {
                "id": "s1",
                "analyses": [
                    {
                        "id": "a1",
                        "coordinates": {
                            "x": [0],
                            "y": [0],
                            "z": [0],
                            "space": "mni152_2mm",
                        },
                    }
                ],
            }
        ],
    }

    # ---- Create temporary files ----
    ds_json = tmp_path / "dataset.json"
    ds_gz = tmp_path / "dataset.json.gz"
    ss_json = tmp_path / "studyset.json"
    ss_gz = tmp_path / "studyset.json.gz"

    write_json_gz(ds_json, dataset_mock)
    write_json_gz(ds_gz, dataset_mock)

    write_json_gz(ss_json, studyset_mock)
    write_json_gz(ss_gz, studyset_mock)

    # ---- Test Dataset loading ----
    ds_normal = Dataset(str(ds_json))
    ds_compressed = Dataset(str(ds_gz))

    assert ds_normal.ids == ds_compressed.ids

    # ---- Test Studyset loading ----
    ss_normal = Studyset(str(ss_json))
    ss_compressed = Studyset(str(ss_gz))

    assert ss_normal is not None
    assert ss_compressed is not None