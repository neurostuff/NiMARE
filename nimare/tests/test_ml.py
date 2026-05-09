"""Tests for the nimare.ml module."""

from __future__ import annotations

import pytest

from nimare.dataset import Dataset
from nimare.ml import MAFeatureDataset, MAFeatureExtractor, make_map_reducer
from nimare.nimads import Studyset


def _build_shared_ml_source():
    """Create one canonical source dict for both Dataset and Studyset builders.

    The two studies intentionally reuse the same contrast ID (``task``) so that
    short analysis IDs are ambiguous and must be resolved via the full
    ``<study_id>-<contrast_id>`` identifier.
    """
    return {
        "study_alpha": {
            "contrasts": {
                "task": {
                    "coords": {
                        "space": "MNI",
                        "x": [1.0, 2.0],
                        "y": [3.0, 4.0],
                        "z": [5.0, 6.0],
                    },
                    "metadata": {"sample_sizes": [20]},
                    "labels": {"alpha_label": 1.0},
                    "text": {"abstract": "Alpha study abstract."},
                }
            }
        },
        "study_beta": {
            "contrasts": {
                "task": {
                    "coords": {
                        "space": "MNI",
                        "x": [-1.0],
                        "y": [-3.0],
                        "z": [-5.0],
                    },
                    "metadata": {"sample_sizes": [30]},
                    "labels": {"beta_label": 1.0},
                    "text": {"abstract": "Beta study abstract."},
                }
            }
        },
    }


def build_shared_dataset():
    """Build a fresh Dataset from the shared source dict."""
    return Dataset(_build_shared_ml_source())


def build_shared_studyset():
    """Build a fresh Studyset from the shared Dataset fixture source."""
    return Studyset.from_dataset(build_shared_dataset())


def test_shared_dataset_builder():
    """The shared Dataset builder should expose the expected tabular fields."""
    dset = build_shared_dataset()

    assert isinstance(dset, Dataset)
    assert dset.ids.tolist() == ["study_alpha-task", "study_beta-task"]
    assert list(dset.coordinates["id"].unique()) == ["study_alpha-task", "study_beta-task"]
    assert "sample_sizes" in dset.metadata.columns
    assert "alpha_label" in dset.annotations.columns
    assert "beta_label" in dset.annotations.columns
    assert "abstract" in dset.texts.columns


def test_shared_studyset_builder_and_ambiguous_short_ids():
    """The shared Studyset builder should preserve full IDs and expose ambiguity."""
    studyset = build_shared_studyset()

    assert isinstance(studyset, Studyset)
    assert studyset.ids.tolist() == ["study_alpha-task", "study_beta-task"]
    assert studyset.filter_ids("task").ids.tolist() == ["study_alpha-task", "study_beta-task"]
    assert "abstract" in studyset.texts.columns
    assert "sample_sizes" in studyset.metadata.columns
    assert "alpha_label" in studyset.annotations_df.columns
    assert "beta_label" in studyset.annotations_df.columns


def test_ma_feature_dataset_initialization():
    """Test that MAFeatureDataset currently raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        MAFeatureDataset()


def test_ma_feature_extractor_initialization():
    """Test that MAFeatureExtractor currently raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        MAFeatureExtractor()


def test_make_map_reducer_placeholder():
    """Test that make_map_reducer currently raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        make_map_reducer()
