"""Tests for the nimare.ml module."""

from __future__ import annotations

import pytest

from nimare.ml import MAFeatureDataset, MAFeatureExtractor, make_map_reducer
from nimare.nimads import Studyset


def _build_studyset_ml_source():
    """Create one canonical source dict for Studyset/NIMADS builder.

    The two studies intentionally reuse the same contrast ID (``task``) so that
    short analysis IDs are ambiguous and must be resolved via the full
    ``<study_id>-<contrast_id>`` identifier.
    """
    return {
        "id": "studyset_ml_source",
        "name": "Studyset ML source",
        "studies": [
            {
                "id": "study_alpha",
                "name": "Study alpha",
                "analyses": [
                    {
                        "id": "task",
                        "name": "Alpha task",
                        "metadata": {"sample_sizes": [20]},
                        "annotations": {"alpha_label": 1.0},
                        "texts": {"abstract": "Alpha study abstract."},
                        "points": [
                            {"space": "MNI", "coordinates": [1.0, 3.0, 5.0]},
                            {"space": "MNI", "coordinates": [2.0, 4.0, 6.0]},
                        ],
                        "images": [],
                    }
                ],
            },
            {
                "id": "study_beta",
                "name": "Study beta",
                "analyses": [
                    {
                        "id": "task",
                        "name": "Beta task",
                        "metadata": {"sample_sizes": [30]},
                        "annotations": {"beta_label": 1.0},
                        "texts": {"abstract": "Beta study abstract."},
                        "points": [{"space": "MNI", "coordinates": [-1.0, -3.0, -5.0]}],
                        "images": [],
                    }
                ],
            },
        ],
    }


def build_studyset():
    """Build a fresh Studyset from the studyset ML source dict."""
    return Studyset(_build_studyset_ml_source())


def test_studyset_builder_and_ambiguous_short_ids():
    """The Studyset builder should preserve full IDs and expose ambiguity."""
    studyset = build_studyset()

    assert isinstance(studyset, Studyset)
    assert studyset.ids.tolist() == ["study_alpha-task", "study_beta-task"]
    assert studyset.study_ids.tolist() == ["study_alpha", "study_beta"]
    assert list(studyset.coordinates["id"].unique()) == ["study_alpha-task", "study_beta-task"]
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
