"""NIMADS-related classes for NiMARE.

The studyset implementation lives in :mod:`nimare.studyset`. This module keeps
the historical import path.
"""

from nimare.studyset.blocks import (
    Comparison,
    CoordinateBlock,
    DesignBlock,
    ImageBlock,
    LabelBlock,
    TextBlock,
)
from nimare.studyset.columns import AnnotationSet
from nimare.studyset.requirements import Coordinates, Images, Labels, PerAnalysis, Texts
from nimare.studyset.io import convert_neurostore_json_to_parquet
from nimare.studyset.studyset import Studyset
from nimare.studyset.view import Context, View

__all__ = [
    "AnnotationSet",
    "Comparison",
    "Context",
    "CoordinateBlock",
    "Coordinates",
    "DesignBlock",
    "ImageBlock",
    "Images",
    "LabelBlock",
    "Labels",
    "PerAnalysis",
    "Studyset",
    "convert_neurostore_json_to_parquet",
    "TextBlock",
    "Texts",
    "View",
]
