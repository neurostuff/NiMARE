"""Machine learning items for masked activation (MA) analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from nimare.base import NiMAREBase

LGR = logging.getLogger(__name__)

__all__ = ["MAFeatureDataset", "MAFeatureExtractor", "make_map_reducer"]


class MAFeatureDataset(NiMAREBase):
    """Placeholder for masked activation feature datasets."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("MAFeatureDataset is not yet implemented.")


class MAFeatureExtractor(NiMAREBase):
    """Placeholder for masked activation feature extraction."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("MAFeatureExtractor is not yet implemented.")


def make_map_reducer(*args: Any, **kwargs: Any):
    """Placeholder for map reducer construction."""

    raise NotImplementedError("make_map_reducer is not yet implemented.")
