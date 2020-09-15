"""
Dataset and trained model downloading functions
"""
import warnings

from .extract import (
    download_nidm_pain,
    download_mallet,
    download_cognitive_atlas,
    download_abstracts,
    download_peaks2maps_model,
    fetch_neurosynth,
)
from . import utils

__all__ = [
    "download_nidm_pain",
    "download_mallet",
    "download_cognitive_atlas",
    "download_abstracts",
    "download_peaks2maps_model",
    "fetch_neurosynth",
    "utils",
]

warnings.simplefilter("default")

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning,
)
