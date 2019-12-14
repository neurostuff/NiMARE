"""
Dataset and trained model downloading functions
"""
import warnings

from .extract import download_nidm_pain, download_mallet

__all__ = ['download_nidm_pain', 'download_mallet']

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
