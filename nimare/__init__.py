"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from .version import __version__
    from .dataset import Dataset


__all__ = [
    "Dataset"
]
