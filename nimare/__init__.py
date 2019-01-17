"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from .dataset import Dataset
    from .meta import cbma
    from .version import __version__

    del cbma, Dataset
