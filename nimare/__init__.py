"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from .dataset import Dataset
    from .meta import cbma
    from .annotate import text
    from .decode import gclda_decode_roi
    from .parcellate import CoordCBP
    from .version import __version__

    del cbma, Dataset, text, gclda_decode_roi, CoordCBP
