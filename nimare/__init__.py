"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from . import dataset
    from . import meta
    from . import resources
    from .version import __version__
