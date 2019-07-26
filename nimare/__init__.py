"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
import warnings
import logging

logging.basicConfig(level=logging.INFO)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from . import base
    from . import dataset
    from . import meta
    from . import correct
    from . import resources
    from . import io
    from . import stats
    from . import utils
    from .version import __version__

    __all__ = ['base', 'dataset', 'meta', 'correct', 'resources', 'io',
               'stats', 'utils', '__version__']
