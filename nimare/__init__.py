"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
import warnings
import logging

from ._version import get_versions

logging.basicConfig(level=logging.INFO)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from . import base
    from . import dataset
    from . import meta
    from . import correct
    from . import annotate
    from . import decode
    from . import resources
    from . import io
    from . import stats
    from . import utils
    from . import workflows

    __version__ = get_versions()["version"]

    __all__ = [
        "base",
        "dataset",
        "meta",
        "correct",
        "annotate",
        "decode",
        "resources",
        "io",
        "stats",
        "utils",
        "workflows",
        "__version__",
    ]

del get_versions
