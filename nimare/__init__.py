"""NiMARE: Neuroimaging Meta-Analysis Research Environment."""
import logging
import warnings

from ._version import get_versions

logging.basicConfig(level=logging.INFO)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from . import (
        annotate,
        base,
        correct,
        dataset,
        decode,
        io,
        meta,
        resources,
        stats,
        utils,
        workflows,
    )

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
