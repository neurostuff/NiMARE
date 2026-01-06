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
        reports,
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
        "reports",
        "workflows",
        "__version__",
    ]

del get_versions

try:
    import nilearn
    from packaging.version import Version

    nilearn_version = Version(nilearn.__version__)
    if nilearn_version < Version("0.12.0") or nilearn_version >= Version("0.14"):
        warnings.warn(
            "NiMARE supports nilearn>=0.12.0,<0.14. " f"Detected nilearn {nilearn.__version__}.",
            UserWarning,
        )
except Exception:
    pass
