"""NiMARE: Neuroimaging Meta-Analysis Research Environment."""
import logging
import sys
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


def _py367_deprecation_warning():
    """Deprecation warnings message.

    Notes
    -----
    Adapted from Nilearn.
    """
    py36_warning = (
        "Python 3.6 and 3.7 support is deprecated and will be removed in release 0.1.0 of NiMARE. "
        "Consider switching to Python 3.8, 3.9 or 3.10."
    )
    warnings.filterwarnings("once", message=py36_warning)
    warnings.warn(message=py36_warning, category=FutureWarning, stacklevel=3)


def _python_deprecation_warnings():
    """Raise deprecation warnings.

    Notes
    -----
    Adapted from Nilearn.
    """
    if sys.version_info.major == 3 and (
        sys.version_info.minor == 6 or sys.version_info.minor == 7
    ):
        _py367_deprecation_warning()


_python_deprecation_warnings()
