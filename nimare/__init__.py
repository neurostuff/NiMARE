"""NiMARE: Neuroimaging Meta-Analysis Research Environment."""

import logging
import warnings

import lazy_loader as lazy
import numpy as np

from ._version import get_versions

logging.basicConfig(level=logging.INFO)

# Ask for float64's machine limits before anything can ask for long double's. NumPy caches
# finfo by dtype, and where long double is double-sized (Windows) the two dtypes compare
# equal, so whichever is requested first answers for both. nibabel requests long double on
# import; after that, np.finfo(float).eps is a long double scalar, and statsmodels' GLM,
# which reads it at import, promotes its arrays to a type numpy.linalg rejects
# (numpy/numpy#32947, since NumPy 2.4).
np.finfo(np.float64)

# Subpackages are imported on first attribute access, so ``import nimare`` stays cheap
# and ``from nimare.meta.cbma import ALE`` loads only what ALE needs. Set the
# environment variable EAGER_IMPORT=1 to import everything up front.
__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submodules=[
        "annotate",
        "base",
        "correct",
        "dataset",
        "decode",
        "diagnostics",
        "estimator",
        "exceptions",
        "extract",
        "generate",
        "io",
        "meta",
        "ml",
        "nimads",
        "reports",
        "resources",
        "results",
        "stats",
        "studyset",
        "transforms",
        "utils",
        "workflows",
    ],
)
del _

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "base",
    "dataset",
    "ml",
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

try:
    from importlib.metadata import version

    from packaging.version import Version

    nilearn_version = Version(version("nilearn"))
    if nilearn_version < Version("0.12.0") or nilearn_version >= Version("0.14"):
        warnings.warn(
            "NiMARE supports nilearn>=0.12.0,<0.14. " f"Detected nilearn {nilearn_version}.",
            UserWarning,
        )
except Exception:
    pass
