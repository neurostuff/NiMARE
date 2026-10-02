"""Torch import guard shared by the CBMR modules.

CBMR is an optional extra, so every module that needs torch routes through here to raise one
consistent error instead of a bare ``ModuleNotFoundError``.
"""

try:
    import torch  # type: ignore[import-not-found]  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Torch is required to use `CBMR` classes. Install with `pip install 'nimare[cbmr]'`."
    ) from e
