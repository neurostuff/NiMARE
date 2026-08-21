"""NIMADS-related classes for NiMARE.

The historical import path. Everything here is re-exported from
:mod:`nimare.studyset`, which is the canonical surface -- this module adds
nothing of its own, so the two cannot drift apart.
"""

from nimare.studyset import *  # noqa: F401,F403
from nimare.studyset import __all__ as _STUDYSET_ALL

__all__ = list(_STUDYSET_ALL)
