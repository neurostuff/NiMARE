"""Common meta-analytic workflows."""

from .ale import ale_sleuth_workflow
from .conperm import conperm_workflow
from .macm import macm_workflow
from .peaks2maps import peaks2maps_workflow
from .scale import scale_workflow
from .neurosynth_compose import run as compose_run

__all__ = [
    "ale_sleuth_workflow",
    "conperm_workflow",
    "macm_workflow",
    "peaks2maps_workflow",
    "scale_workflow",
    "compose_run",
]
