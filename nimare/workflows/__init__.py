"""Common meta-analytic workflows."""

from .ale import ale_sleuth_workflow
from .base import Workflow
from .cbma import CBMAWorkflow, PairwiseCBMAWorkflow
from .macm import macm_workflow

__all__ = [
    "ale_sleuth_workflow",
    "Workflow",
    "CBMAWorkflow",
    "PairwiseCBMAWorkflow",
    "macm_workflow",
]
