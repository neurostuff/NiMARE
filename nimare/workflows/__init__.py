"""
Common meta-analytic workflows
"""

from .ale import ale_sleuth_workflow
from .conperm import conperm_workflow
from .macm import macm_workflow
from .peaks2maps import peaks2maps_workflow

__all__ = ['ale_sleuth_workflow', 'conperm_workflow', 'macm_workflow',
           'peaks2maps_workflow']
