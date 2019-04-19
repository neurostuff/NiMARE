"""
Common meta-analytic workflows
"""

from .ale import ale_sleuth_workflow
from .ibma_perm import con_perm_workflow
from .macm import macm_workflow
from .peaks2maps import peaks2maps_workflow

__all__ = ['ale_sleuth_workflow', 'con_perm_workflow', 'macm_workflow',
           'peaks2maps_workflow']
