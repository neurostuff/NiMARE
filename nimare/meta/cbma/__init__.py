"""
Top-level namespace for meta-analyses.
"""

from .ale import ALE, SCALE
from .mkda import MKDAChi2, MKDADensity, KDA
from .model import BHICP, HPGRF, SBLFR, SBR
from .kernel import ALEKernel, MKDAKernel, KDAKernel, Peaks2MapsKernel
