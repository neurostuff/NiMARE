"""
Top-level namespace for meta-analyses.
"""

from .ibma import (Stouffers, WeightedStouffers, Fishers, RFX_GLM, FFX_GLM,
                   MFX_GLM,
                   stouffers, weighted_stouffers, fishers, rfx_glm, ffx_glm,
                   mfx_glm)

__all__ = ['Stouffers', 'WeightedStouffers', 'Fishers', 'RFX_GLM', 'FFX_GLM',
           'MFX_GLM',
           'stouffers', 'weighted_stouffers', 'fishers', 'rfx_glm', 'ffx_glm',
           'mfx_glm']
