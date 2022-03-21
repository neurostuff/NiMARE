"""

.. _metas_peaks2maps:

================================
Generate MA maps with peaks2maps
================================

.. warning::
    peaks2maps has been deprecated within NiMARE and will be removed in version 0.0.13.
"""
###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
import os

from nilearn.plotting import plot_glass_brain

from nimare.dataset import Dataset
from nimare.meta.kernel import Peaks2MapsKernel
from nimare.utils import get_resource_path

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

###############################################################################
# Run peaks2maps
# -----------------------------------------------------------------------------
k = Peaks2MapsKernel()
imgs = k.transform(dset, return_type="image")

###############################################################################
# Plot modeled activation maps
# -----------------------------------------------------------------------------
for img in imgs:
    display = plot_glass_brain(
        img, display_mode="lyrz", plot_abs=False, colorbar=True, vmax=1, threshold=0
    )
