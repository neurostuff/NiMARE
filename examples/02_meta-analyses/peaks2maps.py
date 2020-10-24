# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas2:

========================================================
 Generate modeled activation maps with peaks2maps
========================================================

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os

from nilearn.plotting import plot_glass_brain

import nimare
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
dset = nimare.dataset.Dataset(dset_file)

###############################################################################
# Run peaks2maps
# --------------------------------------------------
k = nimare.meta.kernel.Peaks2MapsKernel()
imgs = k.transform(dset, return_type="image")

###############################################################################
# Plot modeled activation maps
# --------------------------------------------------
for img in imgs:
    display = plot_glass_brain(
        img, display_mode="lyrz", plot_abs=False, colorbar=True, vmax=1, threshold=0
    )
