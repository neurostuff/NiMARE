# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas3:

========================================================
 Run image-based meta-analyses on 21 pain studies
========================================================

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.

..note::
    This will likely change as we work to shift database querying to a remote
    database, rather than handling it locally with NiMARE.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os
import urllib.request

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask, unmask
from nilearn.plotting import plot_stat_map

import nimare
from nimare.meta.esma import fishers
from nimare.meta.ibma import (Fishers, Stouffers, WeightedStouffers,
                              RFX_GLM, FFX_GLM, ffx_glm)

###############################################################################
# Download data
# --------------------------------
url = "https://raw.githubusercontent.com/tsalo/NiMARE/coco2019/download_test_data.py"
u = urllib.request.urlopen(url)
data = u.read()
u.close()

# write python to file
with open("download_test_data.py", "wb") as f:
    f.write(data)

# download the requisite data
from download_test_data import download_dataset
dset_dir = download_dataset()
os.remove("download_test_data.py")

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(os.path.dirname(nimare.__file__),
                         'tests', 'data', 'nidm_pain_dset.json')
dset = nimare.dataset.Dataset(dset_file)
dset.update_path(dset_dir)

mask_img = dset.masker.mask_img

logp_thresh = -np.log(.05)

###############################################################################
# Fisher's (using functions)
# --------------------------------------------------
# Get images for analysis
files = dset.get_images(imtype='z')
files = [f for f in files if f]
z_imgs = [nib.load(f) for f in files]
z_data = apply_mask(z_imgs, mask_img)
print('{0} studies found.'.format(z_data.shape[0]))

result = fishers(z_data, mask_img)
fishers_result = unmask(result['z'], mask_img)
plot_stat_map(fishers_result, cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Fisher's (using Estimators)
# --------------------------------------------------
# Here is the object-oriented approach
meta = Fishers()
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Stouffer's with fixed-effects inference
# --------------------------------------------------
meta = Stouffers(inference='ffx', null='theoretical', n_iters=None)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Stouffer's with random-effects inference using theoretical null distribution
# -----------------------------------------------------------------------------
meta = Stouffers(inference='rfx', null='theoretical', n_iters=None)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Stouffer's with random-effects inference using empirical null distribution
# -----------------------------------------------------------------------------
meta = Stouffers(inference='rfx', null='empirical', n_iters=1000)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Weighted Stouffer's
# -------------------
meta = WeightedStouffers()
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# RFX GLM with theoretical null distribution
# ------------------------------------------
meta = RFX_GLM(null='theoretical', n_iters=None)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# RFX GLM with empirical null distribution
# ------------------------------------------
meta = RFX_GLM(null='empirical', n_iters=1000)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')
