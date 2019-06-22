# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas2:

========================================================
 Run coordinate-based meta-analyses on 21 pain studies
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
import json
import numpy as np
from glob import glob
from os.path import basename, join, dirname, isfile
import urllib.request
import os

import pandas as pd
import nibabel as nib
from scipy.stats import t
from nilearn.masking import apply_mask
from nilearn.plotting import plot_stat_map

import nimare

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = join(dirname(nimare.__file__), 'tests', 'data', 'nidm_pain_dset.json')
with open(dset_file, 'r') as fo:
    dset_dict = json.load(fo)
dset = nimare.dataset.Dataset(dset_file)

mask_img = dset.mask

###############################################################################
# MKDA density analysis
# --------------------------------------------------
mkda = nimare.meta.cbma.MKDADensity(dset, kernel__r=10)
mkda.fit(n_iters=10, ids=dset.ids, n_cores=1)
plot_stat_map(mkda.results.images['logp_vfwe'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# MKDA Chi2 with FDR correction
# --------------------------------------------------
mkda = nimare.meta.cbma.MKDAChi2(dset, kernel__r=10)
mkda.fit(corr='FDR', ids=dset.ids, ids2=dset.ids, n_cores=1)
plot_stat_map(mkda.results.images['consistency_z'], threshold=1.65, cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# MKDA Chi2 with FWE correction
# --------------------------------------------------
mkda = nimare.meta.cbma.MKDAChi2(dset, kernel__r=10)
mkda.fit(corr='FWE', n_iters=10, ids=dset.ids, ids2=dset.ids, n_cores=1)
plot_stat_map(mkda.results.images['consistency_z'], threshold=1.65, cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# KDA
# --------------------------------------------------
kda = nimare.meta.cbma.KDA(dset, kernel__r=10)
kda.fit(n_iters=10, ids=dset.ids, n_cores=1)
plot_stat_map(kda.results.images['logp_vfwe'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# ALE
# --------------------------------------------------
ale = nimare.meta.cbma.ALE(dset, ids=dset.ids)
ale.fit(n_iters=10, ids=dset.ids, n_cores=1)
plot_stat_map(ale.results.images['logp_cfwe'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# SCALE
# --------------------------------------------------
ijk = np.vstack(np.where(dset.mask.get_data())).T
scale = nimare.meta.cbma.SCALE(dset, ijk=ijk, n_cores=1)
scale.fit(n_iters=10, ids=dset.ids)
plot_stat_map(scale.results.images['vthresh'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')
