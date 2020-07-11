# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas4:

========================================================
 Run image-based meta-analyses on 21 pain studies
========================================================

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.

.. caution::
    Dataset querying will likely change as we work to shift database querying
    to a remote database, rather than handling it locally with NiMARE.

"""
import os

import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask, unmask
from nilearn.plotting import plot_stat_map

import nimare
from nimare.tests.utils import get_test_data_path
from nimare.meta import ibma

###############################################################################
# Download data
# --------------------------------
dset_dir = nimare.extract.download_nidm_pain()

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_test_data_path(), 'nidm_pain_dset.json')
dset = nimare.dataset.Dataset(dset_file)
dset.update_path(dset_dir)
# Calculate missing images
dset.images = nimare.transforms.transform_images(
    dset.images, target='z',
    masker=dset.masker, metadata_df=dset.metadata
)
dset.images = nimare.transforms.transform_images(
    dset.images, target='varcope',
    masker=dset.masker, metadata_df=dset.metadata
)

###############################################################################
# Fisher's
# --------------------------------------------------
meta = ibma.Fishers()
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Stouffer's
# --------------------------------------------------
meta = ibma.Stouffers(use_sample_size=False)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Stouffer's with weighting by sample size
# -----------------------------------------------------------------------------
meta = ibma.Stouffers(use_sample_size=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# RFX GLM with theoretical null distribution
# ------------------------------------------
meta = ibma.RandomEffectsGLM(null='theoretical', n_iters=None)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# RFX GLM with empirical null distribution
# ------------------------------------------
meta = ibma.RandomEffectsGLM(null='empirical', n_iters=100)
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Weighted Least Squares
# ------------------------------------------
meta = ibma.WeightedLeastSquares(method='reml')
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# DerSimonian-Laird
# ------------------------------------------
meta = ibma.Something(method='DerSimonianLaird')
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')

###############################################################################
# Hedge's
# ------------------------------------------
meta = ibma.Something(method='Hedges')
meta.fit(dset)
plot_stat_map(meta.results.get_map('z'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')
