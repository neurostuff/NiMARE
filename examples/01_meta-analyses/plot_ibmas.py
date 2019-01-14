"""

.. _meta2:

=========================================
 Perform image-based meta-analyses
=========================================

Perform image-based meta-analyses on 21 pain studies.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import sys
import json
import os.path as op

import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.plotting import plot_stat_map

import nimare
from nimare.meta.ibma import (stouffers, fishers, weighted_stouffers,
                              rfx_glm, ffx_glm, mfx_glm)

# Load local functions for querying NIDM Results packs
sys.path.append('../utils/')
from nidm_utils import get_files

###############################################################################
# Load data
# --------------------------------
dset_file = op.join(nimare.utils.get_resource_path(),
                    'data/nidm_pain_dset_with_subpeaks.json')
with open(dset_file, 'r') as fo:
    dset_dict = json.load(fo)
db = nimare.dataset.Database(dset_file)
dset = db.get_dataset()
mask_img = dset.mask
cut_coords = [0, 0, 0]

###############################################################################
# Get contrast files, standard error files, and sample sizes
# -----------------------------------------------------------------
con_files, se_files, ns = get_files(dset_dict, ['con', 'se', 'n'])
con_imgs = [nib.load(f) for f in con_files]
se_imgs = [nib.load(f) for f in se_files]
con_data = apply_mask(con_imgs, mask_img)
se_data = apply_mask(se_imgs, mask_img)
sample_sizes = np.array(ns)

###############################################################################
# MFX GLM
# -----------------------------------
mfx_results = mfx_glm(con_data, se_data, sample_sizes, mask_img)
plot_stat_map(mfx_results.images['thresh_z'], cut_coords=cut_coords,
              draw_cross=False, cmap='RdBu_r',
              title='MFX GLM')

###############################################################################
# Contrast Permutation
# -----------------------------------
conperm_results = rfx_glm(con_data, mask_img, null='empirical', n_iters=10,
                          corr='FDR')
plot_stat_map(conperm_results.images['z'], cut_coords=cut_coords,
              threshold=1.96, draw_cross=False, cmap='RdBu_r',
              title='Contrast Permutation')

###############################################################################
# FFX GLM
# -----------------------------------
ffx_results = ffx_glm(con_data, se_data, sample_sizes, mask_img)
plot_stat_map(ffx_results.images['thresh_z'], cut_coords=cut_coords,
              draw_cross=False, cmap='RdBu_r',
              title='FFX GLM')

###############################################################################
# RFX GLM
# -----------------------------------
rfx_results = rfx_glm(con_data, mask_img, null='theoretical', n_iters=None)
plot_stat_map(rfx_results.images['z'], cut_coords=cut_coords,
              threshold=1.96, draw_cross=False, cmap='RdBu_r',
              title='RFX GLM')

###############################################################################
# Get z-value files and sample sizes
# -----------------------------------
z_files, ns = get_files(dset_dict, ['z', 'n'])
z_imgs = [nib.load(f) for f in z_files]
z_data = apply_mask(z_imgs, mask_img)

# T maps to be converted to z
t_files, t_ns = get_files(dset_dict, ['t!z', 'n'])
t_imgs = [nib.load(f) for f in t_files]
t_data_list = [apply_mask(t_img, mask_img) for t_img in t_imgs]
tz_data_list = [nimare.utils.t_to_z(t_data, t_ns[i]-1) for i, t_data
                in enumerate(t_data_list)]
tz_data = np.vstack(tz_data_list)

# Combine
z_data = np.vstack((z_data, tz_data))
ns = np.concatenate((ns, t_ns))
sample_sizes = np.array(ns)

###############################################################################
# Z Permutation
# -----------------------------------
zperm_results = stouffers(z_data, mask_img, inference='rfx', null='empirical',
                          n_iters=10, corr='FDR')
plot_stat_map(zperm_results.images['z'], cut_coords=cut_coords,
              threshold=1.96, draw_cross=False, cmap='RdBu_r',
              title='Z Permutation')

###############################################################################
# Fisher's
# -----------------------------------
fishers_results = fishers(z_data, mask_img)
plot_stat_map(fishers_results.images['z'], cut_coords=cut_coords,
              threshold=1.96, draw_cross=False, cmap='RdBu_r',
              title="Fisher's")

###############################################################################
# Stouffer's
# -----------------------------------
stouffers_results = stouffers(z_data, mask_img, inference='ffx',
                              null='theoretical', n_iters=None)
plot_stat_map(stouffers_results.images['z'], cut_coords=cut_coords,
              threshold=1.96, draw_cross=False, cmap='RdBu_r',
              title="Stouffer's")

###############################################################################
# Stouffers RFX
# -----------------------------------
stouffrfx_results = stouffers(z_data, mask_img, inference='rfx',
                              null='theoretical', n_iters=None)
plot_stat_map(stouffrfx_results.images['z'], cut_coords=cut_coords,
              threshold=1.96, draw_cross=False, cmap='RdBu_r',
              title='Z MFX')

###############################################################################
# Weighted Stouffer's
# -----------------------------------
wstouff_results = weighted_stouffers(z_data, sample_sizes, mask_img)
plot_stat_map(wstouff_results.images['z'], cut_coords=cut_coords,
              threshold=1.96, draw_cross=False, cmap='RdBu_r',
              title="Weighted Stouffer's")
