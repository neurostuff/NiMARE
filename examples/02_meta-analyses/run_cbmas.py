# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas3:

========================================================
 Run coordinate-based meta-analyses on 21 pain studies
========================================================

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.

.. note::
    This will likely change as we work to shift database querying to a remote
    database, rather than handling it locally with NiMARE.

"""
import os

import numpy as np
from nilearn.plotting import plot_stat_map

import nimare
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_test_data_path(), 'nidm_pain_dset.json')
dset = nimare.dataset.Dataset(dset_file)

mask_img = dset.masker.mask_img

###############################################################################
# MKDA density analysis
# --------------------------------------------------
mkda = nimare.meta.cbma.mkda.MKDADensity(kernel__r=10)
mkda.fit(dset)
corr = nimare.correct.FWECorrector(method='permutation', n_iters=10, n_cores=1)
cres = corr.transform(mkda.results)
plot_stat_map(cres.get_map('logp_level-voxel_corr-FWE_method-permutation'),
              cut_coords=[0, 0, -8], draw_cross=False, cmap='RdBu_r')

###############################################################################
# MKDA Chi2 with FDR correction
# --------------------------------------------------
mkda = nimare.meta.cbma.mkda.MKDAChi2(kernel__r=10)
dset1 = dset.slice(dset.ids)
dset2 = dset.slice(dset.ids)
mkda.fit(dset1, dset2)
corr = nimare.correct.FDRCorrector(method='fdr_bh', alpha=0.001)
cres = corr.transform(mkda.results)
plot_stat_map(cres.get_map('consistency_z_FDR_corr-FDR_method-fdr_bh'),
              threshold=1.65, cut_coords=[0, 0, -8], draw_cross=False,
              cmap='RdBu_r')

###############################################################################
# MKDA Chi2 with FWE correction
# --------------------------------------------------
corr = nimare.correct.FWECorrector(method='permutation', n_iters=10, n_cores=1)
cres = corr.transform(mkda.results)
plot_stat_map(cres.get_map('consistency_z'), threshold=1.65,
              cut_coords=[0, 0, -8], draw_cross=False, cmap='RdBu_r')

###############################################################################
# KDA
# --------------------------------------------------
kda = nimare.meta.cbma.mkda.KDA(kernel__r=10)
kda.fit(dset)
corr = nimare.correct.FWECorrector(method='permutation', n_iters=10, n_cores=1)
cres = corr.transform(kda.results)
plot_stat_map(cres.get_map('logp_level-voxel_corr-FWE_method-permutation'),
              cut_coords=[0, 0, -8], draw_cross=False, cmap='RdBu_r')

###############################################################################
# ALE
# --------------------------------------------------
ale = nimare.meta.cbma.ale.ALE()
ale.fit(dset)
corr = nimare.correct.FWECorrector(method='permutation', n_iters=10, n_cores=1)
cres = corr.transform(ale.results)
plot_stat_map(cres.get_map('logp_level-cluster_corr-FWE_method-permutation'),
              cut_coords=[0, 0, -8], draw_cross=False, cmap='RdBu_r')

###############################################################################
# SCALE
# --------------------------------------------------
ijk = np.vstack(np.where(mask_img.get_data())).T
scale = nimare.meta.cbma.ale.SCALE(ijk=ijk, n_iters=10, n_cores=1)
scale.fit(dset)
plot_stat_map(scale.results.get_map('z_vthresh'), cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r')
