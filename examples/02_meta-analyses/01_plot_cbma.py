# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas_cbma:

==============================
Coordinate-based meta-analysis
==============================

Perform CBMAs on a toy dataset.

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.

.. note::
    Creation of the Dataset from the NIDM-Results packs was done with custom
    code. The Results packs for collection 1425 are not completely
    NIDM-Results-compliant, so the nidmresults library could not be used to
    facilitate data extraction.
"""
import os

from nilearn.plotting import plot_stat_map

from nimare.correct import FDRCorrector, FWECorrector
from nimare.dataset import Dataset
from nimare.meta import ALE, KDA, MKDAChi2, MKDADensity
from nimare.utils import get_resource_path

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

mask_img = dset.masker.mask_img

###############################################################################
# MKDA density analysis
# --------------------------------------------------
mkda = MKDADensity(kernel__r=10, null_method="approximate")
mkda.fit(dset)
corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(mkda.results)
plot_stat_map(
    cres.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.001,
)

###############################################################################
# MKDA Chi2 with FDR correction
# --------------------------------------------------
mkda = MKDAChi2(kernel__r=10)
dset1 = dset.slice(dset.ids)
dset2 = dset.slice(dset.ids)
mkda.fit(dset1, dset2)
corr = FDRCorrector(method="bh", alpha=0.001)
cres = corr.transform(mkda.results)
plot_stat_map(
    cres.get_map("z_desc-consistency_level-voxel_corr-FDR_method-bh"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.001,
)

###############################################################################
# MKDA Chi2 with FWE correction
# --------------------------------------------------
# Since we've already fitted the Estimator, we can just apply a new Corrector
# to the estimator.
corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(mkda.results)
plot_stat_map(
    cres.get_map("z_desc-consistencySize_level-cluster_corr-FWE_method-montecarlo"),
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.001,
)

###############################################################################
# KDA
# --------------------------------------------------
kda = KDA(kernel__r=10, null_method="approximate")
kda.fit(dset)
corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(kda.results)
plot_stat_map(
    cres.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.001,
)

###############################################################################
# ALE
# --------------------------------------------------
ale = ALE(null_method="approximate")
ale.fit(dset)
corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(ale.results)
plot_stat_map(
    cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.001,
)
