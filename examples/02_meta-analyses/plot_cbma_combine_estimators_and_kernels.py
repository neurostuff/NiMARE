# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas3:

========================================================
 Test combinations of kernels and estimators for coordinate-based meta-analyses.
========================================================

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.

.. note::
    Creation of the Dataset from the NIDM-Results packs was done with custom
    code. The Results packs for collection 1425 are not completely
    NIDM-Results-compliant, so the nidmresults library could not be used to
    facilitate data extraction.
"""
import os

import numpy as np
from nilearn.plotting import plot_stat_map

import nimare
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
dset = nimare.dataset.Dataset(dset_file)

mask_img = dset.masker.mask_img


###############################################################################
# List possible kernel transformers
# --------------------------------------------------
kernel_transformers = {
    "MKDAKernel": nimare.meta.kernel.MKDAKernel,
    "KDAKernel": nimare.meta.kernel.KDAKernel,
    "ALEKernel": nimare.meta.kernel.ALEKernel,
}


###############################################################################
# MKDA density analysis
# --------------------------------------------------

for kt_name, kt in kernel_transformers.items():
    mkda = nimare.meta.mkda.MKDADensity(kernel_transformer=kt)
    mkda.fit(dset)
    corr = nimare.correct.FWECorrector(
        method="montecarlo", n_iters=10, n_cores=1
    )
    cres = corr.transform(mkda.results)
    plot_stat_map(
        cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo"),
        cut_coords=[0, 0, -8],
        draw_cross=False,
        cmap="RdBu_r",
        title="MKDA estimator with %s" % kt_name,
    )

###############################################################################
# MKDA Chi2
# --------------------------------------------------
for kt_name, kt in kernel_transformers.items():
    mkda = nimare.meta.mkda.MKDAChi2(kernel_transformer=kt)
    dset1 = dset.slice(dset.ids)
    dset2 = dset.slice(dset.ids)
    mkda.fit(dset1, dset2)
    corr = nimare.correct.FWECorrector(
        method="montecarlo", n_iters=10, n_cores=1
    )
    cres = corr.transform(mkda.results)
    plot_stat_map(
        cres.get_map(
            "z_desc-consistency_level-voxel_corr-FWE_method-montecarlo"
        ),
        threshold=1.65,
        cut_coords=[0, 0, -8],
        draw_cross=False,
        cmap="RdBu_r",
        title="MKDA Chi2 estimator with %s" % kt_name,
    )

###############################################################################
# KDA
# --------------------------------------------------
for kt_name, kt in kernel_transformers.items():
    kda = nimare.meta.mkda.KDA(kernel_transformer=kt)
    kda.fit(dset)
    corr = nimare.correct.FWECorrector(
        method="montecarlo", n_iters=10, n_cores=1
    )
    cres = corr.transform(kda.results)
    plot_stat_map(
        cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo"),
        cut_coords=[0, 0, -8],
        draw_cross=False,
        cmap="RdBu_r",
        title="KDA estimator with %s" % kt_name,
    )

###############################################################################
# ALE
# --------------------------------------------------
for kt_name, kt in kernel_transformers.items():
    ale = nimare.meta.ale.ALE(kernel_transformer=kt)
    ale.fit(dset)
    corr = nimare.correct.FWECorrector(
        method="montecarlo", n_iters=10, n_cores=1
    )
    cres = corr.transform(ale.results)
    plot_stat_map(
        cres.get_map("logp_level-cluster_corr-FWE_method-montecarlo"),
        cut_coords=[0, 0, -8],
        draw_cross=False,
        cmap="RdBu_r",
        title="ALE estimator with %s" % kt_name,
    )
