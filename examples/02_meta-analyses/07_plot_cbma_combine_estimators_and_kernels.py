# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas_estimators_and_kernels:

===================================
Combine CBMA kernels and estimators
===================================

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

from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.meta import ALE, KDA, MKDAChi2, MKDADensity
from nimare.meta.kernel import ALEKernel, KDAKernel, MKDAKernel
from nimare.utils import get_resource_path

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

mask_img = dset.masker.mask_img

###############################################################################
# List possible kernel transformers
# --------------------------------------------------
kernel_transformers = {
    "MKDA kernel": MKDAKernel,
    "KDA kernel": KDAKernel,
    "ALE kernel": ALEKernel,
}

###############################################################################
# MKDA density analysis
# --------------------------------------------------
for kt_name, kt in kernel_transformers.items():
    try:
        mkda = MKDADensity(kernel_transformer=kt, null_method="approximate")
        mkda.fit(dset)
        corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
        cres = corr.transform(mkda.results)
        plot_stat_map(
            cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo"),
            cut_coords=[0, 0, -8],
            draw_cross=False,
            cmap="RdBu_r",
            title="MKDA estimator with %s" % kt_name,
        )

    except AttributeError:
        print(
            "\nError: the %s does not currently work with the MKDA meta-analysis method\n"
            % kt_name
        )

###############################################################################
# MKDA Chi2
# --------------------------------------------------
for kt_name, kt in kernel_transformers.items():
    try:
        mkda = MKDAChi2(kernel_transformer=kt)
        dset1 = dset.slice(dset.ids)
        dset2 = dset.slice(dset.ids)
        mkda.fit(dset1, dset2)
        corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
        cres = corr.transform(mkda.results)
        plot_stat_map(
            cres.get_map("z_desc-consistency_level-voxel_corr-FWE_method-montecarlo"),
            threshold=1.65,
            cut_coords=[0, 0, -8],
            draw_cross=False,
            cmap="RdBu_r",
            title="MKDA Chi2 estimator with %s" % kt_name,
        )

    except AttributeError:
        print(
            "\nError: the %s does not currently work with the MKDA Chi2 meta-analysis method\n"
            % kt_name
        )

###############################################################################
# KDA
# --------------------------------------------------
for kt_name, kt in kernel_transformers.items():
    try:
        kda = KDA(kernel_transformer=kt, null_method="approximate")
        kda.fit(dset)
        corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
        cres = corr.transform(kda.results)
        plot_stat_map(
            cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo"),
            cut_coords=[0, 0, -8],
            draw_cross=False,
            cmap="RdBu_r",
            title="KDA estimator with %s" % kt_name,
        )

    except AttributeError:
        print(
            "\nError: the %s does not currently work with the KDA meta-analysis method\n" % kt_name
        )

###############################################################################
# ALE
# --------------------------------------------------
for kt_name, kt in kernel_transformers.items():
    try:
        ale = ALE(kernel_transformer=kt, null_method="approximate")
        ale.fit(dset)
        corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
        cres = corr.transform(ale.results)
        plot_stat_map(
            cres.get_map("logp_desc-size_level-cluster_corr-FWE_method-montecarlo"),
            cut_coords=[0, 0, -8],
            draw_cross=False,
            cmap="RdBu_r",
            title="ALE estimator with %s" % kt_name,
        )

    except AttributeError:
        print(
            "\nError: the %s does not currently work with the ALE meta-analysis method\n" % kt_name
        )
