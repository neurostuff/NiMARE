"""

.. _metas_cbma:

=========================================
Coordinate-based meta-analysis algorithms
=========================================

A tour of CBMA algorithms in NiMARE.

This tutorial is intended to provide a brief description and example of each of
the CBMA algorithms implemented in NiMARE.
For a more detailed introduction to the elements of a coordinate-based
meta-analysis, see other stuff.
"""
###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
# .. note::
#   The data used in this example come from a collection of NIDM-Results packs
#   downloaded from Neurovault collection 1425, uploaded by Dr. Camille Maumet.
#
#   Creation of the Dataset from the NIDM-Results packs was done with custom
#   code. The Results packs for collection 1425 are not completely
#   NIDM-Results-compliant, so the nidmresults library could not be used to
#   facilitate data extraction.
import os

from nilearn.plotting import plot_stat_map

from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.utils import get_resource_path

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

# Some of the CBMA algorithms compare two Datasets,
# so we'll split this example Dataset in half.
dset1 = dset.slice(dset.ids[:10])
dset2 = dset.slice(dset.ids[10:])

###############################################################################
# Multilevel Kernel Density Analysis
# -----------------------------------------------------------------------------
from nimare.meta.cbma.mkda import MKDADensity

meta = MKDADensity()
results = meta.fit(dset)

corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(results)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)
plot_stat_map(
    cres.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)

###############################################################################
# MKDA Chi-Squared
# -----------------------------------------------------------------------------
from nimare.meta.cbma.mkda import MKDAChi2

meta = MKDAChi2(kernel__r=10)
results = meta.fit(dset1, dset2)

corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(results)

plot_stat_map(
    results.get_map("z_desc-consistency"),
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)
plot_stat_map(
    cres.get_map("z_desc-consistencySize_level-cluster_corr-FWE_method-montecarlo"),
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)

###############################################################################
# Kernel Density Analysis
# -----------------------------------------------------------------------------
from nimare.meta.cbma.mkda import KDA

meta = KDA()
results = meta.fit(dset)

corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(results)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)
plot_stat_map(
    cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)

###############################################################################
# Activation Likelihood Estimation
# -----------------------------------------------------------------------------
from nimare.meta.cbma.ale import ALE

meta = ALE()
results = meta.fit(dset)

corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
cres = corr.transform(results)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)
plot_stat_map(
    cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)

###############################################################################
# Specific Co-Activation Likelihood Estimation
# -----------------------------------------------------------------------------
#
# .. important::
#
#   The SCALE algorithm is very memory intensive, so we don't run it within the
#   documentation.
#
# .. code-block:: python
#
#   import numpy as np
#
#   from nimare.meta.cbma.ale import SCALE
#   from nimare.utils import vox2mm
#
#   xyz = vox2mm(
#       np.vstack(np.where(dset.masker.mask_img.get_fdata())).T,
#       dset.masker.mask_img.affine,
#   )
#
#   meta = SCALE(xyz=xyz, n_iters=10)
#   results = meta.fit(dset)

###############################################################################
# ALE-Based Subtraction Analysis
# -----------------------------------------------------------------------------
from nimare.meta.cbma.ale import ALESubtraction

meta = ALESubtraction(n_iters=10, n_cores=1)
results = meta.fit(dset1, dset2)

plot_stat_map(
    results.get_map("z_desc-group1MinusGroup2"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.1,
)
