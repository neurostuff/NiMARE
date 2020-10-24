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

from nilearn.plotting import plot_stat_map

import nimare
from nimare.correct import FWECorrector
from nimare.meta import ibma
from nimare.tests.utils import get_test_data_path

###############################################################################
# Download data
# --------------------------------
dset_dir = nimare.extract.download_nidm_pain()

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
dset = nimare.dataset.Dataset(dset_file)
dset.update_path(dset_dir)
# Calculate missing images
dset.images = nimare.transforms.transform_images(
    dset.images, target="z", masker=dset.masker, metadata_df=dset.metadata
)
dset.images = nimare.transforms.transform_images(
    dset.images, target="varcope", masker=dset.masker, metadata_df=dset.metadata
)

###############################################################################
# Stouffer's
# --------------------------------------------------
meta = ibma.Stouffers(use_sample_size=False)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Stouffer's with weighting by sample size
# -----------------------------------------------------------------------------
meta = ibma.Stouffers(use_sample_size=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Fisher's
# -----------------------------------------------------------------------------
meta = ibma.Fishers()
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Permuted OLS
# -----------------------------------------------------------------------------
meta = ibma.PermutedOLS(two_sided=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Permuted OLS with FWE Correction
# -----------------------------------------------------------------------------
meta = ibma.PermutedOLS(two_sided=True)
meta.fit(dset)
corrector = FWECorrector(method="montecarlo", n_iters=100, n_cores=1)
cresult = corrector.transform(meta.results)
plot_stat_map(
    cresult.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
)

###############################################################################
# Weighted Least Squares
# -----------------------------------------------------------------------------
meta = ibma.WeightedLeastSquares(tau2=0)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# DerSimonian-Laird
# -----------------------------------------------------------------------------
meta = ibma.DerSimonianLaird()
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Hedges
# -----------------------------------------------------------------------------
meta = ibma.Hedges()
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")
