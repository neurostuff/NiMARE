# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas_ibma:

=========================
Image-based meta-analysis
=========================

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.

.. caution::
    Dataset querying will likely change as we work to shift database querying
    to a remote database, rather than handling it locally with NiMARE.
"""
import os

from nilearn.plotting import plot_stat_map

from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.extract import download_nidm_pain
from nimare.meta import ibma
from nimare.transforms import ImageTransformer
from nimare.utils import get_resource_path

###############################################################################
# Download data
# --------------------------------
dset_dir = download_nidm_pain()

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)
dset.update_path(dset_dir)

# Calculate missing images
xformer = ImageTransformer(target=["varcope", "z"])
dset = xformer.transform(dset)

###############################################################################
# Stouffer's
# --------------------------------------------------
meta = ibma.Stouffers(use_sample_size=False, resample=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Stouffer's with weighting by sample size
# -----------------------------------------------------------------------------
meta = ibma.Stouffers(use_sample_size=True, resample=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Fisher's
# -----------------------------------------------------------------------------
meta = ibma.Fishers(resample=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Permuted OLS
# -----------------------------------------------------------------------------
meta = ibma.PermutedOLS(two_sided=True, resample=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Permuted OLS with FWE Correction
# -----------------------------------------------------------------------------
meta = ibma.PermutedOLS(two_sided=True, resample=True)
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
meta = ibma.WeightedLeastSquares(tau2=0, resample=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# DerSimonian-Laird
# -----------------------------------------------------------------------------
meta = ibma.DerSimonianLaird(resample=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")

###############################################################################
# Hedges
# -----------------------------------------------------------------------------
meta = ibma.Hedges(resample=True)
meta.fit(dset)
plot_stat_map(meta.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r")
