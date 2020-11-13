# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas5:

========================================================
 Run a meta-analytic coactivation modeling analysis
========================================================

Meta-analytic coactivation modeling (MACM) is a common coordinate-based
analysis in which task-independent "connectivity" is assessed by selecting
studies within a larger database based on locations of report coordinates.
"""
import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting

import nimare

###############################################################################
# Load Dataset
# --------------------------------------------------
# We will assume that the Neurosynth database has already been downloaded and
# converted to a NiMARE dataset.
dset_file = "neurosynth_nimare_with_abstracts.pkl.gz"
dset = nimare.dataset.Dataset.load(dset_file)

###############################################################################
# Define a region of interest
# --------------------------------------------------
# We'll use the right amygdala from the Harvard-Oxford atlas
atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
img = nib.load(atlas["maps"])

roi_idx = atlas["labels"].index("Right Amygdala")
img_vals = np.unique(img.get_fdata())
roi_val = img_vals[roi_idx]
roi_img = image.math_img("img1 == {}".format(roi_val), img1=img)

###############################################################################
# Select studies with a reported coordinate in the ROI
# ----------------------------------------------------
roi_ids = dset.get_studies_by_mask(roi_img)
dset_sel = dset.slice(roi_ids)
print(
    "{}/{} studies report at least one coordinate in the "
    "ROI".format(len(roi_ids), len(dset.ids))
)

###############################################################################
# Select studies with *no* reported coordinates in the ROI
# --------------------------------------------------------
no_roi_ids = list(set(dset.ids).difference(roi_ids))
dset_unsel = dset.slice(no_roi_ids)
print("{}/{} studies report zero coordinates in the " "ROI".format(len(no_roi_ids), len(dset.ids)))


###############################################################################
# MKDA Chi2 with FWE correction
# --------------------------------------------------
mkda = nimare.meta.MKDAChi2(kernel__r=10)
mkda.fit(dset_sel, dset_unsel)

corr = nimare.correct.FWECorrector(method="montecarlo", n_iters=10000)
cres = corr.transform(mkda.results)

# We want the "specificity" map (2-way chi-square between sel and unsel)
plotting.plot_stat_map(
    cres.get_map("logp_level-cluster_corr-FWE_method-montecarlo"),
    threshold=3.0,
    draw_cross=False,
    cmap="RdBu_r",
)

###############################################################################
# SCALE
# --------------------------------------------------
# Another good option for a MACM analysis is the SCALE algorithm, which was
# designed specifically for MACM. Unfortunately, SCALE does not support
# multiple-comparisons correction.

# First, we must define our null model of reported coordinates in the literature.
# We will use the IJK coordinates in Neurosynth
ijk = dset.coordinates[["i", "j", "k"]].values
scale = nimare.meta.SCALE(ijk=ijk, n_iters=10000, kernel__n=20)
scale.fit(dset_sel)
plotting.plot_stat_map(scale.results.get_map("z_vthresh"), draw_cross=False, cmap="RdBu_r")
