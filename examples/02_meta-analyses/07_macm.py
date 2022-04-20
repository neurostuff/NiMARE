"""

.. _metas_macm:

============================================
Meta-analytic coactivation modeling analysis
============================================

Perform a MACM analysis with Neurosynth data.

Meta-analytic coactivation modeling (MACM) is a common coordinate-based
analysis in which task-independent "connectivity" is assessed by selecting
studies within a larger database based on locations of report coordinates.
"""
import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting

from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.meta.cbma.ale import SCALE
from nimare.meta.cbma.mkda import MKDAChi2

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
# We will assume that the Neurosynth database has already been downloaded and
# converted to a NiMARE dataset.
dset_file = "neurosynth_nimare_with_abstracts.pkl.gz"
dset = Dataset.load(dset_file)

###############################################################################
# Define a region of interest
# -----------------------------------------------------------------------------
# We'll use the right amygdala from the Harvard-Oxford atlas
atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
img = nib.load(atlas["maps"])

roi_idx = atlas["labels"].index("Right Amygdala")
img_vals = np.unique(img.get_fdata())
roi_val = img_vals[roi_idx]
roi_img = image.math_img(f"img1 == {roi_val}", img1=img)

###############################################################################
# Select studies with a reported coordinate in the ROI
# -----------------------------------------------------------------------------
roi_ids = dset.get_studies_by_mask(roi_img)
dset_sel = dset.slice(roi_ids)
print(f"{len(roi_ids)}/{len(dset.ids)} studies report at least one coordinate in the ROI")

###############################################################################
# Select studies with *no* reported coordinates in the ROI
# -----------------------------------------------------------------------------
no_roi_ids = list(set(dset.ids).difference(roi_ids))
dset_unsel = dset.slice(no_roi_ids)
print(f"{len(no_roi_ids)}/{len(dset.ids)} studies report zero coordinates in the ROI")


###############################################################################
# MKDA Chi2 with FWE correction
# -----------------------------------------------------------------------------
mkda = MKDAChi2(kernel__r=10)
results = mkda.fit(dset_sel, dset_unsel)

corr = FWECorrector(method="montecarlo", n_iters=10000)
cres = corr.transform(results)

# We want the "specificity" map (2-way chi-square between sel and unsel)
plotting.plot_stat_map(
    cres.get_map("z_desc-consistency_level-voxel_corr-FWE_method-montecarlo"),
    threshold=3.09,
    draw_cross=False,
    cmap="RdBu_r",
)

###############################################################################
# SCALE
# -----------------------------------------------------------------------------
# Another good option for a MACM analysis is the SCALE algorithm, which was
# designed specifically for MACM. Unfortunately, SCALE does not support
# multiple-comparisons correction.

# First, we must define our null model of reported coordinates in the literature.
# We will use the coordinates in Neurosynth
xyz = dset.coordinates[["x", "y", "z"]].values
scale = SCALE(xyz=xyz, n_iters=10000, n_cores=1, kernel__n=20)
results = scale.fit(dset_sel)
plotting.plot_stat_map(results.get_map("z"), draw_cross=False, cmap="RdBu_r")
