"""

.. _metas_correctors:

==============================================
Multiple comparisons correction and Correctors
==============================================

Here we take a look at multiple comparisons correction in meta-analyses.
"""
import os

import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.plotting import plot_stat_map

from nimare.correct import FDRCorrector, FWECorrector
from nimare.dataset import Dataset
from nimare.extract import download_nidm_pain
from nimare.meta.cbma.ale import ALE
from nimare.meta.ibma import Stouffers
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
xformer = ImageTransformer(target=["z"])
dset = xformer.transform(dset)

mask_img = dset.masker.mask_img

###############################################################################
# Multiple comparisons correction in coordinate-based meta-analyses
# -----------------------------------------------------------------

# First, we need to fit the Estimator to the Dataset.
meta = ALE(null_method="approximate")
results = meta.fit(dset)

# We can check which FWE correction methods are available for the ALE Estimator
# with the ``inspect`` class method.
print(FWECorrector.inspect(results))

###############################################################################
# Apply the Corrector to the MetaResult
# =====================================
# Now that we know what FWE correction methods are available, we can use one.
#
# The "montecarlo" method is a special one that is implemented within the
# Estimator, rather than in the Corrector.
corr = FWECorrector(method="montecarlo", n_iters=50, n_cores=2)
cres = corr.transform(results)

DISTS_TO_PLOT = [
    "values_desc-size_level-cluster_corr-fwe_method-montecarlo",
    "values_desc-mass_level-cluster_corr-fwe_method-montecarlo",
    "values_level-voxel_corr-fwe_method-montecarlo",
]
XLABELS = [
    "Maximum Cluster Size (Voxels)",
    "Maximum Cluster Mass",
    "Maximum Summary Statistic (ALE Value)",
]

fig, axes = plt.subplots(figsize=(8, 8), nrows=3)
null_dists = cres.estimator.null_distributions_

for i_ax, dist_name in enumerate(DISTS_TO_PLOT):
    xlabel = XLABELS[i_ax]
    sns.histplot(x=null_dists[dist_name], bins=40, ax=axes[i_ax])
    axes[i_ax].set_title(dist_name)
    axes[i_ax].set_xlabel(xlabel)
    axes[i_ax].set_xlim(0, None)

fig.tight_layout()

###############################################################################
# Show corrected results
# ======================
MAPS_TO_PLOT = [
    "z",
    "z_desc-size_level-cluster_corr-FWE_method-montecarlo",
    "z_desc-mass_level-cluster_corr-FWE_method-montecarlo",
    "z_level-voxel_corr-FWE_method-montecarlo",
]
TITLES = [
    "Uncorrected z-statistics",
    "Cluster-size FWE-corrected z-statistics",
    "Cluster-mass FWE-corrected z-statistics",
    "Voxel-level FWE-corrected z-statistics",
]

fig, axes = plt.subplots(figsize=(8, 10), nrows=4)

for i_ax, map_name in enumerate(MAPS_TO_PLOT):
    title = TITLES[i_ax]
    plot_stat_map(
        cres.get_map(map_name),
        draw_cross=False,
        cmap="RdBu_r",
        threshold=0.5,
        cut_coords=[0, 0, -8],
        figure=fig,
        axes=axes[i_ax],
    )
    axes[i_ax].set_title(title)

###############################################################################
# Multiple comparisons correction in image-based meta-analyses
# ------------------------------------------------------------
meta = Stouffers(resample=True)
results = meta.fit(dset)
print(FWECorrector.inspect(results))
print(FDRCorrector.inspect(results))

###############################################################################
# Note that the FWECorrector does not support a "montecarlo" method for the
# Stouffers Estimator.
# This is because NiMARE does not have a Monte Carlo-based method implemented
# for most IBMA algorithms.

###############################################################################
# Apply the Corrector to the MetaResult
# =====================================
corr = FDRCorrector(method="indep", alpha=0.05)
cres = corr.transform(results)

###############################################################################
# Show corrected results
# ======================
fig, axes = plt.subplots(figsize=(8, 6), nrows=2)
plot_stat_map(
    cres.get_map("z"),
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.5,
    cut_coords=[0, 0, -8],
    figure=fig,
    axes=axes[0],
)
axes[0].set_title("Uncorrected z-statistics")
plot_stat_map(
    cres.get_map("z_corr-FDR_method-indep"),
    draw_cross=False,
    cmap="RdBu_r",
    threshold=0.5,
    cut_coords=[0, 0, -8],
    figure=fig,
    axes=axes[1],
)
axes[1].set_title("FDR-corrected z-statistics")
