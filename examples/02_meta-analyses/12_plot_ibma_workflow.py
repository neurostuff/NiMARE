"""

.. _ibma_workflow:

====================================================
Run an image-based meta-analysis (IBMA) workflow
====================================================

NiMARE provides a plethora of tools for performing meta-analyses on neuroimaging data.
Sometimes it's difficult to know where to start, especially if you're new to meta-analysis.
This tutorial will walk you through using a IBMA workflow function which puts together
the fundamental steps of a IBMA meta-analysis.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

from nimare.extract import download_nidm_pain

###############################################################################
# Download data
# -----------------------------------------------------------------------------

dset_dir = download_nidm_pain()

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
import nibabel as nib
from nilearn.image import resample_to_img

from nimare.dataset import Dataset
from nimare.transforms import ImageTransformer
from nimare.utils import get_resource_path

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)
dset.update_path(dset_dir)

# Calculate missing images
xformer = ImageTransformer(target=["varcope", "z"])
dset = xformer.transform(dset)

###############################################################################
# Run IBMA Workflow
# -----------------------------------------------------------------------------
# The fit method of a IBMA workflow class runs the following steps:
#
# 1. Runs a meta-analysis using the specified method (default: Stouffers)
# 2. Applies a corrector to the meta-analysis results (default: FDRCorrector, indep)
# 3. Generates cluster tables and runs diagnostics on the corrected results (default: Jackknife)
#
# All in one call!
from nimare.workflows.ibma import IBMAWorkflow

workflow = IBMAWorkflow()
result = workflow.fit(dset)

###############################################################################
# Plot Results
# -----------------------------------------------------------------------------
# The fit method of the IBMA workflow class returns a :class:`~nimare.results.MetaResult` object,
# where you can access the corrected results of the meta-analysis and diagnostics tables.
#
# Corrected map:
img = result.get_map("z_corr-FDR_method-indep")
plot_stat_map(
    img,
    cut_coords=4,
    display_mode="z",
    threshold=1.65,  # voxel_thresh p < .05, one-tailed
    cmap="RdBu_r",
    vmax=4,
)
plt.show()

###############################################################################
# Clusters table
# ``````````````````````````````````````````````````````````````````````````````
result.tables["z_corr-FDR_method-indep_tab-clust"]

###############################################################################
# Contribution table
# ``````````````````````````````````````````````````````````````````````````````
result.tables["z_corr-FDR_method-indep_diag-Jackknife_tab-counts_tail-positive"]

###############################################################################
# Report
# -----------------------------------------------------------------------------
# Finally, a NiMARE report is generated from the MetaResult.
from nimare.reports.base import run_reports

# root_dir = Path(os.getcwd()).parents[1] / "docs" / "_build"
# Use the previous root to run the documentation locally.
root_dir = Path(os.getcwd()).parents[1] / "_readthedocs"
html_dir = root_dir / "html" / "auto_examples" / "02_meta-analyses" / "12_plot_ibma_workflow"
html_dir.mkdir(parents=True, exist_ok=True)

run_reports(result, html_dir)

####################################
# .. raw:: html
#
#     <iframe src="./12_plot_ibma_workflow/report.html" style="border:none;" seamless="seamless"\
#        width="100%" height="1000px"></iframe>
