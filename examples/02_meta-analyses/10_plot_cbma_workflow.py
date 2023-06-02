"""

.. _cbma_workflow:

====================================================
Run a coordinate-based meta-analysis (CBMA) workflow
====================================================

NiMARE provides a plethora of tools for performing meta-analyses on neuroimaging data.
Sometimes it's difficult to know where to start, especially if you're new to meta-analysis.
This tutorial will walk you through using a CBMA workflow function which puts together
the fundamental steps of a CBMA meta-analysis.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

from nimare.dataset import Dataset
from nimare.reports.base import run_reports
from nimare.utils import get_resource_path
from nimare.workflows import cbma_workflow

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

###############################################################################
# Run CBMA Workflow
# -----------------------------------------------------------------------------
# The CBMA workflow function runs the following steps:
#
# 1. Runs a meta-analysis using the specified method (default: ALE)
# 2. Applies a corrector to the meta-analysis results (default: FWECorrector, montecarlo)
# 3. Generates cluster tables and runs diagnostics on the corrected results (default: Jackknife)
#
# All in one function call!
#
# result = cbma_workflow(dset)
#
# For this example, we use an FDR correction because the default corrector (FWE correction with
# Monte Carlo simulation) takes a long time to run due to the high number of iterations that
# are required
result = cbma_workflow(dset, corrector="fdr")

###############################################################################
# Plot Results
# -----------------------------------------------------------------------------
# The CBMA workflow function returns a :class:`~nimare.results.MetaResult` object,
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
result.tables["z_corr-FDR_method-indep_diag-Jackknife_tab-counts"]

###############################################################################
# Report
# -----------------------------------------------------------------------------
# Finally, a NiMARE report is generated from the MetaResult.
root_dir = Path(os.getcwd()).parents[1]
# Use the following path to run the documentation locally:
# html_dir = root_dir / "docs" / "_build" / "html" / "auto_examples" / "02_meta-analyses"
html_dir = root_dir / "_readthedocs" / "html" / "auto_examples" / "02_meta-analyses"
html_dir.mkdir(parents=True, exist_ok=True)

run_reports(result, html_dir)

####################################
# .. raw:: html
#
#     <iframe src="./report.html" style="border:none;" seamless="seamless" width="100%"\
#        height="1000px"></iframe>
