"""

.. _metas_sdm:

==========================================
Seed-based d Mapping (SDM) Meta-Analysis
==========================================

A tutorial on using SDM and SDM-PSI algorithms in NiMARE.

This tutorial demonstrates how to perform meta-analysis using Seed-based d Mapping (SDM),
including both the basic SDM estimator and the advanced SDM-PSI (Permutation of Subject Images)
implementation with multiple imputation and Rubin's rules.

SDM is a hybrid meta-analysis method that can accept both peak coordinates and
whole-brain statistical images, preferentially using images when available.
"""

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
# .. note::
#   The data used in this example come from a collection of NIDM-Results packs
#   downloaded from Neurovault collection 1425, uploaded by Dr. Camille Maumet.
import os
from pprint import pprint

from nilearn.plotting import plot_stat_map

from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.meta import SDM, SDMPSI, SDMKernel
from nimare.utils import get_resource_path

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

# Split dataset for demonstration
dset1 = dset.slice(dset.ids[:10])

###############################################################################
# Basic SDM Meta-Analysis
# -----------------------------------------------------------------------------
# The basic SDM estimator performs coordinate-based meta-analysis using an
# anisotropic Gaussian kernel. When images are available, it uses them directly.
#
# **Key Features:**
#
# - Hybrid input support (coordinates + images)
# - Anisotropic Gaussian kernel (FWHM=20mm default)
# - Simple mean-based meta-analysis
# - Returns stat, z, p, and dof maps

meta = SDM()
results = meta.fit(dset1)

# Plot the results
plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=3,
    title="SDM: Z-statistic Map",
)

print("\\nSDM Description:")
pprint(results.description_)
print("\\nSDM References:")
pprint(results.bibtex_)

###############################################################################
# Custom Kernel Configuration
# -----------------------------------------------------------------------------
# You can customize the SDM kernel's full-width at half-maximum (FWHM)
# parameter to control the smoothness of the reconstructed maps from coordinates.

meta_custom = SDM(SDMKernel, kernel__fwhm=15)
results_custom = meta_custom.fit(dset1)

plot_stat_map(
    results_custom.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=3,
    title="SDM with Custom FWHM=15mm",
)

###############################################################################
# SDM-PSI: Advanced Meta-Analysis with Multiple Imputation
# -----------------------------------------------------------------------------
# SDM-PSI extends SDM with advanced features:
#
# - **Multiple imputation** of missing data
# - **Subject-level image simulation** for more realistic variance estimation
# - **Rubin's rules** for combining results across imputations
# - Variance decomposition (within-imputation and between-imputation variance)
#
# This provides more robust statistical inference compared to basic SDM.

meta_psi = SDMPSI(
    n_imputations=5,  # Number of imputations (default=5)
    n_subjects_sim=50,  # Simulated subjects per study (default=50)
    random_state=42,  # For reproducibility
)
results_psi = meta_psi.fit(dset1)

plot_stat_map(
    results_psi.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=3,
    title="SDM-PSI: Z-statistic Map",
)

print("\\nSDM-PSI Description:")
pprint(results_psi.description_)

###############################################################################
# Variance Decomposition in SDM-PSI
# -----------------------------------------------------------------------------
# SDM-PSI provides variance decomposition showing:
#
# - **Within-imputation variance**: Variance within each imputed dataset
# - **Between-imputation variance**: Variance between different imputations
#
# Higher between-imputation variance indicates greater uncertainty in the
# imputed values.

plot_stat_map(
    results_psi.get_map("within_var"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="hot",
    vmax=0.5,
    title="Within-Imputation Variance",
)

plot_stat_map(
    results_psi.get_map("between_var"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="hot",
    vmax=0.1,
    title="Between-Imputation Variance",
)

###############################################################################
# Multiple Comparison Correction
# -----------------------------------------------------------------------------
# Like other CBMA methods, SDM results can be corrected for multiple comparisons.

corr = FWECorrector(method="montecarlo", n_iters=100, n_cores=1)
cres = corr.transform(results)

plot_stat_map(
    cres.get_map("z_desc-size_level-cluster_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=3,
    title="SDM: FWE-corrected Z-statistic",
)

###############################################################################
# Hybrid Input Mode
# -----------------------------------------------------------------------------
# SDM automatically detects and handles different input types:
#
# - **Coordinates only**: Reconstructs maps using SDM kernel
# - **Images only**: Uses statistical maps directly
# - **Hybrid**: Prefers images, falls back to coordinates
#
# The `input_mode_` attribute indicates which mode was used:

print(f"\\nInput mode used: {meta.input_mode_}")
print(f"SDM-PSI input mode: {meta_psi.input_mode_}")

###############################################################################
# When to Use SDM vs SDM-PSI
# -----------------------------------------------------------------------------
# **Use basic SDM when:**
#
# - You need fast meta-analysis
# - Your data is relatively complete
# - Simple mean-based analysis is sufficient
#
# **Use SDM-PSI when:**
#
# - You have missing data that needs imputation
# - You want more robust variance estimates
# - You need variance decomposition
# - You want to account for imputation uncertainty
#
# **Compared to ALE/MKDA:**
#
# - SDM accepts both coordinates and images (hybrid)
# - SDM provides effect size estimates, not just activation likelihood
# - SDM-PSI offers advanced statistical features
# - SDM may have better sensitivity when images are available
