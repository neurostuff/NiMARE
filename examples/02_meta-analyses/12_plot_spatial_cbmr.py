r"""Spatially varying coordinate-based meta-regression tutorial.

.. _metas_spatial_cbmr:

===================================================
Spatially varying coordinate-based meta-regression
===================================================

A tour of spatially varying coordinate-based meta-regression (Spatial CBMR) in
NiMARE.

Standard CBMR estimates group-specific spatial intensity functions and global
study-level moderator effects. Spatial CBMR extends this framework by allowing
study-level moderator effects to vary smoothly over voxels. For experiment ``m``
and voxel ``v`` in group ``g``, Spatial CBMR models

.. math::

    \log(\lambda_{gmv}) = B(v) \alpha_g + Z_{gm} \beta_g B(v)^T,

where ``B`` is a spatial B-spline basis, ``alpha_g`` is the group-specific
baseline spatial coefficient vector, ``Z`` contains study-level moderators, and
``beta_g`` contains spatially varying moderator coefficients.

This tutorial mirrors the standard CBMR tutorial and uses the same simulated
Studyset construction, but fits spatially varying covariate effects. The example
uses the approximate backend and a coarse spline spacing to keep runtime low.
"""

import numpy as np
from nilearn.plotting import plot_stat_map

from nimare.correct import FDRCorrector
from nimare.generate import create_coordinate_studyset
from nimare.meta import SpatialCBMREstimator
from nimare.transforms import StandardizeField

###############################################################################
# Load Studyset-compatible data
# -----------------------------------------------------------------------------
# We simulate a coordinate-based Studyset with the same structure used in the
# CBMR tutorial: studies have reported foci, sample sizes, diagnosis labels,
# drug-status labels, and continuous study-level moderators. Spatial CBMR can be
# computationally heavier than standard CBMR because it keeps experiment-by-voxel
# foci matrices, so this tutorial uses fewer studies and a coarse spline basis.

ground_truth_foci, studyset = create_coordinate_studyset(
    foci=10,
    sample_size=(20, 40),
    n_studies=200,
)

annotations_df = studyset.annotations_df.copy()
n_rows = annotations_df.shape[0]
annotations_df["diagnosis"] = [
    "schizophrenia" if i % 2 == 0 else "depression" for i in range(n_rows)
]
annotations_df["drug_status"] = ["Yes" if i % 2 == 0 else "No" for i in range(n_rows)]
annotations_df["drug_status"] = annotations_df["drug_status"].sample(frac=1).reset_index(drop=True)
annotations_df["sample_sizes"] = [studyset.metadata.sample_sizes[i][0] for i in range(n_rows)]
annotations_df["avg_age"] = np.arange(n_rows)
studyset.annotations_df = annotations_df

###############################################################################
# Estimate spatially varying moderator effects
# -----------------------------------------------------------------------------
# Spatial CBMR uses standard CBMR preprocessing and grouping, but estimates
# moderator effects as smooth maps. Here, sample size and average age are treated
# as spatially varying covariates within each group.

studyset = StandardizeField(fields=["sample_sizes", "avg_age"]).transform(studyset)

spatial_cbmr = SpatialCBMREstimator(
    group_categories=["diagnosis", "drug_status"],
    moderators=["standardized_sample_sizes", "standardized_avg_age"],
    spline_spacing=100,  # a reasonable analysis choice is 10 or 5; 100 is for speed
    backend="approximate",
    n_iter=10,
    tol=1e3,  # a reasonable analysis choice is 1e-4; 1e3 is for speed
    alpha=1e-3,
    damping=1.0,
    compute_nll=False,
    device="cpu",  # "cuda" is accepted by the full backend if you have a GPU
)
results = spatial_cbmr.fit(dataset=studyset)

###############################################################################
# The fitted result contains baseline spatial intensity maps, spatially varying
# moderator-effect maps, and tables of basis coefficients. The helper below lists
# the spatially varying moderator maps that were produced.

print(results.describe_inference_inputs())
print(results.sv_moderator_names)
print(results.describe_sv_effects())

###############################################################################
# Plot baseline group-specific spatial intensity
# -----------------------------------------------------------------------------
# Spatial CBMR still estimates one baseline spatial intensity map per group.

plot_stat_map(
    results.get_map("spatialIntensity_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Spatial CBMR: Schizophrenia with drug treatment",
    threshold=1e-4,
    vmax=1e-3,
)
plot_stat_map(
    results.get_map("spatialIntensity_group-DepressionNo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Spatial CBMR: Depression without drug treatment",
    threshold=1e-4,
    vmax=1e-3,
)

###############################################################################
# Plot spatially varying covariate effects
# -----------------------------------------------------------------------------
# Unlike standard CBMR, Spatial CBMR estimates a map for each study-level
# moderator in each group. These maps show where the estimated moderator effect
# is stronger or weaker over voxels.

plot_stat_map(
    results.get_map("svModerator_standardized_sample_sizes_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Spatially varying sample-size effect: SchizophreniaYes",
)
plot_stat_map(
    results.get_map("svModerator_standardized_avg_age_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Spatially varying age effect: SchizophreniaYes",
)

###############################################################################
# Inference with sandwich standard errors
# -----------------------------------------------------------------------------
# Spatial CBMR exposes the same result-centered inference helpers as CBMR. The
# default standard-error estimator is a robust sandwich covariance estimator.
# Users can opt into inverse-Fisher information standard errors with
# ``method="FI"``.

contrast_result = results.test_groups(method="sandwich")
print(contrast_result.metadata["spatial_cbmr_inference_method"])

plot_stat_map(
    contrast_result.get_map("z_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Spatial CBMR homogeneity test: SchizophreniaYes",
    threshold=None,
    vmax=10,
)

###############################################################################
# Pairwise group comparisons
# -----------------------------------------------------------------------------
# Group comparisons use the same tuple shorthand as standard CBMR.

contrast_result = results.compare_groups(
    [
        ("SchizophreniaYes", "SchizophreniaNo"),
        ("DepressionYes", "DepressionNo"),
    ]
)

plot_stat_map(
    contrast_result.get_map("z_group-SchizophreniaYes-SchizophreniaNo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Spatial CBMR group comparison: Schizophrenia drug effect",
    threshold=None,
    vmax=2,
)

###############################################################################
# Spatially varying moderator inference
# -----------------------------------------------------------------------------
# The moderator inference maps test whether each spatially varying covariate
# effect differs from zero within each fitted group. These maps are stored as
# voxel-wise p- and z-statistics rather than scalar tables, because the covariate
# effects vary over space.

moderator_result = results.test_moderators()

plot_stat_map(
    moderator_result.get_map("z_svModerator_standardized_sample_sizes_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Test of spatially varying sample-size effect: SchizophreniaYes",
    threshold=None,
    vmax=5,
)

###############################################################################
# Spatial CBMR also supports contrasts between spatially varying covariate
# effects. The following example compares the spatial sample-size effect against
# the spatial age effect within each group.

moderator_comparison_result = results.compare_moderators(
    [("standardized_sample_sizes", "standardized_avg_age")]
)

plot_stat_map(
    moderator_comparison_result.get_map(
        "z_svModerator_standardized_sample_sizes-standardized_avg_age_group-SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Sample-size vs. age effect: SchizophreniaYes",
    threshold=None,
    vmax=2,
)

###############################################################################
# Optional inverse-Fisher standard errors
# -----------------------------------------------------------------------------
# If desired, the same inference helpers can use inverse Fisher information
# instead of the sandwich estimator.

fi_result = results.test_groups(method="FI")
print(fi_result.metadata["spatial_cbmr_inference_method"])

###############################################################################
# FDR correction
# -----------------------------------------------------------------------------
# Correctors operate on Spatial CBMR results in the same way as on standard CBMR
# results. Here, we correct the group homogeneity inference maps.

corr = FDRCorrector(method="indep", alpha=0.05)
cres = corr.transform(results.test_groups())

plot_stat_map(
    cres.get_map("z_group-SchizophreniaYes_corr-FDR_method-indep"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Spatial CBMR homogeneity test: SchizophreniaYes (FDR corrected)",
    threshold=None,
    vmax=10,
)
