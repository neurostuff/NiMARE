r"""
.. _metas_cbmr:
.. _metas_voxelwise_cbmr:

============================================================
Coordinate-based meta-regression with moderator-effect modes
============================================================

A tour of coordinate-based meta-regression (CBMR) in NiMARE.

CBMR is a generative framework for estimating smooth activation intensity
functions from coordinate-based meta-analytic data. The same
:class:`~nimare.meta.cbmr.CBMREstimator` can parameterize moderator effects in
two ways:

* ``moderator_effect="global"`` estimates one scalar coefficient per moderator.
  This assumes the effect of the moderator has a global effect across the entire brain.
* ``moderator_effect="voxelwise"`` estimates a scalar coefficient _per voxel_ per moderator.
  This assumes the effect of the moderator differentially impacts voxels throughout the brain.
  This is likely a more accurate assumption, but requires a lot more data for estimation.

This tutorial fits both versions to the same simulated Studyset with the same
groups and the same standardized moderators, then compares their outputs.
"""

import numpy as np
import scipy
from nilearn.plotting import plot_stat_map

from nimare.correct import FDRCorrector
from nimare.generate import create_coordinate_studyset
from nimare.meta import CBMREstimator, models
from nimare.transforms import StandardizeField

###############################################################################
# Load Studyset-compatible data
# -----------------------------------------------------------------------------
# We simulate a coordinate-based Studyset with reported foci, sample sizes,
# diagnosis labels, drug-status labels, and continuous moderators.
# The example uses a moderate number of studies and coarse B-spline spacing so
# that both global and voxelwise CBMR fits run quickly.

_, studyset = create_coordinate_studyset(
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

studyset = StandardizeField(fields=["sample_sizes", "avg_age"]).transform(studyset)

group_categories = ["diagnosis", "drug_status"]
moderators = ["standardized_sample_sizes", "standardized_avg_age"]

###############################################################################
# Option 1: global moderator effects
# -----------------------------------------------------------------------------
# With ``moderator_effect="global"``, CBMR estimates group-specific baseline
# spatial intensity functions plus one scalar effect for each moderator.
# Here, ``standardized_sample_sizes`` and ``standardized_avg_age`` each receive
# one global coefficient shared over voxels.

global_cbmr = CBMREstimator(
    moderator_effect="global",
    group_categories=group_categories,
    moderators=moderators,
    spline_spacing=100,  # a reasonable analysis choice is 10 or 5; 100 is for speed
    model=models.PoissonEstimator,
    penalty=False,
    lr=1e-1,
    tol=1e3,  # a reasonable analysis choice is 1e-2; 1e3 is for speed
    device="cpu",  # use "cuda" if you have a GPU
)
global_results = global_cbmr.fit(dataset=studyset)

print(global_results.describe_inference_inputs())

###############################################################################
# Plot baseline group-specific spatial intensity
# -----------------------------------------------------------------------------
# Both global and voxelwise CBMR estimate baseline spatial intensity maps. The
# map names are shared across the two moderator-effect parameterizations.

plot_stat_map(
    global_results.get_map("spatialIntensity_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Global CBMR: Schizophrenia with drug treatment",
    threshold=1e-4,
    vmax=1e-3,
)
plot_stat_map(
    global_results.get_map("spatialIntensity_group-DepressionNo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Global CBMR: Depression without drug treatment",
    threshold=1e-4,
    vmax=1e-3,
)

###############################################################################
# Inference on global moderator effects
# -----------------------------------------------------------------------------
# Result-centered helpers run inference without constructing a separate
# inference object explicitly. For global moderator effects, moderator inference
# returns scalar tables.

global_moderator_result = global_results.test_moderators()
print(global_moderator_result.tables["moderators_regression_coef"])
print(global_moderator_result.tables["p_standardized_sample_sizes"])
print(global_moderator_result.tables["p_standardized_avg_age"])

global_moderator_comparison = global_results.compare_moderators(
    [("standardized_sample_sizes", "standardized_avg_age")]
)
print(global_moderator_comparison.tables["p_standardized_sample_sizes-standardized_avg_age"])

###############################################################################
# Group inference and correction for global CBMR
# -----------------------------------------------------------------------------
# Group homogeneity tests and pairwise group comparisons use the same result
# helpers for both moderator-effect modes.

global_group_result = global_results.test_groups()

plot_stat_map(
    global_group_result.get_map("z_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Global CBMR homogeneity test: SchizophreniaYes",
    threshold=scipy.stats.norm.isf(0.05),
    vmax=30,
)

corr = FDRCorrector(method="indep", alpha=0.05)
corrected_global_group_result = corr.transform(global_group_result)

plot_stat_map(
    corrected_global_group_result.get_map("z_group-SchizophreniaYes_corr-FDR_method-indep"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Global CBMR homogeneity test: SchizophreniaYes (FDR corrected)",
    threshold=scipy.stats.norm.isf(0.05),
    vmax=30,
)

global_group_comparison = global_results.compare_groups(
    [
        ("SchizophreniaYes", "SchizophreniaNo"),
        ("DepressionYes", "DepressionNo"),
    ]
)

plot_stat_map(
    global_group_comparison.get_map("z_group-SchizophreniaYes-SchizophreniaNo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Global CBMR group comparison: Schizophrenia drug effect",
    threshold=scipy.stats.norm.isf(0.4),
    vmax=2,
)

###############################################################################
# Flexible GLH tests with contrast matrices
# -----------------------------------------------------------------------------
# CBMR also supports generalized linear hypothesis (GLH) tests by passing raw
# contrast vectors or matrices to ``infer``. The example below tests whether all
# four group-specific spatial intensity estimates are equal.

global_glh_result = global_results.infer(
    group_contrasts=[[[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 1, -1]]],
    moderator_contrasts=False,
)
print("The contrast matrix of GLH_0 is {}".format(global_glh_result.metadata["GLH_groups_0"]))

plot_stat_map(
    global_glh_result.get_map("z_GLH_groups_0"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Global CBMR GLH_groups_0",
    threshold=scipy.stats.norm.isf(0.4),
)

###############################################################################
# Option 2: voxelwise moderator effects
# -----------------------------------------------------------------------------
# The same estimator exposes voxelwise moderator-effect maps through
# ``moderator_effect="voxelwise"``. This option uses the same groups and the same
# standardized moderators as above, but estimates a smooth effect map for each
# moderator within each group. The approximate backend is used here for speed.

voxelwise_cbmr = CBMREstimator(
    moderator_effect="voxelwise",
    group_categories=group_categories,
    moderators=moderators,
    spline_spacing=100,  # a reasonable analysis choice is 10 or 5; 100 is for speed
    backend="approximate",
    n_iter=10,
    tol=1e3,  # a reasonable analysis choice is 1e-4; 1e3 is for speed
    alpha=1e-3,
    damping=1.0,
    compute_nll=False,
    device="cpu",  # the full backend also accepts "cuda" if a GPU is available
)
voxelwise_results = voxelwise_cbmr.fit(dataset=studyset)

print(voxelwise_results.describe_inference_inputs())
print(voxelwise_results.voxelwise_moderator_effect_map_names)
print(voxelwise_results.describe_voxelwise_moderator_effect_maps())

###############################################################################
# Plot voxelwise moderator-effect maps
# -----------------------------------------------------------------------------
# In the voxelwise model, each moderator has a fitted map in each group. This is
# the key difference from global CBMR, where moderator inference is summarized in
# scalar tables.

plot_stat_map(
    voxelwise_results.get_map(
        "voxelwiseModeratorEffect_standardized_sample_sizes_group-SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Voxelwise sample-size effect: SchizophreniaYes",
)
plot_stat_map(
    voxelwise_results.get_map(
        "voxelwiseModeratorEffect_standardized_avg_age_group-SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Voxelwise age effect: SchizophreniaYes",
)

###############################################################################
# Diagnostic maps for per-unit voxelwise moderator changes
# -----------------------------------------------------------------------------
# A fitted :class:`~nimare.meta.cbmr.CBMRInference` object can generate Relative
# Intensity (RI) and Intensity Difference (ID) diagnostic maps showing how a
# user-defined moderator-unit change affects spatial intensity. Users can either
# keep the RI/ID maps for downstream diagnosis or plot RI inside an ID-defined
# region of interest. If no ID threshold is provided, the median absolute ID
# value is used.

voxelwise_inference = voxelwise_results.get_inference(method="FI")
voxelwise_diagnostic_result = voxelwise_inference.generate_voxelwise_moderator_effect_maps(
    moderators=["standardized_sample_sizes", "standardized_avg_age"],
    groups="SchizophreniaYes",
    unit_change=1.0,
)
print(voxelwise_diagnostic_result.metadata["voxelwise_moderator_effect_diagnostic_maps"])

voxelwise_inference.plot_voxelwise_moderator_effects(
    moderators=["standardized_sample_sizes", "standardized_avg_age"],
    groups="SchizophreniaYes",
    unit_change=1.0,
    id_threshold=None,
    cut_coords=[0, 0, -8],
    plot_kwargs={"draw_cross": False},
)

###############################################################################
# Inference on voxelwise moderator effects
# -----------------------------------------------------------------------------
# Voxelwise CBMR supports the same result-centered helpers. Because moderator
# effects vary over space, moderator inference returns maps instead of scalar
# tables. The default method is a robust sandwich covariance estimator; inverse
# Fisher information standard errors can be requested with ``method="FI"``.

voxelwise_moderator_result = voxelwise_results.test_moderators(method="sandwich")
print(voxelwise_moderator_result.metadata["voxelwise_cbmr_inference_method"])

plot_stat_map(
    voxelwise_moderator_result.get_map(
        "z_voxelwiseModeratorEffect_standardized_sample_sizes_group-SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Voxelwise test of sample-size effect: SchizophreniaYes",
    threshold=None,
    vmax=5,
)

voxelwise_moderator_comparison = voxelwise_results.compare_moderators(
    [("standardized_sample_sizes", "standardized_avg_age")]
)

plot_stat_map(
    voxelwise_moderator_comparison.get_map(
        "z_voxelwiseModeratorEffect_standardized_sample_sizes-standardized_avg_age_group-"
        "SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Voxelwise sample-size vs. age effect: SchizophreniaYes",
    threshold=None,
    vmax=2,
)

###############################################################################
# Optional inverse-Fisher standard errors for voxelwise CBMR
# -----------------------------------------------------------------------------
# The same voxelwise inference helpers can use inverse Fisher information, but
# the sandwich estimator is usually the safer default for applied CBMR analyses.
# Inverse-Fisher standard errors are model-based: they are efficient when the
# likelihood, mean-variance relationship, and independence assumptions are
# correctly specified, but can be too optimistic when those assumptions are only
# approximate. Coordinate-based meta-analytic data often have study-level
# clustering, heterogeneous reporting practices, and other departures from the
# idealized Poisson model. Sandwich standard errors use the fitted model for the
# mean structure while estimating covariance from empirical residual variation,
# making inference more robust to this kind of model misspecification.
#
# For that reason, we recommend keeping ``method="sandwich"`` as the default for
# primary voxelwise CBMR inference. ``method="FI"`` can still be useful for
# sensitivity analyses, simulations where the model is known to be correct, or
# comparisons with fully model-based standard errors.

voxelwise_fi_result = voxelwise_results.test_groups(method="FI")
print(voxelwise_fi_result.metadata["voxelwise_cbmr_inference_method"])

###############################################################################
# Summary
# -----------------------------------------------------------------------------
# Use ``moderator_effect="global"`` when the scientific question is whether a
# moderator has an overall effect on activation intensity. Use ``moderator_effect="voxelwise"``
# when the scientific question is where that moderator effect varies across the brain.
# Both options share the same preprocessing, grouping, and result-centered inference interface.
