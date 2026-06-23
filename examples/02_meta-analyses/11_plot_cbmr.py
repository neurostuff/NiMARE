r"""
.. _metas_cbmr:
.. _metas_voxelwise_cbmr:

====================================================================
Coordinate-based meta-regression with global and voxelwise moderators
====================================================================

A tour of coordinate-based meta-regression (CBMR) in NiMARE.

CBMR is a generative framework for estimating smooth activation intensity
functions from coordinate-based meta-analytic data. The current
:class:`~nimare.meta.cbmr.CBMREstimator` implementation exposes one public API
for three moderator-effect parameterizations:

* ``moderator_effect="global"`` estimates one scalar coefficient per moderator.
  This assumes the effect of the moderator has a global effect across the entire brain.
* ``moderator_effect="voxelwise"`` estimates a smooth spatial coefficient map
  for each moderator and group. This allows the moderator effect to vary across
  the brain, but requires more data for stable estimation.
* ``moderator_effect="mixed"`` estimates scalar coefficients for selected
  ``global_moderators`` and spatially varying coefficients for selected
  ``voxelwise_moderators`` in one model.

This tutorial fits all three versions to the same simulated Studyset, shows how
to inspect fitted CBMR results, and demonstrates the result-centered inference
helpers added around :class:`~nimare.meta.cbmr.CBMRInference`.
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
# diagnosis labels, drug-status labels, and continuous moderators. The example
# uses a moderate number of studies and coarse B-spline spacing so that global,
# voxelwise, and mixed CBMR fits run quickly.

_, studyset = create_coordinate_studyset(
    foci=10,
    sample_size=(20, 40),
    n_studies=200,
    seed=100,
)

annotations_df = studyset.annotations_df.copy()
n_rows = annotations_df.shape[0]
group_pattern = [
    ("schizophrenia", "Yes"),
    ("schizophrenia", "No"),
    ("depression", "Yes"),
    ("depression", "No"),
]
annotations_df[["diagnosis", "drug_status"]] = [
    group_pattern[i % len(group_pattern)] for i in range(n_rows)
]
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
    random_state=100,
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
# Group homogeneity tests and pairwise group comparisons use the same
# result-centered helpers for all moderator-effect modes. The helpers now also
# expose the robust covariance options implemented in :class:`~nimare.meta.cbmr.CBMRInference`.

global_group_result = global_results.test_groups(
    method="sandwich",
    sandwich_meat="iid",
    sandwich_correction="hc0",
    ridge=1e-4,
)
print(global_group_result.metadata["global_cbmr_inference_method"])
print(global_group_result.metadata["global_cbmr_sandwich_meat"])
print(global_group_result.metadata["global_cbmr_sandwich_correction"])

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
    random_state=100,
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
# user-defined moderator-unit change affects spatial intensity. The helper
# accepts the same moderator/group selectors as the inference methods and returns
# a CBMRResult copy with named RI and ID maps. Users can keep those maps for
# downstream diagnosis or plot RI inside an ID-defined region of interest. If no
# ID threshold is provided, the median absolute ID value is used.

voxelwise_inference = voxelwise_results.get_inference(
    method="FI",
    incidence_threshold=None,
)
voxelwise_diagnostic_result = voxelwise_inference.generate_voxelwise_moderator_effect_maps(
    moderators=["standardized_sample_sizes", "standardized_avg_age"],
    groups="SchizophreniaYes",
    unit_change=1.0,
)
print(voxelwise_diagnostic_result.metadata["voxelwise_moderator_effect_diagnostic_maps"])

plot_stat_map(
    voxelwise_diagnostic_result.get_map(
        "relativeIntensity_voxelwiseModeratorEffect_standardized_sample_sizes_unit-1_group-"
        "SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    title="RI for one-SD sample-size increase: SchizophreniaYes",
)
plot_stat_map(
    voxelwise_diagnostic_result.get_map(
        "intensityDifference_voxelwiseModeratorEffect_standardized_sample_sizes_unit-1_group-"
        "SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    title="ID for one-SD sample-size increase: SchizophreniaYes",
)

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
# Option 3: mixed global and voxelwise moderator effects
# -----------------------------------------------------------------------------
# Mixed CBMR is useful when some moderators are expected to have a whole-brain
# effect and others are expected to vary spatially. The model below estimates a
# global sample-size effect and a voxelwise age-effect map in the same fit. Mixed
# CBMR currently uses the full Poisson backend.

mixed_cbmr = CBMREstimator(
    moderator_effect="mixed",
    group_categories=group_categories,
    global_moderators=["standardized_sample_sizes"],
    voxelwise_moderators=["standardized_avg_age"],
    backend="full",
    spline_spacing=100,  # a reasonable analysis choice is 10 or 5; 100 is for speed
    n_iter=10,
    lr=1e-1,
    tol=1e3,  # a reasonable analysis choice is 1e-4; 1e3 is for speed
    device="cpu",  # use "cuda" if you have a GPU
    random_state=100,
)
mixed_results = mixed_cbmr.fit(dataset=studyset)

print(mixed_results.describe_inference_inputs())
print(mixed_results.tables["global_moderators_regression_coef"])
print(mixed_results.voxelwise_moderator_effect_map_names)

plot_stat_map(
    mixed_results.get_map("voxelwiseModeratorEffect_standardized_avg_age_group-SchizophreniaYes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Mixed CBMR voxelwise age effect: SchizophreniaYes",
)

###############################################################################
# Inference for mixed CBMR
# -----------------------------------------------------------------------------
# The same result-centered helpers dispatch each mixed-model moderator to the
# right inference path: global moderators return scalar tables, while voxelwise
# moderators return spatial maps. In mixed CBMR, contrast vectors should test
# global and voxelwise moderators separately.

mixed_moderator_result = mixed_results.test_moderators(method="FI")
print(mixed_moderator_result.tables["z_standardized_sample_sizes"])
print(mixed_moderator_result.metadata["global_cbmr_inference_method"])
print(mixed_moderator_result.metadata["voxelwise_cbmr_inference_method"])

plot_stat_map(
    mixed_moderator_result.get_map(
        "z_voxelwiseModeratorEffect_standardized_avg_age_group-SchizophreniaYes"
    ),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Mixed CBMR test of voxelwise age effect: SchizophreniaYes",
    threshold=None,
    vmax=5,
)

###############################################################################
# Summary
# -----------------------------------------------------------------------------
# Use ``moderator_effect="global"`` when the scientific question is whether a
# moderator has an overall effect on activation intensity. Use ``moderator_effect="voxelwise"``
# when the scientific question is where that moderator effect varies across the brain.
# Use ``moderator_effect="mixed"`` when both assumptions are needed in one model.
# All options share the same preprocessing, grouping, and result-centered inference interface.
