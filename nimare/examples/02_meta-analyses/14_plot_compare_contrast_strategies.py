"""

.. _metas_compare_contrast_strategies:

============================================================
Comparing ALE-based pairwise contrast strategies
============================================================

When comparing two groups of neuroimaging studies, NiMARE offers three
complementary approaches:

* :class:`~nimare.meta.cbma.ale.ALESubtraction` — two-sided, whole-mask
  subtraction with a permutation null.
* :class:`~nimare.workflows.cbma.ContrastWorkflow` — main-effect-gated
  subtraction that mirrors the original GingerALE-style logic
  :footcite:p:`laird2005ale`.
* :class:`~nimare.meta.cbma.ale.BalancedALESubtraction` — balanced
  subsampling subtraction that corrects for unequal group sizes
  :footcite:p:`Frahm_Monimu_Hoffstaedter`.

All three use the ALE modelled-activation framework, but they differ in
**how group sizes are handled**, **whether main-effect information gates
directional inference**, and **what output maps are returned**.

**ALESubtraction (standalone)**: evaluates the group difference
(Group 1 − Group 2) in a *two-sided* manner across *all* voxels in the
analysis mask simultaneously, without any prior information about where each
group shows a reliable main effect.  Groups are compared at their natural
sizes; no resampling is performed.  A corrector is applied as a separate
step after fitting.

**ContrastWorkflow**: implements GingerALE-style main-effect-gated
approach :footcite:p:`laird2005ale`.  It first computes and corrects a
within-group ALE for each group, thresholds those corrected maps at a chosen
α, and then passes the surviving voxels as *directional inference masks* to
:class:`~nimare.meta.cbma.ale.ALESubtraction`.  Group 1 > Group 2 differences
are evaluated **only where Group 1 shows a surviving main effect**, and
Group 2 > Group 1 differences are evaluated **only where Group 2 shows a
surviving main effect**.  The workflow also stores the individual main-effect
maps and a voxelwise-minimum conjunction map alongside the thresholded
contrast.  The workflow thresholds the pairwise p-map internally; no additional
corrector call is made after the fact.

**BalancedALESubtraction**: addresses a systematic bias that arises when the
two groups contain different numbers of studies :footcite:p:`Frahm_Monimu_Hoffstaedter`.
Because ALE values grow with sample size, a naive subtraction will spuriously
favour the larger group.  This method draws *matched-size random subsamples*
from each group, averages the balanced ALE differences over many draws, and
builds a Monte Carlo null from balanced resamples of random foci (or permuted
labels).  It also produces *probabilistic activation maps* for each group —
the proportion of subsamples in which each voxel survived cluster thresholding
— and a conjunction of those probability maps.  No separate correction step
is needed.

In this example we apply all three approaches to the same two study groups and
compare their output maps.
"""

###############################################################################
# Load data
# -----------------------------------------------------------------------------
# We use two Sleuth text files bundled with NiMARE that describe pediatric
# semantic cognition studies :footcite:p:`enge2021meta`.
# The first group probed semantic world **knowledge** (e.g., naming an object
# from its description) and the second group asked children to judge semantic
# **relatedness** between words.
#
# .. note::
#   For real analyses, use at least ``n_iters=5000``.  We use 100 iterations
#   throughout this example purely for documentation-build speed.
import os
from pprint import pprint

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

from nimare.io import convert_sleuth_to_studyset
from nimare.utils import get_resource_path

N_ITERS = 100

knowledge_file = os.path.join(get_resource_path(), "semantic_knowledge_children.txt")
related_file = os.path.join(get_resource_path(), "semantic_relatedness_children.txt")

knowledge_studyset = convert_sleuth_to_studyset(knowledge_file)
related_studyset = convert_sleuth_to_studyset(related_file)

print(f"Semantic knowledge: {len(knowledge_studyset.studies)} studies")
print(f"Semantic relatedness: {len(related_studyset.studies)} studies")

###############################################################################
# ALESubtraction (standalone)
# -----------------------------------------------------------------------------
# Running :class:`~nimare.meta.cbma.ale.ALESubtraction` directly compares the
# two groups across the *entire* brain mask in a single two-sided permutation
# test.  Positive z-values indicate knowledge > relatedness; negative values
# indicate relatedness > knowledge.
#
# Unlike the original GingerALE implementation, this NiMARE approach does
# **not** restrict testing to regions where either group has a significant main
# effect, and it does **not** correct for unequal group sizes.  The benefit is
# a whole-mask search without main-effect gating; the trade-off is that it may
# detect differences in regions where neither group shows a reliable main
# effect, and results can be skewed when group sizes differ substantially.
#
# After fitting, a :class:`~nimare.correct.FWECorrector` (or FDR) can be
# applied to the returned :class:`~nimare.results.MetaResult` in the usual way.
from nimare.meta.cbma.ale import ALESubtraction

subtraction = ALESubtraction(n_iters=N_ITERS, n_cores=1)
subtraction_result = subtraction.fit(knowledge_studyset, related_studyset)

print("ALESubtraction uncorrected output maps:")
pprint(list(subtraction_result.maps.keys()))

###############################################################################
# Visualize the uncorrected subtraction z-map.
plot_stat_map(
    subtraction_result.get_map("z_desc-group1MinusGroup2"),
    cut_coords=4,
    display_mode="z",
    title="ALESubtraction (uncorrected): knowledge − relatedness",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=3,
)

###############################################################################
# ContrastWorkflow
# -----------------------------------------------------------------------------
# :class:`~nimare.workflows.cbma.ContrastWorkflow` orchestrates the full
# main-effect-gated pipeline in a single call:
#
# 1. Fit one-sample ALE for Group 1 and correct with the provided corrector
#    (default: FDR with ``method="indep"`` and ``alpha=0.05``).
# 2. Fit one-sample ALE for Group 2 and correct identically.
# 3. Threshold both corrected maps at ``alpha`` to build directional masks.
# 4. Run :class:`~nimare.meta.cbma.ale.ALESubtraction` with those masks:
#    Group 1 > Group 2 differences are tested only inside the Group 1 mask;
#    Group 2 > Group 1 differences are tested only inside the Group 2 mask.
# 5. Threshold the contrast p-map at ``alpha`` and store the result.
#
# The returned :class:`~nimare.results.MetaResult` contains all intermediate
# and final maps. The pairwise p-map is thresholded internally, so no additional
# corrector call is made after the workflow returns.
#
# :class:`~nimare.workflows.cbma.ContrastWorkflow` accepts a class, an
# already-initialized instance, or a string (e.g. ``"alesubtraction"``) for
# all three of its main arguments.  Passing a class or string lets the
# workflow propagate ``n_cores`` automatically; passing an instance uses
# whatever settings that instance already carries.  Here we pass the class
# and then lower ``n_iters`` on the instantiated estimator to keep the
# example fast.
from nimare.correct import FDRCorrector
from nimare.workflows.cbma import ContrastWorkflow

contrast_workflow = ContrastWorkflow(
    pairwise_estimator=ALESubtraction,
    corrector=FDRCorrector(method="indep", alpha=0.05),
    alpha=0.05,
    n_cores=1,
)
contrast_workflow.pairwise_estimator.n_iters = N_ITERS
contrast_result = contrast_workflow.fit(knowledge_studyset, related_studyset)

print("ContrastWorkflow output maps:")
pprint(list(contrast_result.maps.keys()))

###############################################################################
# The four key ContrastWorkflow output maps
# -----------------------------------------------------------------------------
# * **z_desc-group1MainEffect** — FDR-corrected, thresholded ALE z-map for
#   Group 1 (knowledge).  Defines where Group 1 shows reliable activation.
# * **z_desc-group2MainEffect** — same for Group 2 (relatedness).
# * **z_desc-conjunction** — voxelwise minimum of both thresholded main-effect
#   maps; highlights regions where *both* groups converge.
# * **z_desc-contrast** — the main-effect-gated subtraction contrast,
#   thresholded at ``alpha``.  Only voxels that survived in the appropriate
#   main-effect mask contribute to each direction of this map.
fig, axes = plt.subplots(figsize=(14, 12), nrows=4)

plot_stat_map(
    contrast_result.get_map("z_desc-group1MainEffect"),
    cut_coords=4,
    display_mode="z",
    title="Group 1 main effect: knowledge (FDR-corrected, α = 0.05)",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=4,
    axes=axes[0],
    figure=fig,
)

plot_stat_map(
    contrast_result.get_map("z_desc-group2MainEffect"),
    cut_coords=4,
    display_mode="z",
    title="Group 2 main effect: relatedness (FDR-corrected, α = 0.05)",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=4,
    axes=axes[1],
    figure=fig,
)

plot_stat_map(
    contrast_result.get_map("z_desc-conjunction"),
    cut_coords=4,
    display_mode="z",
    title="Conjunction (voxelwise minimum of both main effects)",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=4,
    axes=axes[2],
    figure=fig,
)

plot_stat_map(
    contrast_result.get_map("z_desc-contrast"),
    cut_coords=4,
    display_mode="z",
    title="Contrast: knowledge − relatedness (main-effect-gated, α = 0.05)",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=4,
    axes=axes[3],
    figure=fig,
)

fig.show()

###############################################################################
# BalancedALESubtraction
# -----------------------------------------------------------------------------
# :class:`~nimare.meta.cbma.ale.BalancedALESubtraction`
# :footcite:p:`Frahm_Monimu_Hoffstaedter` corrects for a well-known bias: when
# one group has more studies than the other, ALE values are systematically
# higher in the larger group, inflating apparent differences.  This method
# avoids that bias through the following steps:
#
# 1. **Probabilistic activation maps**: for each group, many random subsamples
#    of size ``target_n = min(n₁, n₂)`` are drawn.  Each subsample is
#    meta-analysed with ALE, cluster-thresholded, and each voxel receives a
#    vote (1 if it survived, 0 if not).  The resulting *probability map* is the
#    proportion of subsamples in which a voxel passed the cluster threshold.
# 2. **Mean balanced difference**: a separate set of ``difference_iterations``
#    matched-size draw pairs is used to estimate the expected (average) ALE
#    difference under the true group labels.
# 3. **Monte Carlo null**: ``n_iters`` additional balanced draw pairs are
#    generated under the null (either random-foci or label-permutation) to
#    build an empirical distribution of the minimum and maximum difference,
#    from which two-tailed p-values are derived without any external corrector.
#
# Key outputs:
#
# * **z_desc-balancedGroup1MinusGroup2** — z-scored balanced difference map
#   (positive = knowledge > relatedness).
# * **stat_desc-conjunction** — voxelwise minimum of the two probability maps;
#   highlights regions both groups activate consistently across subsamples.
# * **prob_desc-group1 / prob_desc-group2** — the per-group probability maps.
#
# No separate corrector call is needed; significance thresholding is handled
# internally via the Monte Carlo extrema.
from nimare.meta.cbma.ale import BalancedALESubtraction

balanced = BalancedALESubtraction(
    n_iters=N_ITERS,
    n_subsamples=100,  # use ≥2500 for real analyses
    difference_iterations=50,  # use ≥1000 for real analyses
    null_method="random-foci",
    n_cores=1,
)
balanced_result = balanced.fit(knowledge_studyset, related_studyset)

print("BalancedALESubtraction output maps:")
pprint(list(balanced_result.maps.keys()))

###############################################################################
# Visualize BalancedALESubtraction outputs.
fig3, axes3 = plt.subplots(figsize=(14, 12), nrows=4)

plot_stat_map(
    balanced_result.get_map("z_desc-balancedGroup1MinusGroup2"),
    cut_coords=4,
    display_mode="z",
    title="Balanced subtraction z-map: knowledge − relatedness",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=3,
    axes=axes3[0],
    figure=fig3,
)

plot_stat_map(
    balanced_result.get_map("prob_desc-group1"),
    cut_coords=4,
    display_mode="z",
    title="Probability map: knowledge (proportion of subsamples surviving cluster threshold)",
    threshold=0.1,
    cmap="hot",
    symmetric_cbar=False,
    vmax=1,
    axes=axes3[1],
    figure=fig3,
)

plot_stat_map(
    balanced_result.get_map("prob_desc-group2"),
    cut_coords=4,
    display_mode="z",
    title="Probability map: relatedness (proportion of subsamples surviving cluster threshold)",
    threshold=0.1,
    cmap="hot",
    symmetric_cbar=False,
    vmax=1,
    axes=axes3[2],
    figure=fig3,
)

plot_stat_map(
    balanced_result.get_map("stat_desc-conjunction"),
    cut_coords=4,
    display_mode="z",
    title="Conjunction (voxelwise minimum of probability maps)",
    threshold=0.1,
    cmap="hot",
    symmetric_cbar=False,
    vmax=1,
    axes=axes3[3],
    figure=fig3,
)

fig3.show()

###############################################################################
# Side-by-side comparison of all three approaches
# -----------------------------------------------------------------------------
# The three z-maps below show the same contrast (knowledge − relatedness) as
# estimated by each method.  Differences in spatial extent and effect
# magnitude reflect whether directional masking or balanced sampling was
# applied.
fig4, axes4 = plt.subplots(figsize=(14, 9), nrows=3)

plot_stat_map(
    subtraction_result.get_map("z_desc-group1MinusGroup2"),
    cut_coords=4,
    display_mode="z",
    title="ALESubtraction (standalone, uncorrected): all voxels, two-sided",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=4,
    axes=axes4[0],
    figure=fig4,
)

plot_stat_map(
    contrast_result.get_map("z_desc-contrast"),
    cut_coords=4,
    display_mode="z",
    title="ContrastWorkflow (main-effect-gated, FDR α = 0.05)",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=4,
    axes=axes4[1],
    figure=fig4,
)

plot_stat_map(
    balanced_result.get_map("z_desc-balancedGroup1MinusGroup2"),
    cut_coords=4,
    display_mode="z",
    title="BalancedALESubtraction (Monte Carlo thresholding, α = 0.05)",
    threshold=1.65,
    cmap="RdBu_r",
    symmetric_cbar=True,
    vmax=4,
    axes=axes4[2],
    figure=fig4,
)

fig4.show()

###############################################################################
# Key differences at a glance
# -----------------------------------------------------------------------------
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 23 24 23
#
#    * - Feature
#      - ALESubtraction
#      - ContrastWorkflow
#      - BalancedALESubtraction
#    * - Corrects for unequal group sizes
#      - No
#      - No
#      - Yes (matched-size subsampling)
#    * - Computes group main effects
#      - No
#      - Yes (one-sample ALE per group)
#      - Yes (probabilistic maps via subsampling)
#    * - Corrects main effects
#      - No
#      - Yes (FDR by default)
#      - Yes (cluster threshold via Monte Carlo)
#    * - Directional inference masking
#      - No — two-sided across all voxels
#      - Yes — gated by thresholded main effects
#      - No — prior-masked two-sided extrema null
#    * - Separate correction step needed
#      - Yes (FWECorrector / FDRCorrector)
#      - No additional corrector call (thresholding is internal)
#      - No additional corrector call (Monte Carlo extrema thresholding)
#    * - Conjunction output
#      - No
#      - Yes (min of main-effect z-maps)
#      - Yes (min of probability maps)
#    * - Output maps
#      - z, p, logp for the difference
#      - Main effects + conjunction z-maps + thresholded contrast
#      - Balanced z/p/stat + probability maps + conjunction
#    * - Null model options
#      - Label permutation (only)
#      - Label permutation (only)
#      - random-foci or label-permutation
#    * - Matches original GingerALE logic
#      - No
#      - GingerALE-style
#      - No (new method)
#    * - Reference
#      - —
#      - :footcite:t:`laird2005ale`
#      - :footcite:t:`Frahm_Monimu_Hoffstaedter`
#
# **When to use each approach**
#
# * Use :class:`~nimare.meta.cbma.ale.ALESubtraction` (standalone) when you
#   want a whole-mask, two-sided test without main-effect gating — for example
#   during exploratory analyses or when you
#   deliberately want to avoid pre-selecting regions based on each group's main
#   effect.  It is the simplest option and requires applying a corrector to the
#   result as a separate step.  **Avoid it when group sizes differ
#   substantially**, as the larger group will have systematically inflated ALE
#   values.
#
# * Use :class:`~nimare.workflows.cbma.ContrastWorkflow` when you want to
#   mirror the original GingerALE-style subtraction logic and your group sizes are
#   roughly equal.  Restricting directional inference to regions where the
#   respective group shows reliable activation reduces the risk of interpreting
#   differences in regions that neither group activates consistently, and
#   produces interpretable intermediate outputs (main effects, conjunction)
#   alongside the final contrast.  This is the closest NiMARE workflow for
#   preregistered analyses that need GingerALE-style main-effect gating.
#
# * Use :class:`~nimare.meta.cbma.ale.BalancedALESubtraction` when the two
#   groups have **unequal numbers of studies**, or when you want a method that
#   explicitly quantifies activation probability per group rather than just
#   thresholded significance.  The probabilistic activation maps and matched-
#   size subsampling make this the most robust option for asymmetric
#   literature searches.  The internal Monte Carlo thresholding means no
#   separate corrector call is needed, but computation time is substantially
#   higher than the other two methods (controlled by ``n_subsamples``,
#   ``difference_iterations``, and ``n_iters``).

###############################################################################
# Boilerplate text and references
# -----------------------------------------------------------------------------
print("ContrastWorkflow description:")
pprint(contrast_result.description_)
print("\nBalancedALESubtraction description:")
pprint(balanced_result.description_)

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
