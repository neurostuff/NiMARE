r"""
.. _metas_cbmr:

====================================================
Coordinate-based meta-regression with a model formula
====================================================

A tour of coordinate-based meta-regression (CBMR) in NiMARE.

CBMR estimates a smooth activation-intensity function from reported coordinates. The model is
written as a formula, in which every term states its own *spatial resolution*::

    CBMR("~ s(diagnosis) + sample_size")

``s()`` crosses a term with the spatial spline basis, so its coefficient becomes a map. Without
it, the term gets a single coefficient that applies to the whole brain. That distinction is the
entire difference between what CBMR used to call global and voxelwise moderator effects, and
because each term declares it separately, a model can freely mix the two.

The parameter cost follows from the same mark. At ``spline_spacing=10`` on the 2 mm brain mask
the basis has 457 columns, so every ``s()`` term costs 457 coefficients per column -- as much as
another group's entire baseline map. That is worth knowing before adding one, which is why the
budget is logged at fit time and available from
:meth:`~nimare.meta.cbmr.CBMRResult.describe_terms`.
"""

import numpy as np
import scipy
from nilearn.plotting import plot_stat_map

from nimare.correct import FDRCorrector
from nimare.generate import create_coordinate_studyset
from nimare.meta.cbmr import CBMR
from nimare.transforms import StandardizeField

###############################################################################
# Simulate a Studyset
# -----------------------------------------------------------------------------
# A coordinate-based Studyset with reported foci, sample sizes, diagnosis and drug-status
# labels, and two continuous moderators. Coarse B-spline spacing keeps the example quick; a real
# analysis would use ``spline_spacing`` of 10 or 5.

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

FIT_KWARGS = dict(
    spline_spacing=100,  # a reasonable analysis choice is 10 or 5; 100 is for speed
    n_iter=200,
    lr=1e-1,
    tol=1e3,  # a reasonable analysis choice is 1e-2; 1e3 is for speed
    device="cpu",  # use "cuda" if you have a GPU
    random_state=100,
)

###############################################################################
# One spatial map per group
# -----------------------------------------------------------------------------
# ``s(diagnosis:drug_status)`` crosses the two factors and gives every combination of levels its
# own intensity map. This is the model the old ``group_categories=["diagnosis", "drug_status"]``
# argument produced -- note that it was always a full interaction, which the formula now says
# out loud.

group_results = CBMR("~ s(diagnosis:drug_status)", **FIT_KWARGS).fit(dataset=studyset)

print(group_results.describe_terms())

plot_stat_map(
    group_results.get_map("spatialIntensity_group-schizophrenia-Yes"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Schizophrenia, on drug treatment",
    threshold=1e-4,
    vmax=1e-3,
)
plot_stat_map(
    group_results.get_map("spatialIntensity_group-depression-No"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Depression, no drug treatment",
    threshold=1e-4,
    vmax=1e-3,
)

###############################################################################
# Testing hypotheses by name
# -----------------------------------------------------------------------------
# Hypotheses are written over the terms and levels of the design. This replaces passing contrast
# matrices positionally, which was unreadable and silently depended on level ordering -- reorder
# the levels and the same matrix tested a different hypothesis.

drug_effect = group_results.test(
    "schizophrenia-Yes = schizophrenia-No",
    name="schizophrenia-drug",
)

plot_stat_map(
    drug_effect.get_map("z_schizophrenia-drug"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Drug effect within schizophrenia",
    threshold=scipy.stats.norm.isf(0.4),
    vmax=2,
)

###############################################################################
# A list of statements is tested jointly, as a generalized linear hypothesis, rather than one at
# a time. This asks whether the drug effect is zero in *both* diagnoses at once.

joint = group_results.test(
    ["schizophrenia-Yes = schizophrenia-No", "depression-Yes = depression-No"],
    name="drug-anywhere",
)

plot_stat_map(
    joint.get_map("chiSquare_drug-anywhere"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="Reds",
    title="Joint test: any drug effect",
)

###############################################################################
# Robust standard errors and multiple comparisons
# -----------------------------------------------------------------------------
# The default standard errors come from the Fisher information, which is correct only if the
# Poisson mean-variance relationship holds. Foci are overdispersed and correlated within an
# experiment, so ``method="sandwich"`` is usually the safer choice; ``meat="cluster"`` allows
# arbitrary correlation among one experiment's own foci.

robust = group_results.test(
    "schizophrenia-Yes = schizophrenia-No",
    name="drug-robust",
    method="sandwich",
    meat="cluster",
    correction="hc1",
)

corrected = FDRCorrector(method="indep", alpha=0.05).transform(robust)

plot_stat_map(
    corrected.get_map("z_drug-robust_corr-FDR_method-indep"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Drug effect, clustered SEs, FDR corrected",
    threshold=scipy.stats.norm.isf(0.4),
    vmax=2,
)

###############################################################################
# Moderators, global and spatially varying
# -----------------------------------------------------------------------------
# Whether a moderator gets one coefficient or a map is decided by ``s()``, per term. Here sample
# size is allowed only to scale the whole map, while age is allowed to reshape it.
#
# Under the log link a global moderator can *only* rescale the intensity -- it multiplies every
# voxel by the same factor. A spatially varying one can change the pattern. That is the real
# distinction, and it is why the two answers arrive in different places: a scalar coefficient in
# a table, a coefficient map among the maps.

mixed_results = CBMR(
    "~ s(diagnosis:drug_status) + standardized_sample_sizes + s(standardized_avg_age)",
    **FIT_KWARGS,
).fit(dataset=studyset)

print(mixed_results.describe_terms())
print(mixed_results.tables["moderatorEffect_standardized_sample_sizes"])

plot_stat_map(
    mixed_results.get_map("voxelwiseModeratorEffect_standardized_avg_age"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Age effect on log intensity (per SD)",
)

###############################################################################
# Reading a moderator map: relative intensity and intensity difference
# -----------------------------------------------------------------------------
# A moderator's coefficient map is a derivative of *log* intensity, which is hard to interpret
# directly. Two derived scales help, for a stated change in the moderator:
#
# * **Relative Intensity (RI)** -- ``exp(unit * coefficient)``, the multiplicative factor on the
#   intensity. An RI of 1.2 means 20% more foci expected at that voxel.
# * **Intensity Difference (ID)** -- ``baseline * (RI - 1)``, the same effect in foci.
#
# ID is the one to threshold. RI is large wherever the baseline is small, so a striking ratio in
# a region nobody reports is not a finding. Plotting them together is the point.

mixed_results.plot_moderator_effects(
    moderator="standardized_avg_age",
    unit_change=1.0,
    group="schizophrenia-Yes",
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

###############################################################################
# Additive spatial factors
# -----------------------------------------------------------------------------
# ``s(diagnosis:drug_status)`` gives every cell a free map. The additive alternative says the two
# factors shift one underlying map independently -- a stronger claim, and far cheaper.
#
# Writing it as ``s(diagnosis) + s(drug_status)`` does not work, and CBMR refuses it: each
# cell-means spatial factor's columns sum to the constant, so their difference is exactly zero
# and the design is rank deficient by a whole basis width whatever the data. ``sz()`` is the
# identified form, after mgcv's ``bs="sz"`` basis. It constrains each factor's coefficients to
# sum to zero across levels, so they measure deviations from a shared baseline instead of
# competing with it.

additive_results = CBMR("~ sz(diagnosis) + sz(drug_status)", **FIT_KWARGS).fit(dataset=studyset)

print(additive_results.describe_terms())

plot_stat_map(
    additive_results.get_map("spatialIntensity_group-Default"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Shared baseline intensity",
    threshold=1e-4,
    vmax=1e-3,
)
plot_stat_map(
    additive_results.get_map("spatialFactorEffect_diagnosis-sz1"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
    title="Diagnosis deviation from the baseline",
)

###############################################################################
# Overdispersion
# -----------------------------------------------------------------------------
# The Poisson model cannot represent excess variance in foci counts. Two alternatives can:
# ``"negativebinomial"``, whose latent variation is independent at each voxel, and
# ``"clusterednegativebinomial"``, whose latent effect belongs to an experiment and is shared
# across the whole brain.
#
# Both are defined on marginals of a mean that factorizes into a spatial part and an
# experiment-level part, so both need *several experiments sharing a spatial map*. A design with
# a continuously varying spatial term -- ``s(avg_age)`` -- gives every experiment its own map and
# is refused, with an error saying so. This is a property of those likelihoods rather than a gap
# in the implementation.

overdispersed = CBMR(
    "~ s(diagnosis:drug_status)",
    distribution="negativebinomial",
    **{**FIT_KWARGS, "lr": 1e-2},
).fit(dataset=studyset)

print(overdispersed.tables["overdispersion"])

###############################################################################
# Migrating from the old interface
# -----------------------------------------------------------------------------
# The five moderator arguments and the ``moderator_effect`` switch are replaced by the formula:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Old
#      - New
#    * - ``CBMREstimator()``
#      - ``CBMR("~ 1")``
#    * - ``group_categories=["a", "b"]``
#      - ``CBMR("~ s(a:b)")``
#    * - ``moderators=["n"], moderator_effect="global"``
#      - ``CBMR("~ s(a:b) + n")``
#    * - ``moderators=["n"], moderator_effect="voxelwise"``
#      - ``CBMR("~ s(a:b) + s(a:b:n)")``
#    * - ``moderator_effect="mixed", global_moderators=["n"], voxelwise_moderators=["age"]``
#      - ``CBMR("~ s(a:b) + n + s(age)")``
#    * - ``model=models.NegativeBinomialEstimator``
#      - ``distribution="negativebinomial"``
#    * - ``infer(group_contrasts=[[[1, -1, 0, 0]]])``
#      - ``result.test("a1-b1 = a1-b2")``
#
# Note the fourth row. The old voxelwise mode keyed moderator coefficients by group, so it was
# always the group-crossed form ``s(a:b:n)``. A moderator map *pooled* across groups, ``s(n)``,
# had no representation at all; nor did a group-specific scalar slope, ``a:n``, since global
# moderator coefficients were shared by construction. Both are now writable.
