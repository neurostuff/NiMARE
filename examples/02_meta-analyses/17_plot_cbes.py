r"""
.. _metas_cbes:

==========================================================
Effect sizes from coordinates, using what studies left out
==========================================================

A tour of coordinate-based effect-size meta-analysis (CBES) in NiMARE.

Most coordinate-based methods ask *where* studies agree. CBES asks *how large* the effect is, and
it gets there from an unusual direction: the coordinate tables are read only for whether a study
reported near a voxel, never for the statistic it printed there.

The reason is that a reported peak height is a poor estimate of the effect at its own location --
it was selected for being large, and it sits wherever the noise pushed the maximum. But the same
selection makes *silence* informative. If a study reported nothing near a voxel, its effect there
did not clear that study's reporting threshold, and that is a genuine upper bound. Collect those
bounds across a literature and they constrain the magnitude.

So CBES needs two kinds of input at once:

* at least one study supplying effect-size **images** (``g`` and ``g_var``), which carry the
  magnitude;
* coordinate **tables** from the rest, which carry the reporting pattern.

The model is a zero-inflated censored likelihood. It separates *how big* the effect is where it
is present (:math:`\mu`, reported as ``g``) from *how often* it is present across studies
(:math:`\pi`, reported as ``prevalence``), and it reports their product as ``g_marginal``.

Which of those to read is a question about the estimand rather than about accuracy, and this
example ends by making the distinction concrete.
"""

import tempfile

import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_stat_map

from nimare.generate import create_effect_size_coordinate_studyset
from nimare.meta.cbma import CBES

###############################################################################
# Simulate a collection with both channels
# -----------------------------------------------------------------------------
# The field simulator builds each study's whole statistic map, thresholds it, and reports the
# local maxima that survive -- so the coordinates are produced by the same selection a paper
# applies. ``n_image_studies=2`` additionally writes ``g``/``g_var`` maps for the first two
# studies, which is what a collection with a couple of shared maps looks like.
#
# Four foci are planted with true effects of 0.2, 0.4, 0.6 and 0.8, so there is a weak effect
# near the detection threshold and a strong one well above it.

FOCI = [(-28, -28, 0), (28, -28, 0), (-28, 28, 0), (28, 28, 0)]
EFFECTS = [0.2, 0.4, 0.6, 0.8]

studyset = create_effect_size_coordinate_studyset(
    FOCI,
    effect_sizes=EFFECTS,
    n_studies=20,
    sample_size=(20, 40),
    tau=0.1,
    seed=7,
    simulate_field=True,
    n_image_studies=2,
    image_dir=tempfile.mkdtemp(),
    noise_extent=48.0,
    field_zooms=4.0,
    blob_fwhm=10.0,
)

n_images = int(studyset.images["g"].notna().sum())
n_foci = len(studyset.coordinates)
print(f"{len(studyset.studies)} studies: {n_images} with images, {n_foci} reported foci")

###############################################################################
# Fit the estimator
# -----------------------------------------------------------------------------
# ``threshold`` is the one number a silence cannot do without: "study k reported nothing here" is
# only evidence about the effect if we know how large an effect would have had to be for k to
# report it. It cannot be inferred from the table, so it is supplied or assumed. The simulator
# records each study's own cut in metadata, so we point at that field; on real data, pass a float
# for the conventional p < 0.001, or the name of a metadata field holding per-study values.
#
# ``null_method="none"`` skips the permutation null to keep the example quick. A real analysis
# leaves it at the default, where each permutation runs a full EM -- so ``n_cores`` is the lever
# that matters for runtime.

estimator = CBES(threshold="reporting_threshold", null_method="none")
results = estimator.fit(studyset)

print("maps:", ", ".join(sorted(results.maps)))

###############################################################################
# What the maps mean
# -----------------------------------------------------------------------------
# ``g`` is the effect size conditional on the effect being present in a study. ``prevalence`` is
# the fraction of studies carrying it, and ``g_marginal`` their product. ``coordinate_share``
# says where the coordinate channel actually acted: 0 means the images carried the estimate
# alone at that voxel, so none of the coordinate caveats apply there.

plot_stat_map(
    results.get_map("g"),
    cut_coords=[0],
    display_mode="z",
    title="g: effect size where present",
    draw_cross=False,
    cmap="RdBu_r",
)
plt.show()

plot_stat_map(
    results.get_map("prevalence"),
    cut_coords=[0],
    display_mode="z",
    title="prevalence: fraction of studies carrying the effect",
    draw_cross=False,
    cmap="viridis",
    vmax=1.0,
)
plt.show()

###############################################################################
# Recovering the planted effects
# -----------------------------------------------------------------------------
# Reading the estimate at each planted focus shows what the two channels together recover. The
# weakest focus sits near the reporting threshold, which is the hardest regime: few studies
# report it, so the silences push the estimate down.

masker = estimator.masker
g_map = results.get_map("g", return_type="array").ravel()
share = results.get_map("coordinate_share", return_type="array").ravel()
prevalence = results.get_map("prevalence", return_type="array").ravel()

affine = masker.mask_img.affine
mask_flat = np.asarray(masker.mask_img.get_fdata() > 0).ravel()
index_of = np.cumsum(mask_flat) - 1

print(f"{'true g':>8} {'g':>8} {'prevalence':>11} {'share':>7}")
for focus, truth in zip(FOCI, EFFECTS):
    ijk = np.rint(np.linalg.inv(affine) @ np.array([*focus, 1.0]))[:3].astype(int)
    flat = np.ravel_multi_index(tuple(ijk), masker.mask_img.shape)
    voxel = int(index_of[flat])
    print(f"{truth:8.2f} {abs(g_map[voxel]):8.3f} {prevalence[voxel]:11.3f} {share[voxel]:7.3f}")

###############################################################################
# Which map answers your question
# -----------------------------------------------------------------------------
# An image-based meta-analysis pools per-study effect maps. Where a fraction :math:`\pi` of
# studies carry an effect :math:`\mu` and the rest carry zero, that pool targets the population
# mean :math:`\pi\mu` -- so **``g_marginal`` is the map comparable to an IBMA, and ``g`` is a
# quantity no image-only method estimates.**
#
# The catch is that :math:`\pi` is the weakest part of the fit. It tracks the true prevalence in
# order but not in level, and it is biased low, so ``g_marginal`` estimates the right quantity
# with a biased multiplier. Read ``g_marginal`` when comparing against an image-based
# meta-analysis or when studies genuinely differ in whether they carry the effect; read ``g``
# when the question is how large the effect is where it is present.

marginal = results.get_map("g_marginal", return_type="array").ravel()
strong = np.abs(g_map) > np.percentile(np.abs(g_map), 99)
print(
    f"at the strongest 1% of voxels: g {np.mean(np.abs(g_map[strong])):.3f}, "
    f"g_marginal {np.mean(np.abs(marginal[strong])):.3f}, "
    f"prevalence {np.mean(prevalence[strong]):.3f}"
)

###############################################################################
# Powering a new study
# -----------------------------------------------------------------------------
# This is the use case that most clearly wants ``g`` rather than ``g_marginal``, and the reason
# is the definition of power. Power is computed *conditional on the alternative being true*: if
# my study has this effect, how likely am I to detect it? The effect size that question needs is
# the effect among studies that have it, which is exactly :math:`\mu`.
#
# Using ``g_marginal`` instead would fold "the effect might be absent in my study" into the
# effect size itself, mixing two different reasons for a small number and under-powering you if
# the effect is in fact present. Plan with ``g``; if you want to reason about the chance your
# paradigm elicits the effect at all, that is ``prevalence``, and it belongs in the conclusion
# rather than inside the effect size.
#
# Three cautions specific to planning from a meta-analysis:
#
# * ``g`` is biased *low* where the effect is weak, because silences push it down. For powering
#   that is the safe direction -- a conservative planning value -- but it does mean the weakest
#   effects here are underestimates rather than estimates.
# * Do not take the interval as a sensitivity range. As the next section shows, it is unbounded
#   at most voxels, so ``g`` is a planning value and not a confidence statement.
# * Choosing the voxel *because* its estimate is the largest re-introduces the selection this
#   estimator exists to correct. Pick the location from anatomy or a pre-registered hypothesis,
#   then read ``g`` there.

###############################################################################
# Uncertainty, and a warning about reading it
# -----------------------------------------------------------------------------
# ``se`` comes from the observed information of the censored likelihood with the prevalence
# profiled out, and it is conservative: against a known truth it runs 1.2 to 2.2 times the
# estimator's actual spread. Most of that is the cost of not knowing :math:`\pi`.
#
# ``interval="profile"`` reports the likelihood region instead, and it is honest in a way that
# may surprise: the bounds are **infinite** wherever the data do not reject :math:`\pi = 0`,
# because "an arbitrarily large effect present in almost no studies" then fits about as well as
# the estimate does. At two image studies that is most of the brain. An unbounded interval is the
# correct answer there, not a failure of the search.
#
# Coverage will not reveal any of this -- every configuration measured covers 0.95 to 1.00,
# including those whose interval admits almost any magnitude. Read the width.

bounded = CBES(threshold="reporting_threshold", null_method="none", interval="profile")
with_interval = bounded.fit(studyset)
lower = with_interval.get_map("g_lower", return_type="array").ravel()
upper = with_interval.get_map("g_upper", return_type="array").ravel()
finite = np.isfinite(lower) & np.isfinite(upper)
print(f"profile interval is bounded at {100 * finite.mean():.1f}% of voxels")

###############################################################################
# When not to use this
# -----------------------------------------------------------------------------
# The correction can make ``g`` worse than pooling the images alone, and the regime where it does
# is not exotic: it is when *every* study really carries the effect. There is then no absence for
# the model to find, and any :math:`\hat\pi < 1` is error. Judged against held-out subjects in
# that regime, an images-only pool recovered 0.85 of the true magnitude where ``g`` recovered
# 0.63.
#
# A magnitude is also only recoverable when the studies differ in *size*. A silence constrains
# :math:`\pi` and :math:`\mu` only through one combination per distinct (sampling error, cutoff)
# pair, and varying the threshold moves both mixture components together while varying the
# sample size moves only one. So a literature of uniformly sized studies leaves the two
# parameters entangled however many studies it has -- which is the opposite of treating
# heterogeneity as a nuisance.
