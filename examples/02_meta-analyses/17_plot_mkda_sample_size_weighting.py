r"""

.. _metas_mkda_weighting:

=============================
Sample-size weighting in MKDA
=============================

Weighting each study contrast by the square root of its sample size is what the
"multilevel" in Multilevel Kernel Density Analysis refers to. A contrast from 90
subjects says more about the population than one from 9, and MKDA encodes that by
weighting the contrast indicator maps before averaging them
:footcite:p:`wager2007meta,wager2009evaluating`:

.. math::

    w_c \propto \delta_c \sqrt{N_c}

where :math:`N_c` is the contrast's sample size and :math:`\delta_c` optionally
discounts contrasts analysed with a fixed-effects study-level model.

This example shows how to turn the weighting on, what it changes in the map, and how
NiMARE's weights compare against the reference MATLAB implementation.

Weighting is opt-in: :class:`~nimare.meta.cbma.mkda.MKDADensity` with no arguments
weights every contrast equally, so existing analyses are unaffected.
"""

###############################################################################
# Load a Studyset
# -----------------------------------------------------------------------------
# Weighting needs a sample size for every contrast. The NIDM pain Studyset shipped
# with NiMARE records one, which most coordinate corpora do not -- see the last
# section.
import os

import numpy as np
import pandas as pd
from nilearn.plotting import plot_stat_map

from nimare.meta.cbma import MKDADensity
from nimare.meta.cbma.weights import StudyWeights, normalize_weights
from nimare.nimads import Studyset
from nimare.utils import get_resource_path

studyset_file = os.path.join(get_resource_path(), "nidm_pain_studyset.json")
studyset = Studyset(studyset_file, target="mni152_2mm")

sample_sizes = {
    row["id"]: int(np.mean(row["sample_sizes"]))
    for _, row in studyset.metadata.iterrows()
    if isinstance(row["sample_sizes"], (list, tuple, np.ndarray))
}
print(
    f"{len(studyset.ids)} contrasts, "
    f"sample sizes {min(sample_sizes.values())}-{max(sample_sizes.values())}"
)

###############################################################################
# The default: every contrast weighted equally
# -----------------------------------------------------------------------------
# ``MKDADensity()`` gives every contrast a weight of exactly 1.0, so its summary
# statistic is the *count* of contrasts activating a voxel.
unweighted = MKDADensity(kernel__r=10.0)
res_unweighted = unweighted.fit(studyset)

weights = np.asarray(unweighted.weight_vec_).ravel()
print("distinct weights:", np.unique(weights))
print("sum:", weights.sum(), "= number of contrasts")

###############################################################################
# Turning the weighting on
# -----------------------------------------------------------------------------
# ``weighting="sample_size"`` is shorthand for ``StudyWeights()``, which reads the
# collection's ``sample_sizes`` metadata and applies :math:`\sqrt{N}`.
weighted = MKDADensity(weighting="sample_size", kernel__r=10.0)
res_weighted = weighted.fit(studyset)

ids = list(weighted.inputs_["id"])
table = pd.DataFrame(
    {
        "id": ids,
        "N": [sample_sizes[study_id] for study_id in ids],
        "weight": np.asarray(weighted.weight_vec_).ravel(),
    }
)
table["sqrt_N"] = np.sqrt(table["N"])

print(f"weights sum to {table['weight'].sum():.6f} (= {len(table)} contrasts)")
print(
    f"spread: {table['weight'].min():.4f} to {table['weight'].max():.4f}, "
    f"a factor of {table['weight'].max() / table['weight'].min():.2f}"
)
print(table.sort_values("N").head(8).to_string(index=False))

###############################################################################
# The weights are normalised to sum to the number of contrasts, so an unweighted
# analysis is the special case where all of them are 1.0. That keeps the statistic on
# the same scale either way; dividing by the contrast count recovers the weighted
# *proportion* of :footcite:t:`wager2009evaluating`.
#
# Note the weights go as :math:`\sqrt{N}`, not as :math:`N`. The largest study here
# has 3.6x the subjects of the smallest but only 1.9x the weight, which is the point
# of the square root: it keeps one large study from dominating the map.
ratio = table["weight"] / table["sqrt_N"]
print(f"weight / sqrt(N) is constant: {ratio.min():.9f} to {ratio.max():.9f}")

###############################################################################
# What it changes in the map
# -----------------------------------------------------------------------------
# Weighting moves which voxels come out on top, because a cluster supported by three
# small studies now counts for less than one supported by three large ones.
for name, result in [("Unweighted", res_unweighted), ("sqrt(N) weighted", res_weighted)]:
    plot_stat_map(
        result.get_map("z"),
        cut_coords=[0, 0, -8],
        draw_cross=False,
        cmap="RdBu_r",
        symmetric_cbar=True,
        threshold=1.65,
        title=name,
    )

###############################################################################
# The two maps are highly correlated -- weighting rescales the same contrasts rather
# than changing which ones enter -- but the ranking of voxels does move, and any
# threshold acts on that ranking.
stat_u = unweighted.masker.transform(res_unweighted.get_map("stat")).ravel()
stat_w = weighted.masker.transform(res_weighted.get_map("stat")).ravel()

both = (stat_u > 0) | (stat_w > 0)
print(f"Pearson r : {np.corrcoef(stat_u[both], stat_w[both])[0, 1]:.6f}")
print(
    "Spearman  : "
    f"{pd.Series(stat_u[both]).corr(pd.Series(stat_w[both]), method='spearman'):.6f}"
)
print(f"peak      : {stat_u.max():.4f} unweighted vs {stat_w.max():.4f} weighted")

###############################################################################
# Other ways to set the weights
# -----------------------------------------------------------------------------
# :class:`~nimare.meta.cbma.weights.StudyWeights` covers the variations of the
# published method. Pass one instead of the ``"sample_size"`` shorthand.

# sqrt(N) is the published transform; "linear" weights by N itself, which is not
# what Wager et al. describe.
linear = StudyWeights(transform="linear")

# Wager et al. discount fixed-effects contrasts. Name the metadata field that records
# each contrast's study-level model; contrasts it does not label are left alone. This
# Studyset has no such field, so the discount is only shown, not applied below --
# NiMARE will not guess which contrasts were fixed-effects, because guessing wrong
# biases every voxel.
ffx = StudyWeights(inference_field="inference", fixed_effects_discount=0.75)
print(f"fixed-effects discount: {ffx.fixed_effects_discount} on field {ffx.inference_field!r}")

# Explicit per-study weights bypass the sample size entirely -- the route for the
# "other study quality measures" the paper mentions.
explicit = StudyWeights(source={study_id: 1.0 for study_id in studyset.ids})

for name, spec in [("sqrt(N)", StudyWeights()), ("linear N", linear), ("explicit", explicit)]:
    raw = spec.raw_weights(studyset, ids)
    w = normalize_weights(raw.to_numpy(), len(ids))
    print(f"{name:10s} min={w.min():.4f} max={w.max():.4f} sum={w.sum():.4f}")

###############################################################################
# Does this match the reference implementation?
# -----------------------------------------------------------------------------
# NiMARE's weighting is meant to reproduce CanLab's ``Canlab_MKDA_MetaAnalysis``
# MATLAB toolbox, so it is worth checking rather than assuming. The two compute the
# same quantity and normalise it differently.
#
# CanLab, ``densityUtility3/Meta_Setup.m``::
#
#     DB.rootn = sqrt(DB.Subjects);
#     w = DB.rootn .* DB.SubjectiveWeights(DB.pointind);   % or just DB.rootn
#     DB.studyweight = w ./ sum(w);                        % sums to 1
#
# NiMARE, :func:`~nimare.meta.cbma.weights.normalize_weights`::
#
#     raw = discount * transform(sample_size)              # delta_c * sqrt(N_c)
#     weight_vec = n_studies * raw / raw.sum()             # sums to n_studies
#
# So CanLab reports a weighted proportion and NiMARE a weighted sum. If the relative
# weights agree, ``nimare_weight / n_contrasts`` should equal CanLab's ``studyweight``.
# The values below were produced by running ``Meta_Setup`` on this same Studyset under
# Octave and dumping ``DB.studyweight``; they are inlined so this example needs no
# MATLAB.
CANLAB_STUDYWEIGHT = {
    "pain_01.nidm-1": 0.0606990726388009,
    "pain_02.nidm-1": 0.0606990726388009,
    "pain_03.nidm-1": 0.0542909010366226,
    "pain_04.nidm-1": 0.0542909010366226,
    "pain_05.nidm-1": 0.0364194435832806,
    "pain_06.nidm-1": 0.0364194435832806,
    "pain_07.nidm-1": 0.0364194435832806,
    "pain_08.nidm-1": 0.0420535511130868,
    "pain_09.nidm-1": 0.0420535511130868,
    "pain_10.nidm-1": 0.0420535511130868,
    "pain_11.nidm-1": 0.0420535511130868,
    "pain_12.nidm-1": 0.0437707237544620,
    "pain_13.nidm-1": 0.0686731613994095,
    "pain_14.nidm-1": 0.0594727023300776,
    "pain_15.nidm-1": 0.0454230267018595,
    "pain_16.nidm-1": 0.0454230267018595,
    "pain_17.nidm-1": 0.0420535511130868,
    "pain_18.nidm-1": 0.0420535511130868,
    "pain_19.nidm-1": 0.0485592581110407,
    "pain_20.nidm-1": 0.0485592581110407,
    "pain_21.nidm-1": 0.0485592581110407,
}

comparison = table.copy()
comparison["nimare_proportion"] = comparison["weight"] / len(comparison)
comparison["canlab"] = comparison["id"].map(CANLAB_STUDYWEIGHT)
comparison["abs_diff"] = (comparison["nimare_proportion"] - comparison["canlab"]).abs()

print(f"NiMARE weights sum to {comparison['weight'].sum():.12f}")
print(f"CanLab weights sum to {comparison['canlab'].sum():.12f}")
print(f"max |NiMARE/n - CanLab| = {comparison['abs_diff'].max():.3e}")
print(f"machine epsilon         = {np.finfo(float).eps:.3e}")

assert np.allclose(comparison["nimare_proportion"], comparison["canlab"], rtol=0, atol=1e-12)
print("\nThe weights are identical up to the normalisation convention.")

###############################################################################
# That is a statement about the weights only, not about the rest of MKDA. The two
# implementations still rasterise their spheres slightly differently, so the maps are
# not bit-identical even though the weights are.

###############################################################################
# When sample sizes are missing
# -----------------------------------------------------------------------------
# Most coordinate corpora record a sample size for only a small fraction of their
# analyses, so this is the common case rather than an edge case. A contrast with a
# missing or non-positive sample size raises by default, because a substituted weight
# is invisible in the output map.
incomplete = studyset.to_dict()
incomplete["studies"][0]["analyses"][0]["metadata"]["sample_sizes"] = None
incomplete = Studyset(incomplete, target="mni152_2mm")

try:
    MKDADensity(weighting="sample_size").fit(incomplete)
except ValueError as exc:
    print(f"ValueError: {exc}")

###############################################################################
# Pass ``on_missing="impute"`` to give those contrasts the mean weight of the rest
# instead, which is what the CanLab implementation does. It keeps the weighted and
# unweighted analyses over the same study set, at the cost of weighting some
# contrasts by a number that is not theirs.
imputing = StudyWeights(on_missing="impute")
raw = imputing.raw_weights(incomplete, list(incomplete.ids))
print(f"{imputing.n_imputed_} contrast(s) imputed, weight {raw.iloc[0]:.4f}")

###############################################################################
# Sample size is also what ALE uses to set its kernel width and what SDM needs for
# its effect-size conversion, so a corpus without it limits more than MKDA.

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
