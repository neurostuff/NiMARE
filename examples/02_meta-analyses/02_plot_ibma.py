"""

.. _metas_ibma:

====================================
Image-based meta-analysis algorithms
====================================

A tour of IBMA algorithms in NiMARE.

This tutorial is intended to provide a brief description and example of each of
the IBMA algorithms implemented in NiMARE.
For a more detailed introduction to the elements of an image-based
meta-analysis, see other stuff.
"""

from nilearn.plotting import plot_stat_map

###############################################################################
# Download data
# -----------------------------------------------------------------------------
# .. note::
#   The data used in this example come from a collection of NIDM-Results packs
#   downloaded from Neurovault collection 1425, uploaded by Dr. Camille Maumet.
from nimare.extract import download_nidm_pain

dset_dir = download_nidm_pain()

###############################################################################
# Load Studyset
# -----------------------------------------------------------------------------
import os
from pprint import pprint

from nimare.nimads import Studyset
from nimare.transforms import ImageTransformer
from nimare.utils import get_resource_path

studyset_file = os.path.join(get_resource_path(), "nidm_pain_studyset.json")
studyset = Studyset(studyset_file, target="mni152_2mm")
studyset.update_path(dset_dir)

# Calculate missing images
xformer = ImageTransformer(target=["varcope", "z"])
studyset = xformer.transform(studyset)

###############################################################################
# Stouffer's
# -----------------------------------------------------------------------------
from nimare.meta.ibma import Stouffers

meta = Stouffers(use_sample_size=False)
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)

###############################################################################
# Stouffer's with weighting by sample size
# -----------------------------------------------------------------------------
meta = Stouffers(use_sample_size=True)
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)

###############################################################################
# Fisher's
# -----------------------------------------------------------------------------
from nimare.meta.ibma import Fishers

meta = Fishers()
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)

###############################################################################
# Fisher's with weighting by sample size
# -----------------------------------------------------------------------------
# Each study receives a weighted-Fisher coefficient equal to its sample size. Note that
# this is a different weighting family from Stouffer's ``use_sample_size``, which uses
# the square root of the sample size.
meta = Fishers(use_sample_size=True)
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)

###############################################################################
# Images that share participants
# -----------------------------------------------------------------------------
# Two maps from the same participants are not two independent pieces of evidence, but
# every estimator above would count them as such unless told otherwise. The ``groupby``
# parameter says which images belong together: by default, those from the same study.
#
# Every study in this studyset contributed exactly one image, so the default grouping
# changes nothing here -- the ``dof`` map counts one degree of freedom per study, minus one.
import numpy as np

meta = Stouffers()
results = meta.fit(studyset)
n_images = len(meta.inputs_["id"])
dof = results.get_map("dof", return_type="array")
print(f"{n_images} images, {np.nanmax(dof):.0f} degrees of freedom")

###############################################################################
# A real studyset usually does repeat: one paper uploads several contrasts, or the same
# participants appear under two task conditions. To see what that costs, pretend the first
# six images came from three studies of two maps each by passing the labels directly.
# ``groupby`` accepts one label per image, in the order the estimator collected them.
labels = [f"pretend-study-{i // 2}" for i in range(6)]
labels += list(meta.inputs_["id"][6:])

grouped = Stouffers(groupby=labels).fit(studyset)
grouped_dof = grouped.get_map("dof", return_type="array")
print(f"after grouping: {np.nanmax(grouped_dof):.0f} degrees of freedom")

plot_stat_map(
    grouped.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

###############################################################################
# Grouping does two things. It stops a prolific study from outvoting its peers, because
# each group contributes one variance-standardized statistic rather than one per map. And
# it corrects the reference distribution for the correlation between the grouped maps,
# which NiMARE estimates from the maps themselves after removing the shared signal.
#
# Pass ``groupby=False`` to opt out and treat every image as independent. That inflates
# significance whenever images really do share participants, so NiMARE warns when the
# grouping it resolved would have found a repeat.

###############################################################################
# Permuted OLS
# -----------------------------------------------------------------------------
from nimare.correct import FWECorrector
from nimare.meta.ibma import PermutedOLS

meta = PermutedOLS(two_sided=True)
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

corrector = FWECorrector(method="montecarlo", n_iters=100, n_cores=1)
cresult = corrector.transform(results)

plot_stat_map(
    cresult.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(cresult.description_)
print("References:")
pprint(cresult.bibtex_)

###############################################################################
# Sample-size-weighted permuted OLS
# -----------------------------------------------------------------------------
# When one study contributes several beta maps, treating them as independent gives
# that study disproportionate weight. ``PermutedOLS`` groups by ``study_id`` by
# default: one mean contribution per study, sign-flipped as a whole exchangeability
# block. ``use_sample_size=True`` also weights each study by its participant count,
# with a CR2 cluster-robust variance.
meta = PermutedOLS(two_sided=True, use_sample_size=True)
results = meta.fit(studyset)
cresult = corrector.transform(results)

plot_stat_map(
    cresult.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(cresult.description_)
print("References:")
pprint(cresult.bibtex_)

###############################################################################
# Weighted Least Squares
# -----------------------------------------------------------------------------
from nimare.meta.ibma import WeightedLeastSquares

meta = WeightedLeastSquares(tau2=0)
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)

###############################################################################
# DerSimonian-Laird
# -----------------------------------------------------------------------------
from nimare.meta.ibma import DerSimonianLaird

meta = DerSimonianLaird()
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)

###############################################################################
# Hedges
# -----------------------------------------------------------------------------
from nimare.meta.ibma import Hedges

meta = Hedges()
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)


###############################################################################
# Why that map looks washed out: beta units
# -----------------------------------------------------------------------------
# ``Hedges`` and ``DerSimonianLaird`` fit raw ``beta`` maps, and in this studyset those are
# not on a common scale. Each study reports betas in whatever units its pipeline produced:
# the per-study spatial standard deviation spans a factor of ~2300 here, and the varcopes
# track it (the ratio between the two varies only ~2.5x), so this is a difference in units,
# not in precision.
#
# Having varcopes does not rescue it. ``tau2`` is estimated from the observed spread of the
# betas, which those units dominate -- its median here is ~2830, orders of magnitude above
# the varcopes, against ~0.05 for the same studies expressed as Hedges’ g. Once ``tau2``
# swamps every varcope, the weights ``1 / (v + tau2)`` all collapse to the same value and
# the fit degenerates into an unweighted mean of incommensurable betas with a badly
# inflated standard error. That is the pale, speckled map above.
#
# Fixed-effect fits such as ``WeightedLeastSquares`` keep a correct false-positive rate, as
# any fixed set of weights does, but they lose power: inverse-variance weighting reads a
# large unit as low precision and downweights that study by the square of its scale.
#
# The estimators that are unaffected are the ones whose inputs are already scale-free, so
# reach for those instead when the betas are not on a shared scale. ``Stouffers`` and
# ``Fishers`` combine ``z`` maps, and ``FixedEffectsHedges`` below turns ``t`` maps into
# Hedges’ g, a standardized effect size. On this studyset those three agree with each other
# to r >= 0.97, while the beta-based fits correlate no better than ~0.87 with them.
#
# Note that ``FixedEffectsHedges`` converts with ``d = t / sqrt(n)``, which is the
# one-sample form. That suits a group activation map, but not a t map from a between-group
# contrast, where the conversion needs both group sizes.
#
# There is no supported random-effects counterpart: ``FixedEffectsHedges`` is fixed-effect
# only, and no NiMARE estimator fits a standardized effect size with a between-study
# variance.
#
# None of this applies if your betas do share a scale -- percent signal change, say, or a
# single pipeline throughout. Then beta and varcope meta-regression is the right tool.

###############################################################################
# Fixed Effects Meta-Analysis with Hedges’ g
# -----------------------------------------------------------------------------
from nimare.meta.ibma import FixedEffectsHedges

meta = FixedEffectsHedges(tau2=0)
results = meta.fit(studyset)

plot_stat_map(
    results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

print("Description:")
pprint(results.description_)
print("References:")
pprint(results.bibtex_)
