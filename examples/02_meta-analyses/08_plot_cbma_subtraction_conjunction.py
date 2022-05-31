"""

.. _metas_subtraction:

============================
Two-sample ALE meta-analysis
============================

Meta-analytic projects often involve a number of common steps comparing two or more samples.

In this example, we replicate the ALE-based analyses from :footcite:t:`enge2021meta`.

A common project workflow with two meta-analytic samples involves the following:

1. Run a within-sample meta-analysis of the first sample.
2. Characterize/summarize the results of the first meta-analysis.
3. Run a within-sample meta-analysis of the second sample.
4. Characterize/summarize the results of the second meta-analysis.
5. Compare the two samples with a subtraction analysis.
6. Compare the two within-sample meta-analyses with a conjunction analysis.
"""
import os

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

###############################################################################
# Load Sleuth text files into Datasets
# -----------------------------------------------------------------------------
# The data for this example are a subset of studies from a meta-analysis on
# semantic cognition in children :footcite:p:`enge2021meta`.
# A first group of studies probed children's semantic world knowledge
# (e.g., correctly naming an object after hearing its auditory description)
# while a second group of studies asked children to decide if two (or more)
# words were semantically related to one another or not.
from nimare.io import convert_sleuth_to_dataset
from nimare.utils import get_resource_path

knowledge_file = os.path.join(get_resource_path(), "semantic_knowledge_children.txt")
related_file = os.path.join(get_resource_path(), "semantic_relatedness_children.txt")

knowledge_dset = convert_sleuth_to_dataset(knowledge_file)
related_dset = convert_sleuth_to_dataset(related_file)

###############################################################################
# Individual group ALEs
# -----------------------------------------------------------------------------
# Computing separate ALE analyses for each group is not strictly necessary for
# performing the subtraction analysis but will help the experimenter to appreciate the
# similarities and differences between the groups.
from nimare.correct import FWECorrector
from nimare.meta.cbma import ALE

ale = ALE(null_method="approximate")
knowledge_results = ale.fit(knowledge_dset)
related_results = ale.fit(related_dset)

corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=100, n_cores=2)
knowledge_corrected_results = corr.transform(knowledge_results)
related_corrected_results = corr.transform(related_results)

fig, axes = plt.subplots(figsize=(12, 10), nrows=2)
knowledge_img = knowledge_corrected_results.get_map(
    "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
)
plot_stat_map(
    knowledge_img,
    cut_coords=4,
    display_mode="z",
    title="Semantic knowledge",
    threshold=2.326,  # cluster-level p < .01, one-tailed
    cmap="RdBu_r",
    vmax=4,
    axes=axes[0],
    figure=fig,
)

related_img = related_corrected_results.get_map(
    "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
)
plot_stat_map(
    related_img,
    cut_coords=4,
    display_mode="z",
    title="Semantic relatedness",
    threshold=2.326,  # cluster-level p < .01, one-tailed
    cmap="RdBu_r",
    vmax=4,
    axes=axes[1],
    figure=fig,
)
fig.show()

###############################################################################
# Characterize the relative contributions of experiments in the ALE results
# -----------------------------------------------------------------------------
# NiMARE contains two methods for this: :class:`~nimare.diagnostics.Jackknife`
# and :class:`~nimare.diagnostics.FocusCounter`.
# We will show both below, but for the sake of speed we will only apply one to
# each subgroup meta-analysis.
from nimare.diagnostics import FocusCounter

counter = FocusCounter(
    target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
    voxel_thresh=None,
)
knowledge_count_table, _ = counter.transform(knowledge_corrected_results)
knowledge_count_table.head(10)

###############################################################################
from nimare.diagnostics import Jackknife

jackknife = Jackknife(
    target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
    voxel_thresh=None,
)
related_jackknife_table, _ = jackknife.transform(related_corrected_results)
related_jackknife_table.head(10)

###############################################################################
# Subtraction analysis
# -----------------------------------------------------------------------------
# Typically, one would use at least 10000 iterations for a subtraction analysis.
# However, we have reduced this to 100 iterations for this example.
from nimare.meta.cbma import ALESubtraction

sub = ALESubtraction(n_iters=100, n_cores=1)
res_sub = sub.fit(knowledge_dset, related_dset)
img_sub = res_sub.get_map("z_desc-group1MinusGroup2")

plot_stat_map(
    img_sub,
    cut_coords=4,
    display_mode="z",
    title="Subtraction",
    cmap="RdBu_r",
    vmax=4,
)

###############################################################################
# Conjunction analysis
# -----------------------------------------------------------------------------
# To determine the overlap of the meta-analytic results, a conjunction image
# can be computed by (a) identifying voxels that were statistically significant
# in *both* individual group maps and (b) selecting, for each of these voxels,
# the smaller of the two group-specific *z* values :footcite:t:`nichols2005valid`.
# Since this is simple arithmetic on images, conjunction is not implemented as
# a separate method in :code:`NiMARE` but can easily be achieved with
# :func:`nilearn.image.math_img`.
from nilearn.image import math_img

formula = "np.where(img1 * img2 > 0, np.minimum(img1, img2), 0)"
img_conj = math_img(formula, img1=knowledge_img, img2=related_img)

plot_stat_map(
    img_conj,
    cut_coords=4,
    display_mode="z",
    title="Conjunction",
    threshold=2.326,  # cluster-level p < .01, one-tailed
    cmap="RdBu_r",
    vmax=4,
)

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
