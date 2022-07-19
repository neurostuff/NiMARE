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

###############################################################################
# Load Sleuth text files into Datasets
# -----------------------------------------------------------------------------
# The data for this example are a subset of studies from a meta-analysis on
# semantic cognition in children :footcite:p:`enge2021meta`.
# A first group of studies probed children's semantic world knowledge
# (e.g., correctly naming an object after hearing its auditory description)
# while a second group of studies asked children to decide if two (or more)
# words were semantically related to one another or not.
import os

from nimare.io import convert_sleuth_to_dataset
from nimare.utils import get_resource_path

knowledge_file = os.path.join(get_resource_path(), "Enge2021_knowledge.txt")
related_file = os.path.join(get_resource_path(), "Enge2021_relatedness.txt")
# Some papers reported MNI coordinates
objects_file_mni = os.path.join(get_resource_path(), "Enge2021_objects_mni.txt")
# Other papers reported Talairach coordinates
objects_file_talairach = os.path.join(get_resource_path(), "Enge2021_objects_talairach.txt")

knowledge_dset = convert_sleuth_to_dataset(knowledge_file)
related_dset = convert_sleuth_to_dataset(related_file)
objects_dset = convert_sleuth_to_dataset(
    [
        objects_file_mni,
        objects_file_talairach,  # NiMARE will automatically convert the Talairach foci to MNI
    ]
)

###############################################################################
# View the contents of one of the Sleuth files
with open(knowledge_file, "r") as fo:
    sleuth_file_contents = fo.readlines()

sleuth_file_contents = sleuth_file_contents[:20]
print("".join(sleuth_file_contents))

###############################################################################
# Meta-analysis of semantic knowledge experiments
# -----------------------------------------------------------------------------
from nimare.meta.cbma import ALE

ale = ALE(null_method="approximate")
knowledge_results = ale.fit(knowledge_dset)

###############################################################################
# Plot the uncorrected statistical map
# `````````````````````````````````````````````````````````````````````````````
from nilearn.plotting import plot_stat_map

plot_stat_map(
    knowledge_results.get_map("z"),
    cut_coords=4,
    display_mode="z",
    title="Semantic knowledge",
    threshold=2.326,  # cluster-level p < .01, one-tailed
    cmap="RdBu_r",
    vmax=4,
)

###############################################################################
# This z-statistic map is not corrected for multiple comparisons.
# In order to account for the many voxel-wise tests that are performed in
# parallel, we must apply some type of multiple comparisons correction.
# To that end, we will use an :class:`~nimare.correct.FWECorrector` with the
# Monte Carlo method.
#
# Multiple comparisons correction with a Monte Carlo procedure
# -----------------------------------------------------------------------------
# We will use the cluster-level corrected map, using a cluster-defining
# threshold of p < 0.001 and 100 iterations.
# In the actual paper, :footcite:`enge2021meta` used 10000 iterations instead.
from nimare.correct import FWECorrector

corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=100, n_cores=2)
knowledge_corrected_results = corr.transform(knowledge_results)

###############################################################################
# Plot the corrected statistical map
# `````````````````````````````````````````````````````````````````````````````
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
)

###############################################################################
# Save the results to disk
# `````````````````````````````````````````````````````````````````````````````
knowledge_corrected_results.save_maps(
    output_dir=".",
    prefix="Enge2021_knowledge",
)

###############################################################################
# Characterize the relative contributions of experiments in the results
# `````````````````````````````````````````````````````````````````````````````
# NiMARE contains two methods for this: :class:`~nimare.diagnostics.Jackknife`
# and :class:`~nimare.diagnostics.FocusCounter`.
# We will show both below, but for the sake of speed we will only apply one to
# each subgroup meta-analysis.
from nimare.diagnostics import Jackknife

jackknife = Jackknife(
    target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
    voxel_thresh=None,
)
knowledge_jackknife_table, _ = jackknife.transform(knowledge_corrected_results)
knowledge_jackknife_table

###############################################################################
# Meta-analysis of semantic relatedness experiments
# -----------------------------------------------------------------------------
ale = ALE(null_method="approximate")
related_results = ale.fit(related_dset)

# Perform Monte Carlo-based multiple comparisons correction
corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=100, n_cores=2)
related_corrected_results = corr.transform(related_results)

###############################################################################
# Plot the resulting statistical map
# `````````````````````````````````````````````````````````````````````````````
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
)

###############################################################################
# Characterize the relative contributions of experiments in the results
# `````````````````````````````````````````````````````````````````````````````
jackknife = Jackknife(
    target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
    voxel_thresh=None,
)
related_jackknife_table, _ = jackknife.transform(related_corrected_results)
related_jackknife_table

###############################################################################
# Meta-analysis of semantic object experiments
# -----------------------------------------------------------------------------
ale = ALE(null_method="approximate")
objects_results = ale.fit(objects_dset)

# Perform Monte Carlo-based multiple comparisons correction
corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=100, n_cores=2)
objects_corrected_results = corr.transform(objects_results)

###############################################################################
# Plot the resulting statistical map
# `````````````````````````````````````````````````````````````````````````````
objects_img = objects_corrected_results.get_map(
    "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
)
plot_stat_map(
    objects_img,
    cut_coords=4,
    display_mode="z",
    title="Semantic objects",
    threshold=2.326,  # cluster-level p < .01, one-tailed
    cmap="RdBu_r",
    vmax=4,
)

###############################################################################
# Characterize the relative contributions of experiments in the results
# `````````````````````````````````````````````````````````````````````````````
jackknife = Jackknife(
    target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
    voxel_thresh=None,
)
objects_jackknife_table, _ = jackknife.transform(objects_corrected_results)
objects_jackknife_table

###############################################################################
# Compare semantic knowledge to the other conditions with subtraction analysis
# -----------------------------------------------------------------------------
# The semantic knowledge experiments can be compared to the experiments from
# the other two conditions by first merging the other two sets of experiments
# into a single Dataset, and then performing a subtraction analysis between the
# semantic knowledge Dataset and the new semantic relatedness/objects Dataset.
#
# In :footcite:t:`enge2021meta`, additional subtraction analyses were performed
# for semantic relatedness vs. (knowledge + objects) and semantic objects vs.
# (knowledge + relatedness).
# However, for the sake of executing this example online, we only perform the
# first of these analyses.
#
# .. important::
#   Typically, one would use at least 10000 iterations for a subtraction analysis.
#   However, we have reduced this to 100 iterations for this example.
from nimare.meta.cbma import ALESubtraction

# First, we combine the relatedness and objects Datasets
related_and_objects_dset = related_dset.merge(objects_dset)

sub = ALESubtraction(n_iters=100, n_cores=1)
res_sub = sub.fit(knowledge_dset, related_and_objects_dset)
img_sub = res_sub.get_map("z_desc-group1MinusGroup2")

plot_stat_map(
    img_sub,
    cut_coords=4,
    display_mode="z",
    title="Knowledge > Other",
    cmap="RdBu_r",
    vmax=4,
)

###############################################################################
# Evaluate convergence across datasets with a conjunction analysis
# -----------------------------------------------------------------------------
# To determine the overlap of the meta-analytic results, a conjunction image
# can be computed by (a) identifying voxels that were statistically significant
# in *both* individual group maps and (b) selecting, for each of these voxels,
# the smaller of the two group-specific *z* values :footcite:t:`nichols2005valid`.
# Since this is simple arithmetic on images, conjunction is not implemented as
# a separate method in :code:`NiMARE` but can easily be achieved with
# :func:`nilearn.image.math_img`.
from nilearn.image import math_img

img_conj = math_img(
    "np.where((img1 * img2 * img3) > 0, np.minimum(img1, np.minimum(img2, img3)), 0)",
    img1=knowledge_img,
    img2=related_img,
    img3=objects_img,
)

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
