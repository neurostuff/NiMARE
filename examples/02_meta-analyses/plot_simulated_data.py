# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _metas6:

========================================================
 Simulate data for coordinate based meta-analysis (CBMA)
========================================================

Simulating data before you run your meta-analysis is a great
way to test your assumptions and see how the meta-analysis
would perform with simplified data
"""
import nimare
from nimare.generate import create_coordinate_dataset
from nilearn.plotting import plot_stat_map
import numpy as np


###############################################################################
# Create function to perform a meta-analysis and plot results
# -----------------------------------------------------------


def analyze_and_plot(dset, ground_truth_foci=None, return_cres=False):
    mkda = nimare.meta.mkda.MKDADensity(kernel__r=10)
    mkda.fit(dset)
    corr = nimare.correct.FWECorrector(method="montecarlo", n_iters=100, n_cores=-1)
    cres = corr.transform(mkda.results)

    # get the z coordinates
    if ground_truth_foci:
        stat_map_kwargs = {
            "cut_coords": [c[2] for c in ground_truth_foci],
        }
    else:
        stat_map_kwargs = {}

    display = plot_stat_map(
        cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo"),
        display_mode="z",
        draw_cross=False,
        cmap="Purples",
        threshold=1.0,
        vmax=15,
        symmetric_cbar=False,
        **stat_map_kwargs,
    )

    if ground_truth_foci:
        # place red dots indicating the ground truth foci
        display.add_markers(ground_truth_foci)

    if return_cres:
        return display, cres
    return display


###############################################################################
# Create Dataset
# --------------------------------------------------
# In this example, each of the 30 generated fake studies
# select 4 coordinates from a probability map representing the probability
# that particular coordinate will be chosen.
# There are 4 "hot" spots centered on 3D gaussian distributions,
# meaning each study will likely select 4 foci that are close
# to those hot spots, but there is still random jittering.
# Each study has a mean ``sample_size`` of 30 with a variance of 10
# so some studies may have fewer than 30 participants and some
# more.

ground_truth_foci, dset = create_coordinate_dataset(
    n_foci=4, fwhm=10.0, sample_size=30, sample_size_interval=10, n_studies=30
)

###############################################################################
# Analyze and plot simple dataset
# -------------------------------

analyze_and_plot(dset, ground_truth_foci)

###############################################################################
# Fine-tune dataset creation
# --------------------------------------------------
# Perhaps you want more control over the studies being generated.
# You can manually specify how many "converging" foci you want specified
# in each study (i.e., ``n_foci``).
# This way you can test what happens if one study has way more reported
# foci than another study.
# You can also:
#   - manually specify the sample sizes for each study (i.e., ``sample_size``).
#   - set the number of false positive foci for each study (i.e., ``n_noise_foci``).
#   - set the weighed probability of each foci being chosen (i.e., ``foci_weights``).
#   - set the dispersion of the gaussian spread for each foci (i.e., ``fwhm``).
#   - set the specific ground truth foci (i.e., ``foci_coords``).

num_studies = 30
n_foci = [4] * num_studies
n_foci[0] = 10
sample_sizes = [30] * num_studies
sample_sizes[0] = 300
n_noise_foci = [10] * num_studies
n_noise_foci[0] = 25
foci_weights = [1, 0.1, 0.5, 0.75]
fwhm = [6.0, 12.0, 10.0, 8.0]
_, manual_dset = create_coordinate_dataset(
    n_foci=n_foci,
    fwhm=fwhm,
    sample_size=sample_sizes,
    n_studies=num_studies,
    foci_coords=ground_truth_foci,
    n_noise_foci=n_noise_foci,
    foci_weights=foci_weights,
)

###############################################################################
# Analyze and plot manual dataset
# -------------------------------

analyze_and_plot(manual_dset, ground_truth_foci)

###############################################################################
# Control percentage of studies with the foci of interest: Strategy #1
# --------------------------------------------------------------------
# There are two ways you can create a dataset where
# a focus is only present in some of the studies.
# One way is to have two ground truth foci specified
# with ``foci_coords``, but only one foci specified
# for ``n_foci``.
# The relative probability of the two foci appearing
# can be controlled with ``foci_weight``.
# For example the weights ``[0.9, 0.1]`` mean
# the first foci is 90% likely to appear in a study and
# the second foci has a 10% chance of appearing in a particular
# study.

_, two_foci_dset = create_coordinate_dataset(
    n_foci=1,
    fwhm=10.0,
    sample_size=30,
    sample_size_interval=10,
    n_studies=30,
    foci_coords=ground_truth_foci[0:2],
    n_noise_foci=0,
    foci_weights=[0.9, 0.1],
)

###############################################################################
# Analyze and plot two foci dataset
# -------------------------------

analyze_and_plot(two_foci_dset, ground_truth_foci[0:2])

###############################################################################
# Control percentage of studies with the foci of interest: Strategy #2
# --------------------------------------------------------------------
# Another method to control what percentage of studies contain
# a focus is to have one focus specified in ``foci_coords``
# and have a list of zeros and ones in ``n_foci`` to specify
# which studies contain the focus of interest.
# The caveat of this strategy is that each study must have at least
# one coordinate to report necessitating ``n_noise_foci`` be set for
# studies without a ground truth foci.

num_studies = 30
# create array of zeros and ones for foci
n_foci = np.random.choice([0, 1], size=num_studies, p=[0.8, 0.2])
# make a noise_foci for studies without ground truth foci
n_noise_foci = 1 - n_foci

_, one_foci_dset = create_coordinate_dataset(
    n_foci=n_foci,
    fwhm=10.0,
    sample_size=30,
    sample_size_interval=10,
    n_studies=num_studies,
    foci_coords=[ground_truth_foci[2]],
    n_noise_foci=n_noise_foci,
)

###############################################################################
# Analyze and plot one foci dataset
# ---------------------------------

analyze_and_plot(one_foci_dset, [ground_truth_foci[2]])

###############################################################################
# Create a null dataset
# --------------------------------------------------------------------
# Perhaps you are interested in the number of false positives your favorite
# meta-analysis algorithm typically gives.
# At an alpha of 0.05 we would expect no more than 5% of results to be false positives.
# To test this, we can create a dataset with no foci that converge, but have many
# distributed foci.

_, no_foci_dset = create_coordinate_dataset(
    n_foci=0,
    fwhm=10.0,
    sample_size=30,
    sample_size_interval=10,
    n_studies=30,
    n_noise_foci=100,
)

###############################################################################
# Analyze and plot no foci dataset
# --------------------------------

display, cres = analyze_and_plot(no_foci_dset, return_cres=True)

logp_values = display.get_map("logp_level-voxel_corr-FWE_method-montecarlo", return_type="array")
# inverse the log transform of the p_values
p_values = 10 ** -logp_values
# what percentage of voxels are not significant?
non_significant_percent = (p_values > 0.05).sum() / p_values.size
print(non_significant_percent)

###############################################################################
# Further exploration
# -------------------
# This notebook covers a few of the presumed use-cases for
# ``create_coordinate_dataset``, but the notebook is not exhaustive.
# For example, this notebook did not cover how varying the fwhm
# could change how well a meta-analysis detects an effect
# or how to generate 1000s of datasets for testing a
# meta-analysis algorithm.
# Hopefully, this notebook gave you an understanding of the fundamentals
# of ``create_coordinate_dataset``.
