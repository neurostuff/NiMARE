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


def analyze_and_plot(dset, ground_truth_foci):
    mkda = nimare.meta.mkda.MKDADensity(kernel__r=10)
    mkda.fit(dset)
    corr = nimare.correct.FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
    cres = corr.transform(mkda.results)

    # get the z coordinates
    cut_coords = [c[2] for c in ground_truth_foci]
    display = plot_stat_map(
        cres.get_map("logp_level-voxel_corr-FWE_method-montecarlo"),
        display_mode="z",
        cut_coords=cut_coords,
        draw_cross=False,
        cmap="RdBu_r",
    )
    display.add_markers(ground_truth_foci)

    return display


###############################################################################
# Create Dataset
# --------------------------------------------------
# In this example, each of the 30 fake studies being generated
# select 4 coordinates from a probability map representing the probability
# that particular coordinate will be chosen.
# There are 4 "hot" spots centered on 3D gaussian distributions.
# Each study has a mean sample_size of 30

ground_truth_foci, dset = create_coordinate_dataset(
    foci_num=4, fwhm=10.0, sample_size=30, sample_size_variance=10, studies=30
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
# in each study (i.e., ``foci_num``).
# This way you can test what happens if one study has way more reported
# foci than another study.
# You can also:
#   - manually specify the sample sizes for each study (i.e., ``sample_size``).
#   - set the number of false positive foci for each study (i.e., ``foci_noise``).
#   - set the weighed probability of each foci being chosen (i.e., ``foci_weights``).
#   - set the dispersion of the gaussian spread for each foci (i.e., ``fwhm``).
#   - set the specific ground truth foci (i.e., ``foci_coords``).

num_studies = 30
foci_num = [4] * num_studies
foci_num[0] = 10
sample_sizes = [30] * num_studies
sample_sizes[0] = 300
foci_noise = [10] * num_studies
foci_noise[0] = 25
foci_weights = [1, 0.1, 0.5, 0.75]
fwhm = [6.0, 12.0, 10.0, 8.0]
_, manual_dset = create_coordinate_dataset(
    foci_num=foci_num,
    fwhm=fwhm,
    sample_size=sample_sizes,
    studies=num_studies,
    foci_coords=ground_truth_foci,
    foci_noise=foci_noise,
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
# for ``foci_num``.
# The relative probability of the two foci appearing
# can be controlled with ``foci_weight``.
# For example the weights ``[0.9, 0.1]`` mean
# the first foci is 90% likely to appear in a study and
# the second foci has a 10% chance of appearing in a particular
# study.

_, two_foci_dset = create_coordinate_dataset(
    foci_num=1,
    fwhm=10.0,
    sample_size=30,
    sample_size_variance=10,
    studies=30,
    foci_coords=ground_truth_foci[0:2],
    foci_noise=0,
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
# and have a list of zeros and ones in ``foci_num`` to specify
# which studies contain the focus of interest.
# The caveat of this strategy is that each study must have at least
# one coordinate to report necessitating ``foci_noise`` be set for
# studies without a ground truth foci.

num_studies = 30
# create array of zeros and ones for foci
foci_num = np.random.choice([0, 1], size=num_studies, p=[0.9, 0.1])
# make a noise_foci for studies without ground truth foci
foci_noise = 1 - foci_num

num_studies = 30
# create array of zeros and ones for foci
foci_num = np.random.choice([0, 1], size=num_studies, p=[0.8, 0.2])
# make a noise_foci for studies without ground truth foci
foci_noise = 1 - foci_num

_, one_foci_dset = create_coordinate_dataset(
    foci_num=foci_num,
    fwhm=10.0,
    sample_size=30,
    sample_size_variance=10,
    studies=num_studies,
    foci_coords=[ground_truth_foci[2]],
    foci_noise=foci_noise,
)

###############################################################################
# Analyze and plot one foci dataset
# -------------------------------

analyze_and_plot(one_foci_dset, [ground_truth_foci[2]])
