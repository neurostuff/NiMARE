"""

.. _metas_estimators:

===================
The Estimator class
===================

An introduction to the Estimator class.

The Estimator class is the base for all meta-analyses in NiMARE.
A general rule of thumb for Estimators is that they ingest Datasets and output
MetaResult objects.
"""
###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
import os

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
from nimare.dataset import Dataset
from nimare.utils import get_resource_path

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

# We will reduce the Dataset to the first 10 studies
dset = dset.slice(dset.ids[:10])

###############################################################################
# The Estimator
# -----------------------------------------------------------------------------
from nimare.meta.cbma.ale import ALE

# First, the Estimator should be initialized with any parameters.
meta = ALE()

# Then, the ``fit`` method takes in the Dataset and produces a MetaResult.
results = meta.fit(dset)

###############################################################################
# Coordinate-based Estimators allow you to provide a specific KernelTransformer
# -----------------------------------------------------------------------------
# Each CBMA Estimator's default KernelTransformer will always be the most
# appropriate type for that algorithm, but you can swap out the kernel as you
# wish.
#
# For example, an ALE Estimator could be initialized with an MKDAKernel,
# though there is no guarantee that the results would make sense.
from nimare.meta.kernel import MKDAKernel

meta = ALE(kernel_transformer=MKDAKernel)
results = meta.fit(dset)

###############################################################################
from nilearn.plotting import plot_stat_map

plot_stat_map(results.get_map("z"), draw_cross=False, cmap="RdBu_r")

###############################################################################
# CBMA Estimators can accept KernelTransformers a few different ways
# -----------------------------------------------------------------------------
from nimare.meta.kernel import ALEKernel

###############################################################################
# Initializing the Estimator with a KernelTransformer class alone will use
# its default settings.
meta = ALE(kernel_transformer=ALEKernel)
print(meta.kernel_transformer)

###############################################################################
# You can also initialize the Estimator with an initialized KernelTransformer
# object.
kernel = ALEKernel()
meta = ALE(kernel_transformer=kernel)
print(meta.kernel_transformer)

###############################################################################
# This is especially useful if you want to initialize the KernelTransformer
# with parameters with non-default values.
kernel = ALEKernel(sample_size=20)
meta = ALE(kernel_transformer=kernel)
print(meta.kernel_transformer)

###############################################################################
# You can also provide specific initialization values to the KernelTransformer
# via the Estimator, by including keyword arguments starting with ``kernel__``.
meta = ALE(kernel__sample_size=20)
print(meta.kernel_transformer)

######################################################################################
# .. _null-method-example:
#
# Most CBMA Estimators have multiple ways to test uncorrected statistical significance
# ------------------------------------------------------------------------------------
# For most Estimators, the two options, defined with the ``null_method``
# parameter, are ``"approximate"`` and ``"montecarlo"``.
# For more information about these options, see :ref:`null methods`.
meta = ALE(null_method="approximate")
results = meta.fit(dset)

######################################################################################
# Note that, to measure significance appropriately with the montecarlo method,
# you need a lot more than 10 iterations.
# We recommend 10000 (the default value).
mc_meta = ALE(null_method="montecarlo", n_iters=10, n_cores=1)
mc_results = mc_meta.fit(dset)

###############################################################################
# The null distributions are stored within the Estimators
# `````````````````````````````````````````````````````````````````````````````
from pprint import pprint

pprint(meta.null_distributions_)

###############################################################################
# As well as the MetaResult, which stores a copy of the Estimator
pprint(results.estimator.null_distributions_)

###############################################################################
# The null distributions also differ based on the null method.
# For example, the ``"montecarlo"`` option creates a
# ``histweights_corr-none_method-montecarlo`` distribution, instead of the
# ``histweights_corr-none_method-approximate`` produced by the
# ``"approximate"`` method.
pprint(mc_meta.null_distributions_)

###############################################################################
import matplotlib.pyplot as plt
import seaborn as sns

with sns.axes_style("whitegrid"):
    fig, axes = plt.subplots(figsize=(8, 8), sharex=True, nrows=3)
    sns.histplot(
        x=meta.null_distributions_["histogram_bins"],
        weights=meta.null_distributions_["histweights_corr-none_method-approximate"],
        bins=100,
        ax=axes[0],
    )
    axes[0].set_xlim(0, None)
    axes[0].set_title("Approximate Null Distribution")
    sns.histplot(
        x=mc_meta.null_distributions_["histogram_bins"],
        weights=mc_meta.null_distributions_["histweights_corr-none_method-montecarlo"],
        bins=100,
        ax=axes[1],
    )
    axes[1].set_title("Monte Carlo Null Distribution")
    sns.histplot(
        x=mc_meta.null_distributions_["histogram_bins"],
        weights=mc_meta.null_distributions_["histweights_level-voxel_corr-fwe_method-montecarlo"],
        bins=100,
        ax=axes[2],
    )
    axes[2].set_title("Monte Carlo Voxel-Level FWE Null Distribution")
    axes[2].set_xlabel("ALE Value")
    fig.tight_layout()
