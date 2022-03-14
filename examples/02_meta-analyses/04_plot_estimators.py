"""

.. _metas_cbma:

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

plot_stat_map(results.get_map("z"), draw_cross=False)

###############################################################################
# CBMA Estimators can accept KernelTransformers a few different ways
# -----------------------------------------------------------------------------
from nimare.meta.kernel import ALEKernel

# -----------------------------------------------------------------------------
# Initializing the Estimator with a KernelTransformer class alone will use
# its default settings.
meta = ALE(kernel_transformer=ALEKernel)
print(meta.kernel_transformer)

# -----------------------------------------------------------------------------
# You can also initialize the Estimator with an initialized KernelTransformer
# object.
kernel = ALEKernel()
meta = ALE(kernel_transformer=kernel)
print(meta.kernel_transformer)

# -----------------------------------------------------------------------------
# This is especially useful if you want to initialize the KernelTransformer
# with parameters with non-default values.
kernel = ALEKernel(sample_size=20)
meta = ALE(kernel_transformer=kernel)
print(meta.kernel_transformer)

# -----------------------------------------------------------------------------
# You can also provide specific initialization values to the KernelTransformer
# via the Estimator, by including keyword arguments starting with ``kernel__``.
meta = ALE(kernel__sample_size=20)
print(meta.kernel_transformer)

###############################################################################
# Most CBMA Estimators also generate null distributions
# -----------------------------------------------------------------------------
# These null distributions determine how the Estimator's summary statistic,
# such as ALE values for ALE meta-analyses or OF values for MKDA meta-analyses,
# convert to test statistics- especially z-statistics.

# First, we fit the Estimator
meta = ALE()
results = meta.fit(dset)

###############################################################################
# The null distributions are available in the fitted Estimator
print(meta.null_distributions_)

###############################################################################
# As well as the MetaResult, which stores a copy of the Estimator
print(results.estimator.null_distributions_)
