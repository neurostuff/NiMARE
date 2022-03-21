"""

.. _datasets_neurovault:

=========================================
Use NeuroVault statistical maps in NiMARE
=========================================

Download statistical maps from NeuroVault, then use them in a meta-analysis,
with NiMARE.
"""
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

###############################################################################
# Neurovault + NiMARE: Load freely shared statistical maps for Meta-Analysis
# -----------------------------------------------------------------------------
# `Neurovault <https://neurovault.org/>`_ is an online platform that hosts
# unthresholded statistical maps, including group statistical maps.
# NiMARE can read these statistical maps when given a list of collection_ids.
# I search "working memory" on neurovault, and find these relevant collections:
#
# * `2884 <https://neurovault.org/collections/2884/>`_
# * `2621 <https://neurovault.org/collections/2621/>`_
# * `3085 <https://neurovault.org/collections/3085/>`_
# * `5623 <https://neurovault.org/collections/5623/>`_
# * `3264 <https://neurovault.org/collections/3264/>`_
# * `3192 <https://neurovault.org/collections/3192/>`_
# * `457 <https://neurovault.org/collections/457/>`_
#
# I can load specific statistical maps from these collections
# into a NiMARE dataset:
from nimare.io import convert_neurovault_to_dataset

# The specific collections I would like to download group level
# statistical maps from
collection_ids = (2884, 2621, 3085, 5623, 3264, 3192, 457)

# A mapping between what I want the contrast(s) to be
# named in the dataset and what their respective group
# statistical maps are named on neurovault
contrasts = {
    "working_memory": (
        "Working memory load of 2 faces versus 1 face - NT2_Tstat|"
        "t-value contrast 2-back minus 0-back|"
        "Searchlight multivariate Decoding 2: visual working memory|"
        "Context-dependent group-specific WM information|"
        "WM working memory zstat1|"
        "WM task over CRT task map|"
        "tfMRI WM 2BK PLACE zstat1"
    )
}

# Convert how the statistical maps on neurovault are represented
# in a NiMARE dataset.
map_type_conversion = {"Z map": "z", "T map": "t"}

dset = convert_neurovault_to_dataset(
    collection_ids,
    contrasts,
    img_dir=None,
    map_type_conversion=map_type_conversion,
)

###############################################################################
# Conversion of Statistical Maps
# -----------------------------------------------------------------------------
# Some of the statistical maps are T statistics and others are Z statistics.
# To perform a Fisher's meta analysis, we need all Z maps.
# Thoughtfully, NiMARE has a class named ``ImageTransformer`` that will
# help us.
from nimare.transforms import ImageTransformer

# Not all studies have Z maps!
dset.images[["z"]]

###############################################################################
z_transformer = ImageTransformer(target="z")
dset = z_transformer.transform(dset)

###############################################################################
# All studies now have Z maps!
dset.images[["z"]]

###############################################################################
# Run a Meta-Analysis
# -----------------------------------------------------------------------------
# With the missing Z maps filled in, we can run a Meta-Analysis
# and plot our results
from nimare.meta.ibma import Fishers

# The default template has a slightly different, but completely compatible,
# affine than the NeuroVault images, so we allow the Estimator to resample
# images during the fitting process.
meta = Fishers(resample=True)

meta_res = meta.fit(dset)

fig, ax = plt.subplots()
display = plot_stat_map(meta_res.get_map("z"), threshold=3.3, axes=ax, figure=fig)
fig.show()
# The result may look questionable, but this code provides
# a template on how to use neurovault in your meta analysis.
