# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _decode_discrete:

============================
Discrete functional decoding
============================

We can use the methods in ``nimare.decode.discrete`` to apply functional
characterization analysis to regions of interest or subsets of the Dataset.
"""
import os

import nibabel as nib
import numpy as np
from nilearn.plotting import plot_roi

import nimare
from nimare.decode import discrete
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load dataset with abstracts
# ---------------------------
# We'll load a small dataset composed only of studies in Neurosynth with
# Angela Laird as a coauthor, for the sake of speed.
dset = nimare.dataset.Dataset(os.path.join(get_test_data_path(), "neurosynth_laird_studies.json"))
dset.annotations.head(5)

###############################################################################
# Create a region of interest
# ---------------------------

# First we'll make an ROI
arr = np.zeros(dset.masker.mask_img.shape, int)
arr[65:75, 50:60, 50:60] = 1
mask_img = nib.Nifti1Image(arr, dset.masker.mask_img.affine)
plot_roi(mask_img, draw_cross=False)

# Get studies with voxels in the mask
ids = dset.get_studies_by_mask(mask_img)

###############################################################################
#
# .. _brain-map-decoder-example:
#
# Decode an ROI image using the BrainMap method
# ---------------------------------------------

# Run the decoder
decoder = discrete.BrainMapDecoder(correction=None)
decoder.fit(dset)
decoded_df = decoder.transform(ids=ids)
decoded_df.sort_values(by="probReverse", ascending=False).head()

###############################################################################
#
# .. _neurosynth-chi2-decoder-example:
#
# Decode an ROI image using the Neurosynth chi-square method
# ----------------------------------------------------------

# Run the decoder
decoder = discrete.NeurosynthDecoder(correction=None)
decoder.fit(dset)
decoded_df = decoder.transform(ids=ids)
decoded_df.sort_values(by="probReverse", ascending=False).head()

###############################################################################
#
# .. _neurosynth-roi-decoder-example:
#
# Decode an ROI image using the Neurosynth ROI association method
# ---------------------------------------------------------------

# This method decodes the ROI image directly, rather than comparing subsets of the Dataset like the
# other two.
decoder = discrete.ROIAssociationDecoder(mask_img)
decoder.fit(dset)

# The `transform` method doesn't take any parameters.
decoded_df = decoder.transform()

decoded_df.sort_values(by="r", ascending=False).head()
