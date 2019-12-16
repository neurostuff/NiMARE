# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _annotations1:

===============================
 Train a GCLDA model and use it
===============================

This example trains a generalized corresponded latent Dirichlet allocation
using abstracts from Neurosynth and then uses it for decoding.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os

import numpy as np
import nibabel as nib

import nimare
from nimare import annotate, decode
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load dataset with abstracts
# ---------------------------
dset = nimare.dataset.Dataset.load(
    os.path.join(get_test_data_path(), 'neurosynth_laird_studies.pkl.gz'))

###############################################################################
# Generate term counts
# --------------------
counts_df = annotate.text.generate_counts(
    dset.texts, text_column='abstract', tfidf=False, max_df=0.99, min_df=0)

###############################################################################
# Run model
# ---------
# Five iterations will take ~10 minutes
model = annotate.topic.GCLDAModel(
    counts_df, dset.coordinates, mask=dset.masker.mask_img)
model.fit(n_iters=5, loglikely_freq=5)
model.save('gclda_model.pkl.gz')

###############################################################################
# Decode an ROI image
# -------------------
# Make an ROI from a single voxel
arr = np.zeros(dset.masker.mask_img.shape, int)
arr[40:44, 45:49, 40:44] = 1
mask_img = nib.Nifti1Image(arr, dset.masker.mask_img.affine)

# Run the decoder
decoded_df, _ = decode.discrete.gclda_decode_roi(model, mask_img)
decoded_df.sort_values(by='Weight', ascending=False).head(10)
