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

###############################################################################
# Load dataset with abstracts
# ---------------------------
out_dir = os.path.abspath('../example_data/')
dset = nimare.dataset.Dataset.load(
    os.path.join(out_dir, 'neurosynth_nimare_with_abstracts.pkl.gz'))

###############################################################################
# Generate term counts
# --------------------
counts_df = nimare.annotate.text.generate_counts(
    dset.texts, text_column='abstract', tfidf=False)

###############################################################################
# Run model
# ---------
# Five iterations will take ~10 minutes
model = nimare.annotate.topic.GCLDAModel(
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
decoded_df, _ = nimare.decode.discrete.gclda_decode_roi(model, mask_img)
decoded_df.sort_values(by='Weight', ascending=False).head(10)
