# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _annotations2:

===============================
 Train an LDA model and use it
===============================

This example trains a latent Dirichlet allocation with MALLET
using abstracts from Neurosynth.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os

import nimare

###############################################################################
# Load dataset with abstracts
# ---------------------------
out_dir = os.path.abspath('../example_data/')
dset = nimare.dataset.Dataset.load(
    os.path.join(out_dir, 'neurosynth_nimare_with_abstracts.pkl.gz'))

###############################################################################
# Download MALLET
# ---------------
# LDAModel will do this automatically.
mallet_dir = nimare.extract.download_mallet()

###############################################################################
# Run model
# ---------
# Five iterations will take ~10 minutes
model = nimare.annotate.topic.LDAModel(dset.texts, text_column='abstract', n_iters=5)
model.fit()
model.save('lda_model.pkl.gz')
