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
from nimare import annotate
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load dataset with abstracts
# ---------------------------
dset = nimare.dataset.Dataset.load(
    os.path.join(get_test_data_path(), 'neurosynth_laird_studies.pkl.gz'))

###############################################################################
# Download MALLET
# ---------------
# LDAModel will do this automatically.
mallet_dir = nimare.extract.download_mallet()

###############################################################################
# Run model
# ---------
# Five iterations will take ~10 minutes
model = annotate.topic.LDAModel(dset.texts, text_column='abstract', n_iters=5)
model.fit()
model.save('lda_model.pkl.gz')
