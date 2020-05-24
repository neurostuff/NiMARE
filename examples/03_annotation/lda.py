# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _annotations2:

===============================
 Train an LDA model and use it
===============================

This example trains a latent Dirichlet allocation model with MALLET
using abstracts from Neurosynth.

"""
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
# MALLET is a Java toolbox for natural language processing.
# While LDA is implemented in some Python libraries, like scikit-learn,
# MALLET appears to do a better job at LDA than other tools.
# LDAModel will download MALLET automatically, but it's included here for clarity.
mallet_dir = nimare.extract.download_mallet()

###############################################################################
# Run model
# ---------
# This may take some time, so we won't run it in the gallery.
model = annotate.lda.LDAModel(dset.texts, text_column='abstract', n_iters=5)
model.fit()
model.save('lda_model.pkl.gz')

# Let's remove the model now that you know how to generate it.
os.remove('lda_model.pkl.gz')
