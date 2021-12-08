# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _annotations_lda:

==================
LDA topic modeling
==================

This example trains a latent Dirichlet allocation model with scikit-learn
using abstracts from Neurosynth.
"""
import os

from nimare import annotate, dataset
from nimare.utils import get_resource_path

###############################################################################
# Load dataset with abstracts
# ---------------------------
dset = dataset.Dataset(os.path.join(get_resource_path(), "neurosynth_laird_studies.json"))

###############################################################################
# Initialize LDA model
# --------------------
model = annotate.lda.LDAModel(n_topics=5, max_iter=1000, text_column="abstract")

###############################################################################
# Run model
# ---------
new_dset = model.transform(dset)

###############################################################################
# View results
# ------------
new_dset.annotations.head()

###############################################################################
model.distributions_["p_topic_g_word_df"].head()
