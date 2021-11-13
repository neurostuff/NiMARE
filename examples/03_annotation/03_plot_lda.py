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

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

import nimare
from nimare import annotate
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load dataset with abstracts
# ---------------------------
dset = nimare.dataset.Dataset(
    os.path.join(get_test_data_path(), "neurosynth_laird_studies.json")
)

###############################################################################
# Extract term counts from the abstracts
# --------------------------------------
counts_df = annotate.text.generate_counts(
    dset.texts,
    text_column="abstract",
    tfidf=False,
    max_df=len(dset.ids) - 2,
    min_df=2,
)
vocabulary = counts_df.columns.tolist()
count_values = counts_df.values
study_ids = counts_df.index.tolist()
N_TOPICS = 5
topic_names = [f"Topic {str(i+1).zfill(3)}" for i in range(N_TOPICS)]

###############################################################################
# Run model
# ---------
# This may take some time, so we won't run it in the gallery.
model = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=1000,
    learning_method="online",
)
doc_topic_weights = model.fit_transform(count_values)
doc_topic_weights_df = pd.DataFrame(
    index=study_ids,
    columns=topic_names,
    data=doc_topic_weights,
)
topic_word_weights = model.components_
topic_word_weights_df = pd.DataFrame(
    index=topic_names,
    columns=vocabulary,
    data=topic_word_weights,
)

###############################################################################
# View results
# ------------
doc_topic_weights_df.head()

###############################################################################
topic_word_weights_df.head()
