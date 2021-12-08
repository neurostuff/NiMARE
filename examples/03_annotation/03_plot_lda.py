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
# Given that the new annotations DataFrame is very wide (many terms),
# but also very short (5 studies), we will transpose it before presenting it.
new_dset.annotations.T.head(10)

###############################################################################
model.distributions_["p_topic_g_word_df"].T.head(10)

###############################################################################
n_top_terms = 10
top_term_df = model.distributions_["p_topic_g_word_df"].T
temp_df = top_term_df.copy()
top_term_df = pd.DataFrame(columns=top_term_df.columns, index=range(n_top_terms))
top_term_df.index.name = "Token"
for col in top_term_df.columns:
    top_tokens = temp_df.sort_values(by=col, ascending=False).index.tolist()[:n_top_terms]
    top_term_df.loc[:, col] = top_tokens

top_term_df
