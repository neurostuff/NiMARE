"""

.. _annotations_lda:

==================
LDA topic modeling
==================

Trains a latent Dirichlet allocation model with scikit-learn using abstracts from Neurosynth.
"""
import os

import pandas as pd

from nimare import annotate
from nimare.dataset import Dataset
from nimare.utils import get_resource_path

###############################################################################
# Load dataset with abstracts
# -----------------------------------------------------------------------------
dset = Dataset(os.path.join(get_resource_path(), "neurosynth_laird_studies.json"))

###############################################################################
# Initialize LDA model
# -----------------------------------------------------------------------------
model = annotate.lda.LDAModel(n_topics=5, max_iter=1000, text_column="abstract")

###############################################################################
# Run model
# -----------------------------------------------------------------------------
new_dset = model.fit(dset)

###############################################################################
# View results
# -----------------------------------------------------------------------------
# This DataFrame is very large, so we will only show a slice of it.
new_dset.annotations[new_dset.annotations.columns[:10]].head(10)

###############################################################################
# Given that this DataFrame is very wide (many terms), we will transpose it before presenting it.
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
