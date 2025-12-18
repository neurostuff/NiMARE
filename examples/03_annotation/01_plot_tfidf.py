"""

.. _annotations_tfidf:

===========================
Simple annotation from text
===========================

Perform simple term count or tf-idf value extraction from texts stored in a Dataset.
"""
import os

import pandas as pd

from nimare import annotate, dataset, utils

###############################################################################
# Load dataset with abstracts
# -----------------------------------------------------------------------------
# We'll load a small dataset composed only of studies in Neurosynth with
# Angela Laird as a coauthor, for the sake of speed.
dset = dataset.Dataset(os.path.join(utils.get_resource_path(), "neurosynth_laird_studies.json"))
dset.texts.head(2)

###############################################################################
# Generate term counts
# -----------------------------------------------------------------------------
# Let's start by extracting terms and their associated counts from article
# abstracts.
counts_df = annotate.text.generate_counts(
    dset.texts,
    text_column="abstract",
    tfidf=False,
    max_df=0.99,
    min_df=0.01,
)
counts_df.head(5)

###############################################################################
# Generate term counts
# -----------------------------------------------------------------------------
# We can also extract term frequency-inverse document frequency (tf-idf)
# values from text using the same function.
# While the terms and values will differ based on the dataset provided and the
# settings used, this is the same general approach used to generate Neurosynth's
# standard features.
tfidf_df = annotate.text.generate_counts(
    dset.texts,
    text_column="abstract",
    tfidf=True,
    max_df=0.99,
    min_df=0.01,
)
tfidf_df.head(5)

###############################################################################
# Add annotations to the Dataset
# -----------------------------------------------------------------------------
# Now we can add the generated annotations back into the Dataset object.
# The annotation functions return DataFrames with 'id' as the index, so we need
# to reset the index to make 'id' a column before assigning to the Dataset.
#
# This will replace any existing annotations. If you want to add to existing
# annotations instead of replacing them, you can merge the DataFrames:
# ``dset.annotations = pd.merge(dset.annotations, tfidf_df.reset_index(), on='id', how='left')``
dset.annotations = tfidf_df.reset_index()

# Now the Dataset has the new annotations
print(f"Dataset now has {len(dset.annotations.columns)} annotation columns")
dset.annotations.head(5)
