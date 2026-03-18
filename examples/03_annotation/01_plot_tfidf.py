"""

.. _annotations_tfidf:

===========================
Simple annotation from text
===========================

Perform simple term count or tf-idf value extraction from texts stored in a Studyset.
"""
import os

from nimare import annotate, dataset, utils
from nimare.nimads import Studyset

###############################################################################
# Load Studyset with abstracts
# -----------------------------------------------------------------------------
# The bundled example file uses the legacy Dataset JSON structure, so we load
# it once and immediately convert it to a Studyset.
dset = dataset.Dataset(os.path.join(utils.get_resource_path(), "neurosynth_laird_studies.json"))
studyset = Studyset.from_dataset(dset)
studyset.texts.head(2)

###############################################################################
# Generate term counts
# -----------------------------------------------------------------------------
# Let's start by extracting terms and their associated counts from article
# abstracts.
counts_df = annotate.text.generate_counts(
    studyset.texts,
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
    studyset.texts,
    text_column="abstract",
    tfidf=True,
    max_df=0.99,
    min_df=0.01,
)
tfidf_df.head(5)

###############################################################################
# Add annotations to the Studyset
# -----------------------------------------------------------------------------
# Now we can add the generated annotations back into the Studyset object.
# The annotation functions return DataFrames with 'id' as the index, so we need
# to reset the index to make 'id' a column before assigning to the Studyset.
#
# This will replace any existing annotations. If you want to add to existing
# annotations instead of replacing them, you can merge the DataFrames:
# ``studyset.annotations_df = studyset.annotations_df.merge(tfidf_df.reset_index(), on='id', how='left')``
studyset.annotations_df = tfidf_df.reset_index()

# Now the Studyset has the new annotations
print(f"Studyset now has {len(studyset.annotations_df.columns)} annotation columns")
studyset.annotations_df.head(5)
