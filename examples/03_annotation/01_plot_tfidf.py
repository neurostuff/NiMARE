"""

.. _annotations_tfidf:

===========================
Simple annotation from text
===========================

Perform simple term count or tf-idf value extraction from texts stored in a Studyset.
"""
import os

from nimare import annotate, utils
from nimare.nimads import Studyset

###############################################################################
# Load Studyset with abstracts
# -----------------------------------------------------------------------------
studyset = Studyset(
    os.path.join(utils.get_resource_path(), "neurosynth_laird_studyset.json"),
    target="mni152_2mm",
)
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
# to reset the index to make 'id' a column.
#
# A Studyset is immutable, so ``with_annotations_df`` returns a new one.
# ``replace=True`` discards any existing annotations; leave it out to add these
# labels alongside them, which ``annotations_df`` will then merge.
studyset = studyset.with_annotations_df(tfidf_df.reset_index(), name="tfidf", replace=True)

# Now the Studyset has the new annotations
print(f"Studyset now has {len(studyset.annotations_df.columns)} annotation columns")
studyset.annotations_df.head(5)
