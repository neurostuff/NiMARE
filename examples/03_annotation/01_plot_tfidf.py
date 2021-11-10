# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _annotations_tfidf:

===========================
Simple annotation from text
===========================

NiMARE contains a function for simple term count or tf-idf value extraction
from texts stored in a Dataset.
"""
import os

import nimare
from nimare import annotate
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load dataset with abstracts
# ---------------------------
# We'll load a small dataset composed only of studies in Neurosynth with
# Angela Laird as a coauthor, for the sake of speed.
dset = nimare.dataset.Dataset(os.path.join(get_test_data_path(), "neurosynth_laird_studies.json"))
dset.texts.head(2)

###############################################################################
# Generate term counts
# --------------------
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
# --------------------
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
