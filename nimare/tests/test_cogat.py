"""
Test nimare.annotate.ontology.cogat (Cognitive Atlas extraction methods).
"""
import os.path as op

import pytest
import pandas as pd

import nimare
from nimare import annotate
from .utils import get_test_data_path


def test_cogat():
    """
    A smoke test for CogAt-related functions.
    """
    # A small test dataset with abstracts
    ns_dset_laird = nimare.dataset.Dataset.load(
        op.join(get_test_data_path(), 'neurosynth_laird_studies.pkl.gz'))
    id_df, rel_df = annotate.ontology.cogat.download_cogat(overwrite=False)
    weights = {'isKindOf': 1, 'isPartOf': 1, 'inCategory': 1}
    weights_df = annotate.ontology.utils._generate_weights(rel_df, weights=weights)
    counts_df, rep_text_df = annotate.ontology.cogat.extract_cogat(
        ns_dset_laird.texts, id_df, text_column='abstract')
    expanded_df = annotate.ontology.cogat.expand_counts(counts_df, rel_df, weights)
    assert isinstance(expanded_df, pd.DataFrame)
