"""
Test nimare.annotate.text (misc. text-based annotation methods).
"""
import os.path as op

import pytest
import pandas as pd

import nimare
from nimare import annotate
from .utils import get_test_data_path


def test_generate_cooccurrence():
    """
    A smoke test for generate_cooccurrence.
    """
    # A small test dataset with abstracts
    ns_dset_laird = nimare.dataset.Dataset.load(
        op.join(get_test_data_path(), 'neurosynth_laird_studies.pkl.gz'))
    df = annotate.text.generate_cooccurrence(
        ns_dset_laird.texts, text_column='abstract',
        vocabulary=['science', 'math'], window=2)
    assert isinstance(df, pd.DataFrame)
