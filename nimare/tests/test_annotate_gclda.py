"""
Test nimare.annotate.gclda (GCLDA).
"""
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd

import nimare
from nimare import annotate, decode

from .utils import get_test_data_path


def test_gclda():
    """
    A smoke test for GCLDA.
    """
    # A small test dataset with abstracts
    dset = nimare.dataset.Dataset.load(
        op.join(get_test_data_path(), "neurosynth_laird_studies.pkl.gz")
    )
    counts_df = annotate.text.generate_counts(
        dset.texts, text_column="abstract", tfidf=False, min_df=1, max_df=1.0
    )
    model = annotate.gclda.GCLDAModel(counts_df, dset.coordinates, mask=dset.masker.mask_img)
    model.fit(n_iters=5, loglikely_freq=5)
    arr = np.zeros(dset.masker.mask_img.shape, int)
    arr[40:44, 45:49, 40:44] = 1
    mask_img = nib.Nifti1Image(arr, dset.masker.mask_img.affine)
    decoded_df, _ = decode.discrete.gclda_decode_roi(model, mask_img)
    assert isinstance(decoded_df, pd.DataFrame)
    decoded_df, _ = decode.continuous.gclda_decode_map(model, mask_img)
    assert isinstance(decoded_df, pd.DataFrame)
    encoded_img, _ = decode.encode.gclda_encode(model, "fmri activation")
    assert isinstance(encoded_img, nib.Nifti1Image)
