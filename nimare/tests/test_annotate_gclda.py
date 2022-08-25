"""Test nimare.annotate.gclda (GCLDA)."""
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nimare import annotate, decode


def test_gclda_symmetric(testdata_laird):
    """A smoke test for GCLDA with symmetric regions."""
    counts_df = annotate.text.generate_counts(
        testdata_laird.texts,
        text_column="abstract",
        tfidf=False,
        min_df=1,
        max_df=1.0,
    )

    with pytest.raises(ValueError):
        annotate.gclda.GCLDAModel(
            counts_df,
            testdata_laird.coordinates,
            mask=testdata_laird.masker.mask_img,
            n_regions=3,
            symmetric=True,
        )

    model = annotate.gclda.GCLDAModel(
        counts_df,
        testdata_laird.coordinates,
        mask=testdata_laird.masker.mask_img,
        n_regions=2,
        symmetric=True,
    )
    model.fit(n_iters=5, loglikely_freq=5)

    # Create ROI to decode
    arr = np.zeros(testdata_laird.masker.mask_img.shape, int)
    arr[40:44, 45:49, 40:44] = 1
    mask_img = nib.Nifti1Image(arr, testdata_laird.masker.mask_img.affine)
    decoded_df, _ = decode.discrete.gclda_decode_roi(model, mask_img)
    assert isinstance(decoded_df, pd.DataFrame)

    # Decode the ROI as a continuous map
    decoded_df, _ = decode.continuous.gclda_decode_map(model, mask_img)
    assert isinstance(decoded_df, pd.DataFrame)

    # Encode text
    encoded_img, _ = decode.encode.gclda_encode(model, "fmri activation")
    assert isinstance(encoded_img, nib.Nifti1Image)


def test_gclda_asymmetric(testdata_laird):
    """A smoke test for GCLDA with three asymmetric regions."""
    counts_df = annotate.text.generate_counts(
        testdata_laird.texts,
        text_column="abstract",
        tfidf=False,
        min_df=1,
        max_df=1.0,
    )
    model = annotate.gclda.GCLDAModel(
        counts_df,
        testdata_laird.coordinates,
        mask=testdata_laird.masker.mask_img,
        n_regions=3,
        symmetric=False,
    )
    model.fit(n_iters=5, loglikely_freq=5)

    # Create ROI to decode
    arr = np.zeros(testdata_laird.masker.mask_img.shape, int)
    arr[40:44, 45:49, 40:44] = 1
    mask_img = nib.Nifti1Image(arr, testdata_laird.masker.mask_img.affine)
    decoded_df, _ = decode.discrete.gclda_decode_roi(model, mask_img)
    assert isinstance(decoded_df, pd.DataFrame)

    # Decode the ROI as a continuous map
    decoded_df, _ = decode.continuous.gclda_decode_map(model, mask_img)
    assert isinstance(decoded_df, pd.DataFrame)

    # Encode text
    encoded_img, _ = decode.encode.gclda_encode(model, "fmri activation")
    assert isinstance(encoded_img, nib.Nifti1Image)
