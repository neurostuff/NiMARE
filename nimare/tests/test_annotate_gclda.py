"""Test nimare.annotate.gclda (GCLDA)."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nimare import annotate, decode


def test_gclda_sampling_rejects_nonpositive_total():
    """Sampling should fail fast on degenerate unnormalized weights."""
    with pytest.raises(ValueError, match="positive"):
        annotate.gclda._sample_from_unnormalized(np.zeros(3, dtype=np.float64))


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
    arr = np.zeros(testdata_laird.masker.mask_img.shape, np.int32)
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


def test_gclda_inv3_logdet_matches_lapack_and_is_symmetric():
    """Closed-form 3x3 inverse must agree with LAPACK and be exactly symmetric."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = rng.normal(size=(3, 3)) * rng.uniform(1, 60)
        sigma = m @ m.T + 50.0 * np.eye(3) * rng.uniform(0.1, 3)

        inv, logdet = annotate.gclda._inv3_logdet(sigma)

        assert np.allclose(inv, np.linalg.inv(sigma), rtol=1e-10)
        _, ref_logdet = np.linalg.slogdet(sigma)
        assert np.isclose(logdet, ref_logdet, rtol=1e-12)
        # Inverse of a symmetric matrix must itself be exactly symmetric.
        assert np.array_equal(inv, inv.T)
        # Deterministic: identical inputs give identical bits.
        inv2, logdet2 = annotate.gclda._inv3_logdet(sigma.copy())
        assert np.array_equal(inv, inv2) and logdet == logdet2


def test_gclda_inv3_logdet_rejects_nonpositive_definite():
    """A non-positive-definite covariance must raise, as the LAPACK path did."""
    singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError):
        annotate.gclda._inv3_logdet(singular)


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
    arr = np.zeros(testdata_laird.masker.mask_img.shape, np.int32)
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
