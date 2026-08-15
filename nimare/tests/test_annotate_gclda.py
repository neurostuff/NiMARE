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


def test_gclda_loglikelihood_uses_zero_indexed_tokens(testdata_laird):
    """Log-likelihood must index documents/words directly, not offset by one.

    Regression test: the indices produced by docidx_mapper are already
    0-indexed, so subtracting 1 wrapped document 0 to the final document.
    """
    counts_df = annotate.text.generate_counts(
        testdata_laird.texts, text_column="abstract", tfidf=False, min_df=1, max_df=1.0
    )
    model = annotate.gclda.GCLDAModel(
        counts_df,
        testdata_laird.coordinates,
        mask=testdata_laird.masker.mask_img,
        n_topics=5,
        n_regions=2,
        symmetric=True,
    )
    model._update_regions()

    # Recompute the word log-likelihood independently, with correct indexing.
    alpha, beta, gamma = model.params["alpha"], model.params["beta"], model.params["gamma"]
    doccounts = model.topics["n_peak_tokens_doc_by_topic"] + gamma
    docprobs_z = doccounts / np.sum(doccounts, axis=1)[:, None]
    wordcounts = model.topics["n_word_tokens_word_by_topic"] + beta
    wordprobs = wordcounts / np.sum(wordcounts, axis=0)[None, :]
    p_w_g_d = np.dot(docprobs_z, wordprobs.T)

    expected_w = 0.0
    for i in range(len(model.data["wtoken_word_idx"])):
        w = model.data["wtoken_word_idx"][i]
        d = model.data["wtoken_doc_idx"][i]
        expected_w += np.log(p_w_g_d[d, w])

    _, w_loglikely, _ = model.compute_log_likelihood(update_vectors=False)
    assert np.isclose(w_loglikely, expected_w)


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
