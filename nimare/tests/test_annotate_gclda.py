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
    delta = model.params["delta"]
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

    # Recompute the peak log-likelihood independently, with correct
    # (non-offset) indexing into ptoken_doc_idx. This is the other half of
    # the off-by-one fix: compute_log_likelihood previously subtracted 1
    # from ptoken_doc_idx (already 0-indexed) as well, which the word-only
    # assertion above cannot detect.
    doccounts_y = model.topics["n_peak_tokens_doc_by_topic"] + alpha
    docprobs_y = doccounts_y / np.sum(doccounts_y, axis=1)[:, None]
    regioncounts = model.topics["n_peak_tokens_region_by_topic"] + delta
    regionprobs = regioncounts / np.sum(regioncounts, axis=0)[None, :]
    peak_probs = model._get_peak_probs(model)

    expected_x = 0.0
    for i in range(len(model.data["ptoken_doc_idx"])):
        doc = model.data["ptoken_doc_idx"][i]
        p_x = 0
        for j_region in range(model.params["n_regions"]):
            p_topic_g_doc = docprobs_y[doc]
            p_region_g_topic = regionprobs[j_region]
            p_region_g_doc = p_topic_g_doc * p_region_g_topic
            p_x_r = peak_probs[i, :, j_region]
            p_x_rd = np.dot(p_region_g_doc, p_x_r)
            p_x += p_x_rd
        expected_x += np.log(p_x)

    x_loglikely, w_loglikely, _ = model.compute_log_likelihood(update_vectors=False)
    assert np.isclose(x_loglikely, expected_x)
    assert np.isclose(w_loglikely, expected_w)


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
