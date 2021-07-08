"""Test nimare.decode.discrete.

Tests for nimare.decode.discrete.gclda_decode_roi are in test_annotate_gclda.
"""
import pandas as pd
import pytest

from nimare.decode import discrete


def test_neurosynth_decode(testdata_laird):
    """Smoke test for discrete.neurosynth_decode."""
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.neurosynth_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )
    assert isinstance(decoded_df, pd.DataFrame)


def test_brainmap_decode(testdata_laird):
    """Smoke test for discrete.brainmap_decode."""
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.brainmap_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )
    assert isinstance(decoded_df, pd.DataFrame)


def test_NeurosynthDecoder(testdata_laird):
    """Smoke test for discrete.NeurosynthDecoder."""
    ids = testdata_laird.ids[:5]
    decoder = discrete.NeurosynthDecoder()
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)


def test_NeurosynthDecoder_featuregroup(testdata_laird):
    """Smoke test for discrete.NeurosynthDecoder with feature group selection."""
    ids = testdata_laird.ids[:5]
    decoder = discrete.NeurosynthDecoder(feature_group="Neurosynth_TFIDF")
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)


def test_NeurosynthDecoder_featuregroup_failure(testdata_laird):
    """Smoke test for NeurosynthDecoder with feature group selection and no detected features."""
    decoder = discrete.NeurosynthDecoder(feature_group="Neurosynth_TFIDF", features=["01", "05"])
    with pytest.raises(Exception):
        decoder.fit(testdata_laird)


def test_BrainMapDecoder(testdata_laird):
    """Smoke test for discrete.BrainMapDecoder."""
    ids = testdata_laird.ids[:5]
    decoder = discrete.BrainMapDecoder()
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)


def test_BrainMapDecoder_failure(testdata_laird):
    """Smoke test for discrete.BrainMapDecoder where there are no features left."""
    decoder = discrete.BrainMapDecoder(features=["doggy"])
    with pytest.raises(Exception):
        decoder.fit(testdata_laird)
