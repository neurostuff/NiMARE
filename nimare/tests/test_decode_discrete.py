"""Test nimare.decode.discrete.

Tests for nimare.decode.discrete.gclda_decode_roi are in test_annotate_gclda.
"""

import pandas as pd
import pytest

from nimare.decode import discrete


def _make_synthetic_decode_data():
    """Build synthetic coordinates/annotations for decode regression tests.

    Uses enough studies (n=12, 6 selected) that per-feature counts clear the
    rare-feature threshold used internally by brainmap_decode, so BH correction
    actually changes the resulting p-values instead of being masked by other
    filtering.
    """
    n = 12
    ids = [f"s{i}" for i in range(n)]
    coordinates = pd.DataFrame({"id": ids, "x": range(n), "y": range(n), "z": range(n)})
    a = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
    b = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    annotations = pd.DataFrame({"id": ids, "a": a, "b": b})
    return coordinates, annotations, ids


@pytest.mark.parametrize("decode_fn", [discrete.neurosynth_decode, discrete.brainmap_decode])
def test_discrete_decode_default_correction_matches_bh(decode_fn):
    """Default ``correction`` must apply BH FDR correction, not silently skip it.

    Regression test: both ``neurosynth_decode`` and ``brainmap_decode`` default to
    ``correction="fdr_bh"``, but the correction dispatch only recognizes ``"bh"``,
    ``"by"``, and ``"bonferroni"``. That mismatch meant the default silently fell
    through to no correction at all, identical to ``correction=None``.
    """
    coordinates, annotations, ids = _make_synthetic_decode_data()
    kwargs = {
        "coordinates": coordinates,
        "annotations": annotations,
        "ids": ids[:6],
        "features": ["a", "b"],
    }
    columns = ["pForward", "pReverse"]

    default_df = decode_fn(**kwargs)
    uncorrected_df = decode_fn(**kwargs, correction=None)
    bh_df = decode_fn(**kwargs, correction="bh")

    # The default must actually apply correction, i.e. differ from uncorrected...
    assert not uncorrected_df[columns].equals(default_df[columns])
    # ...and must match explicitly requesting "bh", since that's the documented default.
    pd.testing.assert_frame_equal(default_df[columns], bh_df[columns])


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
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.NeurosynthDecoder(features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 6)


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
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.BrainMapDecoder(features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 6)


def test_BrainMapDecoder_failure(testdata_laird):
    """Smoke test for discrete.BrainMapDecoder where there are no features left."""
    decoder = discrete.BrainMapDecoder(features=["doggy"])
    with pytest.raises(Exception):
        decoder.fit(testdata_laird)


def test_ROIAssociationDecoder(testdata_laird, roi_img):
    """Smoke test for discrete.ROIAssociationDecoder."""
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.ROIAssociationDecoder(masker=roi_img, features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform()
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 1)
