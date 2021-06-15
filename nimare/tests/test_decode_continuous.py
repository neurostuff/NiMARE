"""Test nimare.decode.continuous.

Tests for nimare.decode.continuous.gclda_decode_map are in test_annotate_gclda.
"""
import pandas as pd
import pytest

from nimare.decode import continuous
from nimare.meta import kernel, mkda


def test_CorrelationDecoder_smoke(testdata_laird, tmp_path_factory):
    """Smoke test for continuous.CorrelationDecoder."""
    tmpdir = tmp_path_factory.mktemp("test_CorrelationDecoder")

    testdata_laird = testdata_laird.copy()
    features = testdata_laird.get_labels(ids=testdata_laird.ids[0])[:5]
    decoder = continuous.CorrelationDecoder(features=features)

    # No basepath
    with pytest.raises(ValueError):
        decoder.fit(testdata_laird)

    # Let's add the path
    testdata_laird.update_path(tmpdir)
    decoder.fit(testdata_laird)

    # Make an image to decode
    meta = mkda.KDA(null_method="approximate")
    res = meta.fit(testdata_laird)
    img = res.get_map("stat")
    decoded_df = decoder.transform(img)
    assert isinstance(decoded_df, pd.DataFrame)


def test_CorrelationDistributionDecoder_smoke(testdata_laird, tmp_path_factory):
    """Smoke test for continuous.CorrelationDistributionDecoder."""
    tmpdir = tmp_path_factory.mktemp("test_CorrelationDistributionDecoder")

    testdata_laird = testdata_laird.copy()
    features = testdata_laird.get_labels(ids=testdata_laird.ids[0])[:5]

    decoder = continuous.CorrelationDistributionDecoder(features=features)

    # No images of the requested type
    with pytest.raises(ValueError):
        decoder.fit(testdata_laird)

    # Let's add the path
    testdata_laird.update_path(tmpdir)

    # Then let's make some images to decode
    kern = kernel.MKDAKernel(r=10, value=1)
    dset = kern.transform(testdata_laird, return_type="dataset")

    # And now we have images we can use for decoding!
    decoder.set_params(target_image=kern.image_type)
    decoder.fit(dset)

    # Make an image to decode
    meta = mkda.KDA(null_method="approximate")
    res = meta.fit(testdata_laird)
    img = res.get_map("stat")
    decoded_df = decoder.transform(img)
    assert isinstance(decoded_df, pd.DataFrame)
