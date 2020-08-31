"""
Test nimare.decode.discrete.

Tests for nimare.decode.discrete.gclda_decode_roi are in test_annotate_gclda.
"""
import pandas as pd
import pytest

from nimare.decode import continuous
from nimare.meta import mkda, kernel


def test_CorrelationDecoder(testdata_laird, tmp_path_factory):
    """
    Smoke test for continuous.CorrelationDecoder
    """
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

    meta = mkda.KDA()
    res = meta.fit(testdata_laird)
    img = res.get_map("of")
    decoded_df = decoder.transform(img)
    assert isinstance(decoded_df, pd.DataFrame)
    raise Exception(decoded_df)


def test_CorrelationDistributionDecoder(testdata_laird, tmp_path_factory):
    """
    Smoke test for continuous.CorrelationDistributionDecoder
    """
    tmpdir = tmp_path_factory.mktemp("test_CorrelationDistributionDecoder")

    testdata_laird = testdata_laird.copy()
    features = testdata_laird.get_labels(ids=testdata_laird.ids[0])[:5]

    decoder = continuous.CorrelationDistributionDecoder(features=features)

    # No basepath
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

    meta = mkda.KDA()
    res = meta.fit(testdata_laird)
    img = res.get_map("of")
    decoded_df = decoder.transform(img)
    assert isinstance(decoded_df, pd.DataFrame)
    raise Exception(decoded_df)
