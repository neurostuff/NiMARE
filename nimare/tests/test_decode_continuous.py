"""
Test nimare.decode.discrete.

Tests for nimare.decode.discrete.gclda_decode_roi are in test_annotate_gclda.
"""
import shutil
import tempfile

import pandas as pd
import pytest

from nimare.decode import continuous
from nimare.meta import mkda


def test_CorrelationDecoder(testdata_laird):
    """
    Smoke test for discrete.neurosynth_decode
    """
    testdata_laird = testdata_laird.copy()
    features = testdata_laird.get_labels(ids=testdata_laird.ids[0])[:5]
    decoder = continuous.CorrelationDecoder(features=features)

    with pytest.raises(ValueError):
        decoder.fit(testdata_laird)

    temp_dir = tempfile.mkdtemp()
    testdata_laird.update_path(temp_dir)
    decoder.fit(testdata_laird)

    meta = mkda.KDA()
    res = meta.fit(testdata_laird)
    img = res.get_map("of")
    decoded_df = decoder.transform(img)
    assert isinstance(decoded_df, pd.DataFrame)
    shutil.rmtree(temp_dir)
