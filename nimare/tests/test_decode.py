"""
Test nimare.meta.cbma.ale (ALE/SCALE meta-analytic algorithms).
"""
import pandas as pd
from nimare.decode import discrete


def test_discrete_neurosynth_decode(testdata_laird):
    """
    Smoke test for discrete.neurosynth_decode
    """
    ids = testdata_laird.ids[:5]
    decoded_df = discrete.neurosynth_decode(testdata_laird.coordinates, testdata_laird.annotations,
                                            ids=ids, correction=None)
    assert isinstance(decoded_df, pd.DataFrame)


def test_discrete_brainmap_decode(testdata_laird):
    """
    Smoke test for discrete.brainmap_decode
    """
    ids = testdata_laird.ids[:5]
    decoded_df = discrete.brainmap_decode(testdata_laird.coordinates, testdata_laird.annotations,
                                          ids=ids, correction=None)
    assert isinstance(decoded_df, pd.DataFrame)


def test_discrete_NeurosynthDecoder(testdata_laird):
    """
    Smoke test for discrete.NeurosynthDecoder
    """
    ids = testdata_laird.ids[:5]
    decoder = discrete.NeurosynthDecoder()
    decoded_df = decoder.transform(testdata_laird, ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)


def test_discrete_BrainMapDecoder(testdata_laird):
    """
    Smoke test for discrete.BrainMapDecoder
    """
    ids = testdata_laird.ids[:5]
    decoder = discrete.BrainMapDecoder()
    decoded_df = decoder.transform(testdata_laird, ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)
