"""Test nimare.annotate.lda (LDA)."""
import numpy as np
import pandas as pd

from nimare import annotate


def test_lda(testdata_laird):
    """A smoke test for LDA."""
    N_TOPICS = 5
    model = annotate.lda.LDAModel(
        n_topics=N_TOPICS,
        max_iter=100,
        text_column="abstract",
    )
    new_dset = model.fit(testdata_laird)
    topic_columns = [c for c in new_dset.annotations.columns if c.startswith("LDA")]
    assert len(topic_columns) == N_TOPICS

    assert hasattr(model, "distributions_")
    assert "p_topic_g_word" in model.distributions_.keys()
    assert isinstance(model.distributions_["p_topic_g_word"], np.ndarray)
    assert model.distributions_["p_topic_g_word"].shape[0] == N_TOPICS
    assert "p_topic_g_word_df" in model.distributions_.keys()
    assert isinstance(model.distributions_["p_topic_g_word_df"], pd.DataFrame)
    assert model.distributions_["p_topic_g_word_df"].shape[0] == N_TOPICS
