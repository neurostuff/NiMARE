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

    # The topics arrive as their own annotation set, so the studyset's existing
    # annotations are untouched and analyses without text are not dropped.
    topics = new_dset.store.annotations[f"LDA{N_TOPICS}"]
    assert len(topics.keys()) == N_TOPICS
    assert all(label.startswith("LDA") for label in topics.keys())
    assert len(new_dset.ids) == len(testdata_laird.ids)

    assert hasattr(model, "distributions_")
    assert "p_topic_g_word" in model.distributions_.keys()
    assert isinstance(model.distributions_["p_topic_g_word"], np.ndarray)
    assert model.distributions_["p_topic_g_word"].shape[0] == N_TOPICS
    assert "p_topic_g_word_df" in model.distributions_.keys()
    assert isinstance(model.distributions_["p_topic_g_word_df"], pd.DataFrame)
    assert model.distributions_["p_topic_g_word_df"].shape[0] == N_TOPICS
