"""Test nimare.annotate.lda (LDA)."""
from nimare import annotate


def test_lda(testdata_laird):
    """A smoke test for LDA."""
    N_TOPICS = 5
    model = annotate.lda.LDAModel(
        n_topics=N_TOPICS,
        max_iter=100,
        text_column="abstract",
    )
    new_dset = model.transform(testdata_laird)
    topic_columns = [c for c in new_dset.annotations.columns if c.startswith("LDA")]
    assert len(topic_columns) == N_TOPICS
