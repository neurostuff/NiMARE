"""Text extraction tools."""

import logging
import os.path as op
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nimare.utils import get_resource_path

LGR = logging.getLogger(__name__)


def _prepare_text_documents(text_df, text_column):
    """Validate text input and return filtered ids/text values."""
    if text_column not in text_df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    orig_ids = text_df["id"].tolist()
    text_df = text_df.fillna("")
    keep_ids = text_df.loc[text_df[text_column] != "", "id"]
    text_df = text_df.loc[text_df["id"].isin(keep_ids)]

    if len(keep_ids) != len(orig_ids):
        LGR.info(f"Retaining {len(keep_ids)}/{len(orig_ids)} studies")

    ids = text_df["id"].tolist()
    text = text_df[text_column].tolist()
    return ids, text


@lru_cache(maxsize=1)
def _load_stop_words():
    """Load the default Neurosynth stop-word list."""
    stoplist = op.join(get_resource_path(), "neurosynth_stoplist.txt")
    with open(stoplist, "r") as fo:
        return fo.read().splitlines()


def _generate_weights(
    text_df,
    text_column="abstract",
    tfidf=True,
    min_df=50,
    max_df=0.5,
    dtype=None,
):
    """Generate a vectorized document-term matrix and associated metadata."""
    ids, text = _prepare_text_documents(text_df, text_column=text_column)
    stop_words = _load_stop_words()

    vectorizer_class = TfidfVectorizer if tfidf else CountVectorizer
    vectorizer_kwargs = {
        "min_df": min_df,
        "max_df": max_df,
        "ngram_range": (1, 2),
        "vocabulary": None,
        "stop_words": stop_words,
    }
    if dtype is not None:
        vectorizer_kwargs["dtype"] = dtype

    vectorizer = vectorizer_class(**vectorizer_kwargs)
    weights = vectorizer.fit_transform(text)

    if hasattr(vectorizer, "get_feature_names_out"):
        names = vectorizer.get_feature_names_out()
    else:
        names = vectorizer.get_feature_names()

    names = [str(name) for name in names]
    return weights, names, ids


def generate_counts(text_df, text_column="abstract", tfidf=True, min_df=50, max_df=0.5):
    """Generate tf-idf weights for unigrams/bigrams derived from textual data.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        A DataFrame with two columns ('id' and 'text'). D = document.

    Returns
    -------
    weights_df : (D x T) :obj:`pandas.DataFrame`
        A DataFrame where the index is 'id' and the columns are the
        unigrams/bigrams derived from the data. D = document. T = term.
    """
    weights, names, ids = _generate_weights(
        text_df,
        text_column=text_column,
        tfidf=tfidf,
        min_df=min_df,
        max_df=max_df,
    )
    weights = weights.toarray()
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = "id"
    return weights_df
