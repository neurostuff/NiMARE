"""Text extraction tools."""
import logging
import os.path as op

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..utils import get_resource_path

LGR = logging.getLogger(__name__)


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
    if text_column not in text_df.columns:
        raise ValueError('Column "{0}" not found in DataFrame'.format(text_column))

    # Remove rows with empty text cells
    orig_ids = text_df["id"].tolist()
    text_df = text_df.fillna("")
    keep_ids = text_df.loc[text_df[text_column] != "", "id"]
    text_df = text_df.loc[text_df["id"].isin(keep_ids)]

    if len(keep_ids) != len(orig_ids):
        LGR.info("Retaining {0}/{1} studies".format(len(keep_ids), len(orig_ids)))

    ids = text_df["id"].tolist()
    text = text_df[text_column].tolist()
    stoplist = op.join(get_resource_path(), "neurosynth_stoplist.txt")
    with open(stoplist, "r") as fo:
        stop_words = fo.read().splitlines()

    if tfidf:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=None,
            stop_words=stop_words,
        )
    else:
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=None,
            stop_words=stop_words,
        )
    weights = vectorizer.fit_transform(text).toarray()
    names = vectorizer.get_feature_names()
    names = [str(name) for name in names]
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = "id"
    return weights_df
