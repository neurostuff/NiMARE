"""
Tf-Idf vectorization of text.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_tfidf(text_df):
    """
    Generate tf-idf weights for unigrams/bigrams derived from textual data.

    Parameters
    ----------
    text_df : :obj:`pandas.DataFrame`
        A DataFrame with two columns ('id' and 'text').

    Returns
    -------
    weights_df : :obj:`pandas.DataFrame`
        A DataFrame where the index is 'id' and the columns are the
        unigrams/bigrams derived from the data.
    """
    ids = text_df['id'].tolist()
    text = text_df['text'].tolist()
    stoplist = '/data/neurosynth/scripts/feature_extraction/stoplist.txt'
    with open(stoplist, 'r') as fo:
        stop_words = fo.read().splitlines()

    vectorizer = TfidfVectorizer(min_df=50, max_df=0.5,
                                 ngram_range=(1, 2), vocabulary=None,
                                 stop_words=stop_words)
    weights = vectorizer.fit_transform(text)
    names = vectorizer.get_feature_names()
    names = [str(name) for name in names]
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = 'id'
    return weights_df
