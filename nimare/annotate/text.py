"""
Text extraction tools.
"""
import logging
import os.path as op

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from ..utils import get_resource_path

LGR = logging.getLogger(__name__)


def generate_counts(text_df, text_column='abstract', tfidf=True,
                    min_df=50, max_df=0.5):
    """
    Generate tf-idf weights for unigrams/bigrams derived from textual data.

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
    orig_ids = text_df['id'].tolist()
    text_df = text_df.fillna('')
    keep_ids = text_df.loc[text_df[text_column] != '', 'id']
    text_df = text_df.loc[text_df['id'].isin(keep_ids)]

    if len(keep_ids) != len(orig_ids):
        LGR.info('Retaining {0}/{1} studies'.format(len(keep_ids),
                                                    len(orig_ids)))

    ids = text_df['id'].tolist()
    text = text_df[text_column].tolist()
    stoplist = op.join(get_resource_path(), 'neurosynth_stoplist.txt')
    with open(stoplist, 'r') as fo:
        stop_words = fo.read().splitlines()

    if tfidf:
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                     ngram_range=(1, 2), vocabulary=None,
                                     stop_words=stop_words)
    else:
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df,
                                     ngram_range=(1, 2), vocabulary=None,
                                     stop_words=stop_words)
    weights = vectorizer.fit_transform(text).toarray()
    names = vectorizer.get_feature_names()
    names = [str(name) for name in names]
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = 'id'
    return weights_df


def generate_cooccurrence(text_df, text_column='abstract', vocabulary=None,
                          window=5):
    """
    Build co-occurrence matrix from documents.
    Not the same approach as used by the GloVe model.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        A DataFrame with two columns ('id' and 'text'). D = document.
    vocabulary : :obj:`list`, optional
        List of words in vocabulary to extract from text.
    window : :obj:`int`, optional
        Window size for cooccurrence. Words which appear within window words
        of one another co-occur.

    Returns
    -------
    df : multi-indexed :obj:`pandas.DataFrame`
        A DataFrame with three indices (id, first_term, and second_term) and
        one column (cooccurrence_count).
    """
    if text_column not in text_df.columns:
        raise ValueError('Column "{0}" not found in DataFrame'.format(text_column))

    ids = text_df['id'].tolist()
    text = text_df[text_column].tolist()
    text = [nltk.word_tokenize(doc) for doc in text]
    text = [[word.lower() for word in doc if word.isalpha()] for doc in text]

    if vocabulary is None:
        all_words = [word for doc in text for word in doc]
        vocabulary = sorted(list(set(all_words)))

    cooc_arr = np.zeros((len(text), len(vocabulary), len(vocabulary)))
    for i, doc in enumerate(text):
        for j, word1 in enumerate(vocabulary):
            if word1 in doc:
                idx1 = [jj for jj, x in enumerate(doc) if x == word1]
                for k, word2 in enumerate(vocabulary):
                    if word2 in doc and k != j:
                        idx2 = [kk for kk, x in enumerate(doc) if x == word2]
                        distances = np.zeros((len(idx1), len(idx2)))
                        for m, idx1_ in enumerate(idx1):
                            for n, idx2_ in enumerate(idx2):
                                distances[m, n] = idx2_ - idx1_

                        cooc = np.sum(np.abs(distances) <= window)
                        cooc_arr[i, j, k] = cooc

    names = ['id', 'first_term', 'second_term']
    index = pd.MultiIndex.from_product([ids, vocabulary, vocabulary], names=names)
    df = pd.DataFrame({'cooccurrence_count': cooc_arr.flatten()}, index=index)
    return df
