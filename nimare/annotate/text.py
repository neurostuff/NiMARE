"""
Text extraction tools.
"""
import re
import os.path as op

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from ..utils import get_resource_path

SPELL_DF = pd.read_csv(op.join(get_resource_path(), 'english_spellings.csv'),
                       index_col='UK')
SPELL_DICT = SPELL_DF['US'].to_dict()


def generate_counts(text_df, tfidf=True):
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
    ids = text_df['id'].tolist()
    text = text_df['text'].tolist()
    stoplist = op.join(get_resource_path(), 'neurosynth_stoplist.txt')
    with open(stoplist, 'r') as fo:
        stop_words = fo.read().splitlines()

    if tfidf:
        vectorizer = TfidfVectorizer(min_df=50, max_df=0.5,
                                     ngram_range=(1, 2), vocabulary=None,
                                     stop_words=stop_words)
    else:
        vectorizer = CountVectorizer(min_df=50, max_df=0.5,
                                     ngram_range=(1, 2), vocabulary=None,
                                     stop_words=stop_words)
    weights = vectorizer.fit_transform(text).toarray()
    names = vectorizer.get_feature_names()
    names = [str(name) for name in names]
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = 'id'
    return weights_df


def generate_cooccurrence(text_df, vocabulary=None, window=5):
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
    df : (V, V, D) :obj:`pandas.Panel`
        One cooccurrence matrix per document in text_df.
    """
    ids = text_df['id'].tolist()
    text = text_df['text'].tolist()
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

    df = pd.Panel(items=ids, major_axis=vocabulary, minor_axis=vocabulary,
                  data=cooc_arr)
    return df


def uk_to_us(text):
    """
    Convert UK spellings to US based on a converter.

    english_spellings.csv: From http://www.tysto.com/uk-us-spelling-list.html

    Parameters
    ----------
    text : :obj:`str`

    Returns
    -------
    text : :obj:`str`
    """
    if isinstance(text, str):
        # Convert British to American English
        pattern = re.compile(r'\b(' + '|'.join(SPELL_DICT.keys()) + r')\b')
        text = pattern.sub(lambda x: SPELL_DICT[x.group()], text)
    return text
