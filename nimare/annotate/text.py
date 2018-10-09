"""
Text extraction tools.
"""
import re
import os.path as op

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
