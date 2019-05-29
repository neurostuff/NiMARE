"""
Text extraction tools.
"""
import re
import logging
import os.path as op

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from ..dataset import Dataset
from ..utils import get_resource_path

LGR = logging.getLogger(__name__)

SPELL_DF = pd.read_csv(op.join(get_resource_path(), 'english_spellings.csv'),
                       index_col='UK')
SPELL_DICT = SPELL_DF['US'].to_dict()


def download_abstracts(dataset, email):
    """
    Download the abstracts for a list of PubMed IDs. Uses the BioPython
    package.

    Parameters
    ----------
    dataset : :obj:`nimare.dataset.Dataset` or :obj:`list` of :obj:`str`
        A Dataset object where IDs are in the form PMID-EXPID or a list of
        PubMed IDs
    email : :obj:`str`
        Email address to use to call the PubMed API

    Returns
    -------
    dataset : :obj:`nimare.dataset.Dataset` or :obj:`list` of :obj:`str`
        Dataset with abstracts added.
    """
    try:
        from Bio import Entrez, Medline
    except:
        raise Exception(
            'Module biopython is required for downloading abstracts from '
            'PubMed.')

    Entrez.email = email

    if isinstance(dataset, Dataset):
        pmids = dataset.coordinates['id'].astype(str).tolist()
        pmids = [pmid.split('-')[0] for pmid in pmids]
        pmids = sorted(list(set(pmids)))
    elif isinstance(dataset, list):
        pmids = [str(pmid) for pmid in dataset]
    else:
        raise Exception(
            'Dataset type not recognized: {0}'.format(type(dataset)))

    records = []
    # PubMed only allows you to search ~1000 at a time. I chose 900 to be safe.
    chunks = [pmids[x: x + 900] for x in range(0, len(pmids), 900)]
    for i, chunk in enumerate(chunks):
        LGR.info('Downloading chunk {0} of {1}'.format(i + 1, len(chunks)))
        h = Entrez.efetch(db='pubmed', id=chunk, rettype='medline',
                          retmode='text')
        records += list(Medline.parse(h))

    # Pull data for studies with abstracts
    data = [[study['PMID'], study['AB']]
            for study in records if study.get('AB', None)]
    df = pd.DataFrame(columns=['id', 'text'], data=data)

    for pmid in dataset.data.keys():
        if pmid in df['id'].tolist():
            abstract = df.loc[df['id'] == pmid, 'text'].values[0]
        else:
            abstract = ""

        for expid in dataset.data[pmid]['contrasts'].keys():
            if 'texts' not in dataset.data[pmid]['contrasts'][expid].keys():
                dataset.data[pmid]['contrasts'][expid]['texts'] = {}
            dataset.data[pmid]['contrasts'][expid]['texts']['abstract'] = abstract

    dataset._load_texts()
    return dataset


def generate_counts(text_df, text_column='abstract', tfidf=True):
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
    df : (V, V, D) :obj:`pandas.Panel`
        One cooccurrence matrix per document in text_df.
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
