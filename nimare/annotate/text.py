"""
Text extraction tools.
"""


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
    pass


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
    pass


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
    pass
