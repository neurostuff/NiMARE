"""Automated annotation of Cognitive Atlas labels."""

import logging
import re

import numpy as np
import pandas as pd

from nimare.annotate import utils
from nimare.extract import download_cognitive_atlas
from nimare.utils import _uk_to_us

LGR = logging.getLogger(__name__)


class CogAtLemmatizer(object):
    """Replace synonyms and abbreviations with Cognitive Atlas identifiers in text.

    Parameters
    ----------
    ontology_df : :obj:`pandas.DataFrame`, optional
        DataFrame with three columns (id, name, alias) and one row for each alias
        (e.g., synonym or abbreviation) for each term in the Cognitive Atlas.
        If None, loads ontology file from resources folder. Default is None.

    Attributes
    ----------
    ontology_ : :obj:`pandas.DataFrame`
        Ontology in DataFrame form.
    regex_ : :obj:`dict`
        Dictionary linking aliases in ontology to regular expressions for lemmatization.

    Notes
    -----
    The Cognitive Atlas :footcite:p:`poldrack2011cognitive` is an ontology for describing
    cognitive neuroscience concepts and tasks.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nimare.extract.download_cognitive_atlas : This function will be called automatically if
                                              ``ontology_df`` is not provided.
    """

    def __init__(self, ontology_df=None):
        if ontology_df is None:
            cogat = download_cognitive_atlas()
            self.ontology_ = pd.read_csv(cogat["ids"])
        else:
            assert isinstance(ontology_df, pd.DataFrame)
            self.ontology_ = ontology_df
        assert "id" in self.ontology_.columns
        assert "name" in self.ontology_.columns
        assert "alias" in self.ontology_.columns

        # Create regex dictionary
        regex_dict = {}
        for term in ontology_df["alias"].values:
            term_for_regex = term.replace("(", r"\(").replace(")", r"\)")
            regex = "\\b" + term_for_regex + "\\b"
            pattern = re.compile(regex, re.MULTILINE | re.IGNORECASE)
            regex_dict[term] = pattern
        self.regex_ = regex_dict

    def transform(self, text, convert_uk=True):
        """Replace terms in text with unique Cognitive Atlas identifiers.

        Parameters
        ----------
        text : :obj:`str`
            Text to convert.
        convert_uk : :obj:`bool`, optional
            Convert British English words in text to American English versions.
            Default is True.

        Returns
        -------
        text : :obj:`str`
            Text with Cognitive Atlas terms replaced with unique Cognitive
            Atlas identifiers.
        """
        if convert_uk:
            text = _uk_to_us(text)

        for term_idx in self.ontology_.index:
            term = self.ontology_["alias"].loc[term_idx]
            term_id = self.ontology_["id"].loc[term_idx]
            text = re.sub(self.regex_[term], term_id, text)
        return text


def extract_cogat(text_df, id_df=None, text_column="abstract"):
    """Extract Cognitive Atlas terms and count instances using regular expressions.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        Pandas dataframe with at least two columns: 'id' and the text.
        D = document.

    id_df : (T x 3) :obj:`pandas.DataFrame`
        Cognitive Atlas ontology dataframe with one row for each term and at least three columns:

        - ``"id"``: A unique identifier for each term.
        - ``"alias"``: A natural language expression for each term.
        - ``"name"``: The preferred name of each term. Currently unused.

    text_column : :obj:`str`, optional
        Name of column in text_df that contains text. Default is 'abstract'.

    Returns
    -------
    counts_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for documents in the corpus.
        One row for each document and one column for each term.

    rep_text_df : (D x 2) :obj:`pandas.DataFrame`
        An updated version of the ``text_df`` DataFrame with terms in the text column replaced
        with their CogAt IDs.

    Notes
    -----
    The Cognitive Atlas :footcite:p:`poldrack2011cognitive` is an ontology for describing
    cognitive neuroscience concepts and tasks.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nimare.extract.download_cognitive_atlas : This function will be called automatically if
                                              ``id_df`` is not provided.
    """
    text_df = text_df.copy()
    if id_df is None:
        cogat = download_cognitive_atlas()
        id_df = pd.read_csv(cogat["ids"])
    gazetteer = sorted(id_df["id"].unique().tolist())
    if "id" in text_df.columns:
        text_df.set_index("id", inplace=True)

    text_df[text_column] = text_df[text_column].fillna("")
    text_df[text_column] = text_df[text_column].apply(_uk_to_us)

    # Create regex dictionary
    regex_dict = {}
    for term in id_df["alias"].values:
        term_for_regex = term.replace("(", r"\(").replace(")", r"\)")
        regex = "\\b" + term_for_regex + "\\b"
        pattern = re.compile(regex, re.MULTILINE | re.IGNORECASE)
        regex_dict[term] = pattern

    # Count
    count_arr = np.zeros((text_df.shape[0], len(gazetteer)), int)
    counts_df = pd.DataFrame(columns=gazetteer, index=text_df.index, data=count_arr)
    for term_idx in id_df.index:
        term = id_df["alias"].loc[term_idx]
        term_id = id_df["id"].loc[term_idx]
        pattern = regex_dict[term]
        counts_df[term_id] += text_df[text_column].str.count(pattern).astype(int)
        text_df[text_column] = text_df[text_column].str.replace(pattern, term_id, regex=True)

    return counts_df, text_df


def expand_counts(counts_df, rel_df=None, weights=None):
    """Perform hierarchical expansion of counts across labels.

    Parameters
    ----------
    counts_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for a corpus. T = term, D = document.
    rel_df : :obj:`pandas.DataFrame`
        Long-form DataFrame of term-term relationships with at least three columns:
        'input', 'output', and 'rel_type'.
    weights : :obj:`dict`
        Dictionary of weights per relationship type. E.g., {'isKind': 1}.
        Unspecified relationship types default to 0.

    Returns
    -------
    weighted_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for a corpus after hierarchical expansion.
    """
    if rel_df is None:
        cogat = download_cognitive_atlas()
        rel_df = pd.read_csv(cogat["relationships"])
    weights_df = utils._generate_weights(rel_df, weights=weights)

    # First reorg counts_df so it has the same columns in the same order as
    # weight_df
    counts_columns = counts_df.columns.tolist()
    weights_columns = weights_df.columns.tolist()
    w_not_c = set(weights_columns) - set(counts_columns)
    c_not_w = set(counts_columns) - set(weights_columns)
    if c_not_w:
        raise Exception(f"Columns found in counts but not weights: {', '.join(c_not_w)}")

    for col in w_not_c:
        counts_df[col] = 0

    counts_df = counts_df[weights_columns]

    # Now matrix multiplication
    counts = counts_df.values
    weights = weights_df.values
    weighted = np.dot(counts, weights)
    weighted_df = pd.DataFrame(index=counts_df.index, columns=counts_df.columns, data=weighted)
    return weighted_df
