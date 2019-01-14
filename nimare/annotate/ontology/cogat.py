"""
Automated annotation of Cognitive Atlas labels.
"""
from ..text import uk_to_us
from ...due import due, Doi


def pull_ontology(out_dir='auto', overwrite=False):
    """
    Download Cognitive Atlas ontology and combine Concepts, Tasks, and
    Disorders to create ID and relationship DataFrames.
    """
    pass


@due.dcite(Doi('10.3389/fninf.2011.00017'),
           description='Introduces the Cognitive Atlas.')
class CogAtLemmatizer(object):
    """
    Replace synonyms and abbreviations with Cognitive Atlas identifiers in
    text.

    Parameters
    ----------
    ontology_df : :obj:`pandas.DataFrame`, optional
        DataFrame with three columns (id, name, alias) and one row for each
        alias (e.g., synonym or abbreviation) for each term in the Cognitive
        Atlas. If None, loads ontology file from resources folder.
    """
    def __init__(self, ontology_df=None):
        pass

    def lemmatize(self, text, convert_uk=True):
        """
        Replace terms in text with unique Cognitive Atlas identifiers.

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
        pass


@due.dcite(Doi('10.3389/fninf.2011.00017'),
           description='Introduces the Cognitive Atlas.')
def extract_cogat(text_df, id_df):
    """
    Extract CogAt terms and perform hierarchical expansion.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        Pandas dataframe with two columns: 'id' and 'text'. D = document.
    id_df : (T x 3) :obj:`pandas.DataFrame`
        Cognitive Atlas ontology dataframe with three columns:
        'id' (unique identifier for term), 'alias' (natural language expression
        of term), and 'name' (preferred name of term; currently unused).
        T = term.

    Returns
    -------
    counts_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for documents in the corpus.
    rep_text_df : (D x 2) :obj:`pandas.DataFrame`
        Text DataFrame with terms replaced with their CogAt IDs.
    """
    pass


def expand_counts(counts_df, rel_df, weights=None):
    """
    Perform hierarchical expansion of CogAt labels.

    Parameters
    ----------
    counts_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for a corpus. T = term, D = document.
    rel_df : :obj:`pandas.DataFrame`
        Long-form DataFrame of term-term relationships with three columns:
        'input', 'output', and 'rel_type'.
    weights : :obj:`dict`
        Dictionary of weights per relationship type. E.g., {'isKind': 1}.
        Unspecified relationship types default to 0.

    Returns
    -------
    weighted_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for a corpus after hierarchical expansion.
    """
    pass
