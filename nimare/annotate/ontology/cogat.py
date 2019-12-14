"""
Automated annotation of Cognitive Atlas labels.
"""
import re
import time
import os
import os.path as op

import numpy as np
import pandas as pd
from cognitiveatlas.api import get_concept
from cognitiveatlas.api import get_task
from cognitiveatlas.api import get_disorder

from . import utils
from ..text import uk_to_us
from ...due import due
from ...utils import get_resource_path
from ... import references


def download_cogat(out_dir='auto', overwrite=False):
    """
    Download Cognitive Atlas ontology and combine Concepts, Tasks, and
    Disorders to create ID and relationship DataFrames.

    Parameters
    ----------
    out_dir : :obj:`str`, optional
        Output directory in which to write Cognitive Atlas files. Default is
        'auto', which writes the files to NiMARE's resources directory.
    overwrite : :obj:`bool`, optional
        Whether or not to overwrite existing files. Default is False.

    Returns
    -------
    aliases : :obj:`pandas.DataFrame`
        DataFrame containing CogAt identifiers, canonical names, and aliases,
        sorted by alias length (number of characters).
    relationships : :obj:`pandas.DataFrame`
        DataFrame containing associations between CogAt items, with three
        columns: input, output, and rel_type (relationship type).
    """
    if out_dir == 'auto':
        out_dir = op.join(get_resource_path(), 'ontologies/cognitive_atlas')
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = op.abspath(out_dir)

    ids_file = op.join(out_dir, 'cogat_aliases.csv')
    rels_file = op.join(out_dir, 'cogat_relationships.csv')
    if overwrite or not all([op.isfile(f) for f in [ids_file, rels_file]]):
        concepts = get_concept(silent=True).pandas
        tasks = get_task(silent=True).pandas
        disorders = get_disorder(silent=True).pandas

        # Identifiers and aliases
        long_concepts = utils._longify(concepts)
        long_tasks = utils._longify(tasks)

        # Disorders currently lack aliases
        disorders['name'] = disorders['name'].str.lower()
        disorders = disorders.assign(alias=disorders['name'])
        disorders = disorders[['id', 'name', 'alias']]

        # Combine into aliases DataFrame
        aliases = pd.concat((long_concepts, long_tasks, disorders), axis=0)
        aliases = utils._expand_df(aliases)
        aliases = aliases.replace('', np.nan)
        aliases = aliases.dropna(axis=0)
        aliases = aliases.reset_index(drop=True)

        # Relationships
        relationship_list = []
        for i, id_ in enumerate(concepts['id'].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, 'isSelf']
            relationship_list.append(row)
            concept = get_concept(id=id_, silent=True).json
            for rel in concept['relationships']:
                reltype = utils._get_concept_reltype(rel['relationship'],
                                                     rel['direction'])
                if reltype is not None:
                    row = [id_, rel['id'], reltype]
                    relationship_list.append(row)

        for i, id_ in enumerate(tasks['id'].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, 'isSelf']
            relationship_list.append(row)
            task = get_task(id=id_, silent=True).json
            for rel in task['concepts']:
                row = [id_, rel['concept_id'], 'measures']
                relationship_list.append(row)
                row = [rel['concept_id'], id_, 'measuredBy']
                relationship_list.append(row)

        for i, id_ in enumerate(disorders['id'].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, 'isSelf']
            relationship_list.append(row)
            disorder = get_disorder(id=id_, silent=True).json
            for rel in disorder['disorders']:
                if rel['relationship'] == 'ISA':
                    rel_type = 'isA'
                else:
                    rel_type = rel['relationship']
                row = [id_, rel['id'], rel_type]
                relationship_list.append(row)

        relationships = pd.DataFrame(columns=['input', 'output', 'rel_type'],
                                     data=relationship_list)
        ctp_df = concepts[['id', 'id_concept_class']]
        ctp_df = ctp_df.assign(rel_type='inCategory')
        ctp_df.columns = ['input', 'output', 'rel_type']
        ctp_df['output'].replace('', np.nan, inplace=True)
        ctp_df.dropna(axis=0, inplace=True)
        relationships = pd.concat((ctp_df, relationships))
        relationships = relationships.reset_index(drop=True)
        aliases.to_csv(ids_file, index=False)
        relationships.to_csv(rels_file, index=False)
    else:
        aliases = pd.read_csv(ids_file)
        relationships = pd.read_csv(rels_file)

    return aliases, relationships


@due.dcite(references.COGNITIVE_ATLAS, description='Introduces the Cognitive Atlas.')
class CogAtLemmatizer(object):
    """
    Replace synonyms and abbreviations with Cognitive Atlas [1]_ identifiers in
    text.

    Parameters
    ----------
    ontology_df : :obj:`pandas.DataFrame`, optional
        DataFrame with three columns (id, name, alias) and one row for each
        alias (e.g., synonym or abbreviation) for each term in the Cognitive
        Atlas. If None, loads ontology file from resources folder.

    Attributes
    ----------
    ontology_ : :obj:`pandas.DataFrame`
        Ontology in DataFrame form.
    regex_ : :obj:`dict`
        Dictionary linking aliases in ontology to regular expressions for
        lemmatization.

    References
    ----------
    .. [1] Poldrack, Russell A., et al. "The cognitive atlas: toward a
        knowledge foundation for cognitive neuroscience." Frontiers in
        neuroinformatics 5 (2011): 17. https://doi.org/10.3389/fninf.2011.00017
    """
    def __init__(self, ontology_df=None):
        if ontology_df is None:
            ontology_file = op.join(get_resource_path(), 'ontology',
                                    'cogat_ids.csv')
            self.ontology_ = pd.read_csv(ontology_file)
        else:
            assert isinstance(ontology_df, pd.DataFrame)
            self.ontology_ = ontology_df
        assert 'id' in self.ontology_.columns
        assert 'name' in self.ontology_.columns
        assert 'alias' in self.ontology_.columns

        # Create regex dictionary
        regex_dict = {}
        for term in ontology_df['alias'].values:
            term_for_regex = term.replace('(', r'\(').replace(')', r'\)')
            regex = '\\b' + term_for_regex + '\\b'
            pattern = re.compile(regex, re.MULTILINE | re.IGNORECASE)
            regex_dict[term] = pattern
        self.regex_ = regex_dict

    def transform(self, text, convert_uk=True):
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
        if convert_uk:
            text = uk_to_us(text)

        for term_idx in self.ontology_.index:
            term = self.ontology_['alias'].loc[term_idx]
            term_id = self.ontology_['id'].loc[term_idx]
            text = re.sub(self.regex_[term], term_id, text)
        return text


@due.dcite(references.COGNITIVE_ATLAS, description='Introduces the Cognitive Atlas.')
def extract_cogat(text_df, id_df, text_column='abstract'):
    """
    Extract Cognitive Atlas [1]_ terms and count instances using regular
    expressions.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        Pandas dataframe with two columns: 'id' and the text. D = document.
    id_df : (T x 3) :obj:`pandas.DataFrame`
        Cognitive Atlas ontology dataframe with three columns:
        'id' (unique identifier for term), 'alias' (natural language expression
        of term), and 'name' (preferred name of term; currently unused).
        T = term.
    text_column : :obj:`str`, optional
        Name of column in text_df that contains text. Default is 'abstract'.

    Returns
    -------
    counts_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for documents in the corpus.
    rep_text_df : (D x 2) :obj:`pandas.DataFrame`
        Text DataFrame with terms replaced with their CogAt IDs.

    References
    ----------
    .. [1] Poldrack, Russell A., et al. "The cognitive atlas: toward a
        knowledge foundation for cognitive neuroscience." Frontiers in
        neuroinformatics 5 (2011): 17. https://doi.org/10.3389/fninf.2011.00017
    """
    gazetteer = sorted(id_df['id'].unique().tolist())
    if 'id' in text_df.columns:
        text_df.set_index('id', inplace=True)

    # Create regex dictionary
    regex_dict = {}
    for term in id_df['alias'].values:
        term_for_regex = term.replace('(', r'\(').replace(')', r'\)')
        regex = '\\b' + term_for_regex + '\\b'
        pattern = re.compile(regex, re.MULTILINE | re.IGNORECASE)
        regex_dict[term] = pattern

    # Count
    count_arr = np.zeros((text_df.shape[0], len(gazetteer)))
    rep_text_df = text_df.copy()
    c = 0
    for i, row in text_df.iterrows():
        text = row[text_column]
        text = uk_to_us(text)
        for term_idx in id_df.index:
            term = id_df['alias'].loc[term_idx]
            term_id = id_df['id'].loc[term_idx]

            col_idx = gazetteer.index(term_id)

            pattern = regex_dict[term]
            count_arr[c, col_idx] += len(re.findall(pattern, text))
            text = re.sub(pattern, term_id, text)
            rep_text_df.loc[i, text_column] = text
        c += 1

    counts_df = pd.DataFrame(columns=gazetteer, index=text_df.index,
                             data=count_arr)
    return counts_df, rep_text_df


def expand_counts(counts_df, rel_df, weights=None):
    """
    Perform hierarchical expansion of counts across labels.

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
    weights_df = utils._generate_weights(rel_df, weights=weights)

    # First reorg count_df so it has the same columns in the same order as
    # weight_df
    counts_columns = counts_df.columns.tolist()
    weights_columns = weights_df.columns.tolist()
    w_not_c = set(weights_columns) - set(counts_columns)
    c_not_w = set(counts_columns) - set(weights_columns)
    if c_not_w:
        raise Exception('Columns found in counts but not weights: '
                        '{0}'.format(', '.join(c_not_w)))

    for col in w_not_c:
        counts_df[col] = 0

    counts_df = counts_df[weights_columns]

    # Now matrix multiplication
    counts = counts_df.values
    weights = weights_df.values
    weighted = np.dot(counts, weights)
    weighted_df = pd.DataFrame(index=counts_df.index,
                               columns=counts_df.columns,
                               data=weighted)
    return weighted_df
