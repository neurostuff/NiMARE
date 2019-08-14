"""
Automated annotation of Cognitive Atlas labels.
"""
import re
import time
import os.path as op

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from cognitiveatlas.api import get_concept
from cognitiveatlas.api import get_task
from cognitiveatlas.api import get_disorder

from ..text import uk_to_us
from ...due import due
from ...utils import get_resource_path
from ... import references


def _longify(df):
    """
    Expand comma-separated lists of aliases in DataFrame into separate rows.
    """
    reduced = df[['id', 'name', 'alias']]
    rows = []
    for index, row in reduced.iterrows():
        if isinstance(row['alias'], str) and ',' in row['alias']:
            aliases = row['alias'].split(', ') + [row['name']]
        else:
            aliases = [row['name']]

        for alias in aliases:
            rows.append([row['id'], row['name'].lower(),
                         alias.lower()])
    out_df = pd.DataFrame(columns=['id', 'name', 'alias'], data=rows)
    out_df = out_df.replace('', np.nan)
    return out_df


def _get_ratio(tup):
    """
    Get fuzzy ratio.
    """
    if all(isinstance(t, str) for t in tup):
        return fuzz.ratio(tup[0], tup[1])
    else:
        return 100


def _gen_alt_forms(term):
    """
    Generate a list of alternate forms for a given term.
    """
    if not isinstance(term, str) or len(term) == 0:
        return [None]

    alt_forms = []
    # For one alternate form, put contents of parentheses at beginning of term
    if '(' in term:
        prefix = term[term.find('(') + 1:term.find(')')]
        temp_term = term.replace('({0})'.format(prefix), '').replace('  ', ' ')
        alt_forms.append(temp_term)
        alt_forms.append('{0} {1}'.format(prefix, temp_term))
    else:
        prefix = ''

    # Remove extra spaces
    alt_forms = [s.strip() for s in alt_forms]

    # Allow plurals
    # temp = [s+'s' for s in alt_forms]
    # temp += [s+'es' for s in alt_forms]
    # alt_forms += temp

    # Remove words "task" and/or "paradigm"
    alt_forms += [term.replace(' task', '') for term in alt_forms]
    alt_forms += [term.replace(' paradigm', '') for term in alt_forms]

    # Remove duplicates
    alt_forms = list(set(alt_forms))
    return alt_forms


def _get_concept_reltype(relationship, direction):
    """
    Convert two-part relationship info (relationship type and direction) to
    more parsimonious representation.
    """
    new_rel = None
    if relationship == 'PARTOF':
        if direction == 'child':
            new_rel = 'hasPart'
        elif direction == 'parent':
            new_rel = 'isPartOf'
    elif relationship == 'KINDOF':
        if direction == 'child':
            new_rel = 'hasKind'
        elif direction == 'parent':
            new_rel = 'isKindOf'
    return new_rel


def _expand_df(df):
    """
    Add alternate forms to DataFrame, then sort DataFrame by alias length
    (for order of extraction from text) and similarity to original name (in
    order to select most appropriate term to associate with alias).
    """
    df = df.copy()
    df['alias'] = df['alias'].apply(uk_to_us)
    new_rows = []
    for index, row in df.iterrows():
        alias = row['alias']
        alt_forms = _gen_alt_forms(alias)
        for alt_form in alt_forms:
            temp_row = row.copy()
            temp_row['alias'] = alt_form
            new_rows.append(temp_row.tolist())
    alt_df = pd.DataFrame(columns=df.columns, data=new_rows)
    df = pd.concat((df, alt_df), axis=0)
    # Sort by name length and similarity of alternate form to preferred term
    # For example, "task switching" the concept should take priority over the
    # "task switching" version of the "task-switching" task.
    df['length'] = df['alias'].str.len()
    df['ratio'] = df[['alias', 'name']].apply(_get_ratio, axis=1)
    df = df.sort_values(by=['length', 'ratio'], ascending=[False, False])
    return df


def pull_ontology(out_dir='auto', overwrite=False):
    """
    Download Cognitive Atlas ontology and combine Concepts, Tasks, and
    Disorders to create ID and relationship DataFrames.
    """
    if out_dir == 'auto':
        out_dir = op.join(get_resource_path(), 'ontology')
    else:
        out_dir = op.abspath(out_dir)

    ids_file = op.join(out_dir, 'cogat_ids.csv')
    rels_file = op.join(out_dir, 'cogat_relationships.csv')
    if overwrite or not all([op.isfile(f) for f in [ids_file, rels_file]]):
        concepts = get_concept(silent=True).pandas
        tasks = get_task(silent=True).pandas
        disorders = get_disorder(silent=True).pandas

        # Identifiers and aliases
        long_concepts = _longify(concepts)
        long_tasks = _longify(tasks)

        # Disorders currently lack aliases
        disorders['name'] = disorders['name'].str.lower()
        disorders = disorders.assign(alias=disorders['name'])
        disorders = disorders[['id', 'name', 'alias']]

        # Combine into id_df
        id_df = pd.concat((long_concepts, long_tasks, disorders), axis=0)
        id_df = _expand_df(id_df)
        id_df = id_df.replace('', np.nan)
        id_df = id_df.dropna(axis=0)
        id_df = id_df.reset_index(drop=True)

        # Relationships
        relationships = []
        for i, id_ in enumerate(concepts['id'].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, 'isSelf']
            relationships.append(row)
            concept = get_concept(id=id_, silent=True).json
            for rel in concept['relationships']:
                reltype = _get_concept_reltype(rel['relationship'],
                                               rel['direction'])
                if reltype is not None:
                    row = [id_, rel['id'], reltype]
                    relationships.append(row)

        for i, id_ in enumerate(tasks['id'].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, 'isSelf']
            relationships.append(row)
            task = get_task(id=id_, silent=True).json
            for rel in task['concepts']:
                row = [id_, rel['concept_id'], 'measures']
                relationships.append(row)
                row = [rel['concept_id'], id_, 'measuredBy']
                relationships.append(row)

        for i, id_ in enumerate(disorders['id'].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, 'isSelf']
            relationships.append(row)
            disorder = get_disorder(id=id_, silent=True).json
            for rel in disorder['disorders']:
                if rel['relationship'] == 'ISA':
                    rel_type = 'isA'
                else:
                    rel_type = rel['relationship']
                row = [id_, rel['id'], rel_type]
                relationships.append(row)

        rel_df = pd.DataFrame(columns=['input', 'output', 'rel_type'],
                              data=relationships)
        ctp_df = concepts[['id', 'id_concept_class']]
        ctp_df = ctp_df.assign(rel_type='inCategory')
        ctp_df.columns = ['input', 'output', 'rel_type']
        ctp_df['output'].replace('', np.nan, inplace=True)
        ctp_df.dropna(axis=0, inplace=True)
        rel_df = pd.concat((ctp_df, rel_df))
        rel_df = rel_df.reset_index(drop=True)
        id_df.to_csv(ids_file, index=False)
        rel_df.to_csv(rels_file, index=False)
    else:
        id_df = pd.read_csv(ids_file)
        rel_df = pd.read_csv(rels_file)

    return id_df, rel_df


def _generate_weights(rel_df, weights):
    """
    Create an IDxID weighting DataFrame based on asserted relationships and
    some weighting scheme that links a weight value to each relationship type
    (e.g., partOf, kindOf).
    """
    # Hierarchical expansion
    def get_weight(rel_type):
        weight = weights.get(rel_type, 0)
        return weight

    t_df = rel_df.copy()
    t_df['rel_type'] = t_df['rel_type'].apply(get_weight)
    weights_df = t_df.pivot_table(index='input', columns='output',
                                  values='rel_type', aggfunc=np.max)
    weights_df = weights_df.fillna(0)
    out_not_in = list(set(t_df['output'].values) - set(t_df['input'].values))
    in_not_out = list(set(t_df['input'].values) - set(t_df['output'].values))

    new_cols = pd.DataFrame(columns=in_not_out,
                            index=weights_df.index,
                            data=np.zeros((weights_df.shape[0],
                                           len(in_not_out))))
    weights_df = pd.concat((weights_df, new_cols), axis=1)
    new_rows = pd.DataFrame(columns=weights_df.columns,
                            index=out_not_in,
                            data=np.zeros((len(out_not_in),
                                           weights_df.shape[1])))
    weights_df = pd.concat((weights_df, new_rows), axis=0)
    all_cols = sorted(weights_df.columns.tolist())
    weights_df = weights_df.loc[all_cols, :]
    weights_df = weights_df.loc[:, all_cols]

    # expanding the hierarchical expansion to all related terms
    # this way, a single dot product will apply counts to all layers
    expanded_df = weights_df.copy()
    mat = weights_df.values

    for i, val in enumerate(weights_df.index):
        res = np.zeros((1, weights_df.shape[0]))
        res[0, i] = 1
        temp = np.zeros((1, weights_df.shape[0]))

        while not np.array_equal(temp != 0, res != 0):
            temp = np.copy(res)
            res = np.dot(res, mat)

            # Constrain weights to <=1.
            # Hopefully this won't mess with weights <1,
            # but will also prevent weights from adding to one another.
            res[res > 1] = 1
        expanded_df.loc[val] = np.squeeze(res)
    return expanded_df


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
    ontology : :obj:`pandas.DataFrame`
        Ontology in DataFrame form.
    regex : :obj:`dict`
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
            self.ontology = pd.read_csv(ontology_file)
        else:
            assert isinstance(ontology_df, pd.DataFrame)
            self.ontology = ontology_df
        assert 'id' in self.ontology.columns
        assert 'name' in self.ontology.columns
        assert 'alias' in self.ontology.columns

        # Create regex dictionary
        regex_dict = {}
        for term in ontology_df['alias'].values:
            term_for_regex = term.replace('(', r'\(').replace(')', r'\)')
            regex = '\\b' + term_for_regex + '\\b'
            pattern = re.compile(regex, re.MULTILINE | re.IGNORECASE)
            regex_dict[term] = pattern
        self.regex = regex_dict

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
        if convert_uk:
            text = uk_to_us(text)

        for term_idx in self.ontology.index:
            term = self.ontology['alias'].loc[term_idx]
            term_id = self.ontology['id'].loc[term_idx]
            text = re.sub(self.regex[term], term_id, text)
        return text


@due.dcite(references.COGNITIVE_ATLAS, description='Introduces the Cognitive Atlas.')
def extract_cogat(text_df, id_df):
    """
    Extract Cognitive Atlas [1]_ terms and perform hierarchical expansion.

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
        text = row['text']
        text = uk_to_us(text)
        for term_idx in id_df.index:
            term = id_df['alias'].loc[term_idx]
            term_id = id_df['id'].loc[term_idx]

            col_idx = gazetteer.index(term_id)

            pattern = regex_dict[term]
            count_arr[c, col_idx] += len(re.findall(pattern, text))
            text = re.sub(pattern, term_id, text)
            rep_text_df.loc[i, 'text'] = text
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
    weights_df = _generate_weights(rel_df, weights=weights)

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
