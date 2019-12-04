"""
Utility functions for ontology tools.
"""
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from ..text import uk_to_us


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


def _generate_weights(rel_df, weights):
    """
    Create an IDxID weighting DataFrame based on asserted relationships and
    some weighting scheme that links a weight value to each relationship type
    (e.g., partOf, kindOf).

    Parameters
    ----------
    rel_df : (X x 3) :obj:`pandas.DataFrame`
        DataFrame with three columns: input, output, and rel_type
        (relationship type).
    weights : :obj:`dict`
        Dictionary defining relationship weights. Each relationship type is a
        key and the associated value is the weight to use for that kind of
        relationship.

    Returns
    -------
    expanded_df : :obj:`pandas.DataFrame`
        Square DataFrame where rows correspond to input items, columns
        correspond to output items, and cells have the weights associated with
        the particular input/output relationship.

    Notes
    -----
    For example, if weights is {'partOf': 1}, the resulting expanded_df will
    have a value of 1 for all cells where the input item (row) is a part of the
    output item (column), and will have zeroes for all other cells.
    """
    # Override isSelf weight
    weights['isSelf'] = 1

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
        row = np.zeros((1, weights_df.shape[0]))
        row[0, i] = 1  # identity
        temp = np.zeros((1, weights_df.shape[0]))

        while not np.array_equal(temp != 0, row != 0):
            temp = np.copy(row)
            row = np.dot(row, mat)

            # Constrain weights to <=1.
            # Hopefully this won't mess with weights <1,
            # but will also prevent weights from adding to one another.
            row[row > 1] = 1
        expanded_df.loc[val] = np.squeeze(row)
    return expanded_df
