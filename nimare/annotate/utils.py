"""Utility functions for ontology tools."""
import numpy as np
import pandas as pd


def _generate_weights(rel_df, weights):
    """Create an IDxID DataFrame linking weight value to each relationship type.

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
    weights["isSelf"] = 1

    # Hierarchical expansion
    def get_weight(rel_type):
        weight = weights.get(rel_type, 0)
        return weight

    t_df = rel_df.copy()
    t_df["rel_type"] = t_df["rel_type"].apply(get_weight)
    weights_df = t_df.pivot_table(
        index="input", columns="output", values="rel_type", aggfunc=np.max
    )
    weights_df = weights_df.fillna(0)
    out_not_in = list(set(t_df["output"].values) - set(t_df["input"].values))
    in_not_out = list(set(t_df["input"].values) - set(t_df["output"].values))

    new_cols = pd.DataFrame(
        columns=in_not_out,
        index=weights_df.index,
        data=np.zeros((weights_df.shape[0], len(in_not_out))),
    )
    weights_df = pd.concat((weights_df, new_cols), axis=1)
    new_rows = pd.DataFrame(
        columns=weights_df.columns,
        index=out_not_in,
        data=np.zeros((len(out_not_in), weights_df.shape[1])),
    )
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
