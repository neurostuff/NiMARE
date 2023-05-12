"""Plot figures for report."""
import numpy as np
import pandas as pd
import plotly.express as px
from nilearn import datasets, plotting
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering


def _reorder_matrix(mat, row_labels, col_labels, reorder):
    """Reorder a matrix.

    This function reorders the provided matrix. It was adaptes from
    nilearn.plotting.plot_matrix._reorder_matrix to reorder non-square matrices.

        License
    -------
    New BSD License
    Copyright (c) 2007 - 2023 The nilearn developers.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    if not row_labels or not col_labels:
        raise ValueError("Labels are needed to show the reordering.")

    # Order rows
    row_linkage_matrix = linkage(mat, method=reorder)
    row_ordered_linkage = optimal_leaf_ordering(row_linkage_matrix, mat)
    row_index = leaves_list(row_ordered_linkage)

    # Order columns
    col_linkage_matrix = linkage(mat.T, method=reorder)
    col_ordered_linkage = optimal_leaf_ordering(col_linkage_matrix, mat)
    col_index = leaves_list(col_ordered_linkage)

    # Make sure labels is an ndarray and copy it
    row_labels = np.array(row_labels).copy()
    col_labels = np.array(col_labels).copy()
    mat = mat.copy()

    # and reorder labels and matrix
    row_labels = row_labels[row_index].tolist()
    col_labels = col_labels[col_index].tolist()
    mat = mat[row_index, :][:, col_index]

    return mat, row_labels, col_labels


def plot_static_brain():
    """Plot static brain."""
    pass


def plot_dynamic_brain(img, out_filename):
    """Plot dynamic brain."""
    template = datasets.load_mni152_template(resolution=1)
    html_view = plotting.view_img(img, bg_img=template, black_bg=False, threshold=2, vmax=4)
    html_view.save_as_html(out_filename)


def plot_heatmap(contribution_table, out_filename):
    """Plot heatmap."""
    mat = contribution_table.to_numpy()
    row_labels, col_labels = (
        contribution_table.index.to_list(),
        contribution_table.columns.to_list(),
    )

    new_mat, new_row_labels, new_col_labels = _reorder_matrix(
        mat,
        row_labels,
        col_labels,
        "single",
    )
    new_df = pd.DataFrame(new_mat, columns=new_col_labels, index=new_row_labels)

    fig = px.imshow(new_df, color_continuous_scale="Reds")
    fig.write_html(out_filename)


def gen_table(clusters_table, out_filename):
    """Generate table."""
    pass
