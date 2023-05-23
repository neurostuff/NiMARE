"""Plot figures for report."""
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from nilearn import datasets
from nilearn.plotting import plot_connectome, plot_stat_map, view_connectome, view_img
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

TABLE_STYLE = [
    dict(
        selector="th, td",
        props=[
            ("text-align", "center"),
            ("font-family", "monospace"),
            ("font-size", "15px"),
            ("padding", "5px 3px"),
            ("margin", "0px 3px"),
            ("border", "1px solid #ddd"),
        ],
    ),
]


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


def plot_static_brain(img, out_filename):
    """Plot static brain."""
    template = datasets.load_mni152_template(resolution=1)
    fig = plot_stat_map(
        img,
        bg_img=template,
        black_bg=False,
        draw_cross=False,
        threshold=2,
        vmax=4,
        display_mode="mosaic",
    )
    fig.savefig(out_filename, dpi=300)
    fig.close()


def plot_coordinates(dataset, out_static_filename, out_interactive_filename, out_legend_filename):
    """Plot static and interactive coordinates."""
    node_coords = dataset.coordinates[["x", "y", "z"]].to_numpy()
    n_coords = len(node_coords)
    adjacency_matrix = np.zeros((n_coords, n_coords))

    # Generate dictionary and array of colors for each unique ID
    ids = dataset.coordinates["id"].to_list()
    unq_ids = np.unique(ids)
    cmap = plt.cm.get_cmap("tab20", len(unq_ids))
    colors_dict = {unq_id: mcolors.to_hex(cmap(i)) for i, unq_id in enumerate(unq_ids)}
    colors = [colors_dict[id_] for id_ in ids]

    fig = plot_connectome(adjacency_matrix, node_coords, node_color=colors)
    fig.savefig(out_static_filename, dpi=300)
    fig.close()

    # Generate legend
    patches_lst = [
        mpatches.Patch(color=color, label=label) for label, color in colors_dict.items()
    ]

    # Plot legeng
    ncol = 10
    labl_fig, ax = plt.subplots(1, 1)
    labl_fig.set_size_inches(ncol, len(patches_lst) / ncol**2)
    labl_fig.legend(
        handles=patches_lst,
        ncol=ncol,
        fontsize=10,
        loc="center",
    )
    ax.axis("off")
    labl_fig.savefig(out_legend_filename, bbox_inches="tight", dpi=300)
    plt.close()

    # Plot interactive connectome
    html_view = view_connectome(
        adjacency_matrix,
        node_coords,
        node_size=10,
        colorbar=False,
        node_color=colors,
    )
    html_view.save_as_html(out_interactive_filename)


def plot_interactive_brain(img, out_filename):
    """Plot interactive brain."""
    template = datasets.load_mni152_template(resolution=1)
    html_view = view_img(img, bg_img=template, black_bg=False, threshold=2, vmax=4)
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
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )
    fig.write_html(out_filename, full_html=True, include_plotlyjs=True)


def gen_table(clusters_table, out_filename):
    """Generate table."""
    clust_ids = clusters_table["Cluster ID"].to_list()
    clusters_table = clusters_table.drop(columns=["Cluster ID"])

    tail = [c_id.split(" ")[0].split("Tail")[0] for c_id in clust_ids]
    ids = [c_id.split(" ")[1] for c_id in clust_ids]
    tuples = list(zip(*[tail, ids]))
    row = pd.MultiIndex.from_tuples(tuples)
    clusters_table.index = row
    clusters_table.index = clusters_table.index.rename(["Tail", "Cluster ID"])

    styled_df = clusters_table.style.format(precision=2).set_table_styles(TABLE_STYLE)
    styled_df.to_html(out_filename)
