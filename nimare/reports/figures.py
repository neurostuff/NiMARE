"""Plot figures for report."""
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from nilearn import datasets
from nilearn.plotting import (
    plot_connectome,
    plot_img,
    plot_roi,
    plot_stat_map,
    view_connectome,
    view_img,
)
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

from nimare.utils import _boolean_unmask

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


def _check_extention(filename, exts):
    if filename.suffix not in exts:
        raise ValueError(
            f'The "out_filename" provided has extension {filename.suffix}. '
            f'Valid extensions are {", ".join(exts)}.'
        )


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
    col_ordered_linkage = optimal_leaf_ordering(col_linkage_matrix, mat.T)
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


def plot_static_brain(img, out_filename, threshold=1e-06):
    """Plot static brain image.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    img : :obj:`~nibabel.nifti1.Nifti1Image`
        Stat image to plot.
    out_filename : :obj:`pathlib.Path`
        The name of an image file to export the plot to.
        Valid extensions are '.png', '.pdf', '.svg'.
    threshold: a number, None, or 'auto', optional
        If None is given, the image is not thresholded. If a number is given, it is
        used to threshold the image: values below the threshold (in absolute value)
        are plotted as transparent. If 'auto' is given, the threshold is determined
        magically by analysis of the image. Default=1e-6.
    """
    _check_extention(out_filename, [".png", ".pdf", ".svg"])

    template = datasets.load_mni152_template(resolution=1)
    fig = plot_stat_map(
        img,
        bg_img=template,
        black_bg=False,
        draw_cross=False,
        threshold=threshold,
        display_mode="mosaic",
    )
    fig.savefig(out_filename, dpi=300)
    fig.close()


def plot_mask(mask, out_filename):
    """Plot mask.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    img : :obj:`~nibabel.nifti1.Nifti1Image`
        Mask image to plot.
    out_filename : :obj:`pathlib.Path`
        The name of an image file to export the plot to.
        Valid extensions are '.png', '.pdf', '.svg'.
    """
    _check_extention(out_filename, [".png", ".pdf", ".svg"])

    template = datasets.load_mni152_template(resolution=1)

    fig = plot_roi(
        mask,
        bg_img=template,
        black_bg=False,
        draw_cross=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
        alpha=0.7,
        display_mode="mosaic",
    )
    fig.savefig(out_filename, dpi=300)
    fig.close()


def plot_coordinates(
    coordinates_df,
    out_static_filename,
    out_interactive_filename,
    out_legend_filename,
):
    """Plot static and interactive coordinates.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    coordinates_df : :obj:`pandas.DataFrame`
        A DataFrame with the coordinates in the dataset.
    out_static_filename : :obj:`pathlib.Path`
        The name of an image file to export the static plot to.
        Valid extensions are '.png', '.pdf', '.svg'.
    out_interactive_filename : :obj:`pathlib.Path`
        The name of an image file to export the interactive plot to.
        Valid extension is '.html'.
    out_legend_filename : :obj:`pathlib.Path`
        The name of an image file to export the legend plot to.
        Valid extensions are '.png', '.pdf', '.svg'.
    """
    _check_extention(out_static_filename, [".png", ".pdf", ".svg"])
    _check_extention(out_interactive_filename, [".html"])
    _check_extention(out_legend_filename, [".png", ".pdf", ".svg"])

    node_coords = coordinates_df[["x", "y", "z"]].to_numpy()
    n_coords = len(node_coords)
    adjacency_matrix = np.zeros((n_coords, n_coords))

    # Generate dictionary and array of colors for each unique ID
    ids = coordinates_df["study_id"].to_list()
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
    max_len_per_page = 200
    max_legend_len = max(len(id_) for id_ in unq_ids)
    ncol = 1 if max_legend_len > max_len_per_page else int(max_len_per_page / max_legend_len)
    labl_fig, ax = plt.subplots(1, 1)
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


def plot_interactive_brain(img, out_filename, threshold=1e-06):
    """Plot interactive brain image.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    img : :obj:`~nibabel.nifti1.Nifti1Image`
        Stat image to plot.
    out_filename : :obj:`pathlib.Path`
        The name of an image file to export the plot to. Valid extension is '.html'.
    threshold: a number, None, or 'auto', optional
        If None is given, the image is not thresholded. If a number is given, it is
        used to threshold the image: values below the threshold (in absolute value)
        are plotted as transparent. If 'auto' is given, the threshold is determined
        magically by analysis of the image. Default=1e-6.
    """
    _check_extention(out_filename, [".html"])

    template = datasets.load_mni152_template(resolution=1)
    html_view = view_img(img, bg_img=template, black_bg=False, threshold=threshold)
    html_view.save_as_html(out_filename)


def plot_heatmap(contribution_table, out_filename):
    """Plot heatmap.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    contribution_table : :obj:`pandas.DataFrame`
        A DataFrame with information about relative contributions of each experiment to each
        cluster in the thresholded map.
    out_filename : :obj:`pathlib.Path`
        The name of an image file to export the plot to.
        Valid extension is '.html'.
    """
    _check_extention(out_filename, [".html"])

    n_studies, n_clusters = contribution_table.shape
    if (n_studies > 2) and (n_clusters > 2):
        # Reorder matrix only if more than 1 cluster/experiment
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
        contribution_table = pd.DataFrame(new_mat, columns=new_col_labels, index=new_row_labels)

    fig = px.imshow(contribution_table, color_continuous_scale="Reds", aspect="equal")

    pxs_per_sqr = 50  # Number of pixels per square in the heatmap
    height = n_studies * pxs_per_sqr
    fig.update_layout(autosize=True, height=height)
    fig.write_html(out_filename, full_html=True, include_plotlyjs=True)


def gen_table(clusters_table, out_filename):
    """Generate table.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    clusters_table : :obj:`pandas.DataFrame`
        A DataFrame with information about each cluster.
    out_filename : :obj:`pathlib.Path`
        The name of an image file to export the plot to.
        Valid extension is '.html'.
    """
    _check_extention(out_filename, [".html"])

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


def plot_clusters(img, out_filename):
    """Plot clusters.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    img : :obj:`~nibabel.nifti1.Nifti1Image`
        Label image to plot.
    out_filename : :obj:`pathlib.Path`
        The name of an image file to export the plot to.
        Valid extensions are '.png', '.pdf', '.svg'.
    """
    _check_extention(out_filename, [".png", ".pdf", ".svg"])

    template = datasets.load_mni152_template(resolution=1)

    # Define cmap depending on the number of clusters
    clust_ids = list(np.unique(img.get_fdata())[1:])
    cmap = plt.cm.get_cmap("tab20", len(clust_ids))

    fig = plot_roi(
        img,
        bg_img=template,
        black_bg=False,
        draw_cross=False,
        cmap=cmap,
        alpha=0.8,
        colorbar=True,
        display_mode="mosaic",
    )
    fig.savefig(out_filename, dpi=300)
    fig.close()


def _plot_ridgeplot(maps_arr, ids_, x_label, out_filename):
    """Plot histograms of the images.

    .. versionadded:: 0.2.0

    Base on: https://seaborn.pydata.org/examples/kde_ridgeplot.html
    """
    # Create dataframe for seaborn
    values = []
    group = []
    for img_i, img_map in enumerate(list(maps_arr)):
        values.append(img_map)
        group.append([ids_[img_i]] * len(img_map))

    data = {x_label: np.hstack(values), "exp_id": np.hstack(group)}
    df = pd.DataFrame(data)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(df, row="exp_id", hue="exp_id", aspect=15, height=0.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, x_label, bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, x_label, clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(values, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label[:20],  # Limit the number of characters in the label
            fontweight="bold",
            color=color,
            fontsize=8,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, x_label)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.savefig(out_filename, dpi=300)
    plt.close()


def _plot_relcov_map(maps_arr, masker, aggressive_mask, out_filename):
    """Plot relative coverage map.

    .. versionadded:: 0.2.0

    """
    _check_extention(out_filename, [".png", ".pdf", ".svg"])

    epsilon = 1e-05

    # Binaries maps and create relative coverage map
    binary_maps_arr = np.where((-epsilon > maps_arr) | (maps_arr > epsilon), 1, 0)
    coverage_arr = np.sum(binary_maps_arr, axis=0) / binary_maps_arr.shape[0]

    # Add bad voxels back to the arr to transform it back to an image
    coverage_arr = _boolean_unmask(coverage_arr, aggressive_mask)
    coverage_img = masker.inverse_transform(coverage_arr)

    # Plot coverage map
    template = datasets.load_mni152_template(resolution=1)
    fig = plot_img(
        coverage_img,
        bg_img=template,
        black_bg=False,
        draw_cross=False,
        threshold=epsilon,
        alpha=0.7,
        colorbar=True,
        cmap="Blues",
        vmin=0,
        vmax=1,
        display_mode="mosaic",
    )
    fig.savefig(out_filename, dpi=300)
    fig.close()
