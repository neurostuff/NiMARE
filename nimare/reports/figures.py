"""Plot figures for report."""

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from nilearn import datasets
from nilearn.plotting import (
    plot_connectome,
    plot_img,
    plot_roi,
    plot_stat_map,
    view_connectome,
    view_img,
)
from ridgeplot import ridgeplot
from scipy import stats
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


PXS_PER_STD = 30  # Number of pixels per study, control the size (height) of Plotly figures
MAX_CHARS = 20  # Maximum number of characters for labels


def _check_extention(filename, exts):
    if filename.suffix not in exts:
        raise ValueError(
            f'The "out_filename" provided has extension {filename.suffix}. '
            f'Valid extensions are {", ".join(exts)}.'
        )


def _reorder_matrix(mat, row_labels, col_labels, symmetric=False, reorder="single"):
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

    # Make sure labels is an ndarray and copy it
    row_labels = np.array(row_labels).copy()

    if not symmetric:
        # Order columns
        col_linkage_matrix = linkage(mat.T, method=reorder)
        col_ordered_linkage = optimal_leaf_ordering(col_linkage_matrix, mat.T)
        col_index = leaves_list(col_ordered_linkage)

        col_labels = np.array(col_labels).copy()
    else:
        col_index = row_index
        col_labels = row_labels

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


def plot_heatmap(
    data_df,
    out_filename,
    symmetric=False,
    reorder="single",
    cmap="Reds",
    zmin=None,
    zmax=None,
):
    """Plot heatmap.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    data_df : :obj:`pandas.DataFrame`
        A DataFrame with the data for the heatmap. It could be a correlation matrix or
        a contribution matrix with information about the relative contributions of
        each experiment to each cluster in the thresholded map.
    out_filename : :obj:`pathlib.Path`
        The name of an image file to export the plot to.
        Valid extension is '.html'.
    symmetric : :obj:`bool`, optional
        Whether to reorder the matrix symmetrically. Use True if using a correlation matrix.
        Default is False.
    reorder : :obj:`str`, optional
        The method to use for reordering the matrix. Default is 'average'.
    cmap : :obj:`str`, optional
        The colormap to use. Default is 'Reds'.
    zmin : :obj:`float`, optional
        The minimum value to use for the colormap. Default is None.
    zmax : :obj:`float`, optional
        The maximum value to use for the colormap. Default is None.
    """
    _check_extention(out_filename, [".html"])

    n_studies, n_clusters = data_df.shape
    if (n_studies > 2) and (n_clusters > 2):
        # Reorder matrix only if more than 1 cluster/experiment
        mat = data_df.to_numpy()
        row_labels, col_labels = (
            data_df.index.to_list(),
            data_df.columns.to_list(),
        )
        new_mat, new_row_labels, new_col_labels = _reorder_matrix(
            mat,
            row_labels,
            col_labels,
            symmetric=symmetric,
            reorder=reorder,
        )

        # Truncate labels to MAX_CHARS characters
        x_labels = [label[:MAX_CHARS] for label in new_col_labels]
        y_labels = [label[:MAX_CHARS] for label in new_row_labels]
        data_df = pd.DataFrame(new_mat, columns=x_labels, index=y_labels)

    fig = px.imshow(data_df, color_continuous_scale=cmap, zmin=zmin, zmax=zmax, aspect="equal")

    height = n_studies * PXS_PER_STD
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


def _plot_true_voxels(maps_arr, ids_, out_filename):
    """Plot percentage of valid voxels.

    .. versionadded:: 0.2.2

    """
    n_studies, n_voxels = maps_arr.shape
    mask = ~np.isnan(maps_arr) & (maps_arr != 0)

    x_label, y_label = "Voxels Included", "ID"
    perc_voxs = mask.sum(axis=1) / n_voxels
    valid_df = pd.DataFrame({y_label: ids_, x_label: perc_voxs})
    valid_sorted_df = valid_df.sort_values(x_label, ascending=True)

    fig = px.bar(
        valid_sorted_df,
        x=x_label,
        y=y_label,
        orientation="h",
        color=x_label,
        color_continuous_scale="blues",
        range_color=(0, 1),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        visible=True,
        showticklabels=False,
        title=None,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        visible=True,
        ticktext=valid_sorted_df[y_label].str.slice(0, MAX_CHARS).tolist(),
    )

    height = n_studies * PXS_PER_STD
    fig.update_layout(
        height=height,
        autosize=True,
        font_size=14,
        plot_bgcolor="white",
        xaxis_gridcolor="white",
        yaxis_gridcolor="white",
        xaxis_gridwidth=2,
        showlegend=False,
    )
    fig.write_html(out_filename, full_html=True, include_plotlyjs=True)


def _plot_ridgeplot(maps_arr, ids_, x_label, out_filename):
    """Plot histograms of the images.

    .. versionadded:: 0.2.0

    """
    n_studies = len(ids_)
    labels = [id_[:MAX_CHARS] for id_ in ids_]  # Truncate labels to MAX_CHARS characters

    mask = ~np.isnan(maps_arr) & (maps_arr != 0)
    maps_lst = [maps_arr[i][mask[i]] for i in range(n_studies)]

    N_KDE_POINTS = 100
    max_val = 8 if x_label == "Z" else 1
    kde_points = np.linspace(-max_val, max_val, N_KDE_POINTS)
    bandwidth = 0.5 if x_label == "Z" else 0.1

    fig = ridgeplot(
        samples=maps_lst,
        labels=labels,
        coloralpha=0.98,
        bandwidth=bandwidth,
        kde_points=kde_points,
        colorscale="Bluered",
        colormode="mean-means",
        spacing=PXS_PER_STD / 100,
        linewidth=2,
    )

    height = n_studies * PXS_PER_STD
    fig.update_layout(
        height=height,
        autosize=True,
        font_size=14,
        plot_bgcolor="white",
        xaxis_gridcolor="white",
        yaxis_gridcolor="white",
        xaxis_gridwidth=2,
        xaxis_title=x_label,
        showlegend=False,
    )
    fig.write_html(out_filename, full_html=True, include_plotlyjs=True)


def _plot_sumstats(maps_arr, ids_, out_filename):
    """Plot summary statistics of the images.

    .. versionadded:: 0.2.2

    """
    n_studies = len(ids_)
    mask = ~np.isnan(maps_arr) & (maps_arr != 0)
    maps_lst = [maps_arr[i][mask[i]] for i in range(n_studies)]

    stats_lbls = [
        "Mean",
        "STD",
        "Var",
        "Median",
        "Mode",
        "Min",
        "Max",
        "Skew",
        "Kurtosis",
        "Range",
        "Moment",
        "IQR",
    ]
    scores, id_lst = [], []
    for id_, map_ in zip(ids_, maps_lst):
        scores.extend(
            [
                np.mean(map_),
                np.std(map_),
                np.var(map_),
                np.median(map_),
                stats.mode(map_)[0],
                np.min(map_),
                np.max(map_),
                stats.skew(map_),
                stats.kurtosis(map_),
                np.max(map_) - np.min(map_),
                stats.moment(map_, moment=4),
                stats.iqr(map_),
            ]
        )
        id_lst.extend([id_] * len(stats_lbls))

    stats_labels = stats_lbls * n_studies
    data_df = pd.DataFrame({"ID": id_lst, "Score": scores, "Stat": stats_labels})

    fig = px.strip(
        data_df,
        y="Score",
        color="ID",
        facet_col="Stat",
        stripmode="group",
        facet_col_wrap=4,
        facet_col_spacing=0.08,
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
    fig.update_yaxes(
        constrain="domain",
        matches=None,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        title=None,
    )
    fig.update_layout(
        height=900,
        autosize=True,
        font_size=14,
        plot_bgcolor="white",
        xaxis_gridcolor="white",
        yaxis_gridcolor="white",
        xaxis_gridwidth=2,
        showlegend=False,
    )
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.write_html(out_filename, full_html=True, include_plotlyjs=True)


def _plot_relcov_map(maps_arr, masker, out_filename):
    """Plot relative coverage map.

    .. versionadded:: 0.2.0

    """
    _check_extention(out_filename, [".png", ".pdf", ".svg"])

    epsilon = 1e-05

    # Binaries maps and create relative coverage map
    binary_maps_arr = np.where((-epsilon > maps_arr) | (maps_arr > epsilon), 1, 0)
    coverage_arr = np.sum(binary_maps_arr, axis=0) / binary_maps_arr.shape[0]

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


def _plot_dof_map(dof_map, out_filename):
    """Plot DoF map.

    .. versionadded:: 0.2.1

    """
    _check_extention(out_filename, [".png", ".pdf", ".svg"])

    epsilon = 1e-05

    # Plot coverage map
    template = datasets.load_mni152_template(resolution=1)
    fig = plot_img(
        dof_map,
        bg_img=template,
        black_bg=False,
        draw_cross=False,
        threshold=epsilon,
        alpha=0.7,
        colorbar=True,
        cmap="YlOrRd",
        vmin=0,
        display_mode="mosaic",
    )
    fig.savefig(out_filename, dpi=300)
    fig.close()
