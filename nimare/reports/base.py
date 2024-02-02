# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache-2.0 terms,
# and this file has been changed.
# The original file this work derives from is found at:
# https://github.com/nipreps/niworkflows/blob/9905f90110879ed4123ea291f512b0a60d7ba207/niworkflows/reports/core.py
#
# [May 2023] CHANGES:
#    * Replace BIDSlayout with code that uses the nimare Dataset and MetaResult class.
#
# ORIGINAL WORK'S ATTRIBUTION NOTICE:
#
#     Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Reports builder for NiMARE's MetaResult object."""
import textwrap
from glob import glob
from pathlib import Path

import jinja2
import pandas as pd
from pkg_resources import resource_filename as pkgrf

from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.reports.figures import (
    _plot_dof_map,
    _plot_relcov_map,
    _plot_ridgeplot,
    _plot_sumstats,
    _plot_true_voxels,
    gen_table,
    plot_clusters,
    plot_coordinates,
    plot_heatmap,
    plot_interactive_brain,
    plot_mask,
    plot_static_brain,
)
from nimare.stats import pearson

PARAMETERS_DICT = {
    "kernel_transformer__fwhm": "FWHM",
    "kernel_transformer__sample_size": "Sample size",
    "kernel_transformer__r": "Sphere radius (mm)",
    "kernel_transformer__value": "Value for sphere",
    "kernel_transformer__memory": "Memory",
    "kernel_transformer__memory_level": "Memory level",
    "kernel_transformer__sum_across_studies": "Sum Across Studies",
    "memory": "Memory",
    "memory_level": "Memory level",
    "null_method": "Null method",
    "n_iters": "Number of iterations",
    "n_cores": "Number of cores",
    "fwe": "Family-wise error rate (FWE) correction",
    "fdr": "False discovery rate (FDR) correction",
    "method": "Method",
    "alpha": "Alpha",
    "prior": "Prior",
    "use_sample_size": "Use sample size for weights",
    "beta": "Parameter estimate",
    "se": "Standard error of the parameter estimate",
    "varcope": "Variance of the parameter estimate",
    "t": "T-statistic",
    "z": "Z-statistic",
}

PNG_SNIPPET = """\
<img class="png-reportlet" src="./{0}" style="width: 100%" /></div>
<div class="elem-filename">
    Get figure file: <a href="./{0}" target="_blank">{0}</a>
</div>
"""

IFRAME_SNIPPET = """\
<div class="igraph-container">
    <iframe class="igraph" src="./{0}"></iframe>
</div>
"""

SUMMARY_TEMPLATE = """\
<ul class="elem-desc">
{meta_text}
</ul>
<details>
<summary>Experiments excluded</summary><br />
<p>{exc_ids}</p>
</details>
"""

ESTIMATOR_TEMPLATE = """\
<ul class="elem-desc">
<li>Estimator: {est_name}</li>
{ker_text}
{est_params_text}
</ul>
"""

CORRECTOR_TEMPLATE = """\
<ul class="elem-desc">
<li> Correction Method: {correction_method}</li>
{cor_params_text}
<li>Parameters: {ext_params_text}</li>
</ul>
"""

DIAGNOSTIC_TEMPLATE = """\
<h2 class="sub-report-group">Target image: {target_image}</h2>
<ul class="elem-desc">
<li>Voxel-level threshold: {voxel_thresh}</li>
<li>Cluster size threshold: {cluster_threshold}</li>
<li>Number of cores: {n_cores}</li>
</ul>
"""


def _get_cbma_summary(dset, sel_ids):
    n_studies = len(dset.coordinates["study_id"].unique())

    mask = dset.masker.mask_img
    sel_ids = dset.get_studies_by_mask(mask)
    sel_dset = dset.slice(sel_ids)

    n_foci = dset.coordinates.shape[0]
    n_foci_sel = sel_dset.coordinates.shape[0]
    n_foci_nonbrain = n_foci - n_foci_sel

    n_exps = len(dset.ids)
    n_exps_sel = len(sel_ids)

    cbma_text = [
        f"<li>Number of studies: {n_studies:d}</li>",
        f"<li>Number of experiments: {n_exps:d}</li>",
        f"<li>Number of experiments included: {n_exps_sel:d}</li>",
        f"<li>Number of foci: {n_foci:d} </li>",
        f"<li>Number of foci outside the mask: {n_foci_nonbrain:d} </li>",
    ]

    return " ".join(cbma_text)


def _get_ibma_summary(dset, sel_ids):
    img_df = dset.images
    n_studies = len(img_df["study_id"].unique())

    ignore_columns = ["id", "study_id", "contrast_id"]
    map_type = [c for c in img_df if not c.endswith("__relative") and c not in ignore_columns]

    n_imgs = len(dset.ids)
    n_sel_ids = len(sel_ids)

    ibma_text = [
        f"<li>Number of studies: {n_studies:d}</li>",
        f"<li>Number of images: {n_imgs:d}</li>",
        f"<li>Number of images included: {n_sel_ids:d}</li>",
    ]

    maptype_text = ["<li>Available maps: ", "<ul>"]
    maptype_text.extend(f"<li>{PARAMETERS_DICT[m]} ({m})</li>" for m in map_type)
    maptype_text.extend(["</ul>", "</li>"])

    ibma_text.extend(maptype_text)
    return " ".join(ibma_text)


def _gen_summary(dset, sel_ids, meta_type, out_filename):
    """Generate preliminary checks from dataset for the report."""
    exc_ids = list(set(dset.ids) - set(sel_ids))
    exc_ids_str = ", ".join(exc_ids)

    meta_text = (
        _get_cbma_summary(dset, sel_ids)
        if meta_type == "CBMA"
        else _get_ibma_summary(dset, sel_ids)
    )

    summary_text = SUMMARY_TEMPLATE.format(
        meta_text=meta_text,
        exc_ids=exc_ids_str,
    )
    (out_filename).write_text(summary_text, encoding="UTF-8")


def _get_kernel_summary(params_dict):
    kernel_transformer = str(params_dict["kernel_transformer"])
    ker_params = {k: v for k, v in params_dict.items() if k.startswith("kernel_transformer__")}
    ker_params_text = ["<ul>"]
    ker_params_text.extend(f"<li>{PARAMETERS_DICT[k]}: {v}</li>" for k, v in ker_params.items())
    ker_params_text.append("</ul>")
    ker_params_text = "".join(ker_params_text)

    return f"<li>Kernel Transformer: {kernel_transformer}{ker_params_text}</li>"


def _gen_est_summary(obj, out_filename):
    """Generate html with parameter use in obj (e.g., estimator)."""
    params_dict = obj.get_params()

    # Add kernel transformer parameters to summary if obj is a CBMAEstimator
    ker_text = _get_kernel_summary(params_dict) if isinstance(obj, CBMAEstimator) else ""

    est_params = {k: v for k, v in params_dict.items() if not k.startswith("kernel_transformer")}
    est_params_text = [f"<li>{PARAMETERS_DICT[k]}: {v}</li>" for k, v in est_params.items()]
    est_params_text = "".join(est_params_text)

    est_name = obj.__class__.__name__

    summary_text = ESTIMATOR_TEMPLATE.format(
        est_name=est_name,
        ker_text=ker_text,
        est_params_text=est_params_text,
    )
    (out_filename).write_text(summary_text, encoding="UTF-8")


def _gen_cor_summary(obj, out_filename):
    """Generate html with parameter use in obj (e.g., corrector)."""
    params_dict = obj.get_params()

    cor_params_text = [f"<li>{PARAMETERS_DICT[k]}: {v}</li>" for k, v in params_dict.items()]
    cor_params_text = "".join(cor_params_text)

    ext_params_text = ["<ul>"]
    ext_params_text.extend(
        f"<li>{PARAMETERS_DICT[k]}: {v}</li>" for k, v in obj.parameters.items()
    )
    ext_params_text.append("</ul>")
    ext_params_text = "".join(ext_params_text)

    summary_text = CORRECTOR_TEMPLATE.format(
        correction_method=PARAMETERS_DICT[obj._correction_method],
        cor_params_text=cor_params_text,
        ext_params_text=ext_params_text,
    )
    (out_filename).write_text(summary_text, encoding="UTF-8")


def _gen_diag_summary(obj, out_filename):
    """Generate html with parameter use in obj (e.g., diagnostics)."""
    diag_dict = obj.get_params()

    summary_text = DIAGNOSTIC_TEMPLATE.format(**diag_dict)
    (out_filename).write_text(summary_text, encoding="UTF-8")


def _no_clusts_found(out_filename):
    """Generate html with single text."""
    null_text = '<h4 style="color:#A30000">No significant clusters found</h4>'
    (out_filename).write_text(null_text, encoding="UTF-8")


def _no_maps_found(out_filename):
    """Generate html with single text."""
    null_text = """\
    <h4 style="color:#A30000">No significant voxels were found above the threshold</h4>
    """
    (out_filename).write_text(null_text, encoding="UTF-8")


def _gen_fig_summary(img_key, threshold, out_filename):
    summary_text = f"""\
    <h2 class="sub-report-group">Corrected meta-analytic map: {img_key}</h2>
    <ul class="elem-desc">
    <li>Voxel-level threshold: {threshold}</li>
    </ul>
    """
    (out_filename).write_text(summary_text, encoding="UTF-8")


def _compute_similarities(maps_arr, ids_):
    """Compute the similarity between maps."""
    corrs = [pearson(img_map, maps_arr) for img_map in list(maps_arr)]

    return pd.DataFrame(index=ids_, columns=ids_, data=corrs)


def _gen_figures(results, img_key, diag_name, threshold, fig_dir):
    """Generate html and png objects for the report."""
    # Plot brain images if not empty
    if (results.maps[img_key] > threshold).any():
        img = results.get_map(img_key)
        plot_interactive_brain(img, fig_dir / "corrector_figure-interactive.html", threshold)
        plot_static_brain(img, fig_dir / "corrector_figure-static.png", threshold)
    else:
        _no_maps_found(fig_dir / "corrector_figure-non.html")

    # Plot clusters table if cluster_table is not empty
    cluster_table = results.tables[f"{img_key}_tab-clust"]
    if cluster_table is not None and not cluster_table.empty:
        gen_table(cluster_table, fig_dir / "diagnostics_tab-clust_table.html")

        # Get label maps and contribution_table
        contribution_tables = []
        heatmap_names = []
        lbl_name = "_".join(img_key.split("_")[1:])
        lbl_name = f"_{lbl_name}" if lbl_name else lbl_name
        for tail in ["positive", "negative"]:
            lbl_key = f"label{lbl_name}_tail-{tail}"
            if lbl_key in results.maps:
                label_map = results.get_map(lbl_key)
                plot_clusters(label_map, fig_dir / f"diagnostics_tail-{tail}_figure.png")

            contribution_table_name = f"{img_key}_diag-{diag_name}_tab-counts_tail-{tail}"
            if contribution_table_name in results.tables:
                contribution_table = results.tables[contribution_table_name]
                if contribution_table is not None and not contribution_table.empty:
                    contribution_table = contribution_table.set_index("id")
                    contribution_tables.append(contribution_table)
                    heatmap_names.append(
                        f"diagnostics_diag-{diag_name}_tab-counts_tail-{tail}_figure.html"
                    )

        # For IBMA plot only one heatmap with both positive and negative tails
        contribution_table_name = f"{img_key}_diag-{diag_name}_tab-counts"
        if contribution_table_name in results.tables:
            contribution_table = results.tables[contribution_table_name]
            if contribution_table is not None and not contribution_table.empty:
                contribution_table = contribution_table.set_index("id")
                contribution_tables.append(contribution_table)
                heatmap_names.append(f"diagnostics_diag-{diag_name}_tab-counts_figure.html")

        # Plot heatmaps
        [
            plot_heatmap(contribution_table, fig_dir / heatmap_name, zmin=0)
            for heatmap_name, contribution_table in zip(heatmap_names, contribution_tables)
        ]

    else:
        _no_clusts_found(fig_dir / "diagnostics_tab-clust_table.html")


class Element(object):
    """Just a basic component of a report."""

    def __init__(self, name, title=None):
        self.name = name
        self.title = title


class Reportlet(Element):
    """Reportlet holds the content of a SubReports.

    A reportlet has title, description and a list of components with either an
    HTML fragment or a path to an SVG file, and possibly a caption. This is a
    factory class to generate Reportlets reusing the config object from a ``Report``
    object.
    """

    def __init__(self, out_dir, config=None):
        if not config:
            raise RuntimeError("Reportlet must have a config object")

        bids_dict = config["bids"]

        # value and suffix are don't need the key, so removing from the bids conform name
        keys_to_skip = ["value", "suffix"]
        bids_name = "_".join("%s-%s" % i for i in bids_dict.items() if i[0] not in keys_to_skip)
        bids_name = f"_{bids_name}" if bids_name else bids_name
        bids_name = f"{bids_dict['value']}{bids_name}_{bids_dict['suffix']}"

        self.name = config.get("name", bids_name)
        self.title = config.get("title")
        self.subtitle = config.get("subtitle")
        self.subsubtitle = config.get("subsubtitle")
        self.description = config.get("description")

        files = glob(str(out_dir / "figures" / f"{self.name}.*"))

        self.components = []
        for file in files:
            src = Path(file)
            ext = "".join(src.suffixes)
            desc_text = config.get("caption")
            iframe = config.get("iframe", False)
            dropdown = config.get("dropdown", False)

            contents = None
            html_anchor = src.relative_to(out_dir)
            if ext == ".html":
                contents = IFRAME_SNIPPET.format(html_anchor) if iframe else src.read_text()
                if dropdown:
                    contents = (
                        f"<details><summary>Advanced ({self.title})</summary>{contents}</details>"
                    )
                    self.title = ""
            elif ext == ".png":
                contents = PNG_SNIPPET.format(html_anchor)

            if contents:
                self.components.append((contents, desc_text))

    def is_empty(self):
        """Check if the reportlet has no components."""
        return len(self.components) == 0


class SubReport(Element):
    """SubReports are sections within a Report."""

    def __init__(self, name, isnested=False, reportlets=None, title=""):
        self.name = name
        self.title = title
        self.reportlets = reportlets or []
        self.isnested = isnested


class Report:
    """The full report object.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    result : :obj:`~nimare.results.MetaResult`
        A MetaResult produced by a coordinate- or image-based meta-analysis.
    out_dir : :obj:`str`
        Output directory in which to save the report.
    out_filename : :obj:`str`, optional
        The name of an html file to export the report to.
        Default is 'report.html'.
    """

    def __init__(
        self,
        results,
        out_dir,
        out_filename="report.html",
    ):
        self.results = results
        meta_type = "CBMA" if issubclass(type(self.results.estimator), CBMAEstimator) else "IBMA"
        self._is_pairwise_estimator = issubclass(
            type(self.results.estimator), PairwiseCBMAEstimator
        )

        # Initialize structuring elements
        self.sections = []
        self.out_dir = Path(out_dir)
        self.out_filename = out_filename

        self.fig_dir = self.out_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        if self._is_pairwise_estimator:
            datasets = [self.results.estimator.dataset1, self.results.estimator.dataset2]
            sel_ids = [
                self.results.estimator.inputs_["id1"],
                self.results.estimator.inputs_["id2"],
            ]
        else:
            datasets = [self.results.estimator.dataset]
            sel_ids = [self.results.estimator.inputs_["id"]]

        for dset_i, (dataset, sel_id) in enumerate(zip(datasets, sel_ids)):
            # Generate summary text
            _gen_summary(
                dataset,
                sel_id,
                meta_type,
                self.fig_dir / f"preliminary_dset-{dset_i+1}_summary.html",
            )

            # Plot mask
            plot_mask(
                dataset.masker.mask_img,
                self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-mask.png",
            )

            if meta_type == "CBMA":
                # Plot coordinates for CBMA estimators
                plot_coordinates(
                    dataset.coordinates,
                    self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-static.png",
                    self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-interactive.html",
                    self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-legend.png",
                )
            elif meta_type == "IBMA":
                # Use "z_maps", for Fishers, and Stouffers; otherwise use "beta_maps".
                key_maps = (
                    "z_maps"
                    if "z_maps" in self.results.estimator.inputs_["raw_data"]
                    else "beta_maps"
                )
                maps_arr = self.results.estimator.inputs_["raw_data"][key_maps]
                ids_ = self.results.estimator.inputs_["id"]
                x_label = "Z" if key_maps == "z_maps" else "Beta"

                if self.results.estimator.aggressive_mask:
                    _plot_relcov_map(
                        maps_arr,
                        self.results.estimator.masker,
                        self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-relcov.png",
                    )
                else:
                    dof_map = self.results.get_map("dof")
                    _plot_dof_map(
                        dof_map,
                        self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-dof.png",
                    )

                _plot_true_voxels(
                    maps_arr,
                    ids_,
                    self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-truevoxels.html",
                )

                _plot_ridgeplot(
                    maps_arr,
                    ids_,
                    x_label,
                    self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-ridgeplot.html",
                )

                _plot_sumstats(
                    maps_arr,
                    ids_,
                    self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-summarystats.html",
                )

                similarity_table = _compute_similarities(maps_arr, ids_)

                plot_heatmap(
                    similarity_table,
                    self.fig_dir / f"preliminary_dset-{dset_i+1}_figure-similarity.html",
                    symmetric=True,
                    cmap="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )

        _gen_est_summary(self.results.estimator, self.fig_dir / "estimator_summary.html")
        _gen_cor_summary(self.results.corrector, self.fig_dir / "corrector_summary.html")
        for diagnostic in self.results.diagnostics:
            img_key = diagnostic.target_image
            diag_name = diagnostic.__class__.__name__
            threshold = diagnostic.voxel_thresh

            _gen_fig_summary(img_key, threshold, self.fig_dir / "corrector_figure-summary.html")
            _gen_diag_summary(diagnostic, self.fig_dir / "diagnostics_summary.html")
            _gen_figures(self.results, img_key, diag_name, threshold, self.fig_dir)

        # Default template from nimare
        self.template_path = Path(pkgrf("nimare", "reports/report.tpl"))
        self._load_config(Path(pkgrf("nimare", "reports/default.yml")))
        assert self.template_path.exists()

    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = settings.get("package", None)

        self.index(settings["sections"])

    def index(self, config):
        """Traverse the reports config definition and instantiate reportlets.

        This method also places figures in their final location.
        """
        for subrep_cfg in config:
            reportlets = [Reportlet(self.out_dir, config=cfg) for cfg in subrep_cfg["reportlets"]]

            if reportlets := [r for r in reportlets if not r.is_empty()]:
                sub_report = SubReport(
                    subrep_cfg["name"],
                    isnested=False,
                    reportlets=reportlets,
                    title=subrep_cfg.get("title"),
                )
                self.sections.append(sub_report)

    def generate_report(self):
        """Once the Report has been indexed, the final HTML can be generated."""
        boilerplate = []
        boiler_idx = 0

        if hasattr(self.results, "description_"):
            text = self.results.description_
            references = self.results.bibtex_
            text = textwrap.fill(text, 99)

            boilerplate.append(
                (
                    boiler_idx,
                    "LaTeX",
                    f"""<pre>{text}</pre>
                        <h3>Bibliography</h3>
                        <pre>{references}</pre>
                    """,
                )
            )
            boiler_idx += 1

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(self.template_path.parent)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        report_tpl = env.get_template(self.template_path.name)
        report_render = report_tpl.render(sections=self.sections, boilerplate=boilerplate)

        # Write out report
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / self.out_filename).write_text(report_render, encoding="UTF-8")


def run_reports(
    results,
    out_dir,
):
    """Run the reports.

    .. versionchanged:: 0.2.1

        * Add similarity matrix to summary for image-based meta-analyses.

    .. versionchanged:: 0.2.0

        * Support for image-based meta-analyses.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    result : :obj:`~nimare.results.MetaResult`
        A MetaResult produced by a coordinate- or image-based meta-analysis.
    out_dir : :obj:`str`
        Output directory in which to save the report.
    """
    return Report(
        results,
        out_dir,
    ).generate_report()
