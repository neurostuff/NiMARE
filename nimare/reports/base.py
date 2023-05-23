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
from pkg_resources import resource_filename as pkgrf

from nimare.reports.figures import (
    gen_table,
    plot_coordinates,
    plot_heatmap,
    plot_interactive_brain,
    plot_static_brain,
)

SVG_SNIPPET = [
    """\
    <object class="svg-reportlet" type="image/svg+xml" data="./{0}">
    Problem loading figure {0}. If the link below works, please try \
    reloading the report in your browser.</object>
    </div>
    <div class="elem-filename">
        Get figure file: <a href="./{0}" target="_blank">{0}</a>
    </div>
    """,
    """\
    <img class="svg-reportlet" src="./{0}" style="width: 100%" />
    </div>
    <div class="elem-filename">
        Get figure file: <a href="./{0}" target="_blank">{0}</a>
    </div>
    """,
]

IFRAME_SNIPPET = """\
    <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" \
    src="./{0}" height="800" width="100%"></iframe>
"""

PARAMETERS_DICT = {
    "kernel_transformer__fwhm": "FWHM",
    "kernel_transformer__sample_size": "Sample size",
    "null_method": "Null method",
    "n_iters": "Number of iterations",
    "n_cores": "Number of cores",
}

SUMMARY_TEMPLATE = """\
<ul class="elem-desc">
<li>Number of studies: {n_exps:d}</li>
<li>Number of studies included: {n_exps_sel:d}</li>
<li>Number of foci: {n_foci:d} </li>
<li>Number of foci outside the mask: {n_foci_nonbrain:d} </li>
</ul>
<details>
<summary>Studies excluded</summary><br />
<p>{exc_ids}</p>
</details>
"""

ESTIMATOR_TEMPLATE = """\
    <ul class="elem-desc">
    <li>Kernel Transformer: {kernel_transformer}{ker_params_text}</li>
    {est_params_text}
    </ul>
"""

CORRECTOR_TEMPLATE = """\
    <ul class="elem-desc">
    {cor_params_text}
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


def gen_est_summary(obj, out_filename):
    """Generate html with parameter use in obj (e.g., estimator)."""
    params_dict = obj.get_params()
    est_params = {k: v for k, v in params_dict.items() if not k.startswith("kernel_transformer")}
    ker_params = {k: v for k, v in params_dict.items() if k.startswith("kernel_transformer__")}

    ker_params_text = ["<ul>"]
    for k, v in ker_params.items():
        ker_params_text.append(f"<li>{PARAMETERS_DICT[k]}: {v}</li>")
    ker_params_text.append("</ul>")
    ker_params_text = "".join(ker_params_text)

    est_params_text = []
    for k, v in est_params.items():
        est_params_text.append(f"<li>{PARAMETERS_DICT[k]}: {v}</li>")
    est_params_text = "".join(est_params_text)

    summary_text = ESTIMATOR_TEMPLATE.format(
        kernel_transformer=str(params_dict["kernel_transformer"]),
        ker_params_text=ker_params_text,
        est_params_text=est_params_text,
    )
    (out_filename).write_text(summary_text, encoding="UTF-8")


def gen_cor_summary(obj, out_filename):
    """Generate html with parameter use in obj (e.g., estimator, corrector, diagnostics)."""
    pass


def gen_diag_summary(obj, out_filename):
    """Generate html with parameter use in obj (e.g., estimator, corrector, diagnostics)."""
    params_dict = obj.get_params()

    summary_text = DIAGNOSTIC_TEMPLATE.format(**params_dict)
    (out_filename).write_text(summary_text, encoding="UTF-8")


def gen_summary(results, out_filename):
    """Generate preliminary checks from dataset for the report."""
    dset = results.estimator.dataset

    mask = dset.masker.mask_img
    sel_ids = dset.get_studies_by_mask(mask)
    sel_dset = dset.slice(sel_ids)

    n_foci = dset.coordinates.shape[0]
    n_foci_sel = sel_dset.coordinates.shape[0]
    n_foci_nonbrain = n_foci - n_foci_sel

    n_exps = len(dset.ids)
    n_exps_sel = len(sel_dset.ids)
    exc_ids = list(set(dset.ids) - set(sel_dset.ids))
    exc_ids_str = ", ".join(exc_ids)

    summary_text = SUMMARY_TEMPLATE.format(
        n_exps=n_exps,
        n_exps_sel=n_exps_sel,
        n_foci=n_foci,
        n_foci_nonbrain=n_foci_nonbrain,
        exc_ids=exc_ids_str,
    )
    (out_filename).write_text(summary_text, encoding="UTF-8")


def gen_figures(results, img_key, diag_name, fig_dir):
    """Generate html and jpeg objects for the report."""
    # Plot brain images
    img = results.get_map(img_key)
    plot_interactive_brain(img, fig_dir / f"{img_key}_figure-interactive.html")
    plot_static_brain(img, fig_dir / f"{img_key}_figure-static.png")

    # Plot clusters table
    cluster_table = results.tables[f"{img_key}_tab-clust"]
    gen_table(cluster_table, fig_dir / f"{img_key}_tab-clust_table.html")
    # plot_clusters(img, fig_dir / f"{img_key}_clust.png")

    if f"{img_key}_diag-{diag_name}_tab-counts" in results.tables:
        contribution_table = results.tables[f"{img_key}_diag-{diag_name}_tab-counts"]
        contribution_table = contribution_table.set_index("id")

        plot_heatmap(
            contribution_table,
            fig_dir / f"{img_key}_diag-{diag_name}_tab-counts_figure.html",
        )


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
        self.description = config.get("description")

        files = glob(str(out_dir / "figures" / f"{self.name}.*"))

        self.components = []
        for file in files:
            src = Path(file)
            ext = "".join(src.suffixes)
            desc_text = config.get("caption")
            iframe = config.get("iframe", False)

            contents = None
            html_anchor = src.relative_to(out_dir)
            if ext == ".html":
                contents = IFRAME_SNIPPET.format(html_anchor) if iframe else src.read_text()
            elif ext == ".png":
                contents = SVG_SNIPPET[config.get("static", True)].format(html_anchor)

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
    """The full report object. This object maintains a BIDSLayout to index all reportlets."""

    def __init__(
        self,
        results,
        out_dir,
        config=None,
        out_filename="report.html",
    ):
        self.results = results

        # Initialize structuring elements
        self.sections = []
        self.out_dir = Path(out_dir)
        self.out_filename = out_filename

        self.fig_dir = self.out_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # Generate summary text and figures
        gen_summary(self.results, self.fig_dir / "preliminary_summary.html")

        # Plot coordinates
        plot_coordinates(
            results.estimator.dataset,
            self.fig_dir / "preliminary_figure-static.png",
            self.fig_dir / "preliminary_figure-interactive.html",
            self.fig_dir / "preliminary_figure-legend.png",
        )

        gen_est_summary(self.results.estimator, self.fig_dir / "estimator_summary.html")
        gen_cor_summary(self.results.corrector, self.fig_dir / "corrector_summary.html")
        for diagnostic in self.results.diagnostics:
            img_key = diagnostic.target_image
            diag_name = diagnostic.__class__.__name__
            gen_diag_summary(diagnostic, self.fig_dir / f"{img_key}_diag-{diag_name}_summary.html")
            gen_figures(self.results, img_key, diag_name, self.fig_dir)

        # Default template from nimare
        self.template_path = Path(pkgrf("nimare", "reports/report.tpl"))
        self._load_config(Path(config or pkgrf("nimare", "reports/default.yml")))
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

            # Filter out empty reportlets
            reportlets = [r for r in reportlets if not r.is_empty()]
            if reportlets:
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
    config=None,
):
    """Run the reports."""
    return Report(
        results,
        out_dir,
        config=config,
    ).generate_report()
