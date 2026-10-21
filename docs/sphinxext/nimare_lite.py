"""Build the JupyterLite site that runs the gallery examples in the browser.

jupyterlite_sphinx builds the site and sphinx-gallery adds the example notebooks to it.
This extension adds the rest:

- the kernel environment from ``jupyterlite/environment.yml``, byte-compiled so the
  browser doesn't compile every module it imports from source;
- the files of the NIDM pain dataset that the examples read, so ``download_nidm_pain``
  finds them instead of going to NeuroVault;
- a first cell in each notebook that turns on inline figures and, in the examples that
  read the pain dataset, points NiMARE at that copy;
- no launch button for the examples in ``UNSUPPORTED``.
"""

import csv
import json
import os
import py_compile
import re
import shutil
import sys
from importlib.util import cache_from_source
from pathlib import Path

import yaml
from sphinx.util import logging

LGR = logging.getLogger(__name__)

# Examples, by path under examples/, that can't run in the browser, and why.
UNSUPPORTED = {
    "01_datasets/02_download_neurosynth": "it downloads the Neurosynth and NeuroQuery databases",
    "01_datasets/03_plot_neurovault_io": "it downloads images from NeuroVault",
    "01_datasets/05_plot_nimads": "it downloads a studyset from NeuroStore",
    # emscripten-forge's pyarrow 23 traps ("table index is out of bounds") in pd.read_parquet.
    "01_datasets/07_plot_parquet_studyset": (
        "pyarrow, which reads parquet files, crashes in the browser"
    ),
    "01_datasets/08_fetch_neurostore": "it downloads studysets from NeuroStore",
    "02_meta-analyses/07_macm": (
        "it needs a local copy of the Neurosynth database and downloads an atlas through nilearn"
    ),
    "02_meta-analyses/11_plot_cbmr": "CBMR needs PyTorch, which has no WebAssembly build",
    "02_meta-analyses/15_plot_predictive_ale": "it needs XGBoost, which has no WebAssembly build",
    "03_annotation/02_plot_cognitive_atlas": "it downloads the Cognitive Atlas",
    "05_machine_learning/01_plot_machine_learning_in_nimare": (
        "it downloads a studyset from NeuroStore and an atlas through nilearn"
    ),
}

# sphinx-gallery's default jupyterlite_contents directory, under the docs source dir.
GALLERY_CONTENTS = "jupyterlite_contents"
DATA_DIR = "nimare_data"


def _build_dir(app):
    return Path(app.srcdir) / "_build" / "jupyterlite"


def _example_name(path):
    """Return an example's path under examples/, without the extension."""
    parts = Path(path).with_suffix("").parts
    return "/".join(parts[parts.index("auto_examples") + 1 :])


def _create_kernel_env(lite_dir, root):
    """Create the kernel environment under ``root`` and byte-compile it.

    jupyterlite-xeus creates environments with ``--no-pyc`` and empack drops ``*.pyc``,
    so the browser would compile every module from source on each page load.
    """
    from empack.pack import DEFAULT_CONFIG_PATH
    from jupyterlite_xeus.create_conda_env import create_conda_env_from_env_file

    spec = yaml.safe_load((lite_dir / "environment.yml").read_text())
    shutil.rmtree(root, ignore_errors=True)
    create_conda_env_from_env_file(root, spec, lite_dir)
    prefix = root / "envs" / spec["name"]

    site = next(prefix.glob("lib/python3.*/site-packages"))
    env_version = site.parent.name.removeprefix("python")
    our_version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if env_version == our_version:
        _compile_prefix(prefix, site)
    else:
        LGR.warning(
            f"The JupyterLite kernel uses Python {env_version} and the docs build Python "
            f"{our_version}, so the kernel ships without .pyc files and imports slowly."
        )

    # empack's default filter, minus its *.pyc exclusion.
    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    for section in [config.get("default", {}), *config.get("packages", {}).values()]:
        section["exclude_patterns"] = [
            p for p in section.get("exclude_patterns", []) if p.get("pattern") != "**/*.pyc"
        ]
    empack_config = root / "empack_config.yaml"
    empack_config.write_text(yaml.safe_dump(config, sort_keys=False))
    return prefix, empack_config


def _compile_prefix(prefix, site):
    """Byte-compile every .py file in the prefix and list each .pyc with its package.

    Bytecode is platform independent, so this Python can compile for the wasm one when the
    minor versions match. The .pyc files are unchecked-hash, because the mtimes empack's
    tarballs give the sources would invalidate timestamp-based ones. empack packs what
    each package lists (``conda-meta/*.json`` ``files``, or the pip ``RECORD``), so the
    .pyc paths are added there.
    """
    owners = {}  # .py path -> (base its listed paths are relative to, metadata file)
    for meta in (prefix / "conda-meta").glob("*.json"):
        for f in json.loads(meta.read_text()).get("files", []):
            if f.endswith(".py"):
                owners[prefix / f] = (prefix, meta)
    for record in site.glob("*.dist-info/RECORD"):
        if (record.parent / "INSTALLER").read_text().strip() != "pip":
            continue  # conda packages' RECORDs duplicate conda-meta, which empack reads
        for row in csv.reader(record.read_text().splitlines()):
            if row and row[0].endswith(".py"):
                owners[(site / row[0]).resolve()] = (site, record)

    added = {}
    for src, (base, meta) in owners.items():
        pyc = cache_from_source(str(src))
        try:
            py_compile.compile(
                str(src),
                cfile=pyc,
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )
        except (py_compile.PyCompileError, OSError):
            continue  # templates and Python 2 files in some packages
        added.setdefault(meta, []).append(Path(pyc).relative_to(base.resolve()).as_posix())

    for meta, paths in added.items():
        if meta.suffix == ".json":
            data = json.loads(meta.read_text())
            data["files"] = sorted(set(data["files"]) | set(paths))
            meta.write_text(json.dumps(data))
        else:
            with open(meta, "a", newline="") as fh:
                csv.writer(fh).writerows([p, "", ""] for p in paths)
    LGR.info(f"[nimare_lite] compiled {sum(map(len, added.values()))} of {len(owners)} modules")


def _bundle_pain_data(dest):
    """Copy the pain dataset files that NiMARE's pain JSON files reference into ``dest``."""
    from nimare.extract import download_nidm_pain
    from nimare.utils import get_resource_path

    def walk(obj):
        if isinstance(obj, dict):
            # A studyset's images give a bare "filename" next to the "url" path.
            obj = [v for k, v in obj.items() if k != "filename"]
        if isinstance(obj, list):
            for value in obj:
                yield from walk(value)
        elif isinstance(obj, str) and obj.endswith(".nii.gz"):
            yield obj

    files = {"description.txt"}
    for name in ["nidm_pain_dset.json", "nidm_pain_studyset.json"]:
        with open(os.path.join(get_resource_path(), name)) as fh:
            files.update(walk(json.load(fh)))

    src = Path(download_nidm_pain())
    shutil.rmtree(dest, ignore_errors=True)
    for f in sorted(files):
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f, dest / f)


def prepare_site(app, exception):
    """Create the kernel environment and the bundled data, and point the build at them.

    Runs on build-finished before jupyterlite_sphinx's handler, which builds the site.
    """
    if exception is not None or app.builder.name not in ["html", "readthedocs"]:
        return

    build = _build_dir(app)
    lite_dir = Path(app.srcdir) / app.config.jupyterlite_dir
    prefix, empack_config = _create_kernel_env(lite_dir, build / "env")
    app.config.jupyterlite_build_command_options = {
        **(app.config.jupyterlite_build_command_options or {}),
        "XeusAddon.prefix": str(prefix),
        "XeusAddon.empack_config": str(empack_config),
    }

    data = build / DATA_DIR
    _bundle_pain_data(data / "nidm_21pain")

    # jupyterlite_sphinx (>= 0.23) puts a contents directory in the site under its own
    # name, but sphinx-gallery links to auto_examples/ at the site root. Listing the
    # gallery contents' children puts auto_examples/ there.
    gallery = str(Path(app.srcdir) / GALLERY_CONTENTS)
    app.config.jupyterlite_contents = [
        f"{gallery}/*" if c == gallery else c for c in app.config.jupyterlite_contents
    ] + [str(data)]


def modify_notebook(notebook, filename):
    """Prepare the notebook for the browser, or say why it can't run there.

    sphinx-gallery calls this on each notebook it copies into the JupyterLite site.
    """
    name = _example_name(filename)
    if name in UNSUPPORTED:
        cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                f"**This example can't run in the browser**, because {UNSUPPORTED[name]}. "
                "Run it in a local NiMARE installation instead."
            ),
        }
    else:
        # The kernel's matplotlib backend is Agg, which plt.show() can't display.
        source = "%matplotlib inline"
        if any("download_nidm_pain" in "".join(c["source"]) for c in notebook["cells"]):
            # The notebook is at auto_examples/<name>.ipynb, so its folder is as many
            # levels below the site root as name has parts.
            data = "/".join([".."] * len(Path(name).parts) + [DATA_DIR])
            source += (
                "\n\n# This site bundles the pain dataset, so download_nidm_pain() finds it "
                "here instead of downloading it.\n"
                "import os\n\n"
                f'os.environ["NIMARE_DATA"] = os.path.abspath("{data}")'
            )
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }
    notebook["cells"].insert(0, cell)


def remove_launch_button(app, docname, source):
    """Drop the JupyterLite button and mention from unsupported examples' pages."""
    if not docname.startswith("auto_examples/"):
        return
    if _example_name(docname) not in UNSUPPORTED:
        return
    text = source[0].replace(" or to run this example in your browser via JupyterLite.", ".")
    source[0] = re.sub(r"\n( *)\.\. container:: lite-badge\n\n(?:\1  .*\n|\n)*", "\n", text)


def setup(app):
    app.connect("build-finished", prepare_site, priority=400)
    app.connect("source-read", remove_launch_button)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
