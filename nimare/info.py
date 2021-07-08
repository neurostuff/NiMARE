import importlib.util
import json
import os.path as op

spec = importlib.util.spec_from_file_location(
    "_version", op.join(op.dirname(__file__), "nimare/_version.py")
)
_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_version)

VERSION = _version.get_versions()["version"]
del _version

# Get list of authors from Zenodo file
with open(op.join(op.dirname(__file__), ".zenodo.json"), "r") as fo:
    zenodo_info = json.load(fo)
authors = [author["name"] for author in zenodo_info["creators"]]
authors = [author.split(", ")[1] + " " + author.split(", ")[0] for author in authors]

AUTHOR = "NiMARE developers"
COPYRIGHT = "Copyright 2018--, NiMARE developers"
CREDITS = authors
LICENSE = "MIT"
MAINTAINER = "Taylor Salo"
EMAIL = "tsalo006@fiu.edu"
STATUS = "Prototype"
URL = "https://github.com/neurostuff/NiMARE"
PACKAGENAME = "NiMARE"
DESCRIPTION = "NiMARE: Neuroimaging Meta-Analysis Research Environment"
LONGDESC = """
NiMARE
======
NiMARE (Neuroimaging Meta-Analysis Research Environment) is a Python package
for coordinate-based and image-based meta-analysis of neuroimaging data.

License
=======
``NiMARE`` is licensed under the terms of the MIT license. See the file
'LICENSE' for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2018--, NiMARE developers
"""

DOWNLOAD_URL = "https://github.com/neurostuff/{name}/archive/{ver}.tar.gz".format(
    name=PACKAGENAME, ver=VERSION
)

REQUIRES = [
    "cognitiveatlas",
    "fuzzywuzzy",
    "indexed_gzip>=1.4.0",
    "matplotlib",
    "nibabel>=3.0.0",
    "nilearn>=0.7.1",
    "numpy",
    "pandas",
    "pymare>=0.0.2",
    "scikit-learn",
    "scipy",
    "seaborn",
    "statsmodels",
    "tqdm",
]

TESTS_REQUIRES = [
    "codecov",
    "coverage",
    "coveralls",
    "flake8-black",
    "flake8-docstrings",
    "flake8-isort",
    "pytest",
    "pytest-cov",
]

EXTRA_REQUIRES = {
    "peaks2maps-cpu": ["tensorflow>=2.0.0", "appdirs"],
    "peaks2maps-gpu": ["tensorflow-gpu>=2.0.0", "appdirs"],
    "doc": [
        "m2r",
        "pillow",
        "recommonmark",
        "sphinx>=3.1.1",
        "sphinx-argparse",
        "sphinx-copybutton",
        "sphinx_gallery",
        "sphinx_rtd_theme",
    ],
    "tests": TESTS_REQUIRES,
    "duecredit": ["duecredit"],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES["all"] = list(set([v for deps in EXTRA_REQUIRES.values() for v in deps]))

ENTRY_POINTS = {"console_scripts": ["nimare=nimare.cli:_main"]}

# Package classifiers
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering",
]
