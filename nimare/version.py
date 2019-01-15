from __future__ import absolute_import, division, print_function

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 1  # use '' for first of series, number for 1 and above
_version_extra = 'a'

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "NiMARE: Neuroimaging Meta-Analysis Research Environment"
# Long description will go up on the pypi page
long_description = """

NiMARE
======
NiMARE (Neuroimaging Meta-Analysis Research Environment) is a Python package
for coordinate-based and image-based meta-analysis of neuroimaging data.

License
=======
``NiMARE`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2018--, NiMARE developers

"""

NAME = "NiMARE"
MAINTAINER = "NiMARE developers"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/neurostuff/NiMARE"
DOWNLOAD_URL = "http://github.com/neurostuff/NiMARE.git"
LICENSE = "MIT"
AUTHOR = "NiMARE developers"
AUTHOR_EMAIL = "http://github.com/neurostuff/NiMARE"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
REQUIRES = ["nibabel", "numpy", "scipy", "pandas", "statsmodels", "nipype",
            "scikit-learn", "nilearn", "duecredit", "pyneurovault", "six",
            "matplotlib", "nltk", "fuzzywuzzy", "cognitiveatlas", "click",
            "tqdm"],
ENTRY_POINTS = {'console_scripts': ['nimare=nimare.cli:cli']}

EXTRAS_REQUIRES = {
    "peaks2maps-cpu": ["tensorflow>=1.0.0", "appdirs"],
    "peaks2maps-gpu": ["tensorflow-gpu>=1.0.0", "appdirs"],
}
