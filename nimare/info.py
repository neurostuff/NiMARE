import json
import os.path as op
import sys
sys.path.append(op.dirname(__file__))

from _version import get_versions
__version__ = get_versions()['version']
del get_versions

# Get list of authors from Zenodo file
code_dir = op.dirname(__file__)
par_dir = op.abspath(op.join(code_dir, op.pardir))
with open(op.join(par_dir, '.zenodo.json'), 'r') as fo:
    zenodo_info = json.load(fo)
authors = [author['name'] for author in zenodo_info['creators']]
authors = [author.split(', ')[1] + ' ' + author.split(', ')[0] for author in authors]

__author__ = 'NiMARE developers'
__copyright__ = 'Copyright 2018--, NiMARE developers'
__credits__ = authors
__license__ = 'MIT'
__maintainer__ = 'Taylor Salo'
__email__ = 'tsalo006@fiu.edu'
__status__ = 'Prototype'
__url__ = 'https://github.com/neurostuff/NiMARE'
__packagename__ = 'NiMARE'
__description__ = 'NiMARE: Neuroimaging Meta-Analysis Research Environment'
__longdesc__ = """
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

DOWNLOAD_URL = (
    'https://github.com/neurostuff/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

REQUIRES = [
    'click',
    'cognitiveatlas',
    'fuzzywuzzy',
    'matplotlib',
    'nibabel<2.3.0',
    'nilearn',
    'nipype',
    'nltk',
    'numpy',
    'pandas',
    'pyneurovault',
    'scikit-learn',
    'scipy',
    'six',
    'statsmodels',
    'tqdm',
    'traits'
]

TESTS_REQUIRES = [
    'codecov',
    'coverage',
    'coveralls',
    'flake8',
    'pytest',
    'pytest-cov'
]

EXTRA_REQUIRES = {
    'peaks2maps-cpu': [
        'tensorflow>=1.0.0',
        'appdirs'
    ],
    'peaks2maps-gpu': [
        'tensorflow-gpu>=1.0.0',
        'appdirs'
    ],
    'doc': [
        'sphinx>=1.5.3',
        'sphinx_rtd_theme',
        'sphinx-argparse',
        'numpydoc'
    ],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = list(set([
    v for deps in EXTRA_REQUIRES.values() for v in deps]))

ENTRY_POINTS = {'console_scripts': ['nimare=nimare.cli:cli']}

# Package classifiers
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering'
]
