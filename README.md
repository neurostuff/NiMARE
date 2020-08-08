# NiMARE: Neuroimaging Meta-Analysis Research Environment
A Python library for coordinate- and image-based meta-analysis.

[![Latest Version](https://img.shields.io/pypi/v/nimare.svg)](https://pypi.python.org/pypi/nimare/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nimare.svg)](https://pypi.python.org/pypi/nimare/)
[![DOI](https://zenodo.org/badge/117724523.svg)](https://zenodo.org/badge/latestdoi/117724523)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CircleCI](https://circleci.com/gh/neurostuff/NiMARE.svg?style=shield)](https://circleci.com/gh/neurostuff/NiMARE)
[![Documentation Status](https://readthedocs.org/projects/nimare/badge/?version=latest)](http://nimare.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/neurostuff/NiMARE/branch/master/graph/badge.svg)](https://codecov.io/gh/neurostuff/nimare)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Join the chat at https://mattermost.brainhack.org/brainhack/channels/nimare](https://img.shields.io/badge/mattermost-join_chat%20%E2%86%92-brightgreen.svg)](https://mattermost.brainhack.org/brainhack/channels/nimare)
[![RRID:SCR_017398](https://img.shields.io/badge/RRID-SCR__017398-blue.svg)](https://scicrunch.org/scicrunch/Resources/record/nlx_144509-1/SCR_017398/resolver?q=nimare&l=nimare)

Currently, NiMARE implements a range of image- and coordinate-based meta-analytic algorithms, as well as several methods for advanced meta-analytic methods, like automated annotation and functional decoding.

## Installation

Please see our [installation instructions](https://nimare.readthedocs.io/en/latest/installation.html)
for information on how to install NiMARE.

### Installation with pip
```
pip install nimare
```

### Local installation (development version)
```
pip install git+https://github.com/neurostuff/NiMARE.git#egg=nimare[peaks2maps-cpu]
```
If you have [TensorFlow configured to take advantage of your local GPU](https://www.tensorflow.org/install/) use
```
pip install git+https://github.com/neurostuff/NiMARE.git#egg=nimare[peaks2maps-gpu]
```

## Contributing

Please see our [contributing guidelines](https://github.com/neurostuff/NiMARE/blob/master/CONTRIBUTING.md)
for more information on contributing to NiMARE.

We ask that all contributions to `NiMARE` respect our [code of conduct](https://github.com/neurostuff/NiMARE/blob/master/CODE_OF_CONDUCT.md).
