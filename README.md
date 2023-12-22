# NiMARE: Neuroimaging Meta-Analysis Research Environment
A Python library for coordinate- and image-based meta-analysis.

[![Latest Version](https://img.shields.io/pypi/v/nimare.svg)](https://pypi.python.org/pypi/nimare/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nimare.svg)](https://pypi.python.org/pypi/nimare/)
[![GitHub Repository](https://img.shields.io/badge/Source%20Code-neurostuff%2Fnimare-purple)](https://github.com/neurostuff/NiMARE)
[![DOI](https://zenodo.org/badge/117724523.svg)](https://zenodo.org/badge/latestdoi/117724523)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Test Status](https://github.com/neurostuff/NiMARE/actions/workflows/testing.yml/badge.svg)](https://github.com/neurostuff/NiMARE/actions/workflows/testing.yml)
[![Documentation Status](https://readthedocs.org/projects/nimare/badge/?version=stable)](http://nimare.readthedocs.io/en/stable/?badge=stable)
[![Codecov](https://codecov.io/gh/neurostuff/NiMARE/branch/main/graph/badge.svg)](https://codecov.io/gh/neurostuff/nimare)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Join the chat at https://mattermost.brainhack.org/brainhack/channels/nimare](https://img.shields.io/badge/mattermost-join_chat%20%E2%86%92-brightgreen.svg)](https://mattermost.brainhack.org/brainhack/channels/nimare)
[![RRID:SCR_017398](https://img.shields.io/badge/RRID-SCR__017398-blue.svg)](https://scicrunch.org/scicrunch/Resources/record/nlx_144509-1/SCR_017398/resolver?q=nimare&l=nimare)
[![Paper](https://img.shields.io/badge/Aperture-10.52294/001c.87681-darkblue.svg)](https://doi.org/10.52294/001c.87681)
[![Preprint](https://neurolibre.org/papers/10.55458/neurolibre.00007/status.svg)](https://doi.org/10.55458/neurolibre.00007)

Currently, NiMARE implements a range of image- and coordinate-based meta-analytic algorithms, as well as several methods for advanced meta-analytic methods, like automated annotation and functional decoding.

## Installation

Please see our [installation instructions](https://nimare.readthedocs.io/en/stable/installation.html)
for information on how to install NiMARE.

### Installation with pip
```
pip install nimare
```

### Local installation (development version)
```
pip install git+https://github.com/neurostuff/NiMARE.git
```

## Citing NiMARE

If you use NiMARE in your research, we recommend citing the Zenodo DOI associated with the NiMARE version you used,
as well as the Aperture Neuro journal article for the NiMARE Jupyter book.
You can find the Zenodo DOI associated with each NiMARE release at https://zenodo.org/record/6642243#.YqiXNy-B1KM.

```BibTeX
# This is the Aperture Neuro paper.
@article{Salo2023,
  doi = {10.52294/001c.87681},
  url = {https://doi.org/10.52294/001c.87681},
  year = {2023},
  volume = {3},
  pages = {1 - 32},
  author = {Taylor Salo and Tal Yarkoni and Thomas E. Nichols and Jean-Baptiste Poline and Murat Bilgel and Katherine L. Bottenhorn and Dorota Jarecka and James D. Kent and Adam Kimbler and Dylan M. Nielson and Kendra M. Oudyk and Julio A. Peraza and Alexandre Pérez and Puck C. Reeders and Julio A. Yanes and Angela R. Laird},
  title = {NiMARE: Neuroimaging Meta-Analysis Research Environment},
  journal = {Aperture Neuro}
}

# This is the Zenodo citation for version 0.0.11.
@software{salo_taylor_2022_5826281,
  author       = {Salo, Taylor and
                  Yarkoni, Tal and
                  Nichols, Thomas E. and
                  Poline, Jean-Baptiste and
                  Kent, James D. and
                  Gorgolewski, Krzysztof J. and
                  Glerean, Enrico and
                  Bottenhorn, Katherine L. and
                  Bilgel, Murat and
                  Wright, Jessey and
                  Reeders, Puck and
                  Kimbler, Adam and
                  Nielson, Dylan N. and
                  Yanes, Julio A. and
                  Pérez, Alexandre and
                  Oudyk, Kendra M. and
                  Jarecka, Dorota and
                  Enge, Alexander and
                  Peraza, Julio A. and
                  Laird, Angela R.},
  title        = {neurostuff/NiMARE: 0.0.11},
  month        = jan,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {0.0.11},
  doi          = {10.5281/zenodo.5826281},
  url          = {https://doi.org/10.5281/zenodo.5826281}
}
```

To cite NiMARE in your manuscript, we recommend something like the following:

> We used NiMARE v0.0.11 (RRID:SCR_017398; Salo et al., 2022a; Salo et al., 2022b).

## Contributing

Please see our [contributing guidelines](https://github.com/neurostuff/NiMARE/blob/main/CONTRIBUTING.md)
for more information on contributing to NiMARE.

We ask that all contributions to `NiMARE` respect our [code of conduct](https://github.com/neurostuff/NiMARE/blob/main/CODE_OF_CONDUCT.md).
