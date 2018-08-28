# NiMARE: Neuroimaging Meta-Analysis Research Environment
A Python library for coordinate- and image-based meta-analysis.

[![CircleCI](https://circleci.com/gh/neurostuff/NiMARE.svg?style=shield)](https://circleci.com/gh/neurostuff/NiMARE)
[![Documentation Status](https://readthedocs.org/projects/NiMARE/badge/?version=latest)](http://NiMARE.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/neurostuff/NiMARE/branch/master/graph/badge.svg)](https://codecov.io/gh/neurostuff/NiMARE)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Supported meta-analytic methods (`nimare.meta`)
- Coordinate-based methods (`nimare.meta.cbma`)
    - Kernel-based methods
        - Activation likelihood estimation (ALE)
        - Specific coactivation likelihood estimation (SCALE)
        - Multilevel kernel density analysis (MKDA)
        - Kernel density analysis (KDA)
    - Model-based methods (`nimare.meta.cbma.model`)
        - Bayesian hierarchical cluster process model (BHICP)
        - Hierarchical Poisson/Gamma random field model (HPGRF)
        - Spatial Bayesian latent factor regression (SBLFR)
        - Spatial binary regression (SBR)
- Image-based methods (`nimare.meta.ibma`)
    - Mixed effects general linear model (MFX-GLM)
    - Random effects general linear model (RFX-GLM)
    - Fixed effects general linear model (FFX-GLM)
    - Stouffer's meta-analysis
    - Random effects Stouffer's meta-analysis
    - Weighted Stouffer's meta-analysis
    - Fisher's meta-analysis

## Additional functionality
- Database extraction (`nimare.dataset.extract`)
    - NeuroVault
    - Neurosynth
    - Brainspell
    - PubMed abstract extraction
- Functional characterization analysis (`nimare.decode`)
    - BrainMap decoding
    - Neurosynth correlation-based decoding
    - Neurosynth MKDA-based decoding
    - Generalized correspondence latent Dirichlet allocation (GCLDA)

## Installation

### Local installation
```
python setup.py install
```

### Installation with Docker
To build the Docker image:
```
docker build -t test/nimare .
```

To run the Docker container:
```
docker run -it -v `pwd`:/home/neuro/code/NiMARE -p8888:8888 test/nimare bash
```

Once inside the container, you can install NiMARE:
```
python /home/neuro/code/NiMARE/setup.py develop
```

## Contributing

Please see our [contributing guidelines](https://github.com/neurostuff/NiMARE/blob/master/CONTRIBUTING.md) for more information on contributing
to NiMARE.

We ask that all contributions to `NiMARE` respect our [code of conduct](https://github.com/neurostuff/NiMARE/blob/master/CODE_OF_CONDUCT.md).
