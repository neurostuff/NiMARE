# NiMARE: Neuroimaging Meta-Analysis Research Environment
A Python library for coordinate- and image-based meta-analysis.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CircleCI](https://circleci.com/gh/neurostuff/NiMARE.svg?style=shield)](https://circleci.com/gh/neurostuff/NiMARE)
[![Documentation Status](https://readthedocs.org/projects/nimare/badge/?version=latest)](http://nimare.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/neurostuff/NiMARE/branch/master/graph/badge.svg)](https://codecov.io/gh/neurostuff/nimare)
[![Join the chat at https://mattermost.brainhack.org/brainhack/channels/nimare](https://img.shields.io/badge/mattermost-join_chat%20%E2%86%92-brightgreen.svg)](https://mattermost.brainhack.org/brainhack/channels/nimare)

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

- Automated annotation (`nimare.annotate`)
    - Tf-idf vectorization of text (`nimare.annotate.tfidf`)
    - Ontology-based annotation (`nimare.annotate.ontology`)
        - Cognitive Paradigm Ontology (`nimare.annotate.ontology.cogpo`)
        - Cognitive Atlas (`nimare.annotate.ontology.cogat`)
    - Topic model-based annotation (`nimare.annotate.topic`)
        - Latent Dirichlet allocation (`nimare.annotate.topic.lda`)
        - Generalized correspondence latent Dirichlet allocation
          (`nimare.annotate.topic.gclda`)
        - Deep Boltzmann machines (`nimare.annotate.topic.boltzmann`)
    - Vector model-based annotation (`nimare.annotate.vector`)
        - Global Vectors for Word Representation model
          (`nimare.annotate.vector.word2brain`)
        - Text2Brain model (`nimare.annotate.vector.text2brain`)
- Database extraction (`nimare.dataset.extract`)
    - NeuroVault
    - Neurosynth
    - Brainspell
    - PubMed abstract extraction
- Functional characterization analysis (`nimare.decode`)
    - BrainMap decoding
    - Neurosynth correlation-based decoding
    - Neurosynth MKDA-based decoding
    - BrainMap decoding
    - Text2brain encoding
    - Generalized correspondence latent Dirichlet allocation (GCLDA)
- Meta-analytic parcellation (`nimare.parcellate`)
    - Meta-analytic parcellation based on text (MAPBOT)
    - Coactivation-base parcellation (CBP)
    - Meta-analytic activation modeling-based parcellation (MAMP)
- Common workflows (`nimare.workflows`)
    - Meta-analytic coactivation modeling (MACM)
    - Meta-analytic clustering analysis
    - Meta-analytic independent components analysis (metaICA)

## Installation

### Local installation (development version)
```
pip install git+https://github.com/neurostuff/NiMARE.git#egg=nimare[peaks2maps-cpu]
```
If you have [TensorFlow configured to take advantage of your local GPU](https://www.tensorflow.org/install/) use
```
pip install git+https://github.com/neurostuff/NiMARE.git#egg=nimare[peaks2maps-gpu]
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
