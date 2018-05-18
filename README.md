# NiMARE: Neuroimaging Meta-Analysis Research Environment
A Python library for coordinate- and image-based meta-analysis.

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
