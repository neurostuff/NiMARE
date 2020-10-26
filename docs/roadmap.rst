.. include:: links.rst

Our Roadmap
===========

NiMARE's primary goal is to consolidate coordinate- and image-based
meta-analysis methods with a simple, shared and comprehensive interface.
This should reduce brand loyalty to any given algorithm, as it should be easy
to employ the most appropriate algorithm for a given project. It also provides
an environment where comparisons between methods are easier to perform.

A secondary goal of NiMARE is to implement some of the more cutting-edge
methods for analyses built on meta-analytic neuroimaging data.
There are many tools or algorithms that use meta-analytic data, including
automated annotation, meta-analytic functional characterization analysis, and
meta-analytic parcellation.
Many of these methods are either tied to a specific meta-analysis package or
never make it from publication to useable (i.e., documented and tested) code.

Ultimately, we plan to support all (or most) of the methods listed below in NiMARE:

- Coordinate-based methods (`nimare.meta`)
    - Kernel-based methods
        - Activation likelihood estimation (ALE)
        - Specific coactivation likelihood estimation (SCALE)
        - Multilevel kernel density analysis (MKDA)
        - Kernel density analysis (KDA)
    - Model-based methods (`nimare.meta.model`)
        - Bayesian hierarchical cluster process model (BHICP)
        - Hierarchical Poisson/Gamma random field model (HPGRF)
        - Spatial Bayesian latent factor regression (SBLFR)
        - Spatial binary regression (SBR)
- Image-based methods (`nimare.meta.ibma`)
- Automated annotation (`nimare.annotate`)
    - TF-IDF vectorization of text (`nimare.annotate.tfidf`)
    - Ontology-based annotation (`nimare.annotate.ontology`)
        - Cognitive Paradigm Ontology (`nimare.annotate.ontology.cogpo`)
        - Cognitive Atlas (`nimare.annotate.ontology.cogat`)
    - Topic model-based annotation (`nimare.annotate.topic`)
        - Latent Dirichlet allocation (`nimare.annotate.topic.LDAModel`)
        - Generalized correspondence latent Dirichlet allocation
          (`nimare.annotate.topic.GCLDAModel`)
        - Deep Boltzmann machines (`nimare.annotate.topic.BoltzmannModel`)
    - Vector model-based annotation (`nimare.annotate.vector`)
        - Global Vectors for Word Representation model
          (`nimare.annotate.vector.Word2BrainModel`)
        - Text2Brain model (`nimare.annotate.vector.Text2BrainModel`)
- Database extraction (`nimare.extract`)
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
    - Prediction framework (e.g. NeuroQuery)
- Meta-analytic parcellation (`nimare.parcellate`)
    - Meta-analytic parcellation based on text (MAPBOT)
    - Coactivation-base parcellation (CBP)
    - Meta-analytic activation modeling-based parcellation (MAMP)
- Common workflows (`nimare.workflows`)
    - Meta-analytic coactivation modeling (MACM)
    - Meta-analytic clustering analysis
    - Meta-analytic independent components analysis (metaICA)
