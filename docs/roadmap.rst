.. include:: links.rst

Our Roadmap
===========

Ultimately, we plan to support all (or most) of the methods listed below in NiMARE:

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
- Automated annotation (`nimare.annotate`)
    - TF-IDF vectorization of text (`nimare.annotate.tfidf`)
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
