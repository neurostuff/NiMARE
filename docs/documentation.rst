.. include:: links.rst

Documentation
=============

NiMARE is a Python package for performing meta-analyses of the neuroimaging literature. 
While meta-analytic packages exist which implement one or two algorithms each, NiMARE 
provides a standard syntax for performing a wide range of analyses and for interacting 
with databases of coordinates and images from fMRI studies (e.g., brainspell, Neurosynth, 
and NeuroVault). 

NiMARE joins a growing Python ecosystem for neuroimaging research, which includes such 
tools as `Nipype`_, `Nistats`_, and `Nilearn`_. As with these other tools, NiMARE is open
source, collaboratively developed, and built with ease of use in mind.

This page has a brief description of each of the tools that are accessible through 
the NiMARE environment.

Overview
````````````

If you have questions, or need help with using NiMARE check out `Neuro Stars`_.

There are two broadly defined types of neuroimaging meta-analyses: coordinate-based 
meta-analyses and image-based meta-analyses. 

A coordinate-based meta-analysis uses a set of coordinates indicating measured BOLD 
activation as its input. `NeuroSynth`_, for example, performs coordinate based meta-analyses 
on peak activation coordinates.

Image-based meta-analyses, on the other hand, use full statistical maps as their input.
You could use a data repository like `NeuroVault`_ to perform an image based meta-analysis.

NiMARE implements tools for doing both types of meta-analyses. It also has the capacity
to pull data from databases.

Databases
````````````

To conduct a meta-analysis you'll need a collection of neuroimaging data. NiMARE has the
ability to pull data from the following databases:

`NeuroVault`_ collects unthresholded statistical maps.

`NeuroSynth`_ collects peak activiation coordinates.

Also it can access `Brain Spell`_ and pull abstracts from PubMed.

See `dataset extraction API`_ for usage.

.. _dataset extraction API: generated/nimare.dataset.extract.html#module-nimare.dataset.extract

Automated annotation
````````````````````````

A necessary step of meta-analysis is annotating data. This can be done manually. NiMARE
also offers a number of automated annotation options, some of which rely on community driven
annotation efforts such as `Cognitive Paradigm Ontology`_ and the `Cognitive Atlas`_. It also
implements several topic models and vector-based annotation models.

Coordinate-based meta-analysis
````````````````````````````````````

Activation Likelihood Estimate (ALE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ALE methods for meta-analysis takes in activation coordinates, then models them as probability
distributions in order to account for spatial uncertainties due to the between-subject and
between-template variability of neuroimaging data. To use NiMARE to run an ALE analysis you will
require a `Sleuth`_ file describing the dataset you want to analyze with ALE. 

See `Activation Likelihood Estimation meta-analysis revisited`_.

.. _Activation Likelihood Estimation meta-analysis revisited: https://www.doi.org/10.1016/j.neuroimage.2011.09.017

.. click:: cli:ale_sleuth_inference
	:prog: nimare ale

Specific Coactivation Likelihood Estimate (SCALE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Peaks2Maps
~~~~~~~~~~~

`peaks2maps`_ is a method for performing coordinate based meta analysis that uses a pretrained deep 
neural network to reconstruct unthresholded maps from peak coordinates. The reconstructed maps are 
evaluated for statistical significance using a permutation based approach with Family Wise Error 
multiple comparison correction.

.. click:: cli:peaks2maps
	:prog: nimare peaks2maps

Multilevel kernel density analysis (MKDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel density analysis (KDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian hierarchical cluster process model (BHICP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hierarchical Poisson/Gamma random field model (HPGRF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spatial Bayesian latent factor regression (SBLFR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spatial binary regression (SBR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image-based meta-analysis
````````````````````````````````````

Mixed effects general linear model (MFX-GLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Random effects general linear model (RFX-GLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Conperm** performs a metaanalysis of contrast maps using random effects and two-sided inference 
with empirical (permutation based) null distribution and Family Wise Error multiple comparison 
correction.

.. click:: cli:con_perm
	:prog: nimare conperm

Fixed effects general linear model (FFX-GLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stouffer's meta-analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Random effects Stouffer's meta-analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weighted Stouffer's meta-analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fisher's meta-analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functional characterization analysis
````````````````````````````````````

Decoding of continuous inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decoding of discrete inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encoding
~~~~~~~~~~~
