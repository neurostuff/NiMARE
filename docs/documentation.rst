.. include:: links.rst

Documentation
=============

NiMARE is a Python package for performing meta-analyses, and derivative analyses
using meta-analytic data, of the neuroimaging literature.
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

If you have questions, or need help with using NiMARE, check out `NeuroStars`_.

There are two broadly defined types of neuroimaging meta-analysis: coordinate-based
and image-based meta-analyses.

A coordinate-based meta-analysis uses a set of coordinates indicating measured BOLD
activation as its input.
`NeuroSynth`_ and `BrainMap`_, for example, perform coordinate-based meta-analyses
on peak activation coordinates.

Image-based meta-analyses, on the other hand, use full statistical maps as their input.
You could use a data repository like `NeuroVault`_ to perform an image based meta-analysis.

NiMARE implements tools for doing both types of meta-analyses.
It also has the capacity to pull data from databases.

In addition to meta-analyses, NiMARE plans to support a range of related methods
that use meta-analytic fMRI data, including automated annotation, functional decoding,
meta-analytic parcellation, meta-analytic clustering, and meta-analytic
coactivation modeling.

Databases
````````````

To conduct a meta-analysis you'll need a collection of neuroimaging data. NiMARE has the
ability to pull data from the following databases:

- `NeuroVault`_, which collects unthresholded statistical maps.

- `BrainMap`_, which collects peak activation coordinates through manual annotation.

- `NeuroSynth`_, which automatically extracts peak activation coordinates from neuroimaging articles online.

- `BrainSpell`_, which allows researchers to manually correct or add data taken from `NeuroSynth`_.

See `dataset extraction API`_ for usage.

.. _dataset extraction API: generated/nimare.dataset.extract.html#module-nimare.dataset.extract
