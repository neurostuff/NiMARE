.. include:: links.rst

A Proposed Meta-Analytic Ecosystem
==================================

.. image:: _static/ecosystem.png
  :width: 100 %

NiMARE aims to fill a gap in the meta-analytic ecosystem currently (partially)
filled by Neurosynth.

Neurosynth 2.0
--------------
`Neurosynth`_ currently stores a coordinated-based database of over 14,000
neuroimaging papers (automatically curated by `ACE`_), provides a web interface
for automated meta-analyses, functional decoding, and gene expression
visualization, and provides a Python package implementing the above methods.

In order to improve modularization, the next iteration of Neurosynth will limit
itself to the web interface, providing a centralized storage for large-scale
automated meta-analyses, but not actually implementing the algorithms used
to run those meta-analyses or to perform the other services provided on the
website (e.g., functional decoding and topic modeling).
The algorithms currently implemented in the Neurosynth Python package will be
implemented (among many others) in NiMARE.
The database currently stored by Neurosynth will instead by stored in the
NeuroStuff database, which will also store other coordinate- and image-based
meta-analytic databases.

NeuroVault
----------

`NeuroVault`_ is a database for unthresholded images.
Users may upload individual maps or `NIDM Results`_, which can be exported from
a number of fMRI analysis tools, like `AfNI`_, `SPM`_, `FSL`_, and
`NeuroScout`_.

NeuroVault also has integrations with `NeuroPower`_ (for power analyses) and
`Neurosynth`_ (for functional decoding), and supports simple meta-analyses.

brainspell
----------
`brainspell`_ is a clone of the Neurosynth database

metaCurious
-----------
`metaCurious`_ is a new frontend (i.e., website) for brainspell, oriented toward
meta-analysts.
MetaCurious provides search and curation tools for researchers to build
meta-analytic samples for analysis.
Search criteria, reasons for exclusion, and other labels may be added by the
researcher and fed back into the underlying database, resulting in
goal-oriented manual annotation.
MetaCurious generates GitHub repositories for meta-analytic samples, which
will also be NiMARE-compatible in the future.

NIMADS
------
NIMADS is a new standard for organizing and representing meta-analytic
neuroimaging data.
NIMADS will be used by NeuroStuff, pyNIMADS, metaCurious, and NiMARE.

NeuroStuff
----------
NeuroStuff will act as a centralized repository for coordinates and maps from
neuroimaging studies, stored in NIMADS format.
Users will be able to query and add to the repository using its API and the
pyNIMADS Python package.

pyNIMADS
--------

NiMARE
------
