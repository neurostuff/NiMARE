.. _examples-datasets:

Working with studysets
----------------------

NiMARE's primary collection type is now :class:`~nimare.nimads.Studyset`.
Studysets can be used directly with estimators, workflows, and several transformers,
while the legacy :class:`~nimare.dataset.Dataset` class remains available for backwards compatibility
and migration workflows.

Additionally, NiMARE contains fetching and conversion tools for a number of meta-analytic resources,
including Neurosynth, NeuroQuery, NeuroVault, and, to a limited extent, BrainMap.
In the examples below, we show how to work with Studysets in NiMARE, along with
legacy Dataset-specific examples for interoperability and migration.
