.. _examples-datasets:

Working with datasets
---------------------

NiMARE stores meta-analytic data in its :class:`~nimare.dataset.Dataset` class.
Dataset objects may contain a range of elements, including coordinates (for coordinate-based meta-analysis),
links to statistical maps (for image-based meta-analysis), article text, label weights, and other metadata.

Additionally, NiMARE contains fetching and conversion tools for a number of meta-analytic resources,
including Neurosynth, NeuroQuery, NeuroVault, and, to a limited extent, BrainMap.
In the examples below, we show what a Dataset can do and exhibit tools for working with data from external meta-analytic resources.
