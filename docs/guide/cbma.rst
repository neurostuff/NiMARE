Coordinate-Based Meta-Analysis
==============================

Coordinate-based meta-analysis (CBMA) is a popular tool for measuring consistency across neuroimaging studies.
While :ref:`ibma.rst` leverage more information than CBMA, and is thus superior,
CBMA is much more popular because the majority of fMRI papers report peaks from significant clusters in tables,
rather than uploading unthresholded statistical maps to a database, like NeuroVault.

.. note::
    This page walks through coordinate-based meta-analysis (CBMA) in NiMARE from a practical perspective.
    The focus is on performing basic CBMAs.
    For a more detailed description of the classes and functions used for CBMA, see
    :ref:`sphx_glr_auto_examples_02_meta-analyses_03_plot_kernel_transformers.py`,
    :ref:`sphx_glr_auto_examples_02_meta-analyses_01_plot_cbma.py`, and
    :ref:`sphx_glr_auto_examples_02_meta-analyses_05_plot_correctors.py`.

Types of CBMA studies
---------------------
1. One large dataset, with multiple subsets.

    1. Create a single, large Dataset with annotations indicating the subsets.
    2. Perform a meta-analysis on the full Dataset.

        - This omnibus analysis is typically interpreted as evaluating convergent results across the subsets.

    3. Slice the Dataset into different subsets.
    4. Perform a meta-analysis on each subset.
    5. Perform a subtraction analysis between each subset and the rest of the subsets (combined in one Dataset).
    6. Perform functional decoding comparing something and something.
    7. Plot significant results.

2. A direct comparison between two datasets.

    1. Create two Dataset objects.
    2. Perform a meta-analysis on each Dataset.
    3. Perform a subtraction analysis comparing the two Datasets.
    4. Perform a conjunction analysis assessing convergence between the two meta-analyses.
    5. Run FociCounter on all meta-analysis results.
    6. Plot significant results.

3. Large-scale analyses on a database.

    1. Download Neurosynth or NeuroQuery.
    2. ...
    3. Plot significant results.

Selecting studies for a meta-analysis
-------------------------------------

Organizing the dataset in NiMARE
--------------------------------

Performing the meta-analysis
----------------------------

.. figure:: ../auto_examples/02_meta-analyses/images/sphx_glr_01_plot_cbma_001.png
    :target: ../auto_examples/02_meta-analyses/01_plot_cbma.ipynb
    :align: center
    :scale: 100

Multiple comparisons correction
-------------------------------

Saving the results
------------------

Performing additional followup analyses
---------------------------------------

References
----------
.. footbibliography::
