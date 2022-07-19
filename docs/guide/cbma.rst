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
    .. note::
        One conundrum meta-analysts often face is that papers may report a mixture of
        within-group analyses and between-group analyses.
        For example, if you are planning to perform a meta-analysis of the n-back task with the
        comparison between 3-back and 1-back conditions,
        you will likely find many papers that report both 3-back > control and 1-back > control results,
        as well as many papers that only report 3-back > 1-back results.

        In the case of the former, the standard approach is to perform a subtraction analysis.
        In the latter case, you would perform a single univariate meta-analysis.
        Unfortunately, you cannot combine the two sets of results.

    1. Create two Dataset objects.
    2. Perform a meta-analysis on each Dataset.
    3. Perform a subtraction analysis comparing the two Datasets.
    4. Perform a conjunction analysis assessing convergence between the two meta-analyses.
    5. Run :class:`~nimare.diagnostics.FociCounter` or :class:`~nimare.diagnostics.Jackknife` on all meta-analysis results.
    6. Run :func:`~nilearn.reporting.get_clusters_table` on all meta-analysis results.
    7. Plot significant results.

3. Large-scale analyses on a database.

    1. Download Neurosynth or NeuroQuery.
    2. ...
    3. Plot significant results.

Selecting studies for a meta-analysis
-------------------------------------

Organizing the dataset in NiMARE
--------------------------------
NiMARE contains several functions for converting common formats to Dataset objects.

Performing the meta-analysis
----------------------------

.. literalinclude:: ../../examples/02_meta-analyses/08_plot_cbma_subtraction_conjunction.py
    :start-at: from nimare.meta.cbma import ALE
    :end-at: knowledge_results = ale.fit(knowledge_dset)

.. figure:: ../auto_examples/02_meta-analyses/images/sphx_glr_06_plot_compare_ibma_and_cbma_001.png
    :target: ../auto_examples/02_meta-analyses/08_plot_cbma_subtraction_conjunction.ipynb
    :align: center
    :scale: 100

Multiple comparisons correction
-------------------------------

.. literalinclude:: ../../examples/02_meta-analyses/08_plot_cbma_subtraction_conjunction.py
    :start-at: from nimare.correct import FWECorrector
    :end-at: knowledge_corrected_results = corr.transform(knowledge_results)

.. figure:: ../auto_examples/02_meta-analyses/images/sphx_glr_06_plot_compare_ibma_and_cbma_002.png
    :target: ../auto_examples/02_meta-analyses/08_plot_cbma_subtraction_conjunction.ipynb
    :align: center
    :scale: 100

Saving the results
------------------

.. literalinclude:: ../../examples/02_meta-analyses/08_plot_cbma_subtraction_conjunction.py
    :start-at: knowledge_corrected_results.save_maps(
    :end-at: )

Performing additional followup analyses
---------------------------------------

.. literalinclude:: ../../examples/02_meta-analyses/08_plot_cbma_subtraction_conjunction.py
    :start-at: from nimare.diagnostics import Jackknife
    :end-at: knowledge_jackknife_table

References
----------
.. footbibliography::
