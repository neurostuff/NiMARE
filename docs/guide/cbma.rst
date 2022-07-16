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

.. figure:: ../auto_examples/02_meta-analyses/images/sphx_glr_01_plot_cbma_001.png
    :target: ../auto_examples/02_meta-analyses/01_plot_cbma.ipynb
    :align: center
    :scale: 100
