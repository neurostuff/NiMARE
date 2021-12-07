.. include:: links.rst

Coordinate-based meta-analysis in NiMARE
========================================

We have implemented a number of coordinate-based meta-analysis algorithms in NiMARE.
Here we discuss the elements of a NiMARE coordinate-based meta-analysis,
including (1) kernels, (2) estimators, (3) null methods, (4) multiple comparisons correction, and (5) Monte Carlo correction outputs.

In other tools for CBMA, such as `GingerALE`_, most of the elements we discuss on this page are combined into a single step.
However, we have chosen to modularize these elements in order to support a range of testing possibilities,
such as combining different kernels with different estimators.

First, let's describe, in basic terms, what each of these elements means.

All of the CBMA algorithms currently implemented in NiMARE are `kernel-based` methods.
In kernel-based CBMA, coordinates are convolved with some kind of kernel to produce a "modeled activation" map for each experiment in the dataset.
The modeled activation map acts as a substitute for the original, unthresholded statistical map from which the coordinates were derived.
The kernel used to create the modeled activation map varies across approaches, but the most common are
the :class:`ALEKernel<nimare.meta.kernel.ALEKernel>`, which convolves coordinates with a 3D Gaussian distribution,
and the :class:`MKDAKernel<nimare.meta.kernel.MKDAKernel>`, which creates a binary sphere around each coordinate.

.. warning::
    While the modeled activation map is an estimate of the original statistical map,
    that doesn't mean that modeled activation maps can actually be used as statistical maps.
    We still need meta-analytic algorithms that are designed for coordinates, rather than images.

Estimators refer to the core meta-analytic algorithm.
The Estimator classes take a kernel object as a parameter, and use that kernel to
(1) transform the coordinates into modeled activation maps,
(2) combine those modeled activation maps into a summary statistic map,
(3) derive a transformation from summary statistic to z-score, and
(4) estimate `uncorrected` significance of the summary statistics.

.. admonition:: Null methods

    In order to accomplish the third step, the Estimator relies on a "null method".
    The null method determines the statistical significance associated with each summary statistic value.
    There are two null methods currently implemented for all CBMA Estimators: "approximate" and "montecarlo".

    The approximate method builds a histogram-based null distribution of summary-statistic values,
    which can then be used to determine the associated p-value for `observed` summary-statistic values.
    The actual implementation of this method varies widely based on the Estimator.

    The montecarlo method uses a large number of permutation,
    within which the coordinates of the Dataset are randomly assigned and
    the distribution of summary statistics from the simulated Dataset is retained.
    Significance in this method is then determined by combining the distributions of summary statistics across all of the permutations,
    and then comparing the summary statistics from the real Dataset to these "null" statistics.
    This method may take a long time, and is only slightly more accurate than the approximate method,
    as long as there are enough iterations.

    In general, we would recommend using the approximate method.

Multiple comparisons correction.

The Monte Carlo FWE correction approach implemented in NiMARE produces three new versions of each of the ``logp`` and ``z`` maps:

-   ``<z|logp>_desc-mass_level-cluster_corr-FWE_method-montecarlo``:
    Cluster-level FWE-corrected map based on cluster mass.
    According to multiple studies, cluster mass-based inference is more powerful than cluster size-based inference,
    so we recommend this for most meta-analyses.
-   ``<z|logp>_desc-size_level-cluster_corr-FWE_method-montecarlo``:
    Cluster-level FWE-corrected map based on cluster size.
    This was previously simply called ``<z|logp>_level-cluster_corr-FWE_method-montecarlo``.
-   ``<z|logp>_level-voxel_corr-FWE_method-montecarlo``:
    Voxel-level FWE-corrected map.
    Voxel-level correction is generally more conservative than cluster-level correction,
    so it is only recommended for very large meta-analyses (i.e., hundreds of studies).
