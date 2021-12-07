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

Kernels
-------

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

Estimators
----------

Estimators refer to the core meta-analytic algorithm.
The Estimator classes take a kernel object as a parameter, and use that kernel to
(1) transform the coordinates into modeled activation maps,
(2) combine those modeled activation maps into a summary statistic map,
(3) derive a transformation from summary statistic to z-score, and
(4) estimate `uncorrected` significance of the summary statistics.

Null methods
````````````

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

Multiple comparisons correction
-------------------------------

The initial Estimator fit (with the null method of choice) will produce a MetaResult with unthresholded, uncorrected statistical maps.
These statistical maps shouldn't be thresholded and interpreted on their own, as they don't account for the multiple comparisons issue.
To correct for multiple comparisons, we have Corrector classes
(:class:`FWECorrector<nimare.correct.FWECorrector>` and :class:`FDRCorrector<nimare.correct.FDRCorrector>`).

These classes ingest MetaResults with uncorrected maps,
then use the Estimator and Dataset that the MetaResult references to perform multiple comparisons correction.
The correction approaches are first broken down into two types:
family-wise error rate correction (FWECorrector) and false discovery rate correction (FDRCorrector).

Additionally, each Corrector type accepts a "method" parameter,
which determines the specific approach used to correct the error rate of choice.
These methods can be broadly separated into two groups: generic methods and Estimator-specific methods.

Generic methods rely on tools like ``statsmodels`` to correct the results as an array,
without accounting for any of the idiosyncrasies of neuroimaging data (e.g., autocorrelation).
One example of a generic method is the "bonferroni" method for the FWECorrector.
**We do not recommend using these methods.**

Estimator-specific methods are approaches that are implemented within the Estimator as class methods
that are then called by the Corrector.
These methods are generally designed specifically for neruoimaging, or event coordinate-based, data,
and are thus generally preferable to generic methods.
One such method is the Monte Carlo method (``method="montecarlo"``).

The Monte Carlo multiple comparisons correction method
``````````````````````````````````````````````````````
:class:`nimare.correct.FWECorrector`, :meth:`nimare.meta.cbma.base.CBMAEstimator.correct_fwe_montecarlo`

For our CBMA algorithms, we strongly recommend using the "montecarlo" method with the FWECorrector.
This is the primary Estimator-specific method, which operates by creating simulated versions of the Dataset,
in which the coordinates are replaced with ones that are randomly drawn from the Estimator's mask image.
A summary statistic map is then calculated for each simulated Dataset, from which relevant information
(e.g., maximum statistic value) is extracted.
This is repeated many times (e.g., 10000x) in order to build null distributions of the relevant measures.

The Monte Carlo FWE correction approach implemented in NiMARE produces three new versions of each of the ``logp`` and ``z`` maps:

-   ``<z|logp>_desc-mass_level-cluster_corr-FWE_method-montecarlo``:
    Cluster-level FWE-corrected map based on cluster mass.
    Cluster mass refers to the sum of the summary statistic values across all voxels in the cluster,
    so in this method the maximum cluster mass is retained from each Monte Carlo permutation and
    used to generate a null distribution.
    Clusters from the meta-analytic map (after a cluster-defining threshold is applied)
    are then assigned significance values based on where each cluster's mass lands on this null distribution.
    **According to multiple studies, cluster mass-based inference is more powerful than cluster size-based inference,
    so we recommend this for most meta-analyses.**
-   ``<z|logp>_desc-size_level-cluster_corr-FWE_method-montecarlo``:
    Cluster-level FWE-corrected map based on cluster size.
    Cluster size refers to the number of voxels in the cluster,
    so in this method the maximum cluster size is retained from each Monte Carlo permutation and
    used to generate a null distribution.
    Clusters from the meta-analytic map (after a cluster-defining threshold is applied)
    are then assigned significance values based on where each cluster's size lands on this null distribution.
    This was previously simply called ``<z|logp>_level-cluster_corr-FWE_method-montecarlo``.
-   ``<z|logp>_level-voxel_corr-FWE_method-montecarlo``:
    Voxel-level FWE-corrected map.
    In this method, the maximum summary statistic value is retained from each Monte Carlo permutation and
    used to generate a null distribution.
    All voxels in the meta-analytic map are then assigned a corrected significance value based on where
    the voxel's summary statistic value lands on this null distribution.
    **Voxel-level correction is generally more conservative than cluster-level correction,
    so it is only recommended for very large meta-analyses (i.e., hundreds of studies).**
