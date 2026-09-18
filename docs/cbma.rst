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
the :class:`~nimare.meta.kernel.ALEKernel`, which convolves coordinates with a 3D Gaussian distribution,
and the :class:`~nimare.meta.kernel.MKDAKernel`, which creates a binary sphere around each coordinate.

.. important::
    While the modeled activation map is an estimate of the original statistical map,
    that doesn't mean that modeled activation maps can actually be used as statistical maps.
    We still need meta-analytic algorithms that are designed for coordinates, rather than images.

Example: :ref:`metas_kernels`

Estimators
----------

Estimators refer to the core meta-analytic algorithm.
The Estimator classes take a kernel object as a parameter, and use that kernel to
(1) transform the coordinates into modeled activation maps,
(2) combine those modeled activation maps into a summary statistic map,
(3) derive a transformation from summary statistic to z-score, and
(4) estimate `uncorrected` significance of the summary statistics.

.. _study weights:

Study weights
`````````````

By default every Estimator weights each contrast equally, so a summary statistic map
reflects how many contrasts report activation at a voxel, not how much evidence each one
brings. :class:`~nimare.meta.cbma.mkda.MKDADensity` can instead weight each contrast by
the square root of its sample size, which is the weighting Multilevel Kernel Density
Analysis was defined with :footcite:p:`wager2007meta,wager2009evaluating`:

.. math::

    w_c \propto \delta_c \sqrt{N_c}

where :math:`N_c` is the contrast's sample size and :math:`\delta_c` optionally discounts
contrasts analysed with a fixed-effects study-level model.

.. code-block:: python

    from nimare.meta.cbma import MKDADensity

    # sqrt(N) weighting, read from the collection's sample_sizes metadata
    meta = MKDADensity(weighting="sample_size")

For anything beyond the default, pass a :class:`~nimare.meta.cbma.weights.StudyWeights`,
which also accepts explicit per-study weights so that study quality measures other than
sample size can be used:

.. code-block:: python

    from nimare.meta.cbma.weights import StudyWeights

    meta = MKDADensity(weighting=StudyWeights(inference_field="inference"))

The weights are normalised to sum to the number of contrasts, so an unweighted analysis is
exactly the case where all of them are 1.0, and the statistic stays on the same scale
either way. Dividing by the contrast count recovers the weighted *proportion* of
:footcite:t:`wager2009evaluating`. A contrast with a missing or non-positive sample size
raises; pass ``StudyWeights(on_missing="impute")`` to give it the mean weight instead.

.. note::
    The weighting applies to the density statistic. The chi-square tests of
    :class:`~nimare.meta.cbma.mkda.MKDAChi2` are unweighted, as they are in the original
    method.

Example: :ref:`metas_mkda_weighting`

.. _null methods:

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

.. tip::
    In general, we recommend using the ``approximate`` method.

Example: :ref:`null-method-example`

.. _multiple comparisons correction:

Multiple comparisons correction
-------------------------------

The initial Estimator fit (with the null method of choice) will produce a MetaResult with unthresholded, uncorrected statistical maps.
These statistical maps shouldn't be thresholded and interpreted on their own, as they don't account for the multiple comparisons issue.
To correct for multiple comparisons, we have Corrector classes
(:class:`~nimare.correct.FWECorrector` and :class:`~nimare.correct.FDRCorrector`).

These classes ingest MetaResults with uncorrected maps,
then use the Estimator and Dataset that the MetaResult references to perform multiple comparisons correction.
The correction approaches are first broken down into two types:
family-wise error rate correction (FWECorrector) and false discovery rate correction (FDRCorrector).

Additionally, each Corrector type accepts a "method" parameter,
which determines the specific approach used to correct the error rate of choice.
These methods can be broadly separated into two groups: generic methods and Estimator-specific methods.

Generic methods rely on internal implementations of common correction approaches to correct the results as an array,
without accounting for any of the idiosyncrasies of neuroimaging data, such as the smoothness of the data.
One example of a generic method is the "bonferroni" method for the FWECorrector.

.. tip::
    We do not recommend using the generic methods.

Estimator-specific methods are approaches that are implemented within the Estimator as class methods
that are then called by the Corrector.
These methods are generally designed specifically for neruoimaging, or event coordinate-based, data,
and are thus generally preferable to generic methods.
One such method is the Monte Carlo method (``method="montecarlo"``).

Example: :ref:`corrector-cbma-example`

The Monte Carlo multiple comparisons correction method
``````````````````````````````````````````````````````
:class:`~nimare.correct.FWECorrector`, :meth:`~nimare.meta.cbma.base.CBMAEstimator.correct_fwe_montecarlo`

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

.. important::

    Starting in version 0.0.13, clusters in the cluster-level corrected images are defined using
    faces connectivity (also known as 1st nearest-neighbor, NN1, or 6 neighbor connectivity),
    which counts voxels sharing a face as connected.
    This is more restrictive than other connectivity structures,
    including faces+edges (aka 2nd nearest-neighbor, NN2, or 18 neighbor connectivity)
    and faces+edges+corners (aka 3rd nearest-neighbor, NN3, or 26 neighbor connectivity).

    Prior to version 0.0.13, clusters were defined using faces+edges connectivity.

    Different tools use different connectivity structures.
    Nilearn uses faces connectivity, like NiMARE, while SPM uses faces+edges.
    FSL allows users to select one of the three connectivity structures, using the ``--connectivity`` parameter.
    Most AFNI programs also allow users to select a connectivity structure,
    though the actual parameter differs across programs.


.. admonition:: What about threshold-free cluster enhancement?

    TFCE :footcite:p:`smith2009threshold` is a voxel-level metric that combines signal magnitude and
    cluster extent to enhance the importance of clusters that are large, have high magnitude, or both.

    It can be applied to coordinate-based meta-analyses as an alternate metric to the
    maximum summary statistic (``level-voxel``), cluster mass (``desc-mass``), or cluster size (``desc-size``).
    However, recent work by Frahm et al. :footcite:p:`frahm2022evaluation` has indicated that the costs of performing
    TFCE-based inference (e.g., massively increased computation time) outweigh any observable benefits.
    As such, we have chosen not to implement TFCE-based correction within NiMARE,
    although there is a closed pull request with an implementation that worked at the time it was closed
    (see `#655 <https://github.com/neurostuff/NiMARE/pull/655>`_).


.. admonition:: Where is SDM?

    Seed Based *d* Mapping (SDM) is currently not implemented in NiMARE because the source code is not publicly available.
    To follow the current discussion on SDM in the context of NiMARE, see `#183 <https://github.com/neurostuff/NiMARE/issues/183>`_.

Reproducible results
--------------------

Monte Carlo null distributions and Monte Carlo FWE correction draw random permutations, so by
default they give slightly different numbers each time they are run.
To get the same numbers back, pass a ``random_state`` to the Estimator::

    from nimare.correct import FWECorrector
    from nimare.meta.cbma.ale import ALE

    meta = ALE(null_method="montecarlo", n_iters=10000, random_state=0)
    result = meta.fit(dset)

    corrector = FWECorrector(method="montecarlo", n_iters=10000)
    corrected = corrector.transform(result)

The seed covers every permutation the Estimator draws, including the ones run later by
``correct_fwe_montecarlo``, and each step draws from its own independent sequence,
so the uncorrected null and the FWE null are never built from the same permutations.
Re-running a seeded analysis --- in the same session or a later one --- reproduces the maps
exactly, and the seed is recorded in ``result.estimator.get_params()`` alongside the other
parameters.

``random_state`` is accepted by :class:`~nimare.meta.cbma.ale.ALE`,
:class:`~nimare.meta.cbma.ale.ALESubtraction`,
:class:`~nimare.meta.cbma.ale.BalancedALESubtraction`,
:class:`~nimare.meta.cbma.ale.SCALE`,
:class:`~nimare.meta.cbma.mkda.MKDADensity`,
:class:`~nimare.meta.cbma.mkda.MKDAChi2`, and
:class:`~nimare.meta.cbma.mkda.KDA`.
Among the image-based estimators, only :class:`~nimare.meta.ibma.PermutedOLS` is stochastic,
and it takes the same parameter (seeded with ``42`` by default).
Outside of meta-analysis, :class:`~nimare.annotate.lda.LDAModel` takes a ``random_state`` too,
:class:`~nimare.annotate.gclda.GCLDAModel` is fixed by its ``seed_init``, and
:func:`~nimare.generate.create_coordinate_dataset` by its ``seed``.

.. note::
    :class:`~nimare.meta.cbma.ale.ALESubtraction` and the label-permutation null of
    :class:`~nimare.meta.cbma.mkda.MKDAChi2` were already deterministic before
    ``random_state`` existed: each permutation was seeded with its own iteration index.
    Leaving ``random_state`` unset keeps that behaviour, so existing results do not change;
    setting it selects a different, equally reproducible set of permutations.

References
----------
.. footbibliography::
