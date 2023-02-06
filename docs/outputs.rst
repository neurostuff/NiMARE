.. include:: links.rst

Outputs of NiMARE
======================

NiMARE includes a wide range of tools with a correspondingly large number of possible outputs.
Here we outline the rules we apply to NiMARE outputs.

File names
----------

NiMARE-generated files, especially ones made by meta-analyses, follow a naming convention somewhat based on BIDS.

Here is the basic naming convention for statistical maps:

.. code-block:: Text

   <value>[_desc-<label>][_level-<cluster|voxel>][_corr-<FWE|FDR>][_method-<label>].nii.gz


First, the ``value`` represents type of data in the map (e.g., z-statistic, t-statistic).
Some of the values found in NiMARE include:

- ``z``: Z-statistic
- ``t``: T-statistic
- ``p``: p-value
- ``logp``: Negative base-ten logarithm of p-value
- ``chi2``: Chi-squared value
- ``prob``: Probability value
- ``stat``: Test value of meta-analytic algorithm (e.g., ALE values for ALE, OF values for MKDA)
- ``est``: Parameter estimate (IBMA only)
- ``se``: Standard error of the parameter estimate (IBMA only)
- ``tau2``: Estimated between-study variance (IBMA only)
- ``sigma2``: Estimated within-study variance (IBMA only)

.. note::
    For one-sided tests, p-values > 0.5 will have negative z-statistics. These values should not 
    be confused with significant negative results. As a result, in NiMARE, these values are 
    replaced by 0.

Next, a series of key/value pairs describe the methods applied to generate the map.

- ``desc``: Description of the data type. Only used when multiple maps with the same data type are produced by the same method.
- ``level``: Level of multiple comparisons correction. Either ``cluster`` or ``voxel``.
- ``corr``: Type of multiple comparisons correction. Either ``FWE`` (familywise error rate) or ``FDR`` (false discovery rate).
- ``method``: Name of the method used for multiple comparisons correction (e.g., "montecarlo" for a Monte Carlo procedure).

File contents
-------------

NiMARE outputs unthresholded statistical maps.
Users may then threshold their results separately.

This may result in some confusion for cluster-level corrected maps,
in which each _cluster_ (after applying a voxel-wise cluster-defining threshold) has an overall significance level.
As such, cluster-level corrected maps contain zeros for all non-significant voxels after applying the cluster-defining threshold,
and each cluster has a single value across all voxels in the cluster.
All clusters surviving the cluster-defining threshold will be included in the map, including clusters that have very high p-values.
