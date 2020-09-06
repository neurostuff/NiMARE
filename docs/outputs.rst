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

   [value]_desc-[description]_level-[cluster|voxel]_corr-[FWE|FDR]_method-[method].nii.gz


First, the `value` represents type of data in the map (e.g., z-statistic, t-statistic).
Some of the values found in NiMARE include:

- z: Z-statistic
- t: T-statistic
- p: p-value
- logp: Negative base-ten logarithm of p-value
- chi2: Chi-squared value
- prob: Probability value
- of: Test value of MKDA and KDA methods
- ale: Test value of ALE and SCALE methods

Next, a series of key/value pairs describe the methods applied to generate the map.

- desc: Description of the data type. Only used when multiple maps with the same data type are produced by the same method.
- level: Level of multiple comparisons correction. Either cluster or voxel.
- corr: Type of multiple comparisons correction. Either FWE (familywise error rate) or FDR (false discovery rate).
- method: Name of the method used for multiple comparisons correction (e.g., "montecarlo" for a Monte Carlo procedure).
