# Why decoding correlation p-values are wrong, and what to do

## The problem

A whole-brain map has ~200,000 voxels, so correlating two maps gives ~200,000 degrees of
freedom and a p-value that is effectively zero for any r. NiMARE says so in
`CorrelationDecoder`'s own docstring:

> Coefficients from correlating two maps have very large degrees of freedom, so almost
> all results will be statistically significant. Do not attempt to evaluate results
> based on significance.

Voxels are not independent samples. Both maps are smooth, and smooth maps correlate with
each other at levels that look impressive against an i.i.d. null. The effective degrees
of freedom are closer to the number of independent spatial features — tens, not hundreds
of thousands.

Parcellating first (Schaefer-400, HCP-MMP) reduces the count but not the problem: parcel
values are still spatially autocorrelated.

## What to do instead

### Option A — report rankings, make no inferential claim

Perfectly respectable, and what most careful papers do. "The five highest-correlating
terms among the 123 tested were ..." with the coefficients shown. No p-values, no stars.

### Option B — spatial-autocorrelation-preserving null

Generate surrogate maps that keep the spatial autocorrelation of your map but destroy
its correspondence with the term map, then compare the observed r to that distribution.

- **Spin test** (Alexander-Bloch et al.): random rotations of the cortical sphere.
  Surface data only. `neuromaps.nulls.alexander_bloch`, `netneurotools`.
- **Variogram / Burt surrogates** (Burt et al., 2020): matches the empirical variogram.
  Works in volume and on parcellated data. `brainsmash`, `neuromaps.nulls.burt2020`.
- **Moran spectral randomization**: `brainspace`.

NiMARE does not implement these; use `neuromaps` or `brainsmash` alongside it. The
pattern in the literature (e.g. via the `JuSpyce` toolbox) is: parcellate both maps,
Spearman-correlate, build 10,000 autocorrelation-preserving surrogates, derive a p-value,
then FDR-correct across terms.

```python
# sketch; see neuromaps docs for the exact null for your space
from neuromaps.nulls import burt2020
from neuromaps.stats import compare_images

nulls = burt2020(my_map, atlas="MNI152", density="2mm", n_perm=10000, seed=0)
r, p = compare_images(my_map, term_map, nulls=nulls)
```

Then correct across terms (Benjamini-Hochberg) and report both.

### Option C — use a decoder that tests over studies

`NeurosynthDecoder` and `BrainMapDecoder` split the database into studies that do and do
not report activation in your ROI and run a chi-square test per term, with FDR
correction (`correction="bh"`). The unit is the study, not the voxel, so the test is
meaningful. The cost is that you must reduce your map to a binary ROI.

This is the right choice whenever you need to *claim* an association rather than
describe a ranking.

## Multiple comparisons across terms

Whatever null you use, you are testing hundreds of terms. `NeurosynthDecoder` and
`BrainMapDecoder` apply Benjamini-Hochberg internally. For correlation-based decoding you
must do it yourself, and only after you have a null that is not the i.i.d. one.

## Reporting

State: the database and version, the vocabulary and how it was restricted, the number of
terms tested, the decoder, the statistic, the null model (or that none was used), the
multiple-comparison correction, and whether the input map was thresholded.
