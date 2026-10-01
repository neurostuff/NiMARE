---
name: nimare-macm
description: Meta-analytic coactivation modeling (MACM) with NiMARE — find the brain regions that coactivate with a seed ROI across the published task-fMRI literature. Use when the user has a seed region, cluster, or ROI and wants its coactivation network, mentions MACM or coactivation-based parcellation, or wants to characterize the functional network of a meta-analysis cluster using Neurosynth, NeuroQuery, NeuroStore, or BrainMap.
license: MIT
---

# Meta-analytic coactivation modeling

MACM = select every study in a database that reports a focus inside your seed, then run a
CBMA on that subset. Whatever else coactivates with the seed across tasks shows up as
convergence.

It is the standard follow-up to an ALE result ("what network is this cluster part of?")
and the engine behind coactivation-based parcellation.

## The pipeline

```python
from nimare.extract import fetch_neurostore
from nimare.meta.cbma import ALE
from nimare.correct import FWECorrector

studyset = fetch_neurostore(version="latest", data_dir="data/")

ids = studyset.get_studies_by_mask("seed_roi.nii.gz")
print(f"{len(ids)} studies report a focus in the seed")
sub = studyset.slice(ids=ids)

est = ALE(kernel_transformer=ALEKernel(fwhm=15))   # databases carry no sample sizes
result = est.fit(sub)
cres = FWECorrector(method="montecarlo", voxel_thresh=0.001,
                    n_iters=5000, n_cores=-1).transform(result)
```

Or the one-liner:
`macm_workflow(dataset_file, mask_file, output_dir=None, prefix=None, n_iters=5000, v_thr=0.001, n_cores=1)`
from `nimare.workflows.macm` — note it takes file *paths*, not objects.

Selection alternatives: `get_studies_by_coordinate(xyz, r=6)` for a spherical seed around
a peak (a 6-10 mm radius is the usual choice), `get_analyses_by_mask` /
`get_analyses_by_coordinate` for analysis-level selection.

## The three decisions that matter

### 1. Kernel width, because databases have no sample sizes

Neurosynth, NeuroQuery and NeuroStore rows do not record N, and ALE's kernel is derived
from N. You must pass a fixed FWHM. The literature convention is **15 mm**
(Salimi-Khorshidi et al., 2009, on correspondence with image-based meta-analysis); 10 mm
also appears. State the value and the reason.

```python
from nimare.meta.kernel import ALEKernel
ALE(kernel_transformer=ALEKernel(fwhm=15))
```

### 2. The null is not uniform — prefer SCALE or MKDAChi2

The studies selected by a seed are, by construction, studies that activate somewhere.
Against a *uniform* spatial null, ALE will find "coactivation" wherever the literature
activates often — the base-rate problem. Two principled fixes:

- **`SCALE`** tests convergence against the empirical base rate of foci in the reference
  database. Pass `xyz` drawn from the full database. Published MACM analyses use this
  precisely to avoid base-rate bias.
- **`MKDAChi2`** contrasts seed-reporting studies against the rest of the database, which
  is a specificity test by construction. This is Neurosynth's own coactivation method.

Plain ALE MACM is common and defensible if you say what the null is, but a frontal-
parietal "network" that looks like the standard task-positive map may be a base-rate
artifact.

```python
import numpy as np
from nimare.meta.cbma import SCALE
xyz = studyset.coordinates[["x", "y", "z"]].values      # database base rate
scale = SCALE(xyz=xyz, n_iters=5000, kernel_transformer=ALEKernel(fwhm=15))
```

```python
from nimare.meta.cbma import MKDAChi2
other = studyset.slice(ids=[i for i in studyset.ids if i not in set(ids)])
res = MKDAChi2().fit(sub, other)   # z_desc-association is the specificity map
```

### 3. The seed is inside its own result

Every selected study has a focus in the seed, so the seed region converges trivially.
Mask the seed out before interpreting, or state that it is excluded from interpretation.

## Checks

- **How many studies were selected?** Under ~20 the map is noise. Print the count and
  stop if it is small. A tight seed in a rarely-reported region may select 5 studies.
- **Is the seed in the database's space?** MNI152, matching the study set's `space`.
- **Did the seed come from the same data as a prior analysis?** Running MACM on clusters
  from your own ALE, over a database that contains those same studies, is circular for
  any claim about replication. It is fine as description.
- Database rows are articles with all foci pooled and activations/deactivations unlabelled
  — a "coactivation" may be an activation and a deactivation in the same paper.

## Coactivation-based parcellation

Run MACM for every voxel in a region, correlate the resulting unthresholded maps
pairwise, then cluster (k-means, hierarchical, spectral — scikit-learn). NiMARE supplies
the MACM step; the clustering and the cluster-number selection are yours. Published
examples run thousands of per-voxel MACMs, so budget compute and cache.

## Reporting

Database and version, how studies were selected (mask or coordinate+radius), how many
were selected, estimator and kernel FWHM with justification, null model, correction, and
whether the seed was masked out.

## References

- `../nimare-cbma-ale/references/ale-parameters.md`
- `../nimare-meta-analysis/references/statistical-guardrails.md` (items 6 and 7)
