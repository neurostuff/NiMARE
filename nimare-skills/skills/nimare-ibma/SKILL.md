---
name: nimare-ibma
description: Image-based meta-analysis with NiMARE — pool unthresholded statistical maps across studies or sites instead of peak coordinates. Use when the user has whole-brain z, t, beta or varcope maps from several studies, is working with NeuroVault collections, is aggregating site-wise summary statistics in a multi-site or mega-analysis design, or mentions Stouffer's, Fisher's, DerSimonian-Laird, Hedges, IBMA, or random-effects meta-analysis of brain maps.
license: MIT
---

# Image-based meta-analysis

IBMA pools whole statistical maps, so it keeps effect magnitudes and sign and is far more
powerful than coordinate-based meta-analysis. The constraint is data availability: most
published studies still share only tables of peaks.

NiMARE's IBMA is also the part of the library with the most careful handling of
statistical dependence — use it.

## Pick the estimator from what you have

| You have | Estimator | Model |
|---|---|---|
| z maps only | `Stouffers(use_sample_size=False, two_sided=True)` | Fixed-effect combination of z's |
| z maps only, want p-combination | `Fishers(two_sided=True)` | Most sensitive, least interpretable |
| beta + varcope maps | `DerSimonianLaird()` | Random effects, method-of-moments tau² |
| beta + varcope maps | `Hedges()` | Random effects, Hedges estimator |
| beta + varcope maps | `WeightedLeastSquares(tau2=0)` | tau² supplied, not estimated |
| beta + varcope maps | `VarianceBasedLikelihood(method="ml"\|"reml")` | Likelihood-based |
| beta maps + sample sizes | `SampleSizeBasedLikelihood(method="ml"\|"reml")` | When variances are unavailable |
| t maps + sample sizes | `FixedEffectsHedges(tau2=0)` | Fixed effects |
| beta maps, want nonparametric | `PermutedOLS(n_jobs=-1, random_state=42)` | Max-statistic FWE by permutation |

`_required_inputs` on each class is authoritative; a missing image type means studies get
silently dropped (`drop_invalid=True`) or the fit fails.

**Combination test vs. random effects.** `Stouffers`/`Fishers` answer "is there *some*
signal here across these studies". `DerSimonianLaird`/`Hedges` estimate a population
effect and generalize beyond the sampled studies. If you have variance maps and more than
a handful of studies, use random effects and say which tau² estimator.

## The dependence parameter — NiMARE's best feature, and easy to miss

```python
from nimare.meta.ibma import DerSimonianLaird
est = DerSimonianLaird(
    groupby=None,              # default: group images by study_id
    weight_scheme="rescale",   # correlated-effects weighting (Hedges et al., 2010)
    rho=0.8,                   # assumed within-group correlation
    small_sample_correction=None,
)
```

- `groupby=None` (default) groups images by `study_id` — right when a paper's several
  maps come from one sample.
- `groupby="<metadata field>"` when a paper contributes genuinely independent samples
  (patients and controls). Pass the field that identifies the sample.
- `groupby=<array>` for one explicit label per image.
- `groupby=False` treats every image as independent and **logs a warning that this
  inflates significance**. Only correct when it is true.

Whenever groups are present, PyMARE switches to **CR2 cluster-robust standard errors**
with Satterthwaite degrees of freedom. That inference is asymptotic in the number of
*groups*, not images: PyMARE warns at <=10 groups and when the Satterthwaite df drop
below about 4. Both are common. Heed the warnings — they mean the p-values are
anti-conservative, not that the run failed.

`weight_scheme`: `"rescale"` (default, divides each image's weight by its group size),
`"individual"` (no adjustment), `"collapse"` (one row per group).
`small_sample_correction="knapp-hartung"` is worth considering with few studies;
it is ignored once group labels are present, since CR2 is the correction then.

Note the contrast with CBMA: **no coordinate-based estimator has any of this.** There,
contrasts from the same sample are counted as independent, silently.

## Masking

By default, IBMA estimators run in "bags" of voxels — each bag is the set of voxels valid
across the same subset of studies — so a voxel is dropped only from the studies missing
it. `aggressive_mask=True` instead removes any voxel that is zero or NaN in *any* input
map. The default preserves coverage; the aggressive option gives one common support.
A bag whose valid images all belong to a single group is skipped and returns NaN.

## Pipeline

```python
from nimare.meta.ibma import DerSimonianLaird
from nimare.correct import FDRCorrector

est = DerSimonianLaird()
res = est.fit(studyset)
cres = FDRCorrector(method="indep", alpha=0.05).transform(res)
```

Or with diagnostics and an HTML report:

```python
from nimare.workflows import IBMAWorkflow
cres = IBMAWorkflow(estimator="stouffers", corrector="fdr",
                    diagnostics="jackknife").fit(studyset)
```

`PermutedOLS` supports `FWECorrector(method="montecarlo")`; the parametric estimators
generally pair with FDR or Bonferroni.

## NeuroVault

NeuroVault holds >238,000 maps but only ~1,473 collections link to a publication, and
mis-annotated, duplicated, thresholded, inverted and non-statistical images are common.
Selection is the analysis. See `references/neurovault-qc.md` for the filter chain:
group-level fMRI-BOLD T or Z maps, N > 10, unthresholded, MNI, brain coverage > 40%,
then heuristic removal of extreme-Z, duplicate, inverted and keyword-flagged
("ICA", "PCA", "correlation") images.

```python
from nimare.generate import create_neurovault_studyset
studyset = create_neurovault_studyset(
    collection_ids={"study1": 1234}, contrasts={"faces": "as-Face"}, img_dir="imgs/")
```

## Transforms

`nimare.transforms` converts between what you have and what the estimator wants:
`t_to_z`, `p_to_z`, `z_to_t`, `t_to_d`, `d_to_g`, `sd_to_varcope`, `se_to_varcope`,
`t_and_varcope_to_beta`, `samplevar_dataset_to_varcope`, `ImageTransformer`,
`transform_images`. `ImagesToCoordinates` goes the other way, extracting peaks from maps
so an IBMA dataset can also be analysed as CBMA.

Converting t to z needs correct degrees of freedom (`sample_sizes_to_dof`); getting this
wrong silently rescales every study's contribution.

## Multi-site / mega-analysis

Fit the same model at each site, then pool the site-wise summary statistics with
`Stouffers` (optionally sample-size weighted) or `Fishers`. This is the IBMMA pattern.
When the pipeline is identical across sites, the usual harmonization steps (smoothing,
design scaling, normalization) can be skipped — but resampling to a common grid cannot.

## Reporting

Estimator and why, `groupby` and the dependence model, `weight_scheme` and `rho`,
masking mode, number of studies and of groups, any PyMARE small-sample warning you
received, correction, and the image selection criteria if the maps came from NeuroVault.

## References

- `references/neurovault-qc.md`
- `../nimare-meta-analysis/references/statistical-guardrails.md`
