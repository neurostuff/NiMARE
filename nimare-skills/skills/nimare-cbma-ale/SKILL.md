---
name: nimare-cbma-ale
description: Run a coordinate-based meta-analysis in NiMARE — ALE, MKDA density, KDA, SCALE, or CBMR — over published activation peaks. Use when the user has a curated set of studies with peak coordinates and wants to know where findings converge, mentions ALE or activation likelihood estimation or GingerALE or Sleuth, or needs kernel choice, null method, multiple-comparison correction, cluster thresholds, power, or convergence diagnostics for a coordinate meta-analysis.
license: MIT
---

# Coordinate-based meta-analysis with NiMARE

ALE asks: do the reported peaks converge in space more than chance? That is a question
about the *literature*, not about the brain, and the answer is only as good as the study
set. Curate first (`nimare-dataset-curation`), then fit.

## Before fitting

1. **Audit independence.** One row per independent subject group, not per contrast.
   `python ../nimare-dataset-curation/scripts/audit_independence.py foci.txt`
2. **Count experiments.** ~17-20 minimum for adequately powered ALE (Müller et al., 2018).
   NiMARE enforces no floor and will return a map from 4 experiments without comment.
3. **Check space and sample sizes.** ALE's kernel width comes from `sample_size`.
4. **Filter out-of-mask foci.** `nimare.diagnostics.FocusFilter`.

## The canonical pipeline

```python
from nimare.io import convert_sleuth_to_studyset
from nimare.meta.cbma import ALE
from nimare.correct import FWECorrector
from nimare.diagnostics import Jackknife, FocusCounter

studyset = convert_sleuth_to_studyset("foci.txt")

est = ALE(null_method="approximate", random_state=42)
result = est.fit(studyset)

corr = FWECorrector(method="montecarlo", voxel_thresh=0.001,
                    n_iters=5000, n_cores=-1)
cres = corr.transform(result)

target = "z_desc-mass_level-cluster_corr-FWE_method-montecarlo"
# siblings: z_desc-size_level-cluster_corr-FWE_method-montecarlo,
#           z_level-voxel_corr-FWE_method-montecarlo
cres = Jackknife(target_image=target, voxel_thresh=None).transform(cres)
cres = FocusCounter(target_image=target, voxel_thresh=None).transform(cres)

cres.save_maps(output_dir="results/")
cres.save_tables(output_dir="results/")
```

`scripts/run_ale.py` is this pipeline with argument parsing, the pre-flight checks, and a
report. Prefer it over retyping.

Or the whole thing as a workflow with an HTML report:

```python
from nimare.workflows import CBMAWorkflow
cres = CBMAWorkflow(estimator="ale", corrector="montecarlo",
                    diagnostics=["jackknife", "focuscounter"], n_cores=-1).fit(studyset)
```

## Decisions that change the answer

### Correction — the default is wrong for ALE

`FWECorrector(method=...)` defaults to `"bonferroni"`. Pass `"montecarlo"`.
Cluster-forming threshold p < 0.001 uncorrected, cluster p_FWE < 0.05, `n_iters >= 5000`.
Uncorrected and FDR thresholds were shown to be invalid for ALE (Eickhoff et al., 2016;
Frahm et al., 2022). Cluster-*mass* (`z_desc-mass_level-cluster`) is generally preferred
to cluster-*size*.

### Kernel

`ALEKernel` derives FWHM per experiment from its sample size (Eickhoff et al., 2012) —
the default and the right choice for a curated set. Override only when sample sizes are
unavailable:

```python
from nimare.meta.kernel import ALEKernel
ALE(kernel_transformer=ALEKernel(fwhm=15))   # database-derived sets; state the value
```

`fwhm` and `sample_size` are mutually exclusive. `MKDAKernel(r=10)` and `KDAKernel(r=10)`
use fixed spheres instead.

### Null method

`null_method="approximate"` (default) builds the null by histogram convolution — fast,
documented as slightly less accurate. `"montecarlo"` with `n_iters` is slower and slightly
more accurate. This governs the *uncorrected* p-map only; it is not the FWE correction.

### Estimator

| | Use when |
|---|---|
| `ALE` | Default. Sample-size-adaptive kernel, convergence against a uniform spatial null. |
| `MKDADensity` | Fixed-radius indicator maps; weights by sample size since 0.22. |
| `KDA` | Raw focus density. Rarely the right primary analysis. |
| `SCALE` | Convergence above the empirical base rate of activation in a reference database. Use for MACM, and wherever a uniform null is implausible. Pass `xyz` from a database. |
| `CBMR` (`nimare.meta.cbmr`) | Study-level covariates, >2 groups, regression framing. Parametric test wants >= ~200 foci per group; otherwise parametric bootstrap. |

## Diagnostics are not optional

A cluster driven by two studies is a finding about two studies.

- `Jackknife` — leave-one-experiment-out contribution per cluster. Note it leaves out one
  *analysis*; if a study contributed several, it stays partly in. Pool first.
- `FocusCounter` — how many foci each experiment put in each cluster.
- `ResampledStability` — repeated subsampling; more informative than a single jackknife
  for larger sets.

Tables land in `cres.tables["<target_image>_diag-<Jackknife|FocusCounter>_tab-counts_tail-positive"]`,
alongside the cluster table `cres.tables["<target_image>_tab-clust"]`. Print `sorted(cres.tables)`
rather than guessing — the suffixes are long and exact.

## Reading the output

`cres.maps` keys follow a BIDS-like scheme. A fitted ALE gives `stat`, `p`, `z`, `logp`;
`FWECorrector(method="montecarlo")` adds, for each of `z` and `logp`:

- `..._level-voxel_corr-FWE_method-montecarlo`
- `..._desc-size_level-cluster_corr-FWE_method-montecarlo`
- `..._desc-mass_level-cluster_corr-FWE_method-montecarlo`   <- usually the one to report

Retrieve with `cres.get_map(key, return_type="image"|"array")`.

After cluster-level FWE every voxel in a surviving cluster carries that cluster's
p-value, so the corrected map itself has no peaks. NiMARE handles this: the
`..._tab-clust` table takes cluster extents from the corrected map but reads peak
coordinates and `Peak Stat` from the corresponding *uncorrected* z map, and lists
subpeaks (`1a`, `1b`, ...). Those are genuine peaks — but say which map the statistic
came from. Some published NiMARE analyses report cluster centres of mass instead; do not
mix the two conventions in one table.

## Common mistakes

- Reporting uncorrected or FDR results as the primary ALE finding.
- Running ALE on Neurosynth/NeuroQuery output and calling it a systematic review: those
  rows are articles with pooled foci, no sample sizes, no contrast selection.
- Entering the same sample's contrasts as separate experiments (see guardrail 1).
- Meta-analyzing a set assembled by searching for an *effect* rather than a *paradigm* —
  that builds the convergence in by construction.
- Interpreting the absence of convergence as the absence of an effect.

## References

- `references/ale-parameters.md` — every parameter that matters, with defaults and
  recommended values.
- `scripts/run_ale.py` — tested end-to-end pipeline.
- `../nimare-meta-analysis/references/statistical-guardrails.md`
