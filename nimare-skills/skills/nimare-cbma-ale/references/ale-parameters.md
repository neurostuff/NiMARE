# ALE and CBMA parameters that change the result

NiMARE 0.22. "Default" is the constructor default; "recommended" is what the ALE
methods literature supports. Where they differ, the difference is the point.

## `ALE(...)`

| Parameter | Default | Recommended | Why |
|---|---|---|---|
| `kernel_transformer` | `ALEKernel()` | same, when sample sizes exist | FWHM per experiment from its N (Eickhoff et al., 2012). |
| `null_method` | `"approximate"` | `"approximate"` | Histogram convolution; documented as slightly less accurate than `"montecarlo"` but much faster. Governs the uncorrected p-map only. |
| `n_iters` | 5000 | 5000+ | Only used when `null_method="montecarlo"`. |
| `random_state` | `None` | set it | Added 0.22. Without it the Monte Carlo null and FWE correction are not reproducible. |
| `n_cores` | 1 | -1 | Monte Carlo only. |
| `memory` / `memory_level` | none | optional | joblib caching for repeated fits. |

## `ALEKernel(...)`

- `fwhm` and `sample_size` are mutually exclusive; passing both raises.
- Neither given: FWHM derived per experiment from the `sample_size` metadata.
- `fwhm=15`: the convention for database-derived sets with no sample sizes
  (Salimi-Khorshidi et al., 2009 found 15 mm gave the closest correspondence to
  image-based meta-analysis). `fwhm=10` also appears in the literature. State the value.
- `sample_size=N`: one constant N applied to every experiment. Rarely right.

`MKDAKernel(r=10)` / `KDAKernel(r=10)` are fixed-radius spheres for MKDA/KDA.

## `FWECorrector(...)`

| Parameter | Default | Recommended |
|---|---|---|
| `method` | **`"bonferroni"`** | **`"montecarlo"`** |
| `voxel_thresh` | 0.001 | 0.001 (cluster-forming, uncorrected p) |
| `n_iters` | varies by estimator | >= 5000 for publication |
| `n_cores` | 1 | -1 |

`method="predictive"` exists for predictive ALE. `FDRCorrector` is available but
Eickhoff et al. (2016) and Frahm et al. (2022) both found FDR inappropriate for ALE.

## Output map keys

A fitted ALE gives `stat`, `p`, `z`, `logp`. `FWECorrector(method="montecarlo")` adds,
for each of `z` and `logp`:

```
..._level-voxel_corr-FWE_method-montecarlo
..._desc-size_level-cluster_corr-FWE_method-montecarlo
..._desc-mass_level-cluster_corr-FWE_method-montecarlo
```

Report the cluster-mass map unless you have a reason not to. Print `sorted(cres.maps)`
rather than guessing.

## Diagnostics

```python
Jackknife(target_image=target, voxel_thresh=None)
FocusCounter(target_image=target, voxel_thresh=None)
ResampledStability(target_image=target, ...)
```

Pass `voxel_thresh=None` when the target is already a corrected cluster-level map.
Tables arrive as:

```
<target>_tab-clust
<target>_diag-Jackknife_tab-counts_tail-positive
<target>_diag-FocusCounter_tab-counts_tail-positive
```

Cluster tables take extents from the corrected map and peak statistics from the matching
uncorrected z map, including subpeaks labelled `1a`, `1b`, ...

## `MKDADensity`

Fixed-radius indicator maps per study, summed. Since 0.22 it takes sample size into
account in the weighting (matching the MKDA toolbox). `kernel__r` sets the radius.

## `SCALE(...)`

Needs `xyz`: an array of foci sampled from a reference database, defining the empirical
base rate. Convergence is then tested against *where the literature activates at all*,
which is the right null for MACM and for any analysis in a region with a high base rate.

## `CBMR` (`nimare.meta.cbmr`)

Spatial intensity as a GLM over spline bases, Poisson / negative binomial / clustered NB,
with study-level covariates and multiple groups. Inference via `build_contrast`,
`generate_hypotheses`, `evaluate_hypotheses`. Parametric voxelwise tests want >= ~200 foci
per group; below that use the parametric bootstrap. Covers group comparisons and covariate
effects (sample size, publication year) that ALE cannot express.

## Reproducibility

Record: NiMARE version, number of experiments **and** independent samples, estimator,
kernel (and FWHM rule), null method, corrector + method + voxel_thresh + n_iters,
`random_state`, and which map you report. `../scripts/run_ale.py` writes this to
`provenance.json`.
