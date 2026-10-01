# Statistical guardrails for NiMARE analyses

Each item states the default NiMARE behaviour, why it is a hazard, and what to do.
Behaviours marked **[verified]** were checked against NiMARE 0.22 source and by running it.

---

## 1. The unit of analysis is the contrast, not the study, not the sample **[verified]**

`Dataset` builds its ids as `f"{study_id}-{contrast_id}"` (`nimare/dataset.py`), and every
CBMA estimator consumes `inputs_["id"]`. Internal variables are named `study_ids` but hold
contrast-level ids — do not be misled by the naming. `Studyset` behaves the same way via
`analysis_full_key`.

Consequences:

- A paper reporting "faces > houses" and "faces > scrambled" from the same 20 subjects
  contributes **two experiments**, each weighted as if it were an independent sample.
- ALE's kernel width is set per experiment from its sample size, so the same 20 subjects
  also donate their precision twice.
- `Jackknife` leaves out one *analysis*, so a leave-one-out that is meant to be
  leave-one-*study*-out will still leave the study's other contrasts in.

No warning is emitted in any of these cases.

Measured effect (NiMARE's bundled `semantic_knowledge_children.txt`, 21 experiments,
`ALE(null_method="approximate")`):

| dataset | units | max z | voxels z>3.1 | voxels z>5 |
|---|---|---|---|---|
| as curated (one contrast per sample) | 21 | 6.76 | 1,587 | 87 |
| each sample entered as two contrasts | 42 | 7.31 | 3,960 | 420 |
| the same 42, after `Studyset.combine_analyses()` | 21 | 6.40 | 1,843 | 86 |

Remedy: one row per independent subject group. Pool the foci of same-sample contrasts into
a single analysis before fitting. `Studyset.combine_analyses()` does this per *study*;
it is correct only when every analysis in a study comes from the same sample. See
`nimare-dataset-curation/references/contrast-independence.md`.

Note that pooling foci and taking the per-experiment maximum is also what the revised ALE
algorithm (Turkeltaub et al., 2012) prescribes for within-experiment foci, so the pooled
dataset is the one the algorithm was designed for.

## 2. No minimum experiment count is enforced **[verified]**

`nimare/diagnostics.py` states plainly: "CBMA estimators declare none [no `_min_analyses`],
which turns off every check that consults this." `ALE().fit()` on 5 experiments returns a
map without comment.

Remedy: count experiments first. Müller et al. (2018) recommend ~17-20 experiments for
adequately powered ALE; Neurosynth Compose points users at the same figure. Below ~17,
report the count and the power limitation prominently or decline the convergence claim.
A single-cluster result from 12 experiments is usually one or two studies.

## 3. Correction defaults are not the recommended settings **[verified]**

`FWECorrector.__init__(self, method="bonferroni", ...)`. Bonferroni over ~200k voxels is
not what the ALE literature validates.

- ALE: cluster-level FWE by Monte Carlo, cluster-forming threshold p < 0.001 uncorrected,
  cluster p_FWE < 0.05, `n_iters >= 5000`. Eickhoff et al. (2016) and Frahm et al. (2022)
  both found uncorrected and FDR thresholding inappropriate for ALE.
- `CBMAWorkflow` passes `_mcc_method = "montecarlo"`, so the workflow route is safe; the
  hand-rolled route is not.

```python
from nimare.correct import FWECorrector
corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=5000, n_cores=-1)
cres = corr.transform(result)
```

## 4. `null_method="approximate"` vs `"montecarlo"`

`ALE` defaults to `"approximate"` (histogram convolution, Eickhoff et al. 2012) — fast and
documented as "slightly less accurate". It is fine for the uncorrected p-value map. It is
not a substitute for the Monte Carlo *correction* in item 3, which is a separate step.

## 5. Decoding correlations are rankings, not tests

NiMARE's `CorrelationDecoder` carries its own warning: "Coefficients from correlating two
maps have very large degrees of freedom, so almost all results will be statistically
significant. Do not attempt to evaluate results based on significance." Voxels are
spatially autocorrelated, so the nominal df is wrong by orders of magnitude.

Remedy: present the top-k ranked terms as a descriptive annotation, or test against a
spatial-autocorrelation-preserving null (spin test / variogram surrogates via `neuromaps`
or `brainsmash`). Of the surveyed articles, only a handful did the latter.

## 6. Neurosynth and NeuroQuery carry no sample sizes

ALE's kernel FWHM is derived from each experiment's sample size, which these databases do
not record. Using ALE on a database-derived study set therefore needs an explicit fixed
kernel. The convention in the literature is `ALEKernel(fwhm=15)`, on the grounds that
15 mm gave the best correspondence with image-based meta-analysis
(Salimi-Khorshidi et al., 2009); 10 mm also appears. State which you used.

```python
from nimare.meta.kernel import ALEKernel
from nimare.meta.cbma import ALE
est = ALE(kernel_transformer=ALEKernel(fwhm=15))
```

Database-derived study sets also mix activations and deactivations without labelling them,
and one "study" may pool several contrasts — i.e. item 1 applies and cannot be fixed.
That is a limitation to report, not a bug to work around.

## 7. Subtraction analysis assumes exchangeability and equal group sizes

`ALESubtraction` builds its null by permuting group labels. That requires the two sets to
be exchangeable, so **the two study sets must be disjoint** — comparing an "all studies"
set against a subset of itself violates the assumption (an error flagged explicitly in one
surveyed paper). ALE values also grow with the number of experiments, so unequal group
sizes bias the subtraction toward the larger group.

Remedies: `BalancedALESubtraction` (matched-size subsampling, Frahm et al.) for unequal
groups, or `ContrastWorkflow` for GingerALE-style main-effect gating. See
`nimare-contrast-conjunction`.

## 8. Conjunction is the voxelwise minimum of corrected maps

`nimare.workflows.misc.conjunction_analysis` takes the minimum statistic across thresholded,
corrected maps. Running it on uncorrected maps gives a map with no interpretable error rate.

## 9. Coordinate spaces must be harmonized before fitting

`nimare.utils.tal2mni` / `mni2tal` implement the Lancaster transform. Sleuth files declare
their space in the header (`// Reference=MNI` or `TAL`) and `convert_sleuth_to_*` applies
the conversion. Hand-assembled datasets do not get this for free — mixing spaces silently
smears convergence. `validate_coordinate_spaces` exists in `nimare/utils.py`; CBMA
estimators call it.

## 10. `Dataset` is deprecated

`Dataset` emits a deprecation warning and is scheduled for removal in NiMARE 1.0.0. New
code should use `nimare.studyset.Studyset`. Most published examples still use `Dataset`;
when adapting them, convert with `Studyset.from_dataset(...)` or the `convert_*_to_studyset`
I/O functions.

---

## Always-run diagnostics

A convergence result is not reportable without knowing how many studies produced it.

```python
from nimare.diagnostics import Jackknife, FocusCounter
target = "z_desc-mass_level-cluster_corr-FWE_method-montecarlo"
cres = Jackknife(target_image=target, voxel_thresh=None).transform(cres)
cres = FocusCounter(target_image=target, voxel_thresh=None).transform(cres)
cres.tables[f"{target}_diag-Jackknife_tab-counts_tail-positive"]
cres.tables[f"{target}_tab-clust"]
```

A cluster that disappears when one study is removed, or whose foci come from two studies,
should be described as such. `ResampledStability` generalizes this to repeated subsampling.
