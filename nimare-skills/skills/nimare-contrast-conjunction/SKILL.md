---
name: nimare-contrast-conjunction
description: Compare two sets of studies or find their common convergence with NiMARE — ALE subtraction, balanced ALE subtraction, GingerALE-style main-effect-gated contrast, MKDA chi-square, and conjunction analysis. Use when the user wants to test where two literatures differ (patients vs controls, task A vs task B, drug vs natural reward) or overlap, mentions subtraction analysis, contrast analysis, or conjunction of meta-analytic maps.
license: MIT
---

# Pairwise contrasts and conjunction

## Choosing

| Method | Use when | Output |
|---|---|---|
| `ALESubtraction` | Two disjoint sets, similar sizes | Two-sided ALE difference over the whole mask, permutation null |
| `ContrastWorkflow` | You want GingerALE-style logic | Difference tested *only where each group has a surviving main effect*, plus main-effect maps and a conjunction |
| `BalancedALESubtraction` | The two sets differ in size | Matched-size subsampling, averaged differences, plus per-group probabilistic activation maps |
| `MKDAChi2` | Specificity framing, or one set vs. a whole database | Uniformity and association maps with chi-square tests |
| `conjunction_analysis` | You want the common effect | Voxelwise minimum of thresholded corrected maps |

## The exchangeability requirement — read this first

`ALESubtraction` builds its null by **permuting group labels** between the two sets. That
requires the sets to be exchangeable under the null, which means **they must be
disjoint**. Comparing an "all studies" set against a subset of itself puts the same
experiments on both sides, violates exchangeability, and invalidates the p-values. This
is a real error that has reached print; one surveyed paper caught it mid-analysis and
restructured into disjoint pairwise comparisons.

If you need "A vs everything", either split into disjoint pairs, or use `MKDAChi2`, whose
framing is "selected vs. the rest" by construction.

## Unequal group sizes bias plain subtraction

ALE values grow with the number of experiments, so a 40-study set will out-converge a
12-study set almost everywhere. `BalancedALESubtraction` fixes this by drawing matched-size
subsamples from both groups, averaging the balanced differences, and building the null
from balanced resamples.

```python
from nimare.meta.cbma.ale import BalancedALESubtraction
est = BalancedALESubtraction(
    target_n=None,          # defaults to the smaller group
    n_subsamples=2500,
    difference_iterations=1000,
    n_iters=1000,
    voxel_thresh=0.001,
    null_method="random-foci",   # or label permutation
    mask_coverage="gm",
    alpha=0.05,
    n_cores=-1, random_state=42,
)
res = est.fit(studyset1, studyset2)
```

It also returns probabilistic activation maps — the proportion of subsamples in which
each voxel survived — and their conjunction. No separate corrector call is needed.

## Plain subtraction

```python
from nimare.meta.cbma.ale import ALESubtraction
from nimare.correct import FWECorrector

est = ALESubtraction(n_iters=5000, voxel_thresh=0.001, n_cores=-1, random_state=42)
res = est.fit(studyset1, studyset2)
cres = FWECorrector(method="montecarlo", n_iters=5000, n_cores=-1).transform(res)
```

`vfwe_only=True` (the default) computes voxel-level nulls only; set `False` to also get
cluster size/mass nulls.

Subtraction is badly underpowered. Each group needs enough experiments in its own right —
apply the ~17-20 rule to **both** groups, not to the total. A null subtraction result
with 12 vs 15 experiments says nothing.

## GingerALE-style gated contrast

```python
from nimare.workflows.cbma import ContrastWorkflow
wf = ContrastWorkflow(main_estimator=ALE, pairwise_estimator=ALESubtraction,
                      alpha=0.05, n_cores=-1, output_dir="results/")
cres = wf.fit(studyset1, studyset2)
```

Fits and corrects a within-group ALE for each set, thresholds at `alpha`, and passes the
surviving voxels to `ALESubtraction` as directional inference masks: "1 > 2" is evaluated
only where group 1 has a main effect, and vice versa. Also stores the two main-effect maps
and their voxelwise-minimum conjunction. It thresholds internally; do not add a corrector.

This matches what most published "ALE contrast" analyses mean, and is usually what a user
asking for "GingerALE subtraction" wants.

## MKDA chi-square

```python
from nimare.meta.cbma import MKDAChi2
res = MKDAChi2(prior=0.5, fwe_null_method="label-permutation",
               random_state=42).fit(studyset1, studyset2)
```

Outputs `z_desc-uniformity` (forward: does this set activate here?) and
`z_desc-association` (reverse: is activation here selective to this set?), plus matching
`chi2_`, `p_` and `logp_` maps. This is Neurosynth's method; `z_desc-association` is what
"Neurosynth association test map" means. Use when `dataset2` is a large reference
database rather than a matched comparison set.

## Conjunction

```python
from nimare.workflows.misc import conjunction_analysis
conj = conjunction_analysis([thresholded_corrected_map1, thresholded_corrected_map2])
```

Voxelwise minimum statistic (Nichols et al., 2005). The inputs must be **thresholded,
corrected** maps — a minimum over uncorrected maps has no interpretable error rate. The
result is "jointly significant in all inputs", which is a stricter and more honest claim
than overlapping two thresholded blobs by eye.

## Reporting

Both group sizes, the estimator and why (balanced? gated?), the null model, the
exchangeability justification (sets disjoint), thresholds for each stage, and — for
conjunction — that the inputs were corrected.

## References

- `../nimare-cbma-ale/SKILL.md`
- `../nimare-meta-analysis/references/statistical-guardrails.md` (items 7 and 8)
- NiMARE example `02_meta-analyses/14_plot_compare_contrast_strategies.py` compares all
  three subtraction routes on the same data.
