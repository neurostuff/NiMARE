# Choosing a NiMARE method

## Coordinate-based (CBMA): you have peaks, not maps

| Estimator | Question it answers | When to use |
|---|---|---|
| `ALE` | Where do foci converge more than chance? | Default for a curated literature set. Kernel width scales with each experiment's sample size. |
| `MKDADensity` | Where do studies' blurred peaks overlap? | Fixed-radius alternative; weights by sample size since 0.22. Original Wager et al. formulation. |
| `KDA` | Raw density of foci | Rarely used alone; mostly a building block. |
| `SCALE` | Convergence above the *base rate* of activation in a reference database | MACM and any analysis where whole-brain uniform null is implausible. Needs a database of foci. |
| `MKDAChi2` | Does this set of studies activate a voxel more than a reference set? | Neurosynth-style "uniformity" and "association" maps; the engine behind term-based decoding. Needs **two** study sets. |
| `ALESubtraction` | Where do two sets differ? | Pairwise comparison with a permutation null. Sets must be disjoint. |
| `BalancedALESubtraction` | Same, with unequal group sizes | Preferred when the two sets differ in size. |
| `CBMR` (`nimare.meta.cbmr`) | Spatial intensity as a regression, with study-level covariates | Group comparisons, covariate effects (sample size, year), multiple groups at once. Needs >= ~200 foci per group for the parametric test; otherwise parametric bootstrap. |

Rule of thumb from the literature: ALE for a hand-curated, PRISMA-screened study set;
MKDAChi2 when the "studies" come from Neurosynth/NeuroQuery and you need a specificity
contrast against the rest of the database; CBMR when you have covariates or >2 groups.

## Image-based (IBMA): you have unthresholded maps

| Estimator | Needs | Model |
|---|---|---|
| `Stouffers` | z maps | Combines z's. `use_sample_size=True` to weight. Fixed-effect combination test. |
| `Fishers` | z maps | Combines p's. Most sensitive, least interpretable effect size. |
| `DerSimonianLaird` | beta + variance maps | Random effects, method-of-moments tau². |
| `Hedges` | beta + variance maps | Random effects, Hedges estimator. |
| `FixedEffectsHedges` | beta + variance maps | Fixed effects. |
| `WeightedLeastSquares` | beta + variance maps, supplied tau² | Fast; tau² is an input, not estimated. |
| `VarianceBasedLikelihood` | beta + variance maps | ML/REML. |
| `SampleSizeBasedLikelihood` | beta maps + sample sizes | ML/REML when variances are unavailable. |
| `PermutedOLS` | beta maps | Nonparametric, max-statistic FWE. |

A *combination test* (Stouffers/Fishers) asks "is there a signal somewhere"; a *random
effects* model (DerSimonianLaird/Hedges) estimates a population effect and generalizes.
Prefer random effects when you have variance maps and enough studies. See `nimare-ibma`.

## Neither: you have one map and want an interpretation

That is functional decoding, not meta-analysis. See `nimare-functional-decoding`.
Deciding factors: is the input a binary ROI (`ROIAssociationDecoder`, `NeurosynthDecoder`,
`BrainMapDecoder`) or an unthresholded continuous map (`CorrelationDecoder`)?

## Specialized

- **MACM** — pick studies by whether they report a focus in a seed, then run a CBMA on
  that subset. See `nimare-macm`.
- **Coactivation-based parcellation** — MACM per voxel in an ROI, then cluster the
  resulting maps. NiMARE supplies the MACM step; the clustering is scikit-learn.
- **Predictive ALE** (`nimare.meta.cbma.predictive`, `FWECorrector(method="predictive")`)
  — asks what a new study is expected to show rather than where past studies converged.
- **Annotation** (`nimare.annotate`) — LDA, GCLDA, Cognitive Atlas term extraction, TF-IDF
  counts, for building your own topic sets rather than using Neurosynth's.
- **Machine learning** (`nimare.ml`, added 0.22) — feature extraction/reduction over
  large NeuroStore study sets.
