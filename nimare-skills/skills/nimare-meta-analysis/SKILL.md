---
name: nimare-meta-analysis
description: Entry point for neuroimaging meta-analysis with NiMARE (Python). Use when the user wants to meta-analyze fMRI/PET/VBM findings, synthesize published coordinates or statistical maps, run ALE/MKDA/KDA/SCALE/CBMR, do image-based meta-analysis, build meta-analytic coactivation (MACM) maps, functionally decode a brain map or ROI against Neurosynth/NeuroQuery/BrainMap, or when they mention NiMARE, Neurosynth, Neurosynth Compose, NeuroQuery, NeuroStore, NiMADS, Sleuth/GingerALE, BrainMap, or NeuroVault. Routes to the right method and enforces the statistical guardrails that distinguish a defensible meta-analysis from a plausible-looking one.
license: MIT
---

# NiMARE meta-analysis

NiMARE is a Python library with a shared interface over most published neuroimaging
meta-analysis algorithms. The library makes it easy to run *an* analysis; this skill exists
because the easy path is frequently not the correct one. Use it to pick a method, then
load the specific skill for that method.

Target version: **NiMARE >= 0.22**. Check with `python -c "import nimare; print(nimare.__version__)"`.
APIs moved substantially between 0.0.x, 0.1-0.5, and 0.2x — code from a paper published
before 2026 will often not run as written.

## Step 1: establish what data the user actually has

Everything downstream follows from this. Ask if it is not stated.

| What the user has | Family | Skill |
|---|---|---|
| Peak coordinates typed out of published papers/tables | CBMA | `nimare-cbma-ale` |
| An existing Sleuth `.txt`, NiMADS bundle, or Neurosynth Compose analysis ID | CBMA | `nimare-dataset-curation` then `nimare-cbma-ale` |
| Unthresholded whole-brain statistical maps (one per study/site) | IBMA | `nimare-ibma` |
| One brain map or ROI of their own, and they want to know "what cognitive function is this?" | Decoding | `nimare-functional-decoding` |
| A seed ROI, and they want "what else coactivates with this?" | MACM | `nimare-macm` |
| Two sets of studies to compare | Pairwise CBMA | `nimare-contrast-conjunction` |
| Nothing yet; they want to query the literature automatically | Database | `nimare-dataset-curation` |

**The most common request is the fourth row.** Of 98 NiMARE-citing articles surveyed
(`docs/use-case-survey.md`), 54 used NiMARE only to decode their own map or ROI against
Neurosynth terms/topics — not to run a meta-analysis at all. Recognize it and do not
over-build.

## Step 2: apply the guardrails before writing code

Read `references/statistical-guardrails.md` in full before any analysis. The five that
cause the most published errors, in order of how often they go unnoticed:

1. **Multiple contrasts from one sample are treated as independent by default.** NiMARE's
   CBMA unit of analysis is the *analysis* (`"<study_id>-<analysis_id>"`), never the study
   and never the subject group. Two contrasts from the same 20 subjects count as two
   experiments. No warning is emitted. In a verified test on NiMARE's own bundled data,
   entering each sample twice raised supra-threshold voxel count from 1,587 to 3,960
   (2.5x) with no new information added. This is the single highest-value check to run;
   see `nimare-dataset-curation` and its audit script,
   `../nimare-dataset-curation/scripts/audit_independence.py`.
2. **CBMA estimators enforce no minimum number of experiments.** `ALE().fit()` on 4
   experiments runs silently and returns a map. Convergence meta-analysis needs ~17-20
   experiments for adequate power (Müller et al., 2018). Below that, say so and stop.
3. **Only cluster-level FWE via Monte Carlo is defensible for ALE.** Uncorrected and FDR
   thresholds were shown to be invalid for ALE (Eickhoff et al., 2016; Frahm et al., 2022).
   `FWECorrector` defaults to `method="bonferroni"` — pass `method="montecarlo"` explicitly.
4. **Decoding correlations are not hypothesis tests.** NiMARE's own `CorrelationDecoder`
   docstring warns that "almost all results will be statistically significant." Report
   rankings, or use a spatial-autocorrelation-preserving null. See
   `nimare-functional-decoding/references/null-models.md`.
5. **IBMA dependence must be declared.** `groupby=None` (the default) groups images by
   `study_id`, which is right for most datasets but wrong when one paper contributes two
   independent samples; `groupby=False` inflates significance. See `nimare-ibma`.

## Step 3: route

Load the one specific skill for the chosen method. Each contains the correct defaults,
the parameters that matter, and a tested reference script. Do not reconstruct a pipeline
from memory — the defaults are not the recommended values in several places.

For a complete worked pipeline (curate → fit → correct → diagnose → report), the
`Workflow` classes wire the steps together and emit an HTML report:

```python
from nimare.workflows import CBMAWorkflow
result = CBMAWorkflow(
    estimator="ale",
    corrector="montecarlo",       # NOT the bonferroni default
    diagnostics=["jackknife", "focuscounter"],
    n_cores=-1,
).fit(studyset)
result.save_maps(output_dir="results/")
result.save_tables(output_dir="results/")
```

## Core objects

- `nimare.studyset.Studyset` — the current container (NiMADS-backed). Prefer it.
- `nimare.dataset.Dataset` — **deprecated**, removal in NiMARE 1.0.0. Still what most
  tutorials and papers show. Convert with `Studyset.from_dataset(dataset)`.
- `nimare.meta.cbma` — ALE, ALESubtraction, BalancedALESubtraction, SCALE, MKDADensity,
  MKDAChi2, KDA. `nimare.meta.cbmr` — coordinate-based meta-regression.
- `nimare.meta.ibma` — Stouffers, Fishers, DerSimonianLaird, Hedges, FixedEffectsHedges,
  WeightedLeastSquares, VarianceBasedLikelihood, SampleSizeBasedLikelihood, PermutedOLS.
- `nimare.correct` — `FWECorrector`, `FDRCorrector`.
- `nimare.diagnostics` — `Jackknife`, `FocusCounter`, `ResampledStability`, `FocusFilter`.
- `nimare.decode` — `CorrelationDecoder`, `ROIAssociationDecoder`, `NeurosynthDecoder`,
  `BrainMapDecoder`, `CorrelationDistributionDecoder`.

## Reporting

Whatever the method, the write-up needs the same facts (version, unit of analysis,
kernel, null method, correction, thresholds, diagnostics). Load `nimare-reporting`
before drafting methods text.

## References

- `references/method-selection.md` — fuller decision tree including CBMR, SCALE, predictive ALE.
- `references/statistical-guardrails.md` — the guardrails above, with citations and code.
- `../../docs/use-case-survey.md` — what 98 citing articles actually did with NiMARE.
