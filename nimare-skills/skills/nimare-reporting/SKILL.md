---
name: nimare-reporting
description: Write up and reproducibly package a NiMARE meta-analysis — methods text, PRISMA flow, what to report for ALE/MKDA/IBMA/decoding, NiMARE's HTML reports, and sharing results on NeuroVault and Neurosynth Compose. Use when drafting the methods or results section of a neuroimaging meta-analysis, preparing a supplement, responding to reviewers about meta-analytic rigor, or packaging an analysis so someone else can rerun it.
license: MIT
---

# Reporting a NiMARE meta-analysis

A coordinate-based meta-analysis is a systematic review with a spatial statistic bolted
on. Reviewers increasingly ask for both halves. These are the facts a reader needs in
order to believe, and reproduce, the result.

## Generate the report first

```python
result.save_maps(output_dir="results/")
result.save_tables(output_dir="results/")
```

The `Workflow` classes (`CBMAWorkflow`, `PairwiseCBMAWorkflow`, `IBMAWorkflow`,
`ContrastWorkflow`) produce an HTML report with maps, cluster tables and diagnostics,
and most carry `generate_description=True`, which drafts a methods paragraph naming the
estimator, kernel, correction and thresholds actually used. Start from that text rather
than writing from memory — it reflects the parameters of the run, including the ones you
left at their defaults.

## Minimum reportable set — all analyses

- NiMARE version (`nimare.__version__`) and Python version. The API changed substantially
  across 0.0.x -> 0.5 -> 0.22; "NiMARE" alone is not a specification.
- Where the data came from: database + version/release date, or the search strategy.
- **Number of articles, number of independent samples, number of experiments entered.**
  Three different numbers. Give all three.
- The rule used to collapse multiple contrasts from one sample (see below).
- Coordinate space, and the transform if any coordinates were converted.
- Estimator, kernel (and the FWHM rule), null method.
- Corrector, method, cluster-forming threshold, corrected alpha, number of iterations.
- `random_state`, so the Monte Carlo result is reproducible.
- Software for visualization and atlas labelling.
- A data and code availability statement. NiMADS/Sleuth input + script is enough.

## The sentence most write-ups are missing

> Where a study reported several contrasts from the same participants, foci were pooled
> into a single experiment, so that each independent subject group contributed one
> experiment. The final analysis comprised N experiments from M articles (K independent
> samples).

NiMARE's unit of analysis is the contrast, and it counts same-sample contrasts as
independent experiments without warning. Stating your rule tells the reader whether that
happened. If you did *not* pool, say so and justify it.

## PRISMA

Coordinate-based meta-analyses are systematic reviews; PRISMA 2020 applies. Report
databases and dates searched, the full query, inclusion and exclusion criteria, screening
counts at each stage, and the flow diagram. Neurosynth Compose generates a
PRISMA-compliant flow from its curation step — worth using for that alone, even if the
model is fitted in Python.

Pre-register where you can. The search-to-result path in a CBMA has many defensible
choices and no way for a reader to see the ones you did not take.

## By method

**ALE / MKDA / KDA.** Kernel and FWHM rule (sample-size-derived, or fixed with the value
and reason); null method; cluster-forming threshold; correction and iterations; which
map is reported (cluster-mass vs cluster-size vs voxel-level); whether cluster
coordinates are peaks from the uncorrected map or centres of mass. Report the
**jackknife and focus-counter tables** — a cluster carried by two experiments must be
described as such.

**Subtraction.** Both group sizes; that the sets are disjoint (exchangeability);
balanced or unbalanced; whether main-effect gating was used; thresholds for every stage.

**Conjunction.** That inputs were thresholded *corrected* maps, and the minimum-statistic
method (Nichols et al., 2005).

**MACM.** Database and version; how studies were selected (mask, or coordinate + radius);
**how many studies were selected**; kernel FWHM and why (databases carry no sample sizes);
null model (uniform ALE, SCALE base-rate, or MKDAChi2); whether the seed was masked out.

**IBMA.** Estimator and whether it is a combination test or random effects; `groupby` and
the dependence model; `weight_scheme` and `rho`; masking mode; number of studies *and*
groups; any PyMARE small-sample warning; image selection criteria for NeuroVault data.

**Decoding.** Database and version; vocabulary and how it was restricted (e.g. the ~123
Neurosynth terms also in the Cognitive Atlas); number of terms tested; decoder; whether
the input map was thresholded; the statistic; **the null model, or an explicit statement
that correlations are reported as rankings rather than tests**; multiple-comparison
correction. Do not call a term correlation "reverse inference" without qualification.

## Language to avoid

| Instead of | Write |
|---|---|
| "region X is responsible for Y" | "foci converged in X across studies of Y" |
| "reverse inference showed the map reflects Y" | "the map's spatial pattern most resembled the meta-analytic map for Y" |
| "no convergence, so X is not involved" | "no convergence was detected; with N experiments the analysis is powered to detect only ..." |
| "N studies" (when you mean contrasts) | "N experiments from M articles" |
| "significantly correlated with the Y term map" | "ranked highest among the T terms tested (r = ...)" |

## Sharing

- **Unthresholded maps to NeuroVault**, with the thresholded ones. This is what makes
  your result reusable in someone else's IBMA, and it costs nothing.
- **The study set**, as NiMADS JSON or a Sleuth text file, in the supplement. Coordinate
  tables retyped from PDFs are the least reproducible part of the whole enterprise.
- **The script**, with pinned versions and the `random_state`.
- **Neurosynth Compose** if you want the curation, specification and execution tracked
  end to end: it stores a reproducible NiMADS bundle with a unique id, runs the same
  NiMARE estimators via `nsc-runner` or a Colab notebook, and pushes results back with a
  NeuroVault link.

## References

- `references/reporting-checklist.md` — a checklist to work through before submission.
