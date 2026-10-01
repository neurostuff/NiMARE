# Agent instructions

This repository is a set of **skills** for neuroimaging meta-analysis with
[NiMARE](https://github.com/neurostuff/NiMARE). It contains no application code.

## How to use it

When a task involves neuroimaging meta-analysis — synthesizing published fMRI/PET/VBM
findings, ALE, MKDA, Neurosynth, NeuroQuery, NeuroVault, Sleuth/GingerALE, BrainMap,
Neurosynth Compose, functional decoding of a brain map, or meta-analytic coactivation
modeling:

1. Read [`skills/nimare-meta-analysis/SKILL.md`](skills/nimare-meta-analysis/SKILL.md).
   It routes to the right method and lists the guardrails.
2. Read the one specific skill it points to. Do not reconstruct a pipeline from memory:
   several NiMARE defaults are not the recommended settings, and the API changed
   substantially across versions.
3. Read a `references/*.md` file only when the skill tells you to. They are written to be
   loaded on demand.
4. Prefer the `scripts/*.py` over retyping a pipeline. They are tested and they record
   provenance.

## Non-negotiables

Before producing or approving any coordinate-based meta-analysis:

- Check whether multiple contrasts from the same subject sample are being counted as
  independent experiments. NiMARE does this silently.
  Run `skills/nimare-dataset-curation/scripts/audit_independence.py`.
- Check the number of **independent samples**, not studies or contrasts. Below ~17-20,
  ALE is underpowered and NiMARE will not tell you.
- Use `FWECorrector(method="montecarlo")`. The default is `"bonferroni"`.
- Run and report `Jackknife` and `FocusCounter`.

Before producing or approving a functional decoding result:

- Restrict the term vocabulary, and say how.
- Do not report p-values from map-to-map correlations. Report rankings, or use a
  spatial-autocorrelation-preserving null.

## Layout

```
skills/<skill-name>/SKILL.md        YAML frontmatter (name, description) + body
skills/<skill-name>/references/     on-demand Markdown
skills/<skill-name>/scripts/        runnable Python
docs/use-case-survey.md             the literature survey the skill set is based on
docs/literature/citing-articles.csv 98 NiMARE-citing articles, tagged by use
```

Target version: NiMARE >= 0.22.
