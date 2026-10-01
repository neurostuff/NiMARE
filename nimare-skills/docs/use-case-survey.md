# What people actually do with NiMARE

The skills in this repository are shaped by a survey of the published literature that
cites NiMARE, not by the structure of the NiMARE API. This document records the survey
so the design can be checked and redone.

## Method

- PubMed Central was searched for the full-text term `NiMARE` (NCBI E-utilities,
  `db=pmc`). 115 records were returned.
- Full text was downloaded for all 115 (`efetch`, JATS XML).
- 98 contained at least one body paragraph mentioning NiMARE; those paragraphs were
  extracted and read. The remaining 17 mention it only in a reference list, in
  supplementary material, or are false positives (one 1999 otolaryngology case report;
  one linguistics paper in which *nimare* is a Hewramî word).
- Google Scholar reports ~143 citing works for the NiMARE paper, so PMC full-text
  coverage is roughly two thirds of the citing literature and is biased toward
  open-access journals.
- Each paper was tagged by which NiMARE capabilities its methods text describes. Tags are
  not exclusive.
- The per-article table is in `literature/citing-articles.csv`.

Survey date: 2026-10-01. Counts below are of the 98 papers with usable methods text.

## What they used it for

| Use | Papers | Share |
|---|---|---|
| Functional decoding of the authors' own map or ROI | 54 | 55% |
| Generating term- or topic-based meta-analytic maps (MKDAChi2, LDA, GC-LDA) | 31 | 32% |
| ALE coordinate-based meta-analysis of a curated study set | 22 | 22% |
| Image-based meta-analysis / NeuroVault | 14 | 14% |
| Meta-analytic coactivation modeling (MACM) | 11 | 11% |
| Fetching or converting a database (Neurosynth, NeuroQuery, Sleuth, NiMADS, pubget) | 10 | 10% |
| Coordinate space transformation (Talairach -> MNI) | 10 | 10% |
| Subtraction or conjunction analysis | 7 | 7% |
| Diagnostics (jackknife, leave-one-out, focus counter) | 3 | 3% |

## What that implies for the skills

**Decoding dominates, and is the least rigorous thing people do.** More than half the
citing literature uses NiMARE purely to attach Neurosynth terms to a map produced some
other way. Typical practice: correlate an unthresholded map against 50-400 term or topic
maps, hand-remove anatomical and methodological terms, report the top 10-25 in a word
cloud. Very few use a spatial-autocorrelation-preserving null, despite NiMARE's own
docstring warning that correlation p-values are meaningless here. Hence
`nimare-functional-decoding` and its `null-models.md` reference.

**Diagnostics are almost never run** — 3 papers out of 98 — even though NiMARE ships
`Jackknife`, `FocusCounter` and `ResampledStability` and the workflows run them by
default. Hence diagnostics are presented as mandatory, not optional, throughout.

**Database-derived analyses inherit problems people do not mention.** The MACM and
decoding papers rely on Neurosynth/NeuroQuery, whose rows are articles with pooled foci,
no sample sizes and no activation/deactivation labelling. The better papers say so
explicitly (and adopt a fixed 15 mm kernel for exactly this reason); most do not.

**Nobody discusses within-study contrast independence.** Not one of the 98 papers
addresses how NiMARE treats multiple contrasts from the same subject group — which it
counts as independent experiments, silently. Several describe curating "contrasts" as
the unit of analysis without saying whether same-sample contrasts were pooled. This is
why `nimare-dataset-curation` leads with it and ships an audit script.

**Version drift is severe.** Versions named in these papers span 0.0.3, 0.0.10, 0.0.11,
0.0.12, 0.0.13, 0.1.1, 0.2.0, 0.2.1, 0.2.2, 0.3.0, 0.4.1, 0.4.2, 0.5.0, 0.5.2, 0.9.0 and
one "0.22.0". Code copied from a paper will frequently not run. The skills target 0.22+
and flag deprecations (`Dataset`, `fetch_neurosynth`, `convert_neurovault_to_dataset`).

**Several papers reimplemented things because NiMARE lacked them.** A GPU ALE
(`nimare-gpu`) for permutation-heavy work; a surface `CorrelationDecoder`, because
NiMARE's nilearn maskers do not transform surface to volume; a MATLAB toolbox (CBMAT)
specifically noting that "removal of multiple experiments from the same paper ... [is]
not implemented in NiMARE". That last one is independent confirmation of the
contrast-independence gap.

## Representative patterns, with sources

**Decoding a gradient or network.** Bin an unthresholded gradient into percentile masks,
decode each with `ROIAssociationDecoder` or `CorrelationDecoder` against Neurosynth
topics, order the terms by hierarchy. (Methods for decoding cortical gradients of
functional connectivity, *Imaging Neuroscience* 2024, PMC12224442.)

**The 123-term Cognitive Atlas convention.** Build MKDAChi2 term maps for the ~123-125
terms present in both Neurosynth and the Cognitive Atlas, project to fsLR, parcellate
with HCP-MMP, correlate. Recurs across at least six papers.
(e.g. *Nature Neuroscience* 2025, PMC12321582.)

**ALE with a proper correction.** Cluster-forming p<0.001, cluster-level FWE p<0.05,
10,000 Monte Carlo iterations, Talairach converted via Lancaster.
(Sleep disorders and structural alterations..., *Scientific Reports* 2026, PMC13046795.)

**ALE plus full diagnostics.** Leave-one-study-out cross-validation plus NiMARE's focus
counter and jackknife to show no single study drove the clusters.
(Brain bases of real-time social interaction, *Aperture Neuro* 2025, PMC12617318.)

**MACM with a fixed kernel and a stated reason.** Neurosynth has no sample sizes, so a
15 mm FWHM kernel is used, citing its correspondence with image-based meta-analysis.
(Cue-reactivity paradigm, *Neurosci Biobehav Rev* 2021, PMC8511211; PTSD connectivity,
*Behav Brain Funct* 2022, PMC9472396.)

**Catching an exchangeability violation.** ALE subtraction between an overall dataset and
a nested disease-specific dataset was abandoned because shared studies violate the
permutation null; restructured as disjoint pairwise comparisons.
(Glucose metabolism across AD/PD/ALS, *medRxiv* 2026, PMC13086094.)

**Systematic NeuroVault IBMA.** A full selection framework — group-level fMRI-BOLD T/Z
maps, N>10, unthresholded, MNI, coverage >40%, then removal of extreme-Z, duplicate,
inverted and keyword-flagged images — before `IBMAWorkflow`.
(Advancing image-based meta-analysis..., *Scientific Reports* 2025, PMC12540767.)

**Multi-site aggregation.** Site-wise models pooled with Stouffer's or Fisher's via
NiMARE. (IBMMA, *NeuroImage* 2025, PMC12875043.)

**The accessibility complaint.** A survey of fMRI meta-analysis software 2019-2024 found
GingerALE still dominant and noted that NiMARE's Python/API-only interface is a barrier
for domain experts without programming skills.
(*Front Hum Neurosci* 2025, PMC12287008.) Neurosynth Compose is the answer to point
non-programmers at; it runs the same NiMARE estimators in the browser.

## Reproducing this survey

```bash
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=NiMARE&retmax=400&retmode=json"
# then efetch db=pmc&retmode=xml for the returned ids, and extract <body> paragraphs
# matching /nimare/i
```

PubMed Central was the data source; see `literature/citing-articles.csv` for the PMCIDs,
PMIDs and DOIs of every article included.
