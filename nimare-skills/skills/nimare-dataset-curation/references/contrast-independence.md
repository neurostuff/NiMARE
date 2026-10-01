# How NiMARE treats multiple analyses within one study

Short answer: **as independent experiments, silently.**

## What the code does

`nimare/dataset.py` constructs ids by concatenation:

```python
# Datasets are organized by study, then experiment
# To generate unique IDs, we combine study ID with experiment ID
id_ = f"{pid}-{expid}"
```

Every CBMA estimator then works from `inputs_["id"]`, i.e. the contrast-level id. Several
internal variables are named `study_ids` but hold these contrast ids; the naming is
misleading, the behaviour is not ambiguous. `Studyset` is the same: `analysis_full_key` is
`"<study id>-<analysis id>"` and `Studyset.ids` returns those.

So for ALE, MKDADensity, KDA, SCALE, MKDAChi2 and the subtraction estimators:

- each contrast gets its own modelled-activation (MA) map;
- each contrast's MA map is weighted by *its own* sample size (ALE kernel FWHM is derived
  per experiment from `sample_size`);
- the across-experiment summary statistic treats them as exchangeable and independent.

Within a single contrast, the revised ALE algorithm does protect against one experiment
dominating a voxel: overlapping foci from the same experiment are combined by taking the
maximum, not summed (Turkeltaub et al., 2012). That protection operates **inside** an
experiment. It does nothing across two experiments that happen to share subjects.

IBMA is different and better: `IBMAEstimator(groupby=...)` defaults to grouping images by
`study_id`, applies the Hedges correlated-effects weighting, and switches to CR2
cluster-robust standard errors with Satterthwaite degrees of freedom. `groupby=False`
opts out and logs a warning that doing so "inflates significance". **No CBMA estimator has
an equivalent.**

## Measured consequence

NiMARE 0.22, bundled `semantic_knowledge_children.txt`, `ALE(null_method="approximate")`,
same foci throughout (the duplicate contrast is the original foci jittered by 4 mm, which
is what a second contrast from the same sample looks like):

| dataset | units fitted | max z | voxels z>3.1 | voxels z>5 |
|---|---|---|---|---|
| one contrast per sample | 21 | 6.76 | 1,587 | 87 |
| each sample entered twice | 42 | 7.31 | 3,960 | 420 |
| the 42 after `combine_analyses()` | 21 | 6.40 | 1,843 | 86 |

2.5x the supra-threshold extent and ~5x the voxels above z=5, from no new data. In a
cluster-level FWE analysis this translates into clusters that survive only because one
sample was counted twice.

## Decision rules

Build the dataset so that **one row = one independent subject group**.

1. **Several contrasts, one sample** (faces>houses and faces>scrambled in the same 20
   subjects) → pool the foci into a single analysis. This is the BrainMap/Sleuth
   convention and what GingerALE users are instructed to do.
2. **Several contrasts, genuinely independent samples** (patients and controls; a
   discovery and a replication cohort) → keep them separate. They are independent
   experiments and should count as two.
3. **Same sample, but the contrasts answer different questions and you want both** →
   you cannot have both in one convergence analysis. Either pick the contrast that matches
   the research question, or run two meta-analyses.
4. **Longitudinal / repeated-measures contrasts from one cohort** → one row.
5. **Can't tell from the paper** → treat as one sample. The conservative direction is to
   pool; the error that inflates results is splitting.

## Doing it

```python
# Case 1, uniform across the study set:
pooled = studyset.combine_analyses()        # one analysis per study
```

`combine_analyses()` concatenates `points`, `images`, `conditions`, `weights` and merges
`metadata`/`texts` per study, naming the merged analysis by joining the component ids with
`_`. It drops annotations, since annotation notes refer to pre-merge analyses.

It groups by **study**, so case 2 breaks it. For a mixed study set, pool selectively:

```python
# Case 2: keep patient and control analyses of the same paper apart.
# Give each independent sample its own study id before loading.
# In a NiMADS/dict source, that means e.g.
#   "smith2010"  -> "smith2010_patients", "smith2010_controls"
# each holding the analyses that belong to that sample, then:
pooled = studyset.combine_analyses()
```

Equivalently, if you are assembling a Sleuth file, name the headers so that the sample is
the study and the contrast is the suffix:

```
// Smith, 2010 patients: encoding > baseline
// Subjects=20
...
// Smith, 2010 patients: retrieval > baseline
// Subjects=20
...
// Smith, 2010 controls: encoding > baseline
// Subjects=22
```

After `convert_sleuth_to_studyset`, `combine_analyses()` yields two rows for Smith 2010 —
the correct answer.

## The Sleuth header trap

`convert_sleuth_to_dict` splits the header on the first `;` or `:` **after a four-digit
year**; with no year it falls back to the first `;` or `:` anywhere; with no separator the
whole header becomes the study name and the contrast becomes `analysis_1`.

So `// smith2010faces` and `// smith2010scrambled` become two *studies*. At that point the
sample link is gone from the data and no downstream function can restore it. NiMARE's own
example file demonstrates the failure mode:

```
// arnoldussen2006nc     // Subjects=11
// arnoldussen2006rm     // Subjects=11
```

Two experiments, same paper, same N, entered as two studies.

**Audit rule:** within a study set, any two analyses from papers with the same first
author and year, and the same sample size, are suspect. `../scripts/audit_independence.py`
flags them.

## What to report

State the unit of analysis explicitly in the methods: how many articles, how many
independent samples, how many experiments entered, and the rule used to collapse
contrasts. Reviewers of coordinate-based meta-analyses ask for this, and it is the
difference between a reproducible dataset and a number.

## Caveat for database-derived study sets

Neurosynth and NeuroQuery rows are *articles*, with all reported foci pooled and no sample
size and no activation/deactivation labelling. Within-study non-independence is therefore
already collapsed (in the pooling direction), but you also lose the ability to select a
contrast. This is a known and acceptable limitation for decoding and MACM; it is not
acceptable as a substitute for a curated ALE.
