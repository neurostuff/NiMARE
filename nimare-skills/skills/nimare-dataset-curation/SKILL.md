---
name: nimare-dataset-curation
description: Build, import, and audit a NiMARE Studyset or Dataset before meta-analysis. Use when getting coordinates into NiMARE from Sleuth/BrainMap text files, NiMADS bundles or Neurosynth Compose analysis IDs, the Neurosynth/NeuroQuery/NeuroStore databases, NeuroVault, or hand-extracted tables; when converting between Dataset and Studyset; when filtering studies by label, mask, coordinate, or metadata; and above all when checking whether multiple contrasts from the same subject group are being double-counted as independent experiments.
license: MIT
---

# Curating a NiMARE study set

Curation decides the answer more than the estimator does. Do this before choosing a method.

## Preferred container

`nimare.studyset.Studyset` (NiMADS-backed). `nimare.dataset.Dataset` still works and is
what most published examples use, but it emits a deprecation warning and is scheduled for
removal in NiMARE 1.0.0.

```python
from nimare.studyset import Studyset
studyset = Studyset.from_dataset(dataset)       # migrate
studyset = Studyset.from_sleuth("foci.txt")
studyset = Studyset.from_nimads("bundle.json")
studyset = Studyset.from_parquet("neurostore_release/")
```

Key properties: `.ids` (full `"<study>-<analysis>"` ids), `.study_ids`, `.coordinates`,
`.annotations`, `.metadata`, `.images`, `.space`.

## Getting data in

| Source | Function | Notes |
|---|---|---|
| Sleuth / BrainMap export | `nimare.io.convert_sleuth_to_studyset(path)` | Also `convert_sleuth_to_dataset`, `_to_dict`, `_to_json`. Accepts a list of files. Applies Talairach→MNI from the header. |
| Neurosynth Compose / NiMADS | `nimare.io.fetch_neurostore_studyset(...)`, `convert_nimads_to_dataset` | The "Reproducible Bundle" a Compose analysis exports. |
| Neurosynth v7 | `nimare.extract.fetch_neurosynth(...)` -> Studyset | **Deprecated in 0.22**; frozen 2018 snapshot. ~14,371 studies, 3,228 terms; also LDA topic sets. Still the source for term annotations used in decoding. |
| NeuroQuery v1 | `nimare.extract.fetch_neuroquery(...)` -> Studyset | Also a frozen snapshot. ~13,459 studies, ~6,145 terms incl. full text. |
| NeuroStore releases | `nimare.extract.fetch_neurostore(version="latest")` | **Preferred source of current coordinate data** (0.22+). `fetch_neurostore_releases()` lists versions. |
| NeuroVault | `nimare.generate.create_neurovault_studyset(...)` | For IBMA. QC is mandatory — see `nimare-ibma`. |
| pubget output | writes NiMARE JSON directly | `pubget` extracts coordinates from PMC full text. Crude; needs manual screening. |
| Abstracts for topic modelling | `nimare.extract.download_abstracts(dset, email)` | Fetches from PubMed by PMID. |
| Cognitive Atlas | `nimare.extract.download_cognitive_atlas()` | Concept/task ontology, used to restrict term lists. |
| Hand-extracted tables | build the NiMARE dict, then `Dataset(d)` / `Studyset(d)` | See `references/data-sources.md` for the schema. |

## The check that matters most: contrast independence

**Run this before every CBMA.** NiMARE's unit of analysis is the *analysis* (contrast),
not the study and not the subject group. Two contrasts from the same subjects count twice,
with no warning. Measured on NiMARE's own example data, that doubles supra-threshold
extent.

```bash
python scripts/audit_independence.py foci.txt            # Sleuth file
python scripts/audit_independence.py studyset.json       # NiMADS
python scripts/audit_independence.py foci.txt --pool out_pooled.json
```

The script reports studies contributing more than one analysis, flags analyses that share
a sample size within a study (the usual signature of one sample reported twice), and warns
about the Sleuth naming trap below. Read
`references/contrast-independence.md` for the decision rules, including when pooling is
*wrong*.

### The Sleuth naming trap

`convert_sleuth_*` splits a header into study and contrast on the first `;` or `:` that
follows a four-digit year.

```
// Smith, 2010: faces > houses     ->  study "Smith, 2010",  analysis "faces > houses"
// Smith, 2010: faces > scrambled  ->  study "Smith, 2010",  analysis "faces > scrambled"
// smith2010faces                  ->  study "smith2010faces", analysis "analysis_1"
// smith2010scrambled              ->  study "smith2010scrambled", analysis "analysis_1"
```

In the second form the two experiments become two *separate studies*, so nothing — not
`combine_analyses()`, not a per-study jackknife — can recover the link. NiMARE's own
bundled `semantic_knowledge_children.txt` has exactly this: `arnoldussen2006nc` and
`arnoldussen2006rm`, both `Subjects=11`, almost certainly the same 11 children, entered as
two independent studies. **Fix the headers at curation time.**

## Pooling same-sample contrasts

```python
pooled = studyset.combine_analyses()   # one analysis per study; foci and images concatenated
```

Correct when every analysis within a study comes from one sample. **Wrong** when a study
contributes genuinely independent samples (patients and controls, two cohorts, two age
groups) — that collapses real information. In that case build the grouping by hand; see
`references/contrast-independence.md`.

## Filtering and slicing

```python
studyset.filter_ids([...])                 # keep these analyses
studyset.filter_study_ids([...])           # keep these studies
studyset.exclude_study_ids([...])
studyset.filter_annotations(["face"], threshold=0.001)
studyset.filter_metadata("sample_size", ">=", 15)
studyset.get_studies_by_mask(roi_img)      # the MACM selection step
studyset.get_analyses_by_coordinate([0, -52, 26], r=10)
studyset.get_studies_by_label(["terms_abstract_tfidf__pain"], label_threshold=0.001)
studyset.slice(ids=[...])
studyset.merge(other)
```

`FocusFilter` (`nimare.diagnostics.FocusFilter`) drops foci falling outside the mask —
worth running, since out-of-brain coordinates are common in hand-extracted tables.

## Pre-flight checklist

1. How many **independent samples**? (not studies, not contrasts) — target >= 17-20 for ALE.
2. Is every coordinate in one space? Sleuth headers declare it; hand-built data does not.
3. Do all experiments have `sample_size` metadata? ALE's kernel needs it; database-derived
   sets lack it and need an explicit `ALEKernel(fwhm=...)`.
4. Are activations and deactivations mixed? NiMARE's algorithms are one-sided; the
   `Dataset` constructor warns if `z_stat` has both signs, but unsigned foci give no signal.
5. Any focus outside the mask? Run `FocusFilter`.
6. Was the search PRISMA-documented? Record it now; see `nimare-reporting`.

## References

- `references/contrast-independence.md` — the full treatment, with measured effects.
- `references/data-sources.md` — schemas, database details, conversion recipes.
- `scripts/audit_independence.py` — runnable audit and pooling helper.
