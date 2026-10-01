# Data sources and conversion recipes

## The NiMARE dict schema (hand-built datasets)

```python
dset_dict = {
    "smith2010": {                                   # study id = one paper
        "contrasts": {
            "faces_gt_houses": {                     # analysis id = one experiment
                "coords": {
                    "space": "MNI",                  # or "TAL"
                    "x": [-42, 38], "y": [-52, -50], "z": [-18, -20],
                },
                "metadata": {"sample_sizes": [20]},  # list, one per group contributing
                "images": {"beta": "...", "varcope": "..."},   # IBMA only
                "labels": {"terms_abstract_tfidf__face": 0.02},
                "text": {"abstract": "..."},
            },
        },
    },
}
from nimare.dataset import Dataset
from nimare.studyset import Studyset
studyset = Studyset.from_dataset(Dataset(dset_dict))
```

`sample_sizes` drives the ALE kernel width. Omit it and `ALEKernel` has nothing to work
from; you must then pass `fwhm` explicitly.

**One study id per independent subject group.** If a paper reports a patient and a control
sample, give them separate study ids (`smith2010_patients`, `smith2010_controls`) so that
pooling and jackknifing group correctly. See `contrast-independence.md`.

## Sleuth / BrainMap

```
// Reference=MNI
// Smith, 2010: faces > houses
// Subjects=20
-42  -52  -18
 38  -50  -20

// Smith, 2010: faces > scrambled
// Subjects=20
...
```

```python
from nimare.io import convert_sleuth_to_studyset, convert_sleuth_to_dataset
studyset = convert_sleuth_to_studyset("foci.txt")       # or a list of paths
```

- The header line declares the space for the whole file; `TAL`/`Talairach` is converted to
  MNI with the Lancaster transform.
- `Subjects=` is required for every experiment or import raises.
- Study/analysis split happens at the first `;` or `:` **after a four-digit year**. Without
  a separator, the entire header becomes the study id. This is the single most common
  curation error — see `contrast-independence.md`.
- `convert_nimads_to_sleuth` goes the other way, for GingerALE cross-checks.

## Neurosynth v7

**Deprecated in 0.22.0**, removal in 1.0.0. The files are a frozen 2018 snapshot. Keep
using it to reproduce published Neurosynth analyses and for the term annotations that
Neurosynth-style decoding needs; use `fetch_neurostore` for current coordinate data.

```python
from nimare.extract import fetch_neurosynth

# 0.22+: returns a Studyset directly.
studyset = fetch_neurosynth(data_dir="data/", version="7", vocab="terms")

# To assemble it yourself (or to reproduce pre-0.22 code):
from nimare.io import convert_neurosynth_to_studyset
files = fetch_neurosynth(data_dir="data/", version="7", vocab="terms",
                         return_type="files")[0]
studyset = convert_neurosynth_to_studyset(
    coordinates_file=files["coordinates"],
    metadata_file=files["metadata"],
    annotations_files=files["features"],
)
```

- ~14,371 articles, 507,891 foci, 3,228 TF-IDF terms from abstracts.
- `vocab="LDA50"` / `"LDA100"` / `"LDA200"` / `"LDA400"` for topic sets instead of terms.
  The 50- and 400-topic sets are the ones the literature uses most.
- Rows are **articles**, with all reported foci pooled. No sample sizes. No
  activation/deactivation labels. Plan around all three.
- Feature column names look like `terms_abstract_tfidf__working memory`. The conventional
  frequency threshold is `0.001`.

## NeuroQuery v1

```python
from nimare.extract import fetch_neuroquery
studyset = fetch_neuroquery(data_dir="data/", version="1", vocab="neuroquery6308")
# return_type="files" for the raw coordinates/metadata/feature paths
```

~13,459 articles and ~6,145 terms drawn from abstract, body, title and keywords, so the
vocabulary is broader and noisier than Neurosynth's. Same converters.

## NeuroStore / Neurosynth Compose

**This is the maintained source of coordinate data** (added 0.22) and should be preferred
over the frozen Neurosynth and NeuroQuery snapshots.

```python
from nimare.extract import fetch_neurostore_releases, fetch_neurostore

releases = fetch_neurostore_releases()               # what is published
studyset = fetch_neurostore(version="latest", data_dir="data/")
# version="nightly", or a published string like "2026-09"
# return_type="files" gives the extracted parquet directory;
# Studyset.from_parquet(path) then loads it.
```

For a specific Compose analysis, the platform exports a NiMADS "Reproducible Bundle"
containing the curated StudySet plus the analysis specification; `Studyset.from_nimads`
loads the study set half. Compose is the right recommendation for users who do not want
to write Python: it runs the same NiMARE estimators in the browser or via its
`nsc-runner` Docker image, and tracks PRISMA curation.

## NeuroVault (image-based)

```python
from nimare.generate import create_neurovault_studyset
studyset = create_neurovault_studyset(
    collection_ids={"study1": 1234, "study2": 5678},   # informative name -> collection id
    contrasts={"face_gt_house": "as-Face"},            # regex matched against image names
    img_dir="data/images/",
)
```

`nimare.io.convert_neurovault_to_dataset` is the deprecated `Dataset`-returning equivalent.

Selection criteria matter more than the code here — see `nimare-ibma/references/neurovault-qc.md`.

## pubget (automated extraction from PMC full text)

`pubget` (from the NeuroQuery authors) downloads open-access PMC articles and extracts
coordinates, writing output in NiMARE's JSON format. Useful for building a candidate pool
quickly; it does **not** replace manual screening, contrast selection, or coordinate
verification. Treat its output as a search result, not a dataset.

## Abstracts and ontologies

```python
from nimare.extract import download_abstracts, download_cognitive_atlas
dset = download_abstracts(dset, "you@example.edu")   # PubMed, by PMID; email required
cogat = download_cognitive_atlas()                   # concepts, tasks, relationships
```

The Cognitive Atlas intersection is the standard way to filter Neurosynth's vocabulary
down to interpretable mental processes: the surveyed literature repeatedly uses the
~123-125 terms present in both Neurosynth and the Cognitive Atlas. `nimare.annotate.cogat`
has `CogAtLemmatizer`, `extract_cogat`, `expand_counts`.

## Coordinate space

```python
from nimare.utils import tal2mni, mni2tal
```

Lancaster transform. Sleuth import applies it from the header; hand-built data does not.
Mixing spaces without converting smears convergence and is invisible in the output.
