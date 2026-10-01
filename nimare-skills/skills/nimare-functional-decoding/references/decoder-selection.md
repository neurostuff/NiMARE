# NiMARE decoders: signatures, outputs, costs

Verified against NiMARE 0.22.

## Signatures

```python
CorrelationDecoder(feature_group=None, features=None, frequency_threshold=0.001,
                   meta_estimator=None, target_image="z_desc-association", n_cores=1)
CorrelationDistributionDecoder(feature_group=None, features=None,
                               frequency_threshold=0.001, target_image="z", n_cores=1)
ROIAssociationDecoder(masker, kernel_transformer=MKDAKernel,
                      feature_group=None, features=None, **kwargs)
NeurosynthDecoder(feature_group=None, features=None, frequency_threshold=0.001,
                  prior=0.5, u=0.05, correction="bh", min_studies=1)
BrainMapDecoder(feature_group=None, features=None, frequency_threshold=0.001,
                u=0.05, correction="bh")
```

`fit(dataset, drop_invalid=True)` **returns `None`** on all of them — it mutates the
decoder. Do not chain `.fit(...).transform(...)`.

| Decoder | `transform` takes | Returns |
|---|---|---|
| `CorrelationDecoder` | an image (path or `Nifti1Image`) | one column `r`, indexed by feature |
| `CorrelationDistributionDecoder` | an image | per-feature correlation distribution summaries |
| `ROIAssociationDecoder` | nothing | one column `r`, indexed by feature |
| `NeurosynthDecoder` | `ids`, optional `ids2` | `pForward, zForward, probForward, pReverse, zReverse, probReverse` |
| `BrainMapDecoder` | `ids`, optional `ids2` | `pForward, zForward, likelihoodForward, pReverse, zReverse, probReverse` |

Note the one column that differs: BrainMap reports `likelihoodForward` (a likelihood
ratio) where Neurosynth reports `probForward` (a posterior probability under `prior`).

The `ids` for the discrete decoders come from the ROI:

```python
ids = studyset.get_studies_by_mask(roi_img)
```

`ids2` supplies an explicit comparison set; omitted, the rest of the database is used.

## Parameters worth setting

- `features` — **always set it.** Fitting over the full vocabulary runs one meta-analysis
  per term. Restrict to a curated list (Cognitive Atlas intersection, or a topic set).
- `feature_group` — e.g. `"terms_abstract_tfidf"`; disambiguates when a study set carries
  several annotation sets.
- `frequency_threshold` — 0.001 is the Neurosynth convention: a study "uses" a term when
  its TF-IDF reaches this. Raising it shrinks and sharpens the per-term study sample.
- `correction` — `"bh"` (Benjamini-Hochberg) or `None`. **Not** `"fdr_bh"`.
- `min_studies` (Neurosynth only) — floor on studies per term; raise it to drop terms
  supported by a handful of papers.
- `prior` (Neurosynth only) — 0.5 by default, i.e. `probReverse` is computed assuming a
  50% base rate for the term. It is a convention that makes terms comparable, not an
  estimate. Do not read `probReverse` as a real posterior.
- `target_image` (CorrelationDecoder) — `"z_desc-association"` by default, the MKDAChi2
  specificity map. `"z_desc-uniformity"` gives the forward-inference map instead.
- `meta_estimator` (CorrelationDecoder) — `MKDAChi2()` by default. Swapping in `ALE()`
  changes what "the term's map" means and is slower.

## Cost

`CorrelationDecoder.fit` runs a full meta-analysis per feature. Over Neurosynth's 3,228
terms this is hours to days. Mitigations:

- restrict `features`;
- `n_cores=-1`;
- fit once, pickle the decoder, reuse;
- `CorrelationDecoder.load_imgs(...)` to supply pre-generated meta-analytic maps and skip
  fitting entirely. This is how the published "123 Cognitive Atlas term maps" pipelines
  work: generate the maps once, reuse them everywhere.

## A gotcha in the Studyset API

`Studyset.annotations` is a **list** of annotation objects. The DataFrame you want is
`Studyset.annotations_df`. Feature columns look like
`terms_abstract_tfidf__working memory`.

## GC-LDA

```python
from nimare.annotate.gclda import GCLDAModel
from nimare.decode.continuous import gclda_decode_map
from nimare.decode.discrete import gclda_decode_roi
from nimare.decode.encode import gclda_encode     # text -> map
```

GC-LDA learns topics jointly over words and coordinates, so each topic has a spatial
distribution `p(topic|voxel)`. NiMARE's default map decoding takes the dot product
`p(word|image) = tau_t . p(word|topic)`. Training is expensive; most users take the
published Neurosynth topic sets instead (`vocab="LDA50"`, `"LDA100"`, `"LDA200"`,
`"LDA400"`).

`gclda_encode` goes the other way — text to a predicted brain map — which is the basis of
NeuroQuery-style synthesis.
