---
name: nimare-functional-decoding
description: Meta-analytic functional decoding of a brain map, ROI, cluster, gradient, or network with NiMARE — "what cognitive functions is this region associated with?". Use when the user wants to annotate their own statistical map or ROI against Neurosynth, NeuroQuery, or BrainMap terms or topics; mentions reverse inference, forward inference, term-based or topic-based decoding, CorrelationDecoder, ROIAssociationDecoder, NeurosynthDecoder, BrainMapDecoder, LDA or GC-LDA topics, or word clouds of cognitive terms; or asks which Neurosynth terms a map resembles.
license: MIT
---

# Meta-analytic functional decoding

This is what most people use NiMARE for. In a survey of 98 NiMARE-citing articles, 54
used it only for this — to attach cognitive terms to a map they produced by other means.
It is an annotation technique, not a meta-analysis, and it is weaker evidence than its
presentation usually implies.

## Pick the decoder from the input

| Input | Decoder | What it computes |
|---|---|---|
| Unthresholded continuous map (z, t, gradient, saliency, eccentricity) | `CorrelationDecoder` | Pearson r between your map and each term's meta-analytic map. |
| Binary ROI mask, moderate-to-large | `ROIAssociationDecoder` | Correlates mean modelled activation in the ROI with each term's weights. |
| Binary ROI / cluster, chi-square framing | `NeurosynthDecoder` | Per-term forward (`P(activation|term)`) and reverse (`P(term|activation)`) inference with a chi-square test and FDR. |
| Small seed ROI, BrainMap-style database | `BrainMapDecoder` | Same two inferences, but accounts for the number of foci per study — the right choice for small spherical seeds. |
| A distribution of maps | `CorrelationDistributionDecoder` | Correlations against each study's map rather than the term map. |

The `NeurosynthDecoder` vs `BrainMapDecoder` distinction is real and routinely got wrong:
BrainMap's algorithm normalizes by foci count, so it suits small seeds; Neurosynth's does
not, so it suits larger ROIs and whole clusters. Published NiMARE analyses have chosen on
exactly this basis.

## Minimal, correct usage

```python
from nimare.extract import fetch_neurosynth
from nimare.decode.continuous import CorrelationDecoder

studyset = fetch_neurosynth(data_dir="data/", version="7", vocab="terms")

decoder = CorrelationDecoder(
    frequency_threshold=0.001,           # drop studies that use the term incidentally
    feature_group="terms_abstract_tfidf",
    features=my_term_list,               # restrict! see below
    target_image="z_desc-association",   # MKDAChi2 specificity map
    n_cores=-1,
)
decoder.fit(studyset)                        # returns None; mutates the decoder
table = decoder.transform("my_map.nii.gz")   # term -> r, ranked
```

Fitting over the full 3,228-term vocabulary runs a MKDAChi2 meta-analysis per term and
takes hours. Restrict `features` first, cache the fitted decoder, or use
`CorrelationDecoder.load_imgs(...)` with pre-generated maps.

For an ROI:

```python
from nimare.decode.discrete import ROIAssociationDecoder, NeurosynthDecoder

roi_dec = ROIAssociationDecoder(masker=roi_img, features=my_term_list)
roi_dec.fit(studyset)              # fit() returns None -- do not chain
table = roi_dec.transform()        # takes no arguments

ns_dec = NeurosynthDecoder(correction="bh", frequency_threshold=0.001,
                           features=my_term_list, min_studies=1)
ns_dec.fit(studyset)
ids = studyset.get_studies_by_mask(roi_img)   # studies reporting a focus in the ROI
table = ns_dec.transform(ids=ids)             # takes the selected ids
```

`correction` is `"bh"` (Benjamini-Hochberg) or `None`, not `"fdr_bh"`.
`BrainMapDecoder` has the same interface minus `prior` and `min_studies`.

## The four things that make a decoding result defensible

### 1. Restrict the term list, and say how

The raw vocabulary is full of anatomy (`insula`), method (`fmri`, `voxel`), and
non-words. Almost every published analysis filters, and the near-universal convention is
the **intersection of Neurosynth with the Cognitive Atlas** — the ~123-125 terms that
name mental processes. Others use the LDA topic sets (`v5-topics-50`, `v7-topics-400`)
and hand-label them. Either is fine. Silently hand-dropping terms after seeing the result
is not; it is the step where confirmation bias enters.

```python
from nimare.extract import download_cognitive_atlas
cogat = download_cognitive_atlas()
```

### 2. Do not report p-values from correlations

NiMARE's `CorrelationDecoder` carries its own warning:

> Coefficients from correlating two maps have very large degrees of freedom, so almost
> all results will be statistically significant. Do not attempt to evaluate results
> based on significance.

Voxels are massively spatially autocorrelated, so the nominal df is wrong by orders of
magnitude. Report **ranked** associations, or test against a spatial-autocorrelation-
preserving null (spin test, variogram surrogates). See `references/null-models.md`.
The chi-square decoders (`NeurosynthDecoder`, `BrainMapDecoder`) *do* produce meaningful
tests, over studies rather than voxels, with FDR correction — prefer them when you need
an inferential claim.

### 3. Decode the unthresholded map

Correlating a thresholded map against unthresholded term maps mixes a thresholding
decision into the similarity. Use the unthresholded statistic map with
`CorrelationDecoder`; use a binary mask with the ROI decoders. Several surveyed papers
bin an unthresholded gradient into percentile masks and decode each bin — a good pattern
for ordered maps.

### 4. Say which direction of inference you mean

- *Forward* / uniformity / consistency: `P(activation | term)` — given studies about X,
  does this region activate?
- *Reverse* / association / specificity: `P(term | activation)` — given activation here,
  is it selectively studies about X?

Neurosynth's current names are **uniformity** and **association** (formerly consistency
and specificity, or forward and reverse inference). `MKDAChi2` emits both:
`z_desc-uniformity` and `z_desc-association`. The association map is the one to decode
against; the uniformity map is not evidence of selectivity. Calling a correlation with
either one "reverse inference" in the Poldrack sense is a stretch — it is a spatial
similarity to a term-based meta-analytic map.

## Surface and parcellated data

NiMARE works in volumetric MNI152 and relies on nilearn maskers, which do not transform
surface to volume. The established pattern is: compute term maps in volume with
`MKDAChi2`, project to `fsLR` with `neuromaps`, parcellate (HCP-MMP, Schaefer), then
correlate in parcel space. BrainStat does the reverse (interpolates a surface map into
the cortical ribbon, then queries NiMARE). Both are documented workarounds, not native
support — state which you used.

## Topics instead of terms

```python
studyset = fetch_neurosynth(data_dir="data/", version="7", vocab="LDA400")
```

LDA topic sets (50/100/200/400) are less noisy than raw terms and need labelling. GC-LDA
(`nimare.annotate.gclda.GCLDAModel`, `nimare.decode.continuous.gclda_decode_map`,
`nimare.decode.discrete.gclda_decode_roi`) adds a spatial prior so topics have brain
distributions. `nimare.annotate.lda.LDAModel` trains your own topics over a corpus
fetched with `download_abstracts`.

## How to write it up

"Meta-analytic decoding indicated the map was most strongly associated with X, Y, Z"
overstates it. Better: "the map's spatial pattern was most similar to meta-analytic maps
for the terms X, Y and Z (r = .21, .17, .12 of 123 Cognitive-Atlas terms tested;
rankings, not hypothesis tests)". If you ran a spin test, say which null and how many
surrogates.

## References

- `references/decoder-selection.md` — decoder-by-decoder parameters and outputs.
- `references/null-models.md` — why correlation p-values fail and what to do instead.
