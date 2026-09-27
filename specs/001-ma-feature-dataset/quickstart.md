# Quickstart: Modeled Activation Feature Dataset

The public workflow for `nimare.ml`. **Revised 2026-09-25** alongside
`contracts/public-api.md`.

## Setup

Install NiMARE with test and documentation dependencies in an isolated
environment:

```bash
python -m pip install -e .[tests,doc]
```

## Convert a Studyset to feature data

```python
bunch = studyset.to_bunch(
    descriptor_fields=["sample_sizes", ("annotations", "Neurosynth_TFIDF__pain")],
    target_field=("annotations", "Neurosynth_TFIDF__emotion"),
)
```

Expected result:

- `bunch.data` is the analysis-by-feature matrix, sparse. Its voxel columns are
  peak counts over the whole image grid of `bunch.masker`; `nimare.ml.MAKernel`
  turns them into MA maps inside the pipeline.
- `bunch.target` is aligned to the rows of `data`.
- `bunch.groups` holds the study each analysis came from.
- `bunch.feature_names`, `bunch.ids` and `bunch.provenance` describe them.

- `bunch.voxel_columns`, `bunch.descriptor_columns`, `bunch.descriptor_names` and
  `bunch.masker` say which columns are voxels and where they came from.
- `bunch.descriptor_categories` says what a categorical descriptor's codes mean.
  A numeric matrix cannot hold a string, so those columns hold the position of a
  category; the encoder is yours to pick, in the pipeline:
  `(OneHotEncoder(), "group_name")`. Passing a code through, or scaling it
  alongside a real number, is refused.

There is no container class, no estimator to configure and no `fit` to call.
Everything after conversion is scikit-learn on ordinary arrays.

A field is named by a bare field name, or by a `(source, field)` tuple. A bare
name is looked up in metadata, annotations and texts in turn, and an ambiguous
one asks which was meant, so the tuple is for the case a bare name cannot
express. Numeric metadata is read the way the
rest of NiMARE reads it, so study-level fields are inherited by their analyses
and `sample_sizes` is reduced rather than rejected.

Analyses without coordinates follow `missing_coordinates`: `"drop"` (the
default) removes them and records their ids in provenance, `"include"` keeps
them as all-zero sparse map rows. Missing descriptor and target values follow
`missing_values`: `"raise"` (the default) names the fields and analyses,
`"drop"` removes those analyses, `"keep"` leaves them for a pipeline to impute.

A mapping sets that per role, since a descriptor gap can be imputed and a
target gap cannot:

```python
missing_values={"target": "drop", "descriptors": "keep"}
```

Pass `memory="/path/to/cache"` to have repeated conversions of the same
Studyset reuse the maps they already generated, in this process and the next.

## Select rows

Select rows on the Studyset, before conversion, since that is the expensive
step:

```python
studyset.slice(["study_0-task0", "study_1-task0"])       # analysis ids
studyset.select_analyses(mask_or_positions)              # mask or positions
```

## Split without study leakage

```python
bunch = studyset.to_bunch(test_size=0.25, random_state=13)

assert set(bunch.groups[bunch.train]).isdisjoint(bunch.groups[bunch.test])
```

`test_size` adds `train` and `test` row positions, grouped by study. Without it
both keys are absent. For several splits, or for cross-validation, pass
`bunch.groups` to a scikit-learn group splitter instead of converting again.

`test_size` is a fraction of *studies*, so analysis counts only approximate it.
For cross-validation, hand `bunch.groups` to any scikit-learn group splitter.

## Build and reduce the voxelwise features

`MAKernel` convolves the peaks into MA maps, and ordinary scikit-learn
transformers reduce them. Both go in the pipeline, which is what keeps every
step that learns across rows fitted on training rows only:

```python
from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAKernel
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    MAKernel(MKDAKernel(r=10), source_masker=bunch.masker),
    TruncatedSVD(n_components=50, random_state=13),
    LogisticRegression(max_iter=1000),
)
scores = cross_val_score(
    pipeline, bunch.data, bunch.target, cv=GroupKFold(5), groups=bunch.groups
)
```

Or by hand, where the fitted reducer carries the fit:

```python
svd = TruncatedSVD(n_components=25, random_state=13)
maps = MAKernel(MKDAKernel(r=10), source_masker=bunch.masker).fit_transform(
    bunch.data[:, bunch.voxel_columns]
)

train_reduced = svd.fit_transform(maps[train])
test_reduced = svd.transform(maps[test])
```

The kernel's own parameters tune like any other, as
`makernel__kernel__r` in a `GridSearchCV`. Skipping it is a feature set too:
an atlas applied straight to the peak columns counts reported coordinates per
region.

Anything that reads sparse input works: truncated SVD, sparse random
projection, variance thresholding. `PCA` takes sparse input too, but only with
its `arpack` or `covariance_eigh` solvers, and it centres, so `TruncatedSVD`
suits a matrix this wide better.

## Find the fields worth using

A release-scale Studyset offers more fields than anyone can read. The 2026-09
NeuroStore release has 76 metadata columns and 924 annotation labels, 875 of
which fewer than one analysis in a hundred reports. `describe_fields` reports
what each one holds, using the reader `from_studyset` uses:

```python
from nimare.extract import fetch_neurostore

studyset = fetch_neurostore()                      # 115,748 analyses, 32,444 studies
fields = ml.describe_fields(studyset, min_coverage=0.5)   # 997 fields -> 29

fields[fields.n_unique.between(2, 12)]             # classification targets
fields[fields.kind == "numeric"].field             # a descriptor_fields list
```

Columns are `source`, `field`, `kind`, `coverage`, `n_unique` and `example`,
ordered by coverage. A field it calls numeric is numeric to `from_studyset`,
because both read it the same way.

Conversion is linear in analyses and an MA row is denser than a Studyset row
(~4,700 non-zeros at a 10 mm radius), so the whole release is roughly 6 GB of
sparse data and does not convert on a 16 GB machine. Slice first:

```python
subset = studyset.slice(analyses=list(studyset.ids)[:4000])
```

## Select annotation labels

An annotation is thousands of mostly-empty columns, so naming labels one at a
time is not a workflow. A glob pattern takes them all, under their own names,
and keeps the block sparse:

```python
bunch = studyset.to_bunch(
    descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")],
    target_field=("annotations", "Neurosynth_TFIDF__pain"),
)

bunch.descriptor_names[:2]   # ['Neurosynth_TFIDF__001', 'Neurosynth_TFIDF__01']
```

A label no analysis carries is a zero rather than a gap, so `missing_values`
has nothing to report about a pattern selection; an exactly named field keeps
the usual value semantics, where absent means missing.

A pattern's brackets are an `fnmatch` character class, which is not what they
mean in an extractor's repeated fields. A pattern matching nothing is retried
with its brackets taken literally, so `*groups[0].*` selects group zero;
`[mt]*`, which already matches, still means a character class.

## Let nilearn transform the voxels

A nilearn masker is already a scikit-learn transformer; it just takes images
where a ColumnTransformer hands out columns of an array. `MaskerTransformer`
is that bridge, and the only transformer NiMARE adds, because it is the only
one that has to know which voxel each column is:

```python
from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMasker

ml.MaskerTransformer(fetch_atlas_difumo(dimension=64), source_masker=bunch.masker)
ml.MaskerTransformer(NiftiMasker(smoothing_fwhm=6), source_masker=bunch.masker)
ml.MaskerTransformer("harvard_oxford", source_masker=bunch.masker,
                     masker_kwargs={"atlas_name": "cort-maxprob-thr25-2mm"})

# in a column transformer the source masker comes from the bundle
ml.make_nimare_column_transformer(bunch, (fetch_atlas_difumo(dimension=64), "maps"))
```

It applies any nilearn masker, or anything nilearn loads as an atlas: a fetched
atlas, an atlas image or file, or the name of a `fetch_atlas_*` function. A 4D
atlas is summarised with a maps masker and a 3D one with a labels masker, and
the atlas's own region names become the feature names. A `NiftiMasker` returns
voxels rather than regions, which is how nilearn's smoothing, standardizing and
detrending reach these features.

## Keep a reducer off the descriptor columns

With map features alone there is nothing to keep a reducer away from, so a
transformer goes straight into the pipeline. Once descriptor columns are there,
they need separate treatment, which is what
[`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
is for. `make_nimare_column_transformer` builds one with the column boundary filled in, the
masker bound into an atlas reducer, and `sparse_threshold=1.0` so a wide sparse
map block is never quietly densified:

```python
from sklearn.impute import SimpleImputer

pipeline = make_pipeline(
    ml.make_nimare_column_transformer(
        bunch,
        (TruncatedSVD(n_components=50, random_state=13), "maps"),
        (SimpleImputer(strategy="median"), "descriptors"),
    ),
    LogisticRegression(max_iter=1000),
)
```

It is `sklearn.compose.make_column_transformer` with the bundle filled in: the
same `(transformer, columns)` pairs, the same step names, and `remainder`,
`sparse_threshold`, `n_jobs`, `verbose` and `verbose_feature_names_out` passed
through. A descriptor may be named by its own field name, so treating them
differently is just more pairs:

```python
from sklearn.preprocessing import StandardScaler

ml.make_nimare_column_transformer(
    bunch,
    (TruncatedSVD(n_components=50, random_state=13), "maps"),
    (SimpleImputer(strategy="median"), "sample_sizes"),
    (StandardScaler(), "year"),
)
```

Leaving a descriptor unclaimed under the default `remainder="drop"` raises
rather than discarding it; say `("drop", "descriptors")` or
`remainder="passthrough"` to mean it.

Descriptors the mapping does not name are passed through, and the columns come
out in the order they went in. `bunch.descriptor_names` lists them.
Transformers are handed the descriptor columns dense, which is what most of
them expect of a few numeric columns -- `StandardScaler` will not centre sparse
data at all -- while the map block stays sparse.

`bunch.voxel_columns` and `bunch.descriptor_columns` are right there, so the
same thing can be written out:

```python
from sklearn.compose import ColumnTransformer

ColumnTransformer(
    [
        ("maps", TruncatedSVD(n_components=50), bunch.voxel_columns),
        ("descriptors", SimpleImputer(), bunch.descriptor_columns),
    ],
    sparse_threshold=1.0,
)
```

With no descriptor columns, `make_nimare_column_transformer` hands the reducer straight
back.

## Use non-numeric fields

Categorical and text fields are not appended to the feature matrix, because
encoding them during extraction would fit the encoder on the analyses you are
about to hold out. Either encode the field yourself and select the numeric
result, or read the raw values from the Studyset's own tables -- indexed by
analysis id -- and encode them inside your pipeline.

For a target, pass a label extractor:

```python
bunch = studyset.to_bunch(
    target_field=("texts", "abstract"),
    target_transformer=lambda texts: [classify(text) for text in texts],
)
```

## Tests

```bash
python -m pytest nimare/tests/test_ml.py
python -m pytest -m performance_smoke nimare/tests/test_ml.py
```

Broader verification before review:

```bash
PYTEST_UNIT_MARKERS="not performance_estimators and not performance_correctors"
PYTEST_UNIT_MARKERS="$PYTEST_UNIT_MARKERS and not performance_smoke and not cbmr_importerror"
python -m pytest -m "$PYTEST_UNIT_MARKERS" --cov-append --cov-report=xml --cov=nimare nimare
make lint
```

## Documentation and examples

```text
examples/05_machine_learning/01_plot_machine_learning_in_nimare.py
```

```bash
make -C docs html
```
