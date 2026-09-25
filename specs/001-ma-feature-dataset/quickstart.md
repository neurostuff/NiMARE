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
from nimare import ml
from nimare.meta.kernel import MKDAKernel

features = ml.FeatureSet.from_studyset(
    studyset,
    kernel_transformer=MKDAKernel(r=10),
    descriptor_fields=["sample_sizes", ("annotations", "Neurosynth_TFIDF__pain")],
    target_field=("annotations", "Neurosynth_TFIDF__emotion"),
)                                        # a FeatureSet

bunch = features.to_sklearn()
```

Expected result:

- `bunch.data` is the analysis-by-feature matrix, sparse while the map features
  are unreduced.
- `bunch.target` is aligned to the rows of `data`.
- `bunch.groups` holds the study each analysis came from.
- `bunch.feature_names`, `bunch.ids` and `bunch.provenance` describe them.

`features.to_sklearn(return_X_y=True)` returns `(X, y)` for callers who want
nothing else. There is no estimator to configure and no `fit` to call: the container builds
itself from a Studyset, and that container is what you work with.

A field is named by a bare field name, by a `(source, field)` tuple, or by a
mapping. A bare name is looked up in metadata, annotations and texts in turn,
and an ambiguous one asks which was meant. Numeric metadata is read the way the
rest of NiMARE reads it, so study-level fields are inherited by their analyses
and `sample_sizes` is reduced rather than rejected.

Analyses without coordinates follow `missing_coordinates`: `"drop"` (the
default) removes them and records their ids in provenance, `"include"` keeps
them as all-zero sparse map rows. Missing descriptor and target values follow
`missing_values`: `"raise"` (the default) names the fields and analyses,
`"drop"` removes those analyses, `"keep"` leaves them for a pipeline to impute.

Pass `memory="/path/to/cache"` to have repeated conversions of the same
Studyset reuse the maps they already generated, in this process and the next.

## Split without study leakage

```python
train, test = features.split(test_size=0.25, random_state=13)

assert set(train.study_ids).isdisjoint(test.study_ids)
```

`test_size` is a fraction of *studies*, so analysis counts only approximate it.
For cross-validation, hand `bunch.groups` to any scikit-learn group splitter.

## Reduce voxelwise map features

Map features are an ordinary sparse matrix, so ordinary scikit-learn
transformers reduce them. Inside a pipeline, which is what keeps the reducer
fitted on training rows only:

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
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

train_reduced = train.fit_transform_maps(svd)
test_reduced = test.transform_maps(svd)   # NotFittedError if svd is unfitted
```

Anything that reads sparse input works: truncated SVD, sparse random
projection, variance thresholding. Dense PCA will ask for dense data.

## Reduce over the regions of an atlas

`AtlasAggregator` is the one reducer NiMARE adds, because it is the one that
has to know which voxel each column is:

```python
from nilearn.datasets import fetch_atlas_difumo

ml.AtlasAggregator(fetch_atlas_difumo(dimension=64), masker=features.masker)
ml.AtlasAggregator("harvard_oxford", masker=features.masker,
                   atlas_kwargs={"atlas_name": "cort-maxprob-thr25-2mm"})

features.make_preprocessor(fetch_atlas_difumo(dimension=64))   # masker supplied
```

An atlas is anything nilearn can load: a fetched atlas, an atlas image or file,
the name of a `fetch_atlas_*` function, or a `NiftiLabelsMasker` or
`NiftiMapsMasker` you configured yourself. A 4D atlas is summarised with a maps
masker and a 3D one with a labels masker, and the atlas's own region names
become the reduced feature names.

## Keep a reducer off the descriptor columns

With map features alone there is nothing to keep a reducer away from, so a
transformer goes straight into the pipeline. Once descriptor columns are there,
they need separate treatment, which is what
[`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
is for. `make_preprocessor` builds one with the column boundary filled in, the
masker bound into an atlas reducer, and `sparse_threshold=1.0` so a wide sparse
map block is never quietly densified:

```python
from sklearn.impute import SimpleImputer

pipeline = make_pipeline(
    features.make_preprocessor(
        TruncatedSVD(n_components=50, random_state=13),
        descriptor_transformer=SimpleImputer(strategy="median"),
    ),
    LogisticRegression(max_iter=1000),
)
```

That applies one transformer to every descriptor. When they want different
treatment, pass a mapping from descriptor name to transformer:

```python
from sklearn.preprocessing import StandardScaler

features.make_preprocessor(
    TruncatedSVD(n_components=50, random_state=13),
    descriptor_transformer={
        "sample_sizes": SimpleImputer(strategy="median"),
        "year": StandardScaler(),
    },
)
```

Descriptors the mapping does not name are passed through, and the columns come
out in the order they went in. `features.descriptor_names` lists them.
Transformers are handed the descriptor columns dense, which is what most of
them expect of a few numeric columns -- `StandardScaler` will not centre sparse
data at all -- while the map block stays sparse.

`features.map_columns` and `features.descriptor_columns` are public, so the
same thing can be written out:

```python
from sklearn.compose import ColumnTransformer

ColumnTransformer(
    [
        ("maps", TruncatedSVD(n_components=50), features.map_columns),
        ("descriptors", SimpleImputer(), features.descriptor_columns),
    ],
    sparse_threshold=1.0,
)
```

With no descriptor columns, `make_preprocessor` hands the reducer straight
back.

## Use non-numeric fields

Categorical and text fields are not appended to the feature matrix, because
encoding them during extraction would fit the encoder on the analyses you are
about to hold out. Either encode the field yourself and select the numeric
result, or read the raw values from `features.descriptors` -- a DataFrame indexed by
analysis id -- and encode them inside your pipeline.

For a target, pass a label extractor:

```python
features = ml.FeatureSet.from_studyset(
    studyset,
    kernel_transformer=MKDAKernel(r=10),
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
