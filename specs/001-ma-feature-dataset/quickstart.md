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

extractor = ml.MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    descriptor_fields=["sample_sizes", ("annotations", "Neurosynth_TFIDF__pain")],
    target_field=("annotations", "Neurosynth_TFIDF__emotion"),
)

data = extractor.transform(studyset)     # an MAFeatureDataset
bunch = data.to_sklearn()                # or extractor.to_sklearn(studyset)
```

Expected result:

- `bunch.data` is the analysis-by-feature matrix, sparse while the map features
  are unreduced.
- `bunch.target` is aligned to the rows of `data`.
- `bunch.groups` holds the study each analysis came from.
- `bunch.feature_names`, `bunch.ids` and `bunch.provenance` describe them.

`extractor.to_sklearn(studyset, return_X_y=True)` returns `(X, y)` for callers
who want nothing else. `MAFeatureExtractor` is not a trainable scikit-learn
estimator and has no `fit`.

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

## Split without study leakage

```python
train, test = data.split(test_size=0.25, random_state=13)

assert set(train.study_ids).isdisjoint(test.study_ids)
```

`test_size` is a fraction of *studies*, so analysis counts only approximate it.
For cross-validation, hand `bunch.groups` to any scikit-learn group splitter.

## Reduce voxelwise map features

Inside a pipeline, which is what keeps the reducer fitted on training rows
only:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    data.make_preprocessor("truncated_svd", n_components=50, random_state=13),
    LogisticRegression(max_iter=1000),
)
scores = cross_val_score(
    pipeline, bunch.data, bunch.target, cv=GroupKFold(5), groups=bunch.groups
)
```

Or by hand, where the fitted reducer carries the fit:

```python
svd = ml.make_map_reducer("truncated_svd", n_components=25, random_state=13)

train_reduced = train.fit_transform_maps(svd)
test_reduced = test.transform_maps(svd)   # NotFittedError if svd is unfitted
```

`make_map_reducer` -- and `make_preprocessor`, which passes what it is given
straight through -- takes more than the three workflow names:

```python
from nilearn.datasets import fetch_atlas_difumo
from sklearn.random_projection import SparseRandomProjection

ml.make_map_reducer("variance_threshold", threshold=0.01)        # a named workflow
ml.make_map_reducer(SparseRandomProjection(n_components=64))     # any transformer
ml.make_map_reducer(SparseRandomProjection, n_components=64)     # or its class
ml.make_map_reducer(fetch_atlas_difumo(dimension=64), masker=data.masker)
ml.make_map_reducer("atlas_aggregation", masker=data.masker,
                    atlas="harvard_oxford",
                    atlas_kwargs={"atlas_name": "cort-maxprob-thr25-2mm"})

data.make_preprocessor(fetch_atlas_difumo(dimension=64))          # masker supplied
```

An atlas is anything nilearn can load: a fetched atlas, an atlas image or file,
the name of a `fetch_atlas_*` function, or a `NiftiLabelsMasker` or
`NiftiMapsMasker` you configured yourself. A 4D atlas is summarised with a maps
masker and a 3D one with a labels masker, and the atlas's own region names
become the reduced feature names.

Map features are sparse, so a reducer has to read sparse input. Truncated SVD,
sparse random projection, variance thresholding and atlas aggregation do; dense
PCA will ask for dense data.

## Use non-numeric fields

Categorical and text fields are not appended to the feature matrix, because
encoding them during extraction would fit the encoder on the analyses you are
about to hold out. Either encode the field yourself and select the numeric
result, or read the raw values from `data.descriptors` -- a DataFrame indexed by
analysis id -- and encode them inside your pipeline.

For a target, pass a label extractor:

```python
extractor = ml.MAFeatureExtractor(
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
examples/05_machine_learning/01_plot_ma_feature_dataset.py
examples/05_machine_learning/02_plot_ma_feature_reduction.py
```

```bash
make -C docs html
```
