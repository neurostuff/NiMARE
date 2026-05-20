# Quickstart: Masked Activation Feature Dataset

This quickstart describes the intended public workflow for the feature. It is
used as planning guidance and later as a basis for Sphinx-Gallery examples.

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
    missing_coordinates="drop",
    descriptor_fields=[
        {"source": "metadata", "field": "sample_sizes", "kind": "numeric"},
        {"source": "annotations", "field": "Neurosynth_TFIDF__pain", "kind": "numeric"},
    ],
    target_field={
        "source": "annotations",
        "field": "Neurosynth_TFIDF__emotion",
        "kind": "numeric",
    },
)

train_sklearn, test_sklearn = extractor.to_sklearn(studyset)
```

Expected result:

- `train_sklearn.data` is analysis-by-feature data.
- `train_sklearn.target` is aligned to rows in `data`.
- `train_sklearn.groups` contains study IDs for grouped splitting.
- `test_sklearn` is either another Bunch or `None` when no split is requested.

`MAFeatureExtractor` provides `to_sklearn(studyset, ...)` as the one-call
public workflow and returns sklearn-ready `(train_bunch, test_bunch)` outputs.
`transform(studyset)` is the advanced workflow that returns
`(train_dataset, test_dataset)` for manual iteration over different reducers.
`MAFeatureExtractor` is not a trainable scikit-learn estimator and does not
expose `fit` or `fit_transform`.

Descriptor fields must be numeric by default. Categorical metadata, annotations,
titles, abstracts, and descriptions require an explicit transformer or
vectorizer before they can be appended to `data`.

Analyses without coordinates are controlled by `missing_coordinates`. Use
`"drop"` to remove them before row construction and record dropped IDs in
provenance, or `"include"` to keep them as all-zero sparse map rows.

## Use explicit preprocessing for non-numeric fields

```python
from sklearn.feature_extraction.text import TfidfVectorizer

extractor = ml.MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    descriptor_fields=[
        {"source": "texts", "field": "abstract", "kind": "text"},
    ],
    descriptor_transformers={
        ("texts", "abstract"): TfidfVectorizer(max_features=100),
    },
)
```

Raw free-text and multi-label fields are not valid targets unless the caller
provides an explicit target transformer or label extractor. Scalar numeric and
scalar categorical metadata or annotation targets can be exported as `y`.

## Split without study leakage

```python
train_data, test_data = feature_dataset.split(test_size=0.25, random_state=13)

assert set(train_data.study_ids).isdisjoint(test_data.study_ids)
```

The split must keep all analyses from one study in one partition.

## Reduce voxelwise map features

```python
variance_reducer = ml.make_map_reducer(method="variance_threshold", threshold=0.0)
svd_reducer = ml.make_map_reducer(method="truncated_svd", n_components=25, random_state=13)

train_reduced = train_data.apply_map_reducer(svd_reducer, fit=True)
test_reduced = test_data.apply_map_reducer(svd_reducer, fit=False)
```

The reducer is fit on training data only, then applied to held-out data.
Unreduced voxelwise map matrices remain sparse. A reducer such as truncated SVD
or atlas aggregation may return a smaller dense matrix only after reducing the
voxel space. Atlas or label aggregation requires a masker or labels image
compatible with the map feature space.

## Tests

Targeted tests for implementation:

```bash
python -m pytest nimare/tests/test_ml.py
```

Performance-sensitive verification for the clarified scale target:

```bash
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

Public API examples must be created as Sphinx-Gallery `.py` files:

```text
examples/05_machine_learning/01_plot_ma_feature_dataset.py
examples/05_machine_learning/02_plot_ma_feature_reduction.py
```

Validate docs conversion with existing infrastructure:

```bash
make -C docs html
```
