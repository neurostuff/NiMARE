# Public API Contract: `nimare.ml`

This contract defines the additive public surface for converting NiMARE
collections into scikit-learn-compatible feature datasets. Names may be refined
during implementation only if the examples, tests, and API docs are updated
together.

## Module

`nimare.ml`

The module must be exported from `nimare/__init__.py` and documented in
`docs/api.rst`.

## Utility Preference Order

For every task, implementation must look for reusable functionality in this
order before adding local helpers:

1. Existing NiMARE utilities, classes, fixtures, and documentation conventions.
2. nilearn utilities for mask/image-aware operations.
3. scikit-learn utilities for dataset containers, splitters, preprocessing,
   decomposition, and pipelines.
4. New local helpers only when none of the above provides the needed behavior.

## `MAFeatureDataset`

Container for aligned map features, descriptor features, targets, groups, and
provenance.

### Required attributes

- `map_features`: sample-by-map-feature matrix. Sparse input must remain sparse
  unless the caller explicitly requests dense output.
- `sample_ids`: one identifier per row.
- `study_ids`: one study-group label per row.
- `sample_metadata`: tabular provenance containing at least sample ID, study ID,
  analysis ID when available, and exclusion status where relevant.
- `masker`: masker defining voxel feature order for unreduced map features.
- `descriptor_features`: optional sample-aligned descriptor values.
- `target`: optional sample-aligned prediction target.
- `feature_names`: names for exported features.
- `exclusion_report`: excluded analysis IDs and reasons.
- `provenance`: map-generation settings and source collection details.

### Required methods

- `to_sklearn(include_descriptors=True, include_target=True, dense=False)`:
  return a `sklearn.utils.Bunch`-compatible dataset object with `data`,
  `target`, `groups`, `sample_metadata`, and `feature_names`. Sparse map
  features remain sparse unless `dense=True`.
- `split(test_size=0.25, random_state=None, cv=None)`: return train/test
  dataset slices using grouped splitting by study ID.
- `apply_map_reducer(reducer, fit=False)`: return a dataset copy with
  map-derived features transformed by a scikit-learn-compatible reducer while
  preserving descriptors, target, sample metadata, and groups.
- `get_feature_names()`: return exported feature names in data-column order.
- `copy()`: return an independent dataset copy.

### Errors

- Raise `ValueError` when feature, target, group, or sample dimensions do not
  align.
- Raise `ValueError` when study groups are missing or ambiguous.
- Raise `ValueError` when a split cannot be created from the available number of
  study groups.
- Raise `ValueError` when a reducer changes sample order or returns a row count
  that does not match the input dataset.

## `MAFeatureExtractor`

Creates `MAFeatureDataset` from Studyset or Dataset inputs.

### Construction

Required parameters:

- `kernel_transformer`: existing NiMARE kernel transformer instance or class.
  No implicit scientific default is selected; public examples must pass an
  explicit kernel transformer.

Optional parameters:

- `descriptor_fields`: list of field selectors from metadata, annotations, or
  texts.
- `descriptor_transformers`: optional mapping from descriptor field selectors to
  explicit transformers or vectorizers for non-numeric descriptor fields.
- `target_field`: optional field selector for `y`.
- `target_transformer`: optional transformer or label extractor for free-text or
  multi-label targets.
- `missing`: one of `raise`, `drop`, or explicit future strategies documented in
  the implementation.
- `memory` and `memory_level`: passed through to existing NiMARE-compatible
  caching when applicable.

### Required methods

- `fit(collection)`: validate the collection, resolve fields, and record
  feature schema.
- `transform(collection)`: produce an `MAFeatureDataset` using the fitted schema.
- `fit_transform(collection)`: fit and transform in one call.

### Required behavior

- Prefer Studyset-native access when input is a Studyset.
- Preserve legacy Dataset compatibility without changing existing Dataset
  behavior.
- Generate MA features through `KernelTransformer.transform(..., return_type="sparse")`.
- Align map rows to sample IDs and report excluded analyses.
- Determine study groups from explicit collection study IDs when available,
  derive from analysis IDs only when unambiguous, and fail clearly otherwise.
- Append numeric descriptor fields directly.
- Reject non-numeric descriptor fields by default unless an explicit descriptor
  transformer or vectorizer is supplied.
- Export scalar numeric and scalar categorical targets as one-dimensional `y`.
- Reject raw free-text and multi-label targets unless an explicit target
  transformer or label extractor is supplied.
- Never split analyses from the same study across train/test partitions.

### Errors and diagnostics

- Missing maps, missing fields, incompatible masks/spaces, insufficient study
  groups, and unusable targets must raise clear exceptions or produce explicit
  reports according to the selected missing-value policy.

## Field Selectors

Selectors identify descriptor or target fields.

Minimum supported selector fields:

- `source`: `metadata`, `annotations`, or `texts`.
- `field`: field name within the selected source.
- `kind`: optional hint such as `numeric`, `categorical`, or `text`.
- `transformer`: optional explicit transformer or vectorizer for non-numeric
  descriptor fields or unsupported target shapes.

Selectors may be dictionaries or lightweight objects, but public examples must
show the most concise supported form.

Default field behavior:

- Descriptor fields must resolve to numeric values unless an explicit
  transformer or vectorizer is supplied.
- Target fields may resolve to scalar numeric or scalar categorical values.
- Raw title, abstract, description, and other free-text fields require explicit
  transformation before they can become descriptor features or targets.
- Multi-label targets require an explicit label extractor or transformer.

## Reduction Helpers

The module must expose convenience constructors for map feature reduction.

Required behavior:

- Return scikit-learn-compatible transformers or pipelines.
- Fit only on training data before transforming held-out data.
- Preserve sample order and study group alignment.
- Keep descriptor features and targets aligned with reduced map features.

Minimum workflows:

- Variance thresholding for sparse or dense map matrices.
- Matrix-appropriate decomposition using PCA for dense matrices and
  `TruncatedSVD` for sparse matrices.
- Atlas or label aggregation when a nilearn-compatible labels image, atlas, or
  masker is supplied and compatible with the map feature space.

Required helper:

- `make_map_reducer(method, **kwargs)`: return a scikit-learn-compatible
  transformer or pipeline for `method` values covering `variance_threshold`,
  `pca`, `truncated_svd`, and `atlas_aggregation` or equivalent documented
  names.

## Documentation Contract

Required public examples:

- `examples/05_machine_learning/01_plot_ma_feature_dataset.py`
- `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

Required docs:

- API autosummary entry in `docs/api.rst`.
- Numpydoc docstrings for all public classes and functions.
- Documentation of the 1,000-study conversion-and-split performance target:
  <=3 minutes and <=5 GB peak memory in the standard development environment.
