# Public API Contract: `nimare.ml`

This contract defines the additive public surface for converting NiMARE
Studysets into scikit-learn-compatible feature datasets. Names may be refined
during implementation only if the examples, tests, and API docs are updated
together.

Terminology: each dataset row is an analysis row.

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

- `ids`: one full Studyset analysis identifier per row, using the
  Studyset `<study_id>-<analysis_id>` convention.
- `study_ids`: one study-group label per row.
- `features`: analysis-by-feature matrix combining map features (sparse
  voxelwise) and optional descriptor features. Unreduced voxelwise features
  must remain sparse. Reduced map features may be dense only when an explicit
  reducer returns a reduced dense representation.
- `feature_names`: names for features in `features` column order, covering both
  voxel features and descriptor features when present.
- `target`: optional analysis-row-aligned prediction target.
- `provenance`: map-generation settings and source Studyset details, including
  `missing_coordinates` and any `dropped_ids`.

### Required methods

- `to_sklearn()`: return a `sklearn.utils.Bunch`-compatible dataset object with
  `data` (same as `features`), `target` (or `None` if not extracted), `groups`
  (same as `study_ids`), and `feature_names`. Unreduced voxelwise feature data
  must remain sparse; reduced feature data may be dense only if an explicit
  reducer produced the reduced dense representation.
- `split(test_size=0.25, random_state=None, cv=None)`: return train/test
  dataset slices using grouped splitting by study ID while preserving row
  alignment across `features`, `target`, `ids`, `study_ids`, and provenance.
- `apply_map_reducer(reducer, fit=False)`: return a dataset copy with
  map-derived features transformed by a scikit-learn-compatible reducer while
  preserving descriptor features, target, `ids`, `study_ids`, and provenance.
- `copy()`: return an independent dataset copy.

### Errors

- Raise `ValueError` when feature, target, group, or row dimensions do not
  align.
- Raise `ValueError` when study groups required for splitting are missing.
- Raise `ValueError` when a split cannot be created from the available number of
  study groups.
- Raise `ValueError` when a reducer changes analysis-row order or returns a row count
  that does not match the input dataset.

## `MAFeatureExtractor`

Orchestrates the full conversion pipeline from Studyset to train/test feature
datasets and sklearn-ready exports. `MAFeatureExtractor` is a NiMARE conversion
helper, not a trainable scikit-learn estimator.

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
- `missing_coordinates`: either `include` or `drop`, defaulting to `drop`.
  `drop` removes analyses without coordinates before row construction and
  records dropped IDs in provenance; `include` retains them as all-zero sparse
  map rows.
- `test_size`: optional float (0 to 1) or int for train/test split; `None`
  (default) means no split, returns full dataset as train and `None` as test.
- `random_state`: random seed for reproducible splits.
- `cache_maps`: boolean flag controlling extractor-level caching of generated
  MA map features across repeated calls; default `True`.
- `memory` and `memory_level`: passed through to existing NiMARE-compatible
  caching when applicable.

### Required methods

- `transform(studyset)`: validate the Studyset, orchestrate extraction and
  optional splitting, and return `(train_dataset, test_dataset)` for users who
  prefer dataset-level iterative workflows.
- `to_sklearn(studyset, map_reducer=None, map_reducer_params=None)`: run the
  full public pipeline convenience wrapper and return sklearn-ready exports as
  `(train_bunch, test_bunch)`. If `map_reducer` is provided, fit on training
  map features only (when split) and apply to both train and test. If
  `test_size` is `None` or `0.0`, return `(full_bunch, None)`.

The initial public API must not expose `fit` or `fit_transform` on
`MAFeatureExtractor`.

### Required behavior

- Use Studyset-native access for IDs, coordinates, masker, metadata,
  annotations, and texts.
- Generate MA features through `KernelTransformer.transform(..., return_type="sparse")`.
- Align map rows to analysis IDs after applying `missing_coordinates`.
- When `missing_coordinates="include"`, analyses with no coordinates must be
  represented as all-zero sparse map rows.
- When `missing_coordinates="drop"`, analyses with no coordinates must be
  removed before row construction and recorded in provenance.
- Determine study groups from Studyset-provided study IDs. MVP inputs are
  assumed to provide unique study IDs and unique analysis IDs.
- If `test_size` is set: use grouped train/test split by study ID to prevent
  study leakage.
- Append numeric descriptor fields directly.
- Reject non-numeric descriptor fields by default unless an explicit descriptor
  transformer or vectorizer is supplied.
- When a split is requested and descriptor transformers are provided, fit all
  descriptor transformers on training data, then apply to train and test
  separately.
- Export scalar numeric and scalar categorical targets as one-dimensional `y`.
- Reject raw free-text and multi-label targets unless an explicit target
  transformer or label extractor is supplied.
- If `to_sklearn(..., map_reducer=...)` is used: fit the reducer on training
  map features only (when split), then apply to both train and test data. If
  no split, fit and apply to the full dataset.
- Never split analyses from the same study across train/test partitions.
- Repeated reducer calls with unchanged Studyset and extractor settings must
  reuse cached MA map features and avoid recomputing kernel maps.

### Errors and diagnostics

- Missing maps, missing fields, incompatible masks/spaces, insufficient study
  groups, and unusable targets must raise clear exceptions or produce explicit
  reports. Coordinate-less analyses follow `missing_coordinates`; other invalid
  map rows must fail clearly.

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
- Preserve analysis-row order and study group alignment.
- Keep descriptor features and targets aligned with reduced map features.

Minimum workflows:

- Variance thresholding for sparse map matrices.
- Sparse-compatible low-rank decomposition using `TruncatedSVD` or an
  equivalent sparse-safe transformer.
- Atlas or label aggregation when a nilearn-compatible labels image, atlas, or
  masker is supplied and compatible with the map feature space.

Required helper:

- `make_map_reducer(method, **kwargs)`: return a scikit-learn-compatible
  transformer or pipeline for `method` values covering `variance_threshold`,
  `truncated_svd`, and `atlas_aggregation` or equivalent documented names.

## Documentation Contract

Required public examples:

- `examples/05_machine_learning/01_plot_ma_feature_dataset.py`
- `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

Required docs:

- API autosummary entry in `docs/api.rst`.
- Numpydoc docstrings for all public classes and functions.
- Documentation of the 1,000-study conversion-and-split performance target:
  <=3 minutes and <=5 GB peak memory in the standard development environment.
