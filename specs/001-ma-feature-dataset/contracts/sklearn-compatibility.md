# Compatibility Contract: Scikit-Learn Dataset Use

This contract defines the shape and behavior expected by downstream
scikit-learn workflows.

## Dataset Export

`MAFeatureDataset.to_sklearn()` must return an object with:

- `data`: two-dimensional analysis-by-feature matrix accepted by scikit-learn
  estimators; same as `MAFeatureDataset.features`.
- `target`: one-dimensional target array or `None`; same as
  `MAFeatureDataset.target`.
- `groups`: one-dimensional study group array aligned to `data` rows; same as
  `MAFeatureDataset.study_ids`.
- `feature_names`: feature names aligned to `data` columns when available; same
  as `MAFeatureDataset.feature_names`.

Exported unreduced voxelwise feature data must remain a sparse numeric matrix.
Dense feature data may be exported only after an explicit reducer produces a
reduced dense representation.

When coordinate-less analyses are retained, their rows must be represented as
all-zero sparse map-feature rows. When coordinate-less analyses are dropped,
all exported arrays and metadata must be aligned to the retained rows only.

**Feature Name Pairing**: Feature names are exported as a separate attribute
rather than paired with dense data (e.g., via DataFrame). This preserves
sparsity essential for neuroimaging datasets. Users working with the exported
bundle are responsible for maintaining name alignment when transforming data,
using standard sklearn patterns like `get_feature_names_out()` or explicit
DataFrame construction when densification is acceptable.

## Study Groups

Study groups must be one-dimensional labels aligned to exported rows.

- Studyset-provided study IDs are the study group source.
- MVP inputs are assumed to provide unique study IDs and unique analysis IDs.
- Missing groups must raise a clear error before export or split.

## Splitting

Grouped splitting must use study group labels. For any split:

- `set(groups[train])` and `set(groups[test])` must be disjoint.
- `data`, `target`, `groups`, and `feature_names` must be sliced with the same
  analysis-row indices, where `groups` equals `study_ids`.
- The same dataset and reproducibility setting must produce the same split.
- Too few study groups must raise a clear error before returning any split.

## Descriptor Features

Descriptor features must be aligned to `ids`.

- Numeric descriptors may be appended directly.
- Categorical/text descriptors must be rejected by default unless an explicit
  transformer or vectorizer is supplied.
- Descriptor transformers must fit only on training data before transforming
  held-out data.
- Missing descriptor values must be reported explicitly unless an explicit
  transformer handles missing values internally.
- A field selected as the prediction target must not be silently reused as a
  descriptor feature by default.

## Prediction Target

Target values must be aligned to exported rows.

- Scalar categorical targets may be strings or encoded values.
- Numeric targets must preserve numeric values.
- Raw free-text and multi-label targets must be rejected unless the caller
  supplies an explicit target transformer or label extractor.
- Missing or constant targets must be diagnosed.

## Reduction Workflows

Reduction workflows must be compatible with scikit-learn estimator workflows.

- Reducers must expose `fit`, `transform`, and preferably `fit_transform`.
- Reducers must be fit on training data only before held-out transformation.
- Reducers must not densify unreduced voxelwise inputs as an intermediary.
- Reduced output must preserve row order.
- Required initial workflows are variance thresholding, truncated SVD or an
  equivalent sparse-compatible low-rank reducer, and atlas/label aggregation
  when a compatible masker or labels image is supplied.

## Acceptance Checks

The following checks must pass in tests:

- A scikit-learn estimator can call `fit(data, target)` on exported data when a
  target is present.
- A group-aware split can be created without study leakage.
- A pipeline can fit a reducer on training data and transform held-out data.
- Exported `groups` can be passed directly to scikit-learn group splitters.
- Non-numeric descriptors fail without an explicit transformer.
- Raw free-text and multi-label targets fail without explicit target handling.
- A representative 1,000-study conversion and grouped split meets the <=3
  minute and <=5 GB peak memory budget under the configured performance check.
